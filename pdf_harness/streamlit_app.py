"""Streamlit control plane for the PDF extraction harness.

Run locally with:
    streamlit run pdf_harness/streamlit_app.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pandas as pd
import requests
import streamlit as st

from pdf_harness.auth import SharedPasswordAuth
from pdf_harness.bootstrap import AppContext, build_context
from pdf_harness.models import Project, ProjectMode, Run, RunStatus
from pdf_harness.secrets import CompositeSecretResolver
from pdf_harness.tools import catalog

DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"], "description": "Document or table title"},
        "rows": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
            "description": "Extracted table rows",
        },
    },
    "required": ["title", "rows"],
    "additionalProperties": False,
}


@st.cache_resource
def context() -> AppContext:
    return build_context()


def _password(ctx: AppContext) -> str:
    if ctx.settings.app_password_secret:
        return CompositeSecretResolver().get(ctx.settings.app_password_secret)
    return ctx.settings.app_password or ""


def require_auth(ctx: AppContext) -> None:
    if st.session_state.get("authenticated"):
        return
    st.title("PDF Extraction Harness")
    st.caption("Internal prototype")
    configured = _password(ctx)
    if not configured:
        st.error("Authentication is not configured. Set HARNESS_APP_PASSWORD locally or HARNESS_APP_PASSWORD_SECRET in production.")
        st.stop()
    with st.form("login"):
        value = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted and SharedPasswordAuth(configured).authenticate(value):
        st.session_state.authenticated = True
        st.rerun()
    if submitted:
        st.error("Invalid password")
    st.stop()


def _dispatch(ctx: AppContext, run_id: str) -> str:
    if ctx.settings.environment != "production":
        if not ctx.repo.queue_run(run_id):
            raise RuntimeError("run is not in a queueable state")
        execution = ctx.dispatcher.dispatch(run_id)
        ctx.repo.update("runs", run_id, {"cloud_execution_name": execution})
        return execution
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token

    audience = ctx.settings.dispatcher_audience or ctx.settings.dispatch_url
    token = id_token.fetch_id_token(Request(), audience)
    response = requests.post(
        ctx.settings.dispatch_url, json={"run_id": run_id},
        headers={"Authorization": f"Bearer {token}"}, timeout=30,
    )
    response.raise_for_status()
    return response.json().get("execution", "queued")


def _dispatch_document(ctx: AppContext, document_id: str) -> str:
    if ctx.settings.environment != "production":
        document = ctx.service.render_document(document_id)
        return f"local:{document.id}"
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token

    audience = ctx.settings.dispatcher_audience or ctx.settings.dispatch_url
    token = id_token.fetch_id_token(Request(), audience)
    response = requests.post(
        ctx.settings.dispatch_url, json={"document_id": document_id},
        headers={"Authorization": f"Bearer {token}"}, timeout=30,
    )
    response.raise_for_status()
    return response.json().get("execution", "queued")


def _projects(ctx: AppContext) -> list[dict[str, Any]]:
    return ctx.repo.find("projects")


def _project_selector(ctx: AppContext) -> Project | None:
    projects = _projects(ctx)
    if not projects:
        return None
    labels = {p["id"]: f"{p['name']} · {p['mode']}" for p in projects}
    selected = st.sidebar.selectbox(
        "Project", list(labels), format_func=lambda item: labels[item],
        index=list(labels).index(st.session_state.project_id) if st.session_state.get("project_id") in labels else 0,
    )
    st.session_state.project_id = selected
    return Project.model_validate(ctx.repo.get("projects", selected))


def page_projects(ctx: AppContext) -> None:
    st.header("Projects")
    with st.expander("Create project", expanded=not bool(_projects(ctx))):
        with st.form("create-project", clear_on_submit=True):
            name = st.text_input("Name")
            description = st.text_area("Description")
            mode = st.selectbox("Mode", [ProjectMode.EXTRACTION.value, ProjectMode.BENCHMARK.value])
            if st.form_submit_button("Create", type="primary"):
                if not name.strip():
                    st.error("Name is required")
                else:
                    project = ctx.service.create_project(name, ProjectMode(mode), description)
                    st.session_state.project_id = project.id
                    st.success("Project created")
                    st.rerun()
    rows = _projects(ctx)
    if rows:
        st.dataframe(pd.DataFrame([{
            "name": row["name"], "mode": row["mode"], "status": row["status"],
            "documents": len(row.get("document_ids", [])), "updated": row.get("updated_at"),
        } for row in rows]), use_container_width=True, hide_index=True)


def page_setup(ctx: AppContext, project: Project) -> None:
    st.header(f"Setup · {project.name}")
    tabs = st.tabs(["Connections", "Documents", "Schema", "Prompts", "Models", "Tools", "Ground truth", "Preflight"])
    with tabs[0]:
        _connections(ctx)
    with tabs[1]:
        _documents(ctx, project)
    with tabs[2]:
        _schema(ctx, project)
    with tabs[3]:
        _prompts(ctx, project)
    with tabs[4]:
        _models(ctx, project)
    with tabs[5]:
        _tools(ctx, project)
    with tabs[6]:
        _ground_truth(ctx, project)
    with tabs[7]:
        _preflight(ctx, project)


def _connections(ctx: AppContext) -> None:
    st.subheader("Platform connections")
    st.info("Credential values are never saved in Mongo. Local values live in `.env`; production values live in Google Secret Manager.")
    for label, check in (
        ("MongoDB", ctx.repo.health), ("Artifact storage", ctx.artifacts.health),
        ("Cloud dispatcher", ctx.dispatcher.health),
    ):
        ok, detail = check()
        (st.success if ok else st.error)(f"{label}: {detail}")
    st.markdown("Required production secrets: shared app password and one API-key secret per model profile. GCS and job dispatch use the Cloud Run service identity rather than downloaded service-account keys.")


def _documents(ctx: AppContext, project: Project) -> None:
    destination = f"gs://{ctx.settings.gcs_bucket}" if ctx.settings.artifact_backend == "gcs" else str(ctx.settings.local_data_dir)
    st.caption(f"Drag PDFs from your computer below. Destination: `{destination}` · limit: {ctx.settings.max_upload_mb} MB per file.")
    uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Upload to storage and queue rendering", disabled=not uploads, type="primary"):
        progress = st.progress(0)
        for index, upload in enumerate(uploads or []):
            try:
                document = ctx.service.add_document(project.id, upload.name, upload.getvalue())
                _dispatch_document(ctx, document.id)
            except Exception as exc:  # noqa: BLE001
                st.error(f"{upload.name}: {exc}")
            progress.progress((index + 1) / len(uploads))
        st.rerun()
    documents = [ctx.repo.get("documents", item_id) for item_id in project.document_ids]
    if documents:
        st.dataframe(pd.DataFrame([{
            "document_id": d["id"], "name": d["name"], "status": d["render_status"], "pages": d.get("page_count"),
            "size_mb": round(d["size_bytes"] / 1_000_000, 2), "sha256": d["sha256"][:12],
            "storage_uri": d.get("source_uri"), "error": d.get("error"),
        } for d in documents if d]), use_container_width=True, hide_index=True)
        retryable = [d for d in documents if d and d.get("render_status") in {"failed", "dispatch_unknown"}]
        if retryable:
            choice = st.selectbox("Failed render to retry", [d["id"] for d in retryable], format_func=lambda value: next(d["name"] for d in retryable if d["id"] == value))
            if st.button("Retry render"):
                _dispatch_document(ctx, choice)
                st.rerun()


def _schema(ctx: AppContext, project: Project) -> None:
    selected = ctx.repo.get("task_specs", project.task_spec_version_id or "")
    initial = selected.get("json_schema") if selected else DEFAULT_SCHEMA
    with st.form("task-schema"):
        name = st.text_input("Task name", value=selected.get("name", "pdf_extraction") if selected else "pdf_extraction")
        schema_text = st.text_area("JSON Schema", value=json.dumps(initial, indent=2), height=360)
        aggregation_prompts = {
            p["id"]: f"{p['name']} v{p['version']}"
            for p in ctx.repo.find("prompt_versions", {"project_id": project.id})
            if p.get("stage") == "aggregation"
        }
        aggregation_options = [None, *aggregation_prompts]
        selected_aggregation = st.selectbox(
            "Optional document aggregation prompt",
            aggregation_options,
            format_func=lambda value: "Disabled (page outputs only)" if value is None else aggregation_prompts[value],
        )
        if st.form_submit_button("Publish immutable schema version", type="primary"):
            try:
                spec = ctx.service.publish_task_spec(
                    project.id, name, json.loads(schema_text),
                    aggregation_prompt_version_id=selected_aggregation,
                )
                st.success(f"Published v{spec.version} · {spec.content_hash[:10]}")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))


def _prompts(ctx: AppContext, project: Project) -> None:
    with st.form("prompt-version"):
        name = st.text_input("Prompt name", value="extract_v1")
        stage = st.selectbox("Stage", ["extraction", "aggregation", "judge"])
        system = st.text_area("System template", value="You extract structured data from document pages. Return only valid JSON matching the supplied schema.")
        user = st.text_area(
            "User template",
            value=(
                "Extract the requested data from {document_name}, page {page_number}." if stage == "extraction"
                else "Combine the page results for {document_name} into one document record:\n{page_outputs}" if stage == "aggregation"
                else "Judge the prediction against the expected result. Prediction: {predicted}\nExpected: {expected}\nDeterministic scores: {deterministic_scores}"
            ),
        )
        if st.form_submit_button("Publish immutable prompt", type="primary"):
            try:
                prompt = ctx.service.publish_prompt(project.id, name, system, user, stage=stage)
                st.success(f"Published {prompt.name} v{prompt.version}")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
    prompts = [ctx.repo.get("prompt_versions", item_id) for item_id in project.extraction_prompt_version_ids]
    if prompts:
        st.dataframe(pd.DataFrame([{"name": p["name"], "version": p["version"], "hash": p["content_hash"][:10]} for p in prompts if p]), hide_index=True)
    all_prompts = ctx.repo.find("prompt_versions", {"project_id": project.id})
    choices = {p["id"]: f"{p['name']} v{p['version']}" for p in all_prompts if p.get("stage") == "extraction"}
    if choices:
        active = st.multiselect("Active prompt versions", list(choices), default=[p for p in project.extraction_prompt_version_ids if p in choices], format_func=lambda value: choices[value])
        if st.button("Save prompt selection"):
            ctx.service.select_prompts(project.id, active)
            st.rerun()


def _models(ctx: AppContext, project: Project) -> None:
    st.caption("Connectors, endpoint hosts, allowed model IDs, and secret bindings are configured by an administrator. Researchers cannot enter credential references or URLs.")
    connectors = ctx.service.connectors
    if not connectors:
        st.error("No administrator-approved LLM connectors are configured.")
        return
    with st.form("model-profile"):
        name = st.text_input("Profile name", value="stanford_model")
        connector_id = st.selectbox(
            "Approved connector", list(connectors),
            format_func=lambda value: connectors[value].label,
        )
        model_id = st.selectbox("Approved model", connectors[connector_id].allowed_models)
        params = st.text_area("Request parameters (JSON)", value='{"temperature": 0}')
        left, right = st.columns(2)
        input_price = left.number_input("Input $ / 1M tokens", min_value=0.0, value=0.0)
        output_price = right.number_input("Output $ / 1M tokens", min_value=0.0, value=0.0)
        if st.form_submit_button("Save immutable model profile", type="primary"):
            try:
                ctx.service.publish_model(
                    project.id, name, connector_id, model_id, json.loads(params),
                    input_price, output_price,
                )
                st.success("Model profile saved; test it before preflight")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
    for item_id in project.model_profile_version_ids:
        model = ctx.repo.get("model_profiles", item_id)
        if not model:
            continue
        cols = st.columns([4, 1])
        cols[0].write(f"**{model['name']} v{model['version']}** · `{model['model_id']}` · {'✅ tested' if model.get('tested_ok') else 'not tested'}")
        if cols[1].button("Test", key=f"test-{item_id}"):
            with st.spinner("Testing model credential and endpoint..."):
                try:
                    ok, detail = ctx.service.test_model(item_id)
                    (st.success if ok else st.error)(detail)
                    if ok:
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
    all_models = ctx.repo.find("model_profiles", {"project_id": project.id})
    model_choices = {m["id"]: f"{m['name']} v{m['version']} · {m['model_id']}" for m in all_models}
    if model_choices:
        active_models = st.multiselect("Active model profiles", list(model_choices), default=[m for m in project.model_profile_version_ids if m in model_choices], format_func=lambda value: model_choices[value])
        if st.button("Save model selection"):
            ctx.service.select_models(project.id, active_models)
            st.rerun()


def _tools(ctx: AppContext, project: Project) -> None:
    definitions = catalog()
    selected = st.multiselect(
        "Trusted built-in MCP functions",
        sorted(definitions), default=["validate_schema", "normalize_text", "parse_date", "coerce_number", "deduplicate_list"],
        format_func=lambda name: f"{name} — {definitions[name].description}",
    )
    if st.button("Publish tool profile", type="primary"):
        try:
            profile = ctx.service.publish_tool_profile(project.id, "default_extraction_tools", selected)
            st.success(f"Published tool profile v{profile.version}")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    st.caption("Users may enable reviewed tools only. Adding executable tools requires a code deployment.")


def _ground_truth(ctx: AppContext, project: Project) -> None:
    if project.mode != ProjectMode.BENCHMARK:
        st.info("Ground truth is not required in extraction mode.")
        return
    st.markdown("CSV must contain `document_id`, optional `page_number`, and one column per schema field. JSON accepts an array of `{document_id, page_number, values}` objects.")
    upload = st.file_uploader("Ground truth CSV or JSON", type=["csv", "json"])
    if st.button("Import and validate labels", disabled=not upload, type="primary"):
        try:
            labels = ctx.service.import_ground_truth(project.id, upload.getvalue(), upload.name)
            st.success(f"Imported {len(labels)} label rows for review.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    labels = ctx.repo.find("ground_truth", {"project_id": project.id})
    if labels:
        st.dataframe(pd.DataFrame([{
            "document_id": row["document_id"], "page": row.get("page_number"),
            "revision": row["revision"], "approved": row["approved"],
            "errors": "; ".join(row.get("validation_errors", [])),
        } for row in labels]), use_container_width=True, hide_index=True)
        choices = {row["id"]: f"{row['document_id']} · page {row.get('page_number') or 'document'} · revision {row['revision']}" for row in labels}
        selected_id = st.selectbox("Review label", list(choices), format_func=lambda value: choices[value])
        selected = ctx.repo.get("ground_truth", selected_id)
        with st.form("review-label"):
            values_text = st.text_area("Values", json.dumps(selected.get("values", {}), indent=2), height=220)
            approval = st.checkbox("Approve this revision", value=bool(selected.get("approved")))
            if st.form_submit_button("Save new revision", type="primary"):
                try:
                    revision = ctx.service.revise_ground_truth(selected_id, json.loads(values_text), approval)
                    st.success(f"Saved revision {revision.revision}")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))


def _preflight(ctx: AppContext, project: Project) -> None:
    report = ctx.service.readiness(project.id)
    st.subheader("Readiness checklist")
    for check in report.checks:
        (st.success if check.passed else st.error)(f"{check.label}: {check.detail}")
    if st.button("Run automatic single-page preflight", disabled=not report.ready, type="primary"):
        try:
            run = ctx.service.create_run(project.id, preflight=True)
            _dispatch(ctx, run.id)
            st.success(f"Preflight launched: {run.id}")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    preflights = [r for r in ctx.repo.find("runs", {"project_id": project.id, "is_preflight": True})]
    if preflights:
        latest = Run.model_validate(preflights[0])
        st.write(f"Latest preflight: **{latest.status.value}**")
        if latest.preflight_result:
            (st.success if latest.preflight_result.passed else st.error)(
                "Schema-valid result; full batch can be approved." if latest.preflight_result.passed else "; ".join(latest.preflight_result.errors)
            )
            if latest.preflight_result.extraction_id:
                extraction = ctx.repo.get("extractions", latest.preflight_result.extraction_id)
                st.json(extraction.get("output") if extraction else {})
                if extraction:
                    cols = st.columns(4)
                    cols[0].metric("Input tokens", extraction.get("input_tokens", 0))
                    cols[1].metric("Output tokens", extraction.get("output_tokens", 0))
                    cols[2].metric("Latency", f"{extraction.get('latency_seconds', 0):.1f}s")
                    cols[3].metric("Est. cost", f"${extraction.get('estimated_cost_usd', 0):.4f}")


def page_runs(ctx: AppContext, project: Project) -> None:
    st.header(f"Runs · {project.name}")
    preflights = [Run.model_validate(r) for r in ctx.repo.find("runs", {"project_id": project.id, "is_preflight": True})]
    approved = next((r for r in preflights if r.preflight_result and r.preflight_result.passed), None)
    judge_prompts = {
        p["id"]: f"{p['name']} v{p['version']}"
        for p in ctx.repo.find("prompt_versions", {"project_id": project.id})
        if p.get("stage") == "judge"
    }
    models = {
        m["id"]: f"{m['name']} v{m['version']}"
        for m in ctx.repo.find("model_profiles", {"project_id": project.id})
        if m.get("tested_ok")
    }
    judge_enabled = st.checkbox(
        "Also run experimental LLM judge (benchmark mode only)",
        disabled=project.mode != ProjectMode.BENCHMARK or not judge_prompts or not models,
    )
    judge_model = st.selectbox("Judge model", list(models), format_func=lambda value: models[value], disabled=not judge_enabled) if models else None
    judge_prompt = st.selectbox("Judge prompt", list(judge_prompts), format_func=lambda value: judge_prompts[value], disabled=not judge_enabled) if judge_prompts else None
    if approved and st.button("Approve preflight and launch full model × prompt matrix", type="primary"):
        try:
            batch = ctx.service.approve_preflight(
                approved.id, llm_judge_enabled=judge_enabled,
                judge_model_profile_version_id=judge_model if judge_enabled else None,
                judge_prompt_version_id=judge_prompt if judge_enabled else None,
            )
            execution = _dispatch(ctx, batch.id)
            st.success(f"Batch {batch.id} launched as {execution}")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    runs = ctx.repo.find("runs", {"project_id": project.id})
    if not runs:
        st.info("No runs yet.")
        return
    st.dataframe(pd.DataFrame([{
        "run_id": r["id"], "preflight": r.get("is_preflight"), "status": r["status"],
        "progress": f"{r.get('completed_items', 0) + r.get('failed_items', 0)}/{r.get('total_items', 0)}",
        "failures": r.get("failed_items", 0), "tokens": r.get("input_tokens", 0) + r.get("output_tokens", 0),
        "cost_usd": round(r.get("estimated_cost_usd", 0), 5), "created": r.get("created_at"),
    } for r in runs]), use_container_width=True, hide_index=True)
    selectable = {r["id"]: r for r in runs}
    selected = st.selectbox("Run detail", list(selectable))
    run = selectable[selected]
    if run["status"] in {RunStatus.DISPATCHING.value, RunStatus.DISPATCH_UNKNOWN.value, RunStatus.QUEUED.value, RunStatus.RUNNING.value} and st.button("Request cancellation"):
        ctx.repo.update("runs", selected, {"status": RunStatus.CANCEL_REQUESTED.value})
        st.warning("Cancellation requested")
        st.rerun()
    extractions = ctx.repo.find("extractions", {"run_id": selected})
    if extractions:
        st.download_button(
            "Download extraction JSON", json.dumps(extractions, default=str, indent=2),
            file_name=f"run-{selected}-extractions.json", mime="application/json",
        )
        for extraction in extractions[:20]:
            with st.expander(f"Page {extraction.get('page_number')} · {extraction['model_profile_version_id'][:8]} · {'valid' if extraction['schema_valid'] else 'invalid'}"):
                left, right = st.columns([1, 1])
                document = ctx.repo.get("documents", extraction["document_id"])
                page_uri = (document or {}).get("page_uris", {}).get(str(extraction.get("page_number")))
                if page_uri:
                    try:
                        left.image(ctx.artifacts.get(page_uri), caption=f"{document['name']} · page {extraction.get('page_number')}", use_container_width=True)
                    except Exception as exc:  # noqa: BLE001
                        left.warning(f"Page preview unavailable: {type(exc).__name__}")
                right.json(extraction.get("output"))
                if extraction.get("validation_errors"):
                    st.error("; ".join(extraction["validation_errors"]))
                evaluations = ctx.repo.find("evaluations", {"extraction_id": extraction["id"]})
                if evaluations:
                    st.markdown("**Deterministic benchmark scores**")
                    st.json(evaluations[0].get("deterministic", {}))
                    if evaluations[0].get("llm_judge"):
                        st.markdown("**Experimental LLM judge (separate from deterministic quality)**")
                        st.json(evaluations[0]["llm_judge"])
                label_key = f"{extraction['document_id']}:{extraction.get('page_number')}"
                label_id = run.get("snapshot", {}).get("ground_truth_ids", {}).get(label_key)
                if label_id:
                    label = ctx.repo.get("ground_truth", label_id)
                    st.markdown("**Snapshotted ground truth**")
                    st.json((label or {}).get("values", {}))
                st.text_area("Raw provider response", extraction.get("raw_response") or "", disabled=True, key=f"raw-{extraction['id']}")
                if extraction.get("repair_response"):
                    st.text_area("Structured-output repair response", extraction["repair_response"], disabled=True, key=f"repair-{extraction['id']}")


def page_dashboard(ctx: AppContext, project: Project) -> None:
    st.header(f"Dashboard · {project.name}")
    project_runs = ctx.repo.find("runs", {"project_id": project.id})
    run_labels = {r["id"]: f"{r['created_at']} · {r['status']} · {'preflight' if r.get('is_preflight') else 'batch'}" for r in project_runs}
    selected_runs = st.multiselect(
        "Runs included", list(run_labels),
        default=[r["id"] for r in project_runs if not r.get("is_preflight") and r.get("status") in {"completed", "completed_with_errors"}] or ([project_runs[0]["id"]] if project_runs else []),
        format_func=lambda value: run_labels[value],
    )
    extractions = [
        item for item in ctx.repo.find("extractions", {"project_id": project.id})
        if item["run_id"] in selected_runs
    ]
    if not extractions:
        st.info("Run a preflight or batch to populate the dashboard.")
        return
    evaluations = {e["extraction_id"]: e for e in ctx.repo.find("evaluations")}
    models = {m["id"]: m["name"] for m in ctx.repo.find("model_profiles", {"project_id": project.id})}
    prompts = {p["id"]: f"{p['name']} v{p['version']}" for p in ctx.repo.find("prompt_versions", {"project_id": project.id})}
    rows = []
    for extraction in extractions:
        evaluation = evaluations.get(extraction["id"], {})
        deterministic = evaluation.get("deterministic", {})
        judge = evaluation.get("llm_judge") or {}
        rows.append({
            "model": models.get(extraction["model_profile_version_id"], extraction["model_profile_version_id"]),
            "prompt": prompts.get(extraction["prompt_version_id"], extraction["prompt_version_id"]),
            "schema_valid": extraction["schema_valid"],
            "quality": deterministic.get("overall_score"),
            "coverage": deterministic.get("required_field_coverage"),
            "latency": extraction.get("latency_seconds", 0),
            "tokens": extraction.get("input_tokens", 0) + extraction.get("output_tokens", 0),
            "cost": extraction.get("estimated_cost_usd", 0),
            "judge_score": (judge.get("result") or {}).get("score"),
            "judge_cost": judge.get("estimated_cost_usd", 0),
        })
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "prompt"], dropna=False).agg(
        runs=("schema_valid", "size"), schema_valid_rate=("schema_valid", "mean"),
        quality=("quality", "mean"), coverage=("coverage", "mean"),
        avg_latency=("latency", "mean"), total_tokens=("tokens", "sum"), total_cost=("cost", "sum"),
        judge_score=("judge_score", "mean"), judge_cost=("judge_cost", "sum"),
    ).reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.bar_chart(summary.set_index(["model", "prompt"])[["quality", "schema_valid_rate"]])


def page_traces(ctx: AppContext, project: Project) -> None:
    st.header(f"Agent trace · {project.name}")
    runs = ctx.repo.find("runs", {"project_id": project.id})
    if not runs:
        st.info("No traces yet.")
        return
    run_id = st.selectbox("Run", [r["id"] for r in runs], key="trace-run")
    items = ctx.repo.find("work_items", {"run_id": run_id})
    if not items:
        st.info("No work items for this run.")
        return
    labels = {i["id"]: f"page {i.get('page_number')} · {i['status']} · {i['id'][:8]}" for i in items}
    work_id = st.selectbox("Work item", list(labels), format_func=lambda value: labels[value])
    events = sorted(ctx.repo.find("traces", {"work_item_id": work_id}), key=lambda e: e["sequence"])
    for event in events:
        with st.expander(f"{event['sequence']:02d} · {event['event_type']} · {event['name']}", expanded=True):
            if event.get("duration_seconds") is not None:
                st.caption(f"{event['duration_seconds']:.2f}s")
            st.json(event.get("payload", {}))


def main() -> None:
    st.set_page_config(page_title="PDF Extraction Harness", page_icon="📄", layout="wide")
    try:
        ctx = context()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Application startup failed: {type(exc).__name__}: {exc}")
        st.info("Check HARNESS_MONGO_URI, HARNESS_MONGO_DB, artifact backend settings, and Google application credentials.")
        st.stop()
    require_auth(ctx)
    st.sidebar.title("PDF Harness")
    page = st.sidebar.radio("Navigate", ["Projects", "Setup", "Runs", "Dashboard", "Agent trace"])
    project = _project_selector(ctx)
    if st.sidebar.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    if page == "Projects":
        page_projects(ctx)
    elif not project:
        st.info("Create a project first.")
    elif page == "Setup":
        page_setup(ctx, project)
    elif page == "Runs":
        page_runs(ctx, project)
    elif page == "Dashboard":
        page_dashboard(ctx, project)
    else:
        page_traces(ctx, project)


if __name__ == "__main__":
    main()
