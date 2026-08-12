# PDF Extraction Harness

This package generalizes the repository's TV-guide pipeline into a project-based,
page-first PDF-to-structured-data harness. Streamlit is the control plane, MongoDB is
the durable system of record, GCS stores immutable artifacts, and Cloud Run Jobs execute
the model × prompt × page work matrix.

## Local start

1. Install `requirements-dev.txt` in the repo-local virtual environment.
2. Copy the harness variables from `.env.example` into the gitignored `.env`.
3. Set `HARNESS_MONGO_URI` to a development Mongo database and keep
   `HARNESS_ARTIFACT_BACKEND=local`.
4. Put the Stanford key in `.env`. Administrators may define additional approved
   connector/model/secret bindings with `HARNESS_LLM_CONNECTORS_JSON`.
5. Start Streamlit:

```bash
source .venv/bin/activate
streamlit run pdf_harness/streamlit_app.py
```

To preview the interface before Mongo credentials are available, use the development-only
in-memory repository. Projects and uploads created in this mode disappear when Streamlit
restarts, so do not use it for real runs:

```bash
HARNESS_REPOSITORY_BACKEND=memory \
HARNESS_APP_PASSWORD=preview-only \
streamlit run pdf_harness/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

The local dispatcher executes synchronously for easy debugging. Production Streamlit
submits to the private dispatcher function, which starts a Cloud Run Job. Run a queued
job manually with `HARNESS_RUN_ID=<uuid> python -m pdf_harness.worker`.
Render a queued document manually with
`HARNESS_DOCUMENT_ID=<uuid> python -m pdf_harness.worker`.

The optional task-scoped MCP server is:

```bash
HARNESS_RUN_ID=<uuid> python -m pdf_harness.mcp_server
```

It exposes only reviewed tools and refuses work-item access outside `HARNESS_RUN_ID`.

## Workflow

1. Create an extraction or benchmark project.
2. Upload PDFs from the browser; production stores the source in GCS immediately and
   queues page rendering on a durable Cloud Run Job. Sources and pages receive
   content-addressed artifact paths.
3. Publish an object JSON Schema, prompts, model profiles, and a built-in tool profile.
4. Test every model profile. The database records only its Secret Manager reference.
5. For benchmark mode, import CSV/JSON labels and explicitly approve a revision.
6. Resolve every readiness check and run the automatic single-page preflight.
7. Review output, validation, latency, tokens, cost, and trace, then approve the batch.
8. Optionally aggregate page outputs into document records and/or run a separately
   reported LLM judge on labeled data.
9. Compare paired model/prompt results in the dashboard or download run JSON.

Published task, prompt, model, and tool versions are immutable. A run stores their exact
IDs plus label revisions and code version. Work items use atomic leases and deterministic
identities, making dispatch and retry safe.

## Credentials

Never paste values into chat, source, Mongo, or Terraform state.

| Credential | Local development | Production |
|---|---|---|
| Shared app password | `HARNESS_APP_PASSWORD` in `.env` | `HARNESS_APP_PASSWORD_SECRET` resource reference |
| Mongo URI | `HARNESS_MONGO_URI` in `.env` | Cloud Run secret environment binding |
| LLM API key | approved connector binding (Stanford key by default) | Secret Manager resource in the admin connector registry |
| GCS access | Application Default Credentials | attached app/worker service accounts |
| Dispatcher | local worker | OIDC from app service account to private Gen-2 function |
| W&B Weave | `WANDB_API_KEY` + `HARNESS_WEAVE_PROJECT` | Secret environment binding; optional |

W&B is a best-effort sanitized observability mirror. Mongo remains authoritative, and
W&B failures never change run status. See `deploy/README.md` for GCP setup and IAM.

## Collections

The new database uses `projects`, `documents`, version registries, `ground_truth`, `runs`,
`work_items`, `extractions`, `evaluations`, `traces`, and `audit_events`. It does not modify
the legacy `llm_outputs` or `agentic_runs` collections.

## Tests

```bash
EVAL_DISABLE_WEAVE=1 python -m pytest pdf_harness/tests agent_eval -q
```

Tests use in-memory repositories and fake model/secret adapters; they make no network or
credential calls.
