# Repository Guide for Coding Agents

## Start here

- This repository contains two pipelines. Establish which pipeline the task concerns before editing.
- Read `README.md` for the project overview, `PROJECT_STATUS.md` for the dated archival handoff,
  `agent_eval/README.md` for the harness design and its rationale, and `NEXT_STEPS.md` for the
  chronological experiment log.
- Inspect the current branch and `git status`; do not assume a branch name or disturb existing work.

## Archival handoff

- Branch `mcp-metric-calc` is an intentionally broad project snapshot. It contains supported code,
  unfinished writing, notebooks, reference examples, and local-model experiments so a future
  maintainer can recover the full context. Do not treat that breadth as accidental cleanup debt.
- Treat claims in `PROJECT_STATUS.md`, `NEXT_STEPS.md`, backend pricing, and linked GitHub issues as
  dated evidence. Verify model availability, rates, infrastructure, and issue state before acting on
  them months later.
- Preserve safe WIP unless a task explicitly asks to remove or reorganize it. Secrets, credentials,
  sensitive source data, caches, and generated data-laden result pages remain excluded.

## Pipeline boundaries

- **Upstream inference (`scripts/`)** runs models over images and writes `llm_outputs` to MongoDB.
  Its configuration lives in `inputs/models.json` and `inputs/benchmarks.json`.
- **Agentic evaluation (`agent_eval/` and `analysis/`)** reads those outputs through an MCP agent,
  scores agent behavior, stores results, and builds dashboards. Use the `agentic-eval` skill for
  detailed commands, extension procedures, and task-specific gotchas.
- Do not confuse `scripts/3_create_mapping.py` with the registries under `agent_eval/registry/`.
  `model_id` names the model under test; judge backends use separate model identifiers.

## Environment and commands

- Use the repository-local environment: `source .venv/bin/activate`.
- Run module and SLURM commands from the repository root.
- Run the offline suite after changes to the agentic harness:

  ```bash
  cd agent_eval && EVAL_DISABLE_WEAVE=1 python -m pytest
  ```

- `EVAL_DISABLE_WEAVE=1` must be set for local tests so Weave is a no-op.
- Secrets belong in the ignored `.env`; use `.env.example` as the template. Never print or commit
  secret values.
- MongoDB, GPU, SLURM, and live model paths are integration surfaces. Keep decision logic pure when
  practical so it can be tested offline, and document any verification that requires infrastructure.

## Implementation rules

- Keep changes surgical and consistent with the existing module boundaries.
- Prefer data-only backend and prompt configuration where the harness already supports it.
- Keep generic runtime, runner, reporting, and storage behavior separate from task-specific tools,
  scorers, registries, and prompts.
- A task-specific MCP tool must be registered by the server and included in that task's tool subset.
- Preserve the separation between grading truth and agent-visible data. Never expose expected answers
  to the agent merely to simplify implementation or testing.
- Preserve version-keyed result behavior and client-side stamping of `eval_id` and `git_commit` unless
  the requested behavior explicitly changes those contracts.

## Behavior-first tests

- State observable behavior with concrete Given/When/Then examples before changing production code.
- For features, write and run a focused failing test before implementation. Confirm it fails because
  the behavior is missing, then make the smallest change that passes it.
- For bugs, reproduce the failure with a regression test first.
- Before refactoring untested behavior, add characterization tests that capture the current contract.
- Test stable inputs, outputs, errors, persistence effects, and CLI behavior rather than private
  implementation details. Do not weaken a valid test to fit an implementation.
- After the focused test passes, run the relevant module tests and then the full offline suite.
- Explicitly explain exceptions for documentation, mechanical metadata, disposable spikes, or
  infrastructure-only behavior, and use the strongest practical alternative check.

## Code review rules

- Flag changes that conflate upstream inference with agentic evaluation.
- Flag prompt names whose prefix selects the wrong tool set or silently falls back to metric-eval.
- Flag task tools that are registered but not included in the task's allowed tool subset.
- Flag any path that leaks grading truth such as expected dates or original values into an agent
  prompt or tool result.
- Flag result-store changes that break version-keyed comparisons, skip-existing semantics, or joins
  by `eval_id` without an explicit migration plan.
- Keep **Misrouted** distinct from **Unscored**, and keep declared metric selection distinct from a
  clean routing path in dashboards and reports.
- Reserve formatting and style enforcement for automated checks; reviews should prioritize behavior,
  data integrity, security, regressions, and missing tests.

## Git and handoff

- Never commit, push, rebase shared history, or change remote GitHub settings unless explicitly asked.
- Keep new changes and pull requests single-purpose. The existing `mcp-metric-calc` branch is the
  documented exception: it is an intentionally broad archival handoff and should not be rewritten
  merely to make its history or review scope smaller.
- At each meaningful checkpoint, report the promised behavior, why the implementation works, tests
  run (including the initial failing test), recommended review order, and remaining risks.
