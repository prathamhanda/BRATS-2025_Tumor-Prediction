---
name: resumeNotebookRun
description: Resume and robustly execute a notebook end-to-end, patching blockers as needed.
argument-hint: Notebook path, current progress context, target milestone, constraints (env/paths/deps)
---
You are working in VS Code on a Jupyter notebook and must **continue execution forward** from the current state until you reach the user’s target milestone (or a genuine hard blocker). The user wants you to **keep running cells after you make fixes**, and to **not stop unless necessary**.

## Inputs
- The current notebook (the active editor notebook unless a path is provided): `<NOTEBOOK_PATH>`
- What “done” means (target milestone): `<TARGET_MILESTONE>`
- Current environment constraints (OS, remote kernel, GPU, permissions, read-only mounts, etc.): `<ENV_CONSTRAINTS>`
- Any known failures or flaky spots (missing packages, corrupted samples, stale env vars): `<KNOWN_BLOCKERS>`

## Operating rules
1. **Inspect before editing**: Read the notebook’s current contents and detect recent changes before applying new edits.
2. **Minimal, surgical patches**: Fix root causes, avoid unrelated refactors, preserve existing APIs and style.
3. **Execution-first**: After each fix, re-run the smallest necessary set of prerequisite cells and then continue forward.
4. **Fail-soft**: One bad sample or missing optional dependency must not crash the full run.
5. **State-aware**: Assume kernels can reset. If variables are missing, re-run required setup cells in strict order.
6. **Filesystem-safe**: Prefer writable locations for exports/artifacts; detect and avoid read-only snapshot paths.
7. **Reproducible artifacts**: Write key outputs (JSON/CSV/logs) into a stable results folder with timestamps.
8. **Concise progress updates**: After 3–5 tool calls or any major change, report what’s done and what’s next.

## Step-by-step workflow
### A) Establish notebook run context
- Grep/scan the notebook for:
  - configuration flags (e.g., `DO_EXPORT`, `RUN_*`)
  - dataset root variables / environment variables
  - “milestone” cells (e.g., folds creation, export, training commands)
  - places where hardcoded paths or permissions might fail
- Determine the expected execution order and dependencies between cells.

### B) Validate environment and paths
- Run/verify environment cells: Python version, CUDA availability (if relevant), key package imports.
- Validate dataset roots:
  - If an env var points to a non-existent path, **self-heal** by unsetting it and re-detecting.
  - Print existence checks and a short directory probe to confirm correctness.

### C) Make the pipeline robust to bad data and missing deps
- Implement or verify a **training-safe case list**:
  - build it from authoritative artifacts (e.g., fold JSON) when possible
  - explicitly track excluded/failed cases and reasons
  - ensure downstream steps always use this safe list
- Optional dependencies (radiomics, sklearn, etc.):
  - if missing, provide a fallback strategy (simple stratification, heuristic bins, or skip with warning)

### D) Continue execution forward, patching only when blocked
- Execute cells sequentially toward `<TARGET_MILESTONE>`.
- When a cell fails:
  1) Capture the full error and the immediate context (inputs, paths, key variable values).
  2) Decide if it’s a data issue, dependency issue, path/permission issue, or kernel reset.
  3) Apply the smallest fix (guard clauses, path fallback, retry logic, better error message).
  4) Re-run only the required upstream cells, then re-run the failing cell.

### E) Handle common notebook blockers (apply as needed)
- **Kernel reset / lost state**: detect via `NameError`/missing variables; re-run setup → indexing → helpers → feature/folds → export.
- **Read-only filesystem**: redirect outputs to a writable base (e.g., a workspace-local folder); confirm with a small write test.
- **Missing CLI/tooling**: detect import/CLI availability; if missing, either install (if allowed) or stop with a precise blocker.
- **Corrupted samples**: skip and log; ensure fold/training lists exclude them.

## Output expectations
- Keep moving until:
  - you reach `<TARGET_MILESTONE>`, or
  - you hit a hard blocker that cannot be resolved with code/config changes in the repo.
- At the end, provide:
  - what ran successfully
  - what changed (files/cells)
  - where artifacts were written
  - the next command(s)/cell(s) to run, if any
  - any remaining blockers with concrete remediation steps
