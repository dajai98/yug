# YUG — Final Package

This archive contains the complete YUG project as it currently stands:
architecture documentation, the full source repository, and the full test
suite.

**Start here, in this order:**

1. `REPORT/YUG_FINAL_ARCHITECTURE_DECISION_REPORT.md` — the complete,
   single-file summary of verified state, frozen decisions, open items,
   risks, and the exact next step. Read this first.
2. `ARCHITECTURE/NATIVE_BRAIN_ARCHITECTURE.md` — the canonical, living
   architecture record. This is the authoritative source if it ever
   disagrees with the report above (it's updated per-milestone; the
   report is a point-in-time synthesis).
3. `ARCHITECTURE/YUG_HANDOFF.md` — a prior conversation-handoff summary,
   kept for continuity.
4. `SOURCE/TEST_STATUS.md` — **read this before trusting any test result.**
   Real `pytest` and PyTorch have not been available in the development
   environment that produced this package (no network access); see this
   file and the main report's §6 for exactly what has and hasn't been
   genuinely verified.
5. `SOURCE/` — the complete, self-contained package: `yug/`, `tests/`,
   `pyproject.toml`, and `TEST_STATUS.md`. Installable via
   `pip install -e ./SOURCE` (or copy `SOURCE/*` into a fresh repo root).

## Most important open item

`yug/native/model.py` (the native Transformer implementation) has been
written but **never executed** — PyTorch is not installed in the
environment that produced this package. Before continuing any work on
this project, install PyTorch (`pip install -e ".[native]"` from the
`SOURCE/` directory) and run `pytest SOURCE/tests/test_native_model.py -v`
to find out whether it actually works. See the final report, §1 and §6,
for full detail.

## Quick orientation

- **What works and is tested right now:** Phase 0 (config, logging, CLI),
  Phase 1 (`BrainInterface` + mock/llama.cpp adapters), NB-02 (tokenizer),
  NB-03 (model configuration) — 169 tests passing.
- **What's written but unverified:** NB-04 (the native Transformer itself).
- **What doesn't exist yet:** training pipeline, checkpoint format,
  `YugNativeAdapter`, generation/sampling, anything past NB-04.
