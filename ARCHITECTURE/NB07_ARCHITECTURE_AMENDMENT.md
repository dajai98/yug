# Architecture Amendment — NB-07: From-Scratch Foundation Brain Authorized

**Status: AUTHORIZED by explicit project owner decision.**
**Supersedes: the "no foundation model training from scratch" decision below, for this repository going forward.**

---

## Previous decision

> No foundation model training from scratch.

This decision originated in the outer `YUG_FINAL_ARCHITECTURE_REPORT` planning
package's `33_FINAL_ARCHITECTURE_DECISION.md` (Frozen Decision #2), which was
used as an external reference during repository audits but was never part of
this repository's own shipped documentation. **[VERIFIED IN REPOSITORY]** —
no file under this repository's `README.md`, `REPORT/`, or `ARCHITECTURE/`
ever stated this constraint; it was only ever an external planning document's
position, surfaced during audit conversations, not a decision recorded here.

Separately, **[VERIFIED IN REPOSITORY]** — this repository's own
`ARCHITECTURE/NATIVE_BRAIN_ARCHITECTURE.md`, Architectural Decision #16,
already authorized building `yug/native/` (config, model, generation) as a
from-scratch Transformer track, prior to and independent of this amendment.
That existing authorization is consistent with, not contradicted by, the
decision below — this amendment formalizes and extends it into training
infrastructure specifically, and resolves in writing the external/internal
documentation conflict multiple prior audits of this repository flagged.

## New decision

> **YUG will train and own its foundation brain from scratch.**

YUG owns: model architecture, tokenizer, training pipeline, dataset/data
pipeline, checkpoint format and lifecycle, and inference/runtime integration
for its native model. The native model (`yug/native/`) is the long-term
canonical brain. External model backends (currently: `LlamaCppAdapter`) may
be retained as optional baselines, bootstrapping, debugging, or comparison
tools — they are explicitly not YUG's final brain, and their removal is a
separate future decision, not implied by this one.

## Reason

Explicit architectural decision by the project owner, communicated directly
as an override of the previous "no foundation training" position. No
technical motivation is asserted here beyond that instruction — this
document records the decision, not a justification invented for it.

## Consequences

- Training (`training/` or equivalent, not yet created — see NB-07 status
  below) becomes a first-class subsystem of this repository.
- `yug/native/` (`ModelConfig`, `YugTransformer`, `generation.py`) becomes
  the strategic core of YUG's brain, superseding its previous status as an
  unauthorized/legacy track relative to the external planning document.
- `LlamaCppAdapter` and `BrainInterface`'s external-backend path become
  optional baselines/development tools, not YUG's permanent brain — but
  remain in place; **no removal was performed as part of this amendment**.
- Training infrastructure must remain architecturally separate from runtime:
  `training -> yug` is an allowed dependency direction; `yug -> training` is
  not. **[VERIFIED IN REPOSITORY]** — no `training/` package exists yet, so
  this boundary has nothing to violate yet; it is recorded here as a
  constraint on whatever is built next.
- Checkpoint lifecycle, dataset handling, and evaluation become first-class
  concerns for this repository, not yet implemented (see companion audit
  report for this milestone for exact current-state evidence).

## Scope note

This document records the architecture decision only. It does not itself
implement any training infrastructure. See this milestone's audit report
for the current verified repository state, environment constraints, and
implementation status of NB-07 (YUG From-Scratch Training Foundation).
