# YUG Self-Improvement Principle (design principle only)

**Status: RECORDED — NOT IMPLEMENTED. NOT AUTHORIZED FOR IMPLEMENTATION.**
Authority: project owner. Recorded during Stage 1C of the representative-corpus expansion.
No agent, self-modification, sandbox runner, or approval tooling exists or is requested by this document.

## Principle

YUG must not receive unrestricted authority to rewrite or upgrade itself. Any future
self-improvement capability must follow this controlled flow, and every step that changes
code or configuration must be gated by explicit owner approval:

1. Detect a problem.
2. Analyze the repository.
3. Propose a change.
4. Apply the change **only inside an isolated sandbox / worktree**.
5. Run tests and checks.
6. Present the diff and the evidence to the owner.
7. Require **explicit owner approval**.
8. Apply the approved change.
9. Verify.
10. Maintain a rollback path.

## Constraints that follow from the principle (for any future design)

- No change reaches the live repository, a model, or a runtime configuration without step 7.
- The sandbox cannot write to the live tree, and its evidence (diff, test output) is produced
  outside the proposing component's control.
- A rollback path exists before step 8 runs, not after.
- Approval is per change; no standing approval, no self-granted permission.
- This principle does not change the current boundary: `yug/` must not import `training/`,
  and nothing in the data-pipeline stages may modify production code.

## Not covered here

Agent architecture, tool permissions, sandbox technology, review UI, and rollback mechanics are
undecided and out of scope. They need their own design and owner approval before any implementation.