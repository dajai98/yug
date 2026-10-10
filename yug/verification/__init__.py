"""Verification subsystem: deterministic, explicit-caller-selected
checks -- schema shape and arithmetic claims, for now.

Deliberately independent of ``yug.tools`` and ``yug.brain``: nothing
here imports either. The safe AST-evaluation pattern proven in
``yug.tools.builtin.calculator`` is ADAPTED (re-implemented as its own
copy in :mod:`yug.verification.math`), not imported, so this
subsystem has no dependency edge on Tools -- the two are meant to stay
independent per the current architecture map, and a future consumer of
Verification should not need to pull in the Tools subsystem to use it.

No automatic invocation: nothing in this package calls itself. A
caller (test code today; a future Agent/Runtime integration, NOT
implemented here) must explicitly choose a verifier and pass explicit
data to it via :class:`yug.verification.engine.VerificationEngine`.
"""
