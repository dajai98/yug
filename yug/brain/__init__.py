"""Brain subsystem: the replaceable language-model interface.

Per architectural invariant #1 ("the model must remain replaceable"),
nothing outside this package (and its adapters) may depend on a
specific inference backend. Callers depend only on
:class:`yug.brain.interface.BrainInterface` and the types in
:mod:`yug.brain.types`.
"""
