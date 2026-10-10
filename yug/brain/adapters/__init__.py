"""Concrete BrainInterface implementations.

Each adapter owns all backend-specific logic. Core YUG code must never
import from this package directly except to construct an adapter at
startup (wiring/composition root) -- all runtime use goes through
:class:`yug.brain.interface.BrainInterface`.
"""
