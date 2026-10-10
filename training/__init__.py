"""YUG NB-07: from-scratch training foundation for ``yug.native``.

VERIFICATION STATUS: every module under this package was written
without a PyTorch installation available in the development sandbox
(no network access to install it -- see
``ARCHITECTURE/NB07_ARCHITECTURE_AMENDMENT.md`` and the accompanying
NB-07 environment-gate audits for the full evidence trail). **None of
this code has been executed.** It is written to be run by whoever has
a real PyTorch + pytest environment available -- install with
``pip install -e ".[native]"`` and run
``pytest tests/test_training_*.py -v``. Until that happens, every
claim about this package is [PROPOSAL] or [SOURCE-DERIVED] from the
existing ``yug.native``/``yug.tokenizer`` APIs it calls -- never
[VERIFIED BY TEST].

Architectural boundary (unchanged from the NB-07 architecture
amendment): this package may import from ``yug`` (specifically
``yug.native`` and ``yug.tokenizer``, both stable, already-existing
interfaces). Nothing under ``yug/`` may import from ``training/`` --
training is not a runtime dependency of the production system.

Package layout:

    training/config.py       TrainingConfig (deterministic, explicit)
    training/dataset.py      text -> tokens -> (input, target) sequences -> batches
    training/loss.py         next-token cross-entropy
    training/checkpoint.py   save/load/resume (no bare pickle of the
                              TrainingConfig object -- see its module
                              docstring for the exact format)
    training/loop.py         Trainer: forward/loss/backward/step, plus
                              its no-grad evaluate() method and fit()
    training/determinism.py  seed_everything()
    training/errors.py       TrainingError hierarchy (no torch dependency)
"""
