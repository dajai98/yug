"""NB-08: Controlled Small-Scale Language-Modeling Experiment.

See ``ARCHITECTURE/NB08_CONTROLLED_LM_EXPERIMENT.md`` for the full
experiment design, results, and interpretation.

VERIFICATION STATUS: same as the rest of ``training/`` -- written
without a PyTorch installation available in the development sandbox.
``nb08_corpus.py`` and ``training.split`` (used by this experiment)
have no PyTorch dependency and their behavior IS genuinely verified by
real test execution in that sandbox; everything downstream of
tokenizer training that touches ``YugTransformer`` (model
construction, training, validation loss, checkpointing, generation) is
NOT executed and remains [UNKNOWN] pending a real PyTorch environment.
"""
