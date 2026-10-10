"""Tests proving the candidate configurations quoted in the NB-03 report
are real, reproducible calculations -- not numbers pasted in by hand.

None of these candidates is frozen as the production choice (see
docs/architecture/NATIVE_BRAIN_ARCHITECTURE.md, Open Decisions);
proving they're calculated correctly is what's tested here, not
endorsing any particular one.
"""

from __future__ import annotations

from yug.native import ModelConfig, estimate_parameter_count

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64

# Mirrors the candidates presented in the NB-03 report.
_CANDIDATES = {
    "A_tiny_deep": dict(vocab_size=4096, hidden_size=192, num_layers=8, num_heads=6, ffn_size=768, context_length=512),
    "B_tiny_wide": dict(vocab_size=4096, hidden_size=320, num_layers=4, num_heads=8, ffn_size=1280, context_length=512),
    "C_mid": dict(vocab_size=8192, hidden_size=256, num_layers=6, num_heads=8, ffn_size=1024, context_length=1024),
    "D_mid_deeper": dict(vocab_size=8192, hidden_size=256, num_layers=10, num_heads=8, ffn_size=1024, context_length=1024),
    "E_larger": dict(vocab_size=8192, hidden_size=384, num_layers=8, num_heads=8, ffn_size=1536, context_length=1024),
    "F_30M_target": dict(vocab_size=8192, hidden_size=512, num_layers=8, num_heads=8, ffn_size=2048, context_length=1024),
    "G_50M_ceiling": dict(vocab_size=8192, hidden_size=512, num_layers=12, num_heads=8, ffn_size=2048, context_length=1024),
}

# Expected tied-embedding totals, computed and cross-checked by hand
# against the formula in yug.native.config.estimate_parameter_count.
_EXPECTED_TIED_TOTALS = {
    "A_tiny_deep": 4_328_640,
    "B_tiny_wide": 6_228_800,
    "C_mid": 6_819_072,
    "D_mid_deeper": 9_966_848,
    "E_larger": 17_308_032,
    "F_30M_target": 29_368_832,
    "G_50M_ceiling": 41_955_840,
}


def _build(name: str, tie_embeddings: bool) -> ModelConfig:
    return ModelConfig(
        tokenizer_identity=_PLACEHOLDER_IDENTITY,
        tie_embeddings=tie_embeddings,
        **_CANDIDATES[name],
    )


def test_all_candidates_construct_without_validation_error():
    for name in _CANDIDATES:
        _build(name, tie_embeddings=True)


def test_candidate_parameter_counts_match_reported_values():
    for name, expected in _EXPECTED_TIED_TOTALS.items():
        cfg = _build(name, tie_embeddings=True)
        actual = estimate_parameter_count(cfg).total
        assert actual == expected, f"{name}: expected {expected:,}, got {actual:,}"


def test_candidates_e_f_g_fall_inside_the_10_to_50m_proof_of_capability_range():
    for name in ("E_larger", "F_30M_target", "G_50M_ceiling"):
        cfg = _build(name, tie_embeddings=True)
        total = estimate_parameter_count(cfg).total
        assert 10_000_000 <= total <= 50_000_000, f"{name}: {total:,} not in [10M, 50M]"


def test_candidates_a_through_d_fall_below_the_10m_floor():
    """Documented, not fixed: these are useful lower-bound examples,
    not disqualified designs -- just outside the specific 10-50M
    proof-of-capability target discussed in the architecture doc."""
    for name in ("A_tiny_deep", "B_tiny_wide", "C_mid", "D_mid_deeper"):
        cfg = _build(name, tie_embeddings=True)
        total = estimate_parameter_count(cfg).total
        assert total < 10_000_000, f"{name}: {total:,} unexpectedly >= 10M"


def test_untied_variant_always_has_more_parameters_than_tied():
    for name in _CANDIDATES:
        tied = estimate_parameter_count(_build(name, tie_embeddings=True)).total
        untied = estimate_parameter_count(_build(name, tie_embeddings=False)).total
        assert untied > tied
