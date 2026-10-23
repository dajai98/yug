# MIT heading presentation fix

Date: 2026-10-14 (Asia/Riyadh).

Changed training/source_approval.py and tests/test_source_approval.py. Other
working-tree changes were preserved. The MIT root parser accepts an optional
MIT License/The MIT License title before or after copyright notices. Both retained
test repositories omit the title entirely. Copyright attribution and the complete
whitespace-normalized MIT terms remain mandatory; extra restrictions, modified or
truncated clauses, duplicate headings and missing notices fail closed. No filename,
license-family, project-provenance, file-level evidence or downstream rule changed.

Approval policy identity advanced to package-or-mit-attributed-file-grants-v3 so
old sealed decisions are not silently reinterpreted. Fresh artifacts preserve the
previous REVIEW receipts. No downloads, normalization or tokenizer changes.

| Source | Pin | Cleared | REVIEW | EXCLUDED | New accepted tokens |
|---|---|---:|---:|---:|---|
| hynek/doc2dash | 9edcc6448189cc02046f7d329c386c0a7464aa1f | 9 | 52 | 3 | 9,926 train |
| hynek/pem | 91ec16ac3b1915ebf0009d203fb7165533d6d943 | 4 | 37 | 0 | 5,004 validation |
| Total | | 13 | 89 | 3 | 14,930 |

Remaining REVIEW reasons: 72 missing/conflicting file attribution, 15 outside
supported scope, one additional provenance/license signal, one generated-material
signal. Exclusions: one binary/media extension, one denied path, one non-UTF8 file.

All 13 cleared files passed the existing production pipeline. Only new grants were
executed; 461 carried-forward plan entries were not re-admitted. Retained split:
Doc2dash train (Linux/systems/tools, CLI), PEM validation (security, secure development).
New training tokens: 9,926; new validation tokens: 5,004. Production training total:
407,479 + 9,926 = **417,405**. Validation tokens are not credited to that total.

Cross-batch dedup checked 520 prior indexed items and found no conflicts. Production
retained all 59 holdout references. Prior production database hashes were unchanged.
Four-item bounded checkpoints reached COMPLETE. Replaying with fetch and encode
disabled preserved accounting and both shard identities. Raw cached bytes were
verified and no new payload download was performed.

Shard checksums:
- train-000000.bin: 9,926 tokens, sha256:1a378eaf0c35e79821d72d270530c03de04d423f4c6a1d92c0ae22cc798aa10f
- validation-000000.bin: 5,004 tokens, sha256:e6105cfd903eebe827f3d6f7d8cfc7871a6f831d726e2351082de1da9ffb63ce

Full production receipt: production/result.json. Evidence, exact file grants,
admission receipts and checkpoints are retained in this directory. classify.py and
process_approved.py invoke existing APIs; neither creates a new approval route.

Focused command:
`python -B -m pytest -q tests/test_source_approval.py tests/test_clearance_evidence.py tests/test_source_admission.py tests/test_conditional_license.py`

109 passed, 0 failed, 0 skipped in 84.37 seconds.

Full `python -B -m pytest -q`: **1455 passed, 0 failed, 2 skipped** in
360.28 seconds. Existing skips require a real GGUF model and CUDA FP16 hardware.
`git diff --check` passed. Deterministic clearance/admission replay matched.

Final result: **READY**. No implementation blocker remains for this bounded
parser correction and processed scope. The 89 REVIEW files remain unapproved.

No corpus allocation, architecture, tokenizer or training semantics changed.
No pretraining started. Remaining REVIEW files are not approved.
