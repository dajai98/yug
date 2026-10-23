# Autonomous production runner — validation completion

Date: 2026-10-10 (Asia/Riyadh).

Verdict: READY for the existing approved immutable inventory. This is not a declaration of 2B corpus readiness or authorization for pretraining.

Resumed committed runner at 98abb2b. Previous test process was unavailable, so the production-focused suite was rerun. No new batch, source review, acquisition or training was started.

Two confirmed defects fixed:
- Runner progress reports inherited the coordinator identity after adding fields. Reports are now sealed after all additions, including final export metadata. The CLI regression verifies the seal.
- Windows subprocess tests could inherit an invalid stdin handle (WinError 6). Test subprocesses now explicitly use DEVNULL. Production subprocess behavior was not changed.

Tracked files changed:
- training/production_build.py
- tests/test_production_build.py

Final production-focused command:
python -B -m pytest -q tests/test_production_build.py tests/test_production_acquisition.py tests/test_production_spec.py tests/test_rep_corpus_policy.py tests/test_rep_corpus_sources.py tests/test_rep_corpus_license_verify.py tests/test_training_corpus_prep.py tests/test_representative_mini_corpus.py tests/test_production_pipeline.py
Result: 296 passed, 0 failed, 0 skipped in 47.52 seconds.

Full regression: python -B -m pytest -q
Result: 1193 passed, 0 failed, 2 skipped in 103.20 seconds.
Skips: optional real llama.cpp model and CUDA FP16 GradScaler resume, unavailable in this environment.

git diff --check: passed.

Coverage includes fresh-process crash/resume, interrupted download, normalization/tokenization/shard recovery, duplicate protection, deterministic replay, all-category accounting, target boundary/cache reuse, exhausted inventory, unknown-license skipping, and corruption rejection. The initially added progress-seal assertion failed before the correction; final suites pass. A separate runner rerun exposed the Windows stdin harness failure, also fixed and verified.

Real Batch 1 resume validation explicitly prohibited Fetcher calls and native tokenizer encode calls. It reached the existing 365,997-token target without changing any of 1,106 raw/normalized/cache files (hashes and modification times checked). Before and after: 426 planned, 126 ACCEPTED, 300 PREPARED. No network downloads or retokenization. The 365,997 count is the active Batch 1 ledger; older Requests artifacts are not added again.

Evidence: verify_existing.py and resume_verification.json. Existing Batch 1 runner progress and token-prefix shard artifacts were validated/refreshed, without advancing admission. See resume_verification.json for their identities and export path. These local artifacts are ignored by Git under existing policy.

Next command (from C:\YUG\yug1) to continue only the existing approved inventory:

python -B -m training.production_build --spec artifacts/production_batch_1_v1/admission/spec.json --registry artifacts/production_batch_1_v1/admission/registry.json --plan artifacts/production_batch_1_v1/admission/plan.jsonl --directory artifacts/production_batch_1_v1/production --target-tokens 10000000 --resume

The target is a ceiling, not a promise of yield or permission to expand this inventory. This command was NOT executed in this validation. Exit 3 means target boundary or inventory exhaustion; inspect runner/progress.json. It cannot fetch other sources automatically.

Remaining limitations: the cleared inventory is insufficient for a 10M validation target; the base registry is not a substitute for scoped per-file clearance. New ambiguous licensing still needs a policy/evidence decision. Million-file/2B-scale performance remains unmeasured. Global dedup settles across the fixed inventory before accounting, so a token target is not a download-byte bound. No remaining failure was found in the tested approved-inventory runner contract.
