# Evidence signing rehearsal: TEST_ONLY trust path

This path exercises the existing canonical manifest, exact artifact/file, multi-file decision, Ed25519 signature, public PEM hash, and externally supplied trust-pin checks. It never grants source rights, intended use, admission, export, accounting, or training. `YUG-EVIDENCE-KEY-001` was a separate production key ID, is permanently ineligible for production activation, and must not be used for rehearsal. KEY-002's active production trust pin cannot be substituted for a test pin.

`training.evidence_trust.build_test_trust_pin` constructs an in-memory `yug-evidence-test-trust-pin-v1` record. Its exact fields include `environment=TEST_ONLY`, `production_eligible=false`, `authority=NONE`, `status=TEST_ONLY`, an ID matching `YUG-EVIDENCE-TEST-KEY-[0-9]{3,}`, exact public PEM byte SHA256, relative public-key path, and canonical identity. `verify_test_trust_pin` requires an independently supplied exact identity and checks canonical bytes, duplicate fields, path containment, identity and PEM bytes. It accepts no production key ID. The production `verify_trust_pin` accepts only the unchanged production schema and ID pattern; it rejects this test schema even if someone renames, reseals, or copies the file. The [test-only schema](../../training/specifications/evidence_test_trust_pin_v1.schema.json) is structural documentation, not trust authority.

`training.evidence_test_ceremony.verify_test_bundle` requires a `development` manifest whose source ID starts with `TEST_ONLY-`, at least two exact files, bound source inventory, license, source-file bytes and a v2 decision record. The decision is synthetic shape evidence, not human approval. The function calls the ordinary `verify_bundle` for canonical manifest/artifact/Git-blob/decision checks, then verifies an Ed25519 v2 signature whose key ID exactly matches the test pin. Its result always includes `environment=TEST_ONLY`, `production_eligible=false`, `authority=NONE`. The production verifier never calls the test-pin verifier, and a production-mode manifest is rejected by the test verifier. A local bundled public key without the independently retained test-pin identity is insufficient.

## Human-operated interactive step — not executed here

Choose an isolated throwaway `TEST_ONLY` directory away from production evidence; do not put the test private key in Git. On the approved local terminal, the human operator replaces the placeholders with absolute paths and executes the following. OpenSSL prompts for a throwaway passphrase **directly in that terminal**. Never send it to Codex or ChatGPT, put it in arguments, scripts, environment files, shell history, logs, or evidence artifacts. Stop if the output is not encrypted PKCS#8 PEM. The exact saved public PEM bytes are the fingerprint target.

```powershell
& "<OPENSSL_EXE>" genpkey -algorithm ED25519 -aes-256-cbc -outform PEM -out "<TEST_ONLY_PRIVATE_KEY_PATH>"
& "<OPENSSL_EXE>" pkey -in "<TEST_ONLY_PRIVATE_KEY_PATH>" -pubout -outform PEM -out "<TEST_ONLY_PUBLIC_KEY_PATH>"
& "<OPENSSL_EXE>" pkey -pubin -in "<TEST_ONLY_PUBLIC_KEY_PATH>" -pubcheck -noout
& "<OPENSSL_EXE>" pkey -in "<TEST_ONLY_PRIVATE_KEY_PATH>" -pubout -outform PEM -out "<TEST_ONLY_DERIVED_PUBLIC_PATH>"
Get-FileHash -Algorithm SHA256 -LiteralPath "<TEST_ONLY_PUBLIC_KEY_PATH>"
Get-FileHash -Algorithm SHA256 -LiteralPath "<TEST_ONLY_DERIVED_PUBLIC_PATH>"
```

The two full SHA256 values must match. The operator then provides **only** the non-secret test directory and public PEM path/fingerprint to Codex. That is the first resume point: Codex may prepare a synthetic two-file development manifest, test pin, exact external identity pins, and domain-separated message bytes. The test source ID and decision attribution should explicitly say `TEST_ONLY NON_PRODUCTION authority=NONE`; the manifest mode is `development`, and the detached v2 signature names `YUG-EVIDENCE-TEST-KEY-001`. Strict existing manifest/decision/signature shapes cannot carry extra fields, so the test ID, development mode, synthetic content, isolated directory and test-pin markers collectively distinguish them from production. No attrs or other real source is signed.

When Codex has prepared the message file, the human runs the separate interactive signing command below. Codex must not unlock the key or receive the passphrase. The human supplies the non-secret signature-file path afterward; Codex packages its Base64 bytes into the v2 detached test signature artifact and resumes offline verification.

```powershell
& "<OPENSSL_EXE>" pkeyutl -sign -inkey "<TEST_ONLY_PRIVATE_KEY_PATH>" -rawin -in "<TEST_ONLY_MESSAGE_PATH>" -out "<TEST_ONLY_SIGNATURE_BYTES_PATH>"
```

The test verifier command, run only after the test pin and signature exist, is:

```text
python -B -m training.evidence_test_ceremony --root <TEST_ONLY_EVIDENCE_ROOT> --manifest <TEST_ONLY_MANIFEST> --expected-identity sha256:<INDEPENDENTLY_RETAINED_TEST_MANIFEST_IDENTITY> --signature <TEST_ONLY_SIGNATURE_JSON> --test-trust-pin <TEST_ONLY_TRUST_PIN> --expected-test-trust-pin-identity sha256:<INDEPENDENTLY_RETAINED_TEST_PIN_IDENTITY> --trust-root <TEST_ONLY_PUBLIC_KEY_ROOT> --openssl <OPENSSL_EXE>
```

The independent test identities must be recorded before verification from a channel separate from the mutable test bundle. The result is useful only to rehearse integrity; `signature=VERIFIED` never changes `authority=NONE`. A missing timestamp is reported `MISSING`; the target production policy remains `SIGNATURE_REQUIRED` plus `TIMESTAMP_REQUIRED`. TSA trust-anchor preparation is retained, but no production RFC 3161 response has been requested. A test signature without verified trusted time is not fully production-sealed. No TSA request is part of this rehearsal.

## Backup, recovery and disposal

The human makes two encrypted throwaway backup copies on separate test media, never beside production keys. Each recovery check occurs on an isolated/offline device: restore one backup, unlock interactively, derive a temporary public PEM, and compare its full SHA256 to the independently retained test public fingerprint. Repeat independently for the second backup; use a test-only signing challenge if needed. This proves technical recoverability only, not future physical storage security. After verification, remove the original test private key, both backup copies, and temporary recovered private/public copies by exact path under the approved local cleanup procedure. Keep only non-secret fingerprint, test pin/manifest identities, verification result, recovery result, and destruction confirmation if useful. Do not retain a passphrase or recoverable private material.
