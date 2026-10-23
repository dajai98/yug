# YUG

The authoritative live tree is root `yug/`, `training/`, `tests/` and
`pyproject.toml`. `SOURCE/` is an ignored historical snapshot, not an install target.

From the repository root, with Python 3.11 or newer:

```powershell
python -m pip install -e ".[dev,native,dataset]"
python -B -m pytest -q
```

Native model, tokenizer, training, checkpoint and corpus automation code have been
exercised. The Phase 1 closure audit recorded 1469 passed, zero failures and two
optional hardware/model skips. This historical result does not replace fresh tests.

YUG uses its native transformer trained from scratch. The frozen target is
100,951,680 parameters, context 2048 and vocabulary 4096. This checkpoint does not
authorize production pretraining.

See [checkpoint evidence and restoration boundaries](docs/evidence/phase1/README.md).
Tests use tracked resources, not local corpora. Production acquisition still needs
verified clearance, holdout and prior-production evidence and fails closed without it.
No production corpus or model checkpoint is bundled.

`ARCHITECTURE/` and `REPORT/` retain historical decisions: respect milestone dates
and scope. The external research ledger is documentation, not corpus policy.
