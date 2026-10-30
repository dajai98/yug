# Phase 2 recorded NOTICE and attribution integrity v1

This is an integrity contract for YUG's retained source-approval evidence, not
an independent legal determination. The controlled source-approval workflow
remains the authority for license scope and applicable obligations.

New production plan items name the sealed `source_clearance_identity`. Admission,
verification, resume, accounting and export reopen that source clearance and
require its source/revision/manifest scope, attribution and complete evidence
reference list to match the item. Every referenced evidence file is rehashed.
The per-file manifest remains the authority for the item's license ID and
content hash. A changed source clearance needs a fresh item, plan and external
plan/ownership approval; merely resealing local metadata is insufficient.

Completed historical plans without this new field are not rewritten or newly
admitted. Their existing externally pinned plan and retained item identities
remain the compatibility authority. The legacy route is limited to terminal
builds with the original plan identity and no new corpus reservation; it still
requires exact source/revision and evidence-reference agreement. A historical
item's approved attribution text is preserved even if the older source
clearance used a different summary. Generic historical shard decoding does
not gain intended-use authorization.

For a new prepared export, `notice_attribution.json` is assembled from **all
and only accepted items**, grouped by identical source obligations and sorted
by source/revision/license/obligation identity. Each entry records source ID,
revision, locator, license ID, approved attribution, complete hash-bound
license evidence, source-clearance identity, any separately referenced NOTICE
file, applicable conditional approval/obligations and accepted-item count.
`UNDETERMINED` means no separate NOTICE applicability was established by the
retained evidence; it does not assert that no notice is legally required.
Requests has a retained NOTICE reference. Cobra's retained evidence has a
LICENSE reference but no separate NOTICE reference; this contract does not
invent one.

The canonical JSON has schema `yug-notice-attribution-v1` and a content-derived
identity. `prepared.json` binds its filename, schema, identity and byte SHA256;
the shard-embedded prepared manifest inherits that binding. Production export
reuse and authorized replay rederive the inventory from accepted records,
check exact canonical content and hash, and refuse absent, changed, extra or
missing entries. An existing historical export is never rewritten in place;
use a fresh destination for a newly bound export. The generic shard decoder
remains an integrity-only tool and does not grant production authorization.

This artifact proves consistency with YUG's approved retained records. It does
not prove that those records are legally complete or that an operator-supplied
approval is authentic. Revocation and supersession governance are separate
work and are not implemented by this policy.
