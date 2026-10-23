# Conditional license approval contract

The existing ALLOWED set in source_prepare.LICENSES is unchanged. Supported
conditional identifiers are exactly CC-BY-SA-3.0, CC-BY-SA-4.0, MPL-2.0 and
Apache-2.0 WITH LLVM-exception. Other versions, ports and license expressions
are not implicitly recognized.

Discovery records conditional metadata claims as REVIEW, with a conditional_license
descriptor containing exact license, required obligation summary and approval_required.
GitHub catalog entries have status CONDITIONAL and may enter review-cache preparation;
HF catalog entries remain REVIEW and may enter bounded review transport. Neither path
grants rights, allocates tokens, or makes a source eligible for production. The
automated source approval generator remains restricted to its previous ALLOWED set.
Dataset cards alone never CLEAR conditional records.

## Explicit approval

An authorized project reviewer must supply a sealed local JSON approval. The code
does not create approvals, promote source status, infer legal rights or authenticate
the reviewer cryptographically. As with existing clearance manifests, the repository
approval workflow is the trust boundary. Hashes establish integrity, not authority.
Production use requires all existing source/file grants in addition to this approval.

The source registry record references it as conditional_approval with path and sha256.
Each approved file/plan record includes conditional_approval_identity matching the
sealed approval's identity. Approval fields:

- policy: source-specific-conditional-license-v1
- decision: APPROVED
- source_id, locator, revision, license_id: exact source binding
- approved_by, approved_at: accountable reviewer and approval date
- files: nonempty list of exact path and sha256 pairs; no glob grants
- intended_use: all five keys production_preparation, commercial_training,
  dataset_distribution, model_distribution, hosted_inference; each has decision
  APPROVED or FORBIDDEN and a nonempty rationale
- obligations: nonempty attribution, notices, modifications, reciprocity and
  release_controls plans, with exact version-specific interpretation and responsible
  enforcement described by the reviewer
- evidence: hash-bound local documents with role, path and sha256; all roles
  rights_scope, provenance, obligations and approval are required

The rights_scope evidence must establish the grant covers these exact files;
provenance evidence must address contributor/upstream authority. The obligations
and approval evidence must justify the chosen intended uses and reciprocal-license
interpretation. Nonempty fields alone do not prove the legal conclusions. A reviewer
must not fill them with unsupported claims or placeholders.

The production acquisition validator requires production_preparation APPROVED and
revalidates approval/evidence hashes for every conditional item. It requires exact
file hash and matching source/file license. Mixed conditional scopes require separate
source scopes. Revoking/replacing evidence or changing revision/content invalidates
the existing approval reference; it must not be silently rebound on resume.

The compiler propagates the file's conditional_approval_identity. Existing
clearance_evidence and source_admission call the same acquisition validator for
per-file grants, so an existing grant cannot bypass conditional approval checks.
An ACCEPT source status without conditional approval is insufficient; a CONDITIONAL
source status with approval is still insufficient until the existing source approval
requirements are explicitly satisfied.

## Boundaries

This implements review and acquisition gating, not an automatic legal opinion or a
model/dataset release system. Intended-use decisions are retained evidence, not
authorization to start training. Release obligations require the approved operational
controls; this module does not generate notices or license model weights. No blanket
decision is made that training triggers or avoids ShareAlike/copyleft.

Historical discovery receipts for sources whose classification changes from REJECTED
to conditional REVIEW must be kept immutable: use a new output directory under the
new policy. Do not overwrite old sealed decisions or treat them as new approval.
