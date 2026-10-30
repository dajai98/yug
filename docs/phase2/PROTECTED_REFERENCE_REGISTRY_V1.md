# Protected reference registry v1

The pinned held-out JSONL inventory now requires `yug-protected-reference-v1`
records. Every row binds a canonical `work_id`, `work_version`, `item_id`, unique
`reference_id`, `role`, `protected_split`, `aliases`, provenance, text, content type,
language, raw UTF-8 SHA256 and content-type-normalized UTF-8 SHA256. Supported
roles are `question`, `answer`, `solution`, and `reference` for retained non-Q/A
holdouts. Supported protected splits are `validation`, `test`, and `evaluation`.
Question, answer and solution rows for one item share work/version/item and split;
the coordinator indexes each row's actual text through the existing dedup policy.
It cannot infer an answer from a question. Aliases are retained identity metadata;
they do not currently expand exclusion matching, substitute for text or authorize
data use.

The versioned validator refuses missing or unknown fields, unsupported values,
duplicate JSON keys and reference IDs, conflicting work/item/role or split, empty
text/provenance, and incorrect raw or normalized hashes. Existing unstructured
inventories are not grandfathered into new admission. The retained-holdout
producer emits explicit `reference` rows with its prior text unchanged and
provenance retained. Its source revision or pinned artifact hash supplies the
recorded work version where applicable. Historical files are not rewritten;
operators must generate a new pinned inventory and use a new compatible build.
Generic old shard integrity decoding confers no training authorization.

A synthetic answer that was absent from the inventory previously entered as 277
byte-level test-double tokens. Listing the answer under the same work/version/item
as its question and solution makes the answer candidate from a different source
`DUPLICATE`: zero accepted/accounted tokens and zero training replay contribution.
Exact and currently supported edited/excerpt question and solution variants remain
blocked. This is a completeness repair for listed records, not a claim that the
registry contains every benchmark or that hashes without text enable near matching.

The fixed dedup policy is unchanged: five-token shingles, 128 anchors, Jaccard
0.5, containment 0.8 and 10,000 candidate limit. Its measured residuals remain:
renamed-variable Python solutions can escape lexical matching, and benign short
four-word fragments can be falsely clustered. Registry completeness, source aliases,
near-copy recall and false positives remain separate Phase 2 review work.
