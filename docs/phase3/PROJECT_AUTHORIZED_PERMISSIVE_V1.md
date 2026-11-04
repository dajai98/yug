# Project-authorized permissive first-time file clearance v1

This is an alternative *evidence profile* for the existing, externally pinned
first-time source-clearance decision. It is not a grant, a legal conclusion, or
an automatic consequence of an open-source repository. It does not change the
existing Python package/file profile, revision-supersession route, downstream
corpus gates, or the license allowlist. An unapproved file remains `REVIEW`.

The Phase 2 baseline requires trusted review of publisher rights statements;
it does not require a separate signature from every contributor. The earlier
first-time implementation accepted only narrow Python package/file evidence.
Treating that limitation as a universal contributor-proof rule would be an
accidental restriction, not a documented governance decision.

For this v1 profile, the materializer accepts only an exact English Markdown
documentation file under MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause. A
human-reviewed `APPROVE_EXACT_FILES` decision, pinned by a SHA256 supplied
outside the mutable output, must identify its source, canonical locator,
40-character revision, path, raw SHA256, byte size, owner, subcategory, split,
bounded cap, attribution, obligations, and file-rights record. The verified
cache must also reproduce the file's Git blob identity. A different source,
revision, path, content, owner, decision, or evidence hash requires a new
review and decision.

The per-file rights record must bind that same scope and retain separate,
hash-pinned evidence for (1) the exact-revision project license, (2) explicit
project publication of the relevant documentation under that license, (3) the
project contribution/review terms or workflow, and (4) this file's history.
The reviewer must examine whether these support project distribution authority
for the *exact bytes* without assuming a CLA or individual contributor
assignment. Root license, SPDX, repository membership, and file extension are
each insufficient by themselves. The reviewer must explicitly find no
conflicting file license and no unresolved copied, adapted, vendored, or
third-party provenance. A visible contrary signal fails closed; an absent
signal is not itself proof of ownership.

Generated, binary, media, translated and separately licensed material are
outside this profile. All local links and includes in a Markdown file must be
recorded as exclusions; unsupported or unresolvable link forms fail closed.
Inline and reference-style Markdown links and HTML `src`/`href` references are
checked against the exact saved file bytes. They do not inherit clearance;
each referenced file would need a separate decision. Required copyright/license attribution and
NOTICE obligations must be retained in the source clearance and checked by
the existing production gates. Evidence files are hash-verified at
materialization; the human review and its external decision pin are the trust
boundary. This profile does not authenticate the human operator supplying the
pin.

The Stride Docs `en/manual/index.md` file at revision
`0cb2f68bc735e12ce7071ebc50409e3f64a464c7` has a retained exact-file
clearance and a separate training-use decision. Its linked image and included
changelog files remain outside scope. The one-file production preflight passed,
but its live admission decision remains pending; no production corpus item is
admitted by this profile or preflight.
