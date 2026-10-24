"""Bounded Slice 2 rules, not a complete PII detector or legal approval.

Evidence is re-derived from bytes at eligibility boundaries: a public hash seal
alone cannot authenticate a claimed scan. Findings contain no matched values.
"""
import re
import unicodedata

from training.production_corpus import identity, canonical
from training.production_spec import need, seal, verify_seal

POLICY = 'yug-screening-v1'
SCHEMA = 'yug-screening-result-v1'
MAX_BYTES = 1024 * 1024
MAX_FINDINGS = 256
MAX_MATCHES = 512
# This table is part of the detector identity, including contexts and limits.
RULES = {
    'private_key': 'EXCLUDE: PEM private-key header, including incomplete blocks',
    'provider_token': 'EXCLUDE: AKIA 16 uppercase/digits; ghp_ 36 alphanumerics; sk- 32+ alphanumerics',
    'credential': 'REVIEW: password/passwd/api_key/access_token/secret assignments, bare or matching single/double quoted keys; EXCLUDE non-placeholder value >=12 chars',
    'email': 'REVIEW unless reserved example domain or same-line copyright/contact/support/documentation attribution',
    'phone': 'REVIEW: international + number with 8-15 digits',
    'personal_id': 'REVIEW: SSN shape or citizenship/passport/national-ID labels (English/Nepali)',
    'view': 'raw and normalized; NFKC, Unicode decimal digits to ASCII, remove zero-width U+200B/U+200C/U+200D/U+FEFF',
    'placeholder': 'credential values starting FAKE_, EXAMPLE_, YOUR_, DUMMY_, TEST_, < or ${; exact changeme/redacted/none/null',
}
DETECTOR = identity(dict(policy=POLICY, rules=RULES, max_bytes=MAX_BYTES,
                         max_findings=MAX_FINDINGS, max_matches=MAX_MATCHES, implementation=2,
                         unicode_version=unicodedata.unidata_version))


def _view(text):
    text = unicodedata.normalize('NFKC', text)
    return ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c
                   for c in text if c not in '\u200b\u200c\u200d\ufeff')


def _placeholder(value):
    return (value.upper().startswith(('FAKE_', 'EXAMPLE_', 'YOUR_', 'DUMMY_', 'TEST_', '<', '${'))
            or value.casefold() in ('changeme', 'redacted', 'none', 'null'))


def _find(text, channel):
    findings = []
    matches = 0
    def consume():
        nonlocal matches
        matches += 1
        need(matches <= MAX_MATCHES, 'Screening match budget exceeded')
    def add(rule, outcome, match):
        need(len(findings) < MAX_FINDINGS, 'Screening finding budget exceeded')
        findings.append(dict(rule=rule, outcome=outcome, channel=channel,
                             start=match.start(), end=match.end(), location_view='canonical_scan_view',
                             summary='[MASKED]'))
    patterns = (
        ('private_key', 'EXCLUDE', r'-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----'),
        ('provider_token', 'EXCLUDE', r'\b(?:AKIA[A-Z0-9]{16}|ghp_[A-Za-z0-9]{36}|sk-[A-Za-z0-9]{32,128})\b'),
        ('phone', 'REVIEW', r'(?<!\w)\+[0-9](?:[0-9 ()-]{6,24})[0-9]'),
        ('personal_id', 'REVIEW', r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b|(?:passport|national[ _-]?id|citizenship|नागरिकता|राहदानी)\s*[:=]\s*\S{3,80}'),
    )
    for rule, outcome, pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE if rule == 'personal_id' else 0):
            consume()
            if rule == 'phone' and not 8 <= sum(c.isdigit() for c in match[0]) <= 15:
                continue
            add(rule, outcome, match)
    # A quoted key must close with its opening quote. Keep the same assignment
    # separators, whitespace, value limits and placeholder rules as bare keys.
    for match in re.finditer(r'''(?i)(?<!\w)(?P<quote>["']?)(?:password|passwd|api_key|access_token|secret)\b(?P=quote)\s*[:=]\s*["']?(?P<value>[^\s"';,]{1,256})''', text):
        consume()
        value = match['value']
        if not _placeholder(value):
            add('credential', 'EXCLUDE' if len(value) >= 12 else 'REVIEW', match)
    for match in re.finditer(r'[\w.+-]{1,64}@[\w.-]{1,190}\.[A-Za-z]{2,24}', text):
        consume()
        domain = match[0].rsplit('@', 1)[1].lower()
        line = text[text.rfind('\n', 0, match.start())+1:text.find('\n', match.end()) if '\n' in text[match.end():] else len(text)]
        reserved = domain in ('example.com', 'example.org', 'example.net') or domain.endswith(('.invalid', '.example'))
        public = re.search(r'(?i)\b(copyright|contact|support|documentation)\b|©', line)
        if not reserved and not public:
            add('email', 'REVIEW', match)
    return findings


def screen(raw, normalized, document_identity, *, policy=POLICY):
    """Return sealed deterministic evidence; errors never return ACCEPT."""
    import hashlib
    need(policy == POLICY, 'Unsupported screening policy')
    need(type(raw) is bytes and len(raw) <= MAX_BYTES, 'Invalid/bounded raw screening input required')
    need(type(normalized) is str, 'Invalid normalized screening input')
    encoded = normalized.encode('utf-8')
    need(len(encoded) <= MAX_BYTES, 'Normalized screening byte budget exceeded')
    need(isinstance(document_identity, str) and re.fullmatch(r'sha256:[0-9a-f]{64}', document_identity), 'Invalid screening document identity')
    text = raw.decode('utf-8-sig', errors='strict')
    findings = _find(_view(text), 'raw') + _find(_view(normalized), 'normalized')
    need(len(findings) <= MAX_FINDINGS, 'Screening finding budget exceeded')
    outcomes = {f['outcome'] for f in findings}
    outcome = 'EXCLUDE' if 'EXCLUDE' in outcomes else 'REVIEW' if outcomes else 'ACCEPT'
    sha = lambda b: 'sha256:' + hashlib.sha256(b).hexdigest()
    return seal(dict(schema=SCHEMA, policy=POLICY, detector=DETECTOR,
                     document_identity=document_identity, raw_sha256=sha(raw),
                     normalized_sha256=sha(encoded), outcome=outcome, findings=findings))


def verify_result(result, raw, normalized, document_identity):
    verify_seal(result)
    expected = screen(raw, normalized, document_identity)
    need(canonical(result) == canonical(expected), 'Screening evidence mismatch/stale/forged')
    need(result['outcome'] == 'ACCEPT', 'Screening withholds eligibility')
    return result['identity']
