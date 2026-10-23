"""Conservative retained-text resolution; never grants repository/file clearance."""
import hashlib

# Whitespace-normalized COMPLETE Apache-2.0 terms, through END OF TERMS AND
# CONDITIONS, matching retained approved Serde LICENSE-APACHE evidence. No
# truncation, phrase matching, suffix removal, or filename-based inference.
APACHE_TERMS_SHA256 = '59d8f0ba87ad9a2f1a431123c8d16646e5b89ba53653e818f16d136d77263c99'


def resolve_license(metadata_id, payload):
    if metadata_id not in (None, '', 'NOASSERTION'):
        return metadata_id
    normalized = ' '.join(payload.decode('utf-8').split()).encode('utf-8')
    if hashlib.sha256(normalized).hexdigest() == APACHE_TERMS_SHA256:
        return 'Apache-2.0'
    return None


def package_license_matches(project, license_id, license_path):
    """Equivalent explicit PEP 621 forms, no expression guessing or globbing."""
    declaration = project.get('license')
    files = project.get('license-files')
    if isinstance(declaration, str):
        return declaration == license_id and files == [license_path]
    if not isinstance(declaration, dict) or files not in (None, [license_path]):
        return False
    return declaration == {'text': license_id} or declaration == {'file': license_path}
