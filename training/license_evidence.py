"""Conservative retained-text resolution; never grants repository/file clearance."""
import hashlib
import re

# Whitespace-normalized COMPLETE Apache-2.0 terms, through END OF TERMS AND
# CONDITIONS, matching retained approved Serde LICENSE-APACHE evidence. No
# truncation, phrase matching, suffix removal, or filename-based inference.
APACHE_TERMS_SHA256 = '59d8f0ba87ad9a2f1a431123c8d16646e5b89ba53653e818f16d136d77263c99'

# Complete standard terms only. Copyright lines may identify different owners,
# but no extra grants, restrictions, or alternate terms may be discarded.
MIT_TERMS = """Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""
BSD_2_TERMS = """Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""
BSD_3_CONDITION = """3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission."""


def _terms(value):
    return ' '.join(value.split())


def _standard_profile(payload):
    try:
        lines = payload.decode('utf-8').splitlines()
    except UnicodeError:
        return None
    lines = [line.strip() for line in lines if line.strip()]
    while lines and lines[0] in ('MIT License', 'The MIT License (MIT)',
                                 'BSD 2-Clause License', 'BSD 3-Clause License'):
        lines.pop(0)
    copyright_count = 0
    while lines and re.fullmatch(r'Copyright (?:\(c\)|©) .+', lines[0], re.IGNORECASE):
        lines.pop(0)
        copyright_count += 1
    if not copyright_count:
        return None
    body = _terms(' '.join(lines))
    if body == _terms(MIT_TERMS):
        return 'MIT'
    if body == _terms(BSD_2_TERMS):
        return 'BSD-2-Clause'
    if body == _terms(BSD_2_TERMS.replace('THIS SOFTWARE', BSD_3_CONDITION +
                                          '\nTHIS SOFTWARE', 1)):
        return 'BSD-3-Clause'
    return None


def resolve_license(metadata_id, payload):
    if metadata_id not in (None, '', 'NOASSERTION'):
        return metadata_id
    normalized = ' '.join(payload.decode('utf-8').split()).encode('utf-8')
    if hashlib.sha256(normalized).hexdigest() == APACHE_TERMS_SHA256:
        return 'Apache-2.0'
    return _standard_profile(payload)


def package_license_matches(project, license_id, license_path):
    """Equivalent explicit PEP 621 forms, no expression guessing or globbing."""
    declaration = project.get('license')
    files = project.get('license-files')
    if isinstance(declaration, str):
        return declaration == license_id and files == [license_path]
    if not isinstance(declaration, dict) or files not in (None, [license_path]):
        return False
    return declaration == {'text': license_id} or declaration == {'file': license_path}
