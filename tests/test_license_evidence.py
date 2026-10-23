from pathlib import Path
import pytest
from training.license_evidence import resolve_license, package_license_matches

TERMS = (Path(__file__).parent/'fixtures/license_evidence/Apache-2.0.txt').read_bytes()

def test_full_terms_only():
    assert resolve_license('NOASSERTION', TERMS) == 'Apache-2.0'
    assert resolve_license(None, b' '.join(TERMS.split())) == 'Apache-2.0'

@pytest.mark.parametrize('data',[TERMS+b'Additional restriction', TERMS[:200],
    TERMS.replace(b'perpetual',b'temporary'), b'See LICENSE.APACHE or LICENSE.BSD'])
def test_unrecognized_or_modified_terms(data):
    assert resolve_license('NOASSERTION', data) is None

def test_conflicting_metadata_is_not_overridden():
    assert resolve_license('GPL-3.0', TERMS) == 'GPL-3.0'

@pytest.mark.parametrize('declaration',[{'text':'MIT'}, {'file':'LICENSE'}])
def test_legacy_explicit_declarations(declaration):
    assert package_license_matches({'license':declaration}, 'MIT', 'LICENSE')

@pytest.mark.parametrize('project',[
    {'license':{'text':'MIT OR Apache-2.0'}}, {'license':{'file':'../LICENSE'}},
    {'license':{'text':'MIT','file':'LICENSE'}}, {'license':{'file':'other'}},
    {'license':{'text':'MIT'},'license-files':['*']}, {'license':'MIT'},
    {'license':{'text':'MIT','extra':True}}])
def test_ambiguous_declarations(project):
    assert not package_license_matches(project,'MIT','LICENSE')
