"""Exact-file authority checks independent of ignored review-cache artifacts."""
import copy

import pytest

from training.production_corpus import canonical, identity
from training.production_spec import seal, SpecificationError
from training.source_revision_supersession import (
    LOCATOR, OLD_REVISION, POLICY, REVISION, SOURCE, verify_decision_scope)


def sample():
    paths=['src/attr/_funcs.py']+[f'src/attr/file_{i:02}.py' for i in range(15)]
    old=[dict(path=p,sha256='sha256:'+'a'*64) for p in paths]
    files=copy.deepcopy(old)
    files[0]['sha256']='sha256:'+'b'*64
    files.sort(key=lambda x:x['path'])
    candidates={x['path']:dict(x,source_id=SOURCE,revision=REVISION,
        owner_claim='software_engineering',subcategory_claim='curated_subset',
        license_id='MIT') for x in files}
    d=seal(dict(schema=POLICY,source_id=SOURCE,locator=LOCATOR,revision=REVISION,
        prior_revision=OLD_REVISION,owner='software_engineering',
        subcategory='curated_subset',license_id='MIT',decision='APPROVE_EXACT_FILES',
        reviewer='YUG owner',attribution='MIT copyright and license',
        notice_obligation='preserve_required_mit_attribution',files=files))
    return d,old,candidates


def test_exact_revision_supersession_and_historical_input_immutable():
    d,old,candidates=sample();before=canonical(old)
    assert verify_decision_scope(d,old,candidates)=={'src/attr/_funcs.py'}
    assert canonical(old)==before
    assert d['prior_revision']!=d['revision']


@pytest.mark.parametrize('field,replacement',[
    ('source_id','gh-other-project'),('locator','https://github.com/other/project'),
    ('revision',OLD_REVISION),('owner','code'),('subcategory','python'),
    ('license_id','Apache-2.0'),('notice_obligation','none'),('attribution',''),
])
def test_decision_source_revision_ownership_and_obligations_fail_closed(field,replacement):
    d,old,candidates=sample();d[field]=replacement
    with pytest.raises(SpecificationError):verify_decision_scope(d,old,candidates)


@pytest.mark.parametrize('field,replacement',[
    ('source_id','gh-other-project'),('revision',OLD_REVISION),('path','src/other.py'),
    ('sha256','sha256:'+'c'*64),('owner_claim','code'),
    ('subcategory_claim','python'),('license_id','Apache-2.0')])
def test_candidate_substitution_fails_closed(field,replacement):
    d,old,candidates=sample();candidates['src/attr/_funcs.py'][field]=replacement
    with pytest.raises(SpecificationError):verify_decision_scope(d,old,candidates)


def test_unchanged_content_still_requires_new_revision():
    d,old,candidates=sample()
    candidates['src/attr/file_00.py']['revision']=OLD_REVISION
    with pytest.raises(SpecificationError):verify_decision_scope(d,old,candidates)


def test_changed_funcs_requires_independent_new_content_hash():
    d,old,candidates=sample()
    d['files'][0]['sha256']=old[0]['sha256']
    with pytest.raises(SpecificationError):verify_decision_scope(d,old,candidates)


def test_omission_duplicate_and_extra_path_fail_closed():
    for transform in (lambda d:d['files'].pop(),
                      lambda d:d['files'].__setitem__(1,copy.deepcopy(d['files'][0])),
                      lambda d:d['files'][1].__setitem__('path','src/other.py')):
        d,old,candidates=sample();transform(d)
        with pytest.raises(SpecificationError):verify_decision_scope(d,old,candidates)


def test_decision_replay_identity_deterministic_and_changed_obligation_changes_identity():
    d,old,candidates=sample();first=canonical(d)
    assert verify_decision_scope(d,old,candidates)==verify_decision_scope(copy.deepcopy(d),old,copy.deepcopy(candidates))
    assert canonical(d)==first and identity(d)==identity(copy.deepcopy(d))
    changed=seal(dict(d,attribution='different MIT attribution'))
    assert changed['identity']!=d['identity']
