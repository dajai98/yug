"""Fake canaries only; no real credentials or personal records."""
import json
import pytest
from training import content_screening as cs
from training.production_spec import seal
from training.production_corpus import canonical
from tests.test_corpus_gate import setup, bundle, builder, Tokenizer

DOC='sha256:'+'a'*64


@pytest.mark.parametrize('key',['password','passwd','api_key','access_token','secret'])
@pytest.mark.parametrize('quote',['', '"', "'"])
@pytest.mark.parametrize('separator',['=', ':', ' \t:\n ', '\r\n=\t'])
def test_equivalent_quoted_credentials(key,quote,separator):
    text='{'+quote+key+quote+separator+'"CanaryOnlyNotValid123"}'
    r=cs.screen(text.encode(),text,DOC)
    assert r['outcome']=='EXCLUDE'
    assert {f['channel'] for f in r['findings']}=={'raw','normalized'}
    assert all(f['summary']=='[MASKED]' for f in r['findings'])
    assert 'CanaryOnlyNotValid123' not in canonical(r).decode()
    assert r==cs.screen(text.encode(),text,DOC)


@pytest.mark.parametrize('quote',['"', "'"])
@pytest.mark.parametrize('channel',['raw','normalized'])
def test_quoted_single_channel(quote,channel):
    bad=quote+'password'+quote+': '+quote+'CanaryOnlyNotValid123'+quote
    r=cs.screen((bad if channel=='raw' else 'benign').encode(),bad if channel=='normalized' else 'benign',DOC)
    assert r['outcome']=='EXCLUDE'
    assert {f['channel'] for f in r['findings']}=={channel}


@pytest.mark.parametrize('value',['EXAMPLE_NOT_REAL','FAKE_CanaryOnlyNotValid123','YOUR_TOKEN','DUMMY_VALUE','TEST_VALUE','<token>','${TOKEN}','changeme','redacted','none','null'])
def test_quoted_placeholder_policy_unchanged(value):
    text='{"password": "'+value+'"}'
    assert cs.screen(text.encode(),text,DOC)['outcome']=='ACCEPT'


def test_bare_assignment_inside_string_still_detected():
    text="'password = CanaryOnlyNotValid123'"
    assert cs.screen(text.encode(),text,DOC)['outcome']=='EXCLUDE'


def test_exact_quoted_key_exploit_no_tokens(tmp_path):
    text='Configuration with credential assignment: {"password": "CanaryOnlyNotValid123"}'
    assert len(text.encode())==79
    data=setup(tmp_path,texts=[text,'Independent benign source document.']);p=bundle(tmp_path,data)
    with builder(tmp_path,data) as b:
        result=b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)
        assert result['stages']=={'FILTERED':1}
        assert result['rejected_reasons']=={'screening_EXCLUDE':1}
        assert result['accepted_clean_train_tokens']==0
        assert b.db.execute('SELECT count(*) FROM totals').fetchone()[0]==0
        out=tmp_path/'build/authorized'
        b.export_shards(out,intended_use='commercial_training',decision_receipts=p)
        assert json.loads((out/'prepared.json').read_bytes())['counts']=={'train':0,'validation':0}
        assert list(b.replay_for_use(out,split='train',intended_use='commercial_training',decision_receipts=p))==[]


def test_revision_one_evidence_cannot_be_reused():
    old_rules=dict(cs.RULES,credential='REVIEW: password/passwd/api_key/access_token/secret assignments; EXCLUDE non-placeholder value >=12 chars')
    from training.production_corpus import identity
    import unicodedata
    old=identity(dict(policy=cs.POLICY,rules=old_rules,max_bytes=cs.MAX_BYTES,max_findings=cs.MAX_FINDINGS,max_matches=cs.MAX_MATCHES,implementation=1,unicode_version=unicodedata.unidata_version))
    assert old!=cs.DETECTOR
    raw=b'Benign technical documentation.';r=cs.screen(raw,raw.decode(),DOC)
    with pytest.raises(Exception):cs.verify_result(seal(dict(r,detector=old)),raw,raw.decode(),DOC)


@pytest.mark.parametrize('text,outcome',[
    ('password = "CanaryOnlyNotValid123"','EXCLUDE'),
    ('-----BEGIN RSA PRIVATE KEY-----\nnot-a-key\n-----END RSA PRIVATE KEY-----','EXCLUDE'),
    ('api_\u200bkey = "CanaryOnlyNotValid123"','EXCLUDE'),
    ('नागरिकता: १२३४५६७','REVIEW'),
    ('passport: 12345678','REVIEW'),
    ('Phone +977 9812345678','REVIEW'),
    ('alice@private.test','REVIEW'),
    ('Copyright Alice <alice@project.test>','ACCEPT'),
    ('Documentation contact: support@project.test','ACCEPT'),
    ('password = "EXAMPLE_NOT_REAL"','ACCEPT'),
    ('Contact audit-person@example.invalid','ACCEPT'),
    ('ghp_'+'a'*36,'EXCLUDE'),
    ('password=x','REVIEW'),
])
def test_rules_and_masking(text,outcome):
    r=cs.screen(text.encode(),text,DOC)
    assert r['outcome']==outcome
    assert r==cs.screen(text.encode(),text,DOC)
    for finding in r['findings']:
        assert finding['summary']=='[MASKED]'
        assert set(finding)=={'rule','outcome','channel','start','end','location_view','summary'}
    assert 'CanaryOnlyNotValid123' not in canonical(r).decode()


@pytest.mark.parametrize('channel',['raw','normalized'])
def test_both_views(channel):
    bad='password=CanaryOnlyNotValid123'
    r=cs.screen((bad if channel=='raw' else 'benign').encode(),bad if channel=='normalized' else 'benign',DOC)
    assert r['outcome']=='EXCLUDE'
    assert {f['channel'] for f in r['findings']}=={channel}


@pytest.mark.parametrize('damage',['policy','raw','normalized','document','outcome','findings','malformed','seal'])
def test_forged_results(damage):
    raw=b'benign prose';normal=raw.decode();r=cs.screen(raw,normal,DOC)
    if damage=='malformed':r=[]
    elif damage=='seal':r['identity']='bad'
    else:
        field={'raw':'raw_sha256','normalized':'normalized_sha256','document':'document_identity'}.get(damage,damage)
        r[field]='stale';r=seal(r)
    with pytest.raises(Exception):cs.verify_result(r,raw,normal,DOC)


def test_resealed_sensitive_accept_cannot_pass():
    raw=b'password=CanaryOnlyNotValid123'
    r=cs.screen(raw,raw.decode(),DOC)
    r=seal(dict(r,outcome='ACCEPT',findings=[]))
    with pytest.raises(Exception):cs.verify_result(r,raw,raw.decode(),DOC)


@pytest.mark.parametrize('kind',['raw_budget','normal_budget','findings','policy','utf8','type'])
def test_limits(kind):
    raw=b'benign';normal='benign';kw={}
    if kind=='raw_budget':raw=b'a'*(cs.MAX_BYTES+1)
    if kind=='normal_budget':normal='a'*(cs.MAX_BYTES+1)
    if kind=='findings':normal=('x@private.test\n')*(cs.MAX_FINDINGS+1)
    if kind=='policy':kw['policy']='unknown'
    if kind=='utf8':raw=b'\xff'
    if kind=='type':raw='not bytes'
    with pytest.raises(Exception):cs.screen(raw,normal,DOC,**kw)


def test_exploit_with_sensitive_canary_filtered_before_accounting(tmp_path):
    data=setup(tmp_path,texts=['Synthetic diagnostic password="CanaryOnlyNotValid123". Never real data.','Other harmless independent document.'])
    p=bundle(tmp_path,data)
    with builder(tmp_path,data) as b:
        result=b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)
        assert result['stages']=={'FILTERED':1}
        assert result['accepted_clean_train_tokens']==0
        assert result['rejected_reasons']=={'screening_EXCLUDE':1}


@pytest.mark.parametrize('boundary',['resume','export','replay','account'])
@pytest.mark.parametrize('damage',['missing','resealed','corrupt'])
def test_boundaries(tmp_path,boundary,damage):
    data=setup(tmp_path);p=bundle(tmp_path,data)
    with builder(tmp_path,data) as b:
        b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)
        out=tmp_path/'build/authorized'
        b.export_shards(out,intended_use='commercial_training',decision_receipts=p)
        assert list(b.replay_for_use(out,split='train',intended_use='commercial_training',decision_receipts=p))
        item,record=b.db.execute("SELECT item,record FROM items WHERE stage='ACCEPTED'").fetchone()
        path=b.path('screening',item,'.json')
        if damage=='missing':path.unlink()
        elif damage=='corrupt':path.write_text('{')
        else:
            r=json.loads(path.read_bytes());r['detector']='stale';path.write_bytes(canonical(seal(r)))
        before=list(b.db.execute('SELECT * FROM totals'))
        with pytest.raises(ValueError,match='Screening'):
            if boundary=='resume':b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)
            elif boundary=='export':b.export_shards(out,intended_use='commercial_training',decision_receipts=p)
            elif boundary=='account':b._account(item,json.loads(record),619)
            else:b.replay_for_use(out,split='train',intended_use='commercial_training',decision_receipts=p)
        assert list(b.db.execute('SELECT * FROM totals'))==before


def test_scanner_exception_is_sanitized(tmp_path,monkeypatch):
    data=setup(tmp_path)
    def fail(*args,**kwargs):raise RuntimeError('CanaryOnlyNotValid123')
    monkeypatch.setattr(cs,'screen',fail)
    with builder(tmp_path,data) as b:
        with pytest.raises(ValueError,match='Screening failed'):
            b.run(data[3][0],Tokenizer())
        assert b.db.execute('SELECT count(*) FROM totals').fetchone()[0]==0
        assert 'CanaryOnlyNotValid123' not in b.get('last_error')


def test_benign_match_budget():
    text='Contact x@example.invalid\n'*(cs.MAX_MATCHES+1)
    with pytest.raises(Exception,match='budget'):cs.screen(text.encode(),text,DOC)


def test_missing_result_before_accounting(tmp_path,monkeypatch):
    data=setup(tmp_path)
    with builder(tmp_path,data) as b:
        original=b._prepare
        def prepare(item,record):
            result=original(item,record)
            b.path('screening',item,'.json').unlink()
            return result
        monkeypatch.setattr(b,'_prepare',prepare)
        with pytest.raises(ValueError,match='Screening'):
            b.run(data[3][0],Tokenizer())
        assert b.db.execute('SELECT count(*) FROM totals').fetchone()[0]==0
        assert b.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED'").fetchone()[0]==0


def test_missing_normalized_identity_cannot_skip_resume(tmp_path):
    data=setup(tmp_path)
    with builder(tmp_path,data) as b:
        b.run(data[3][0],Tokenizer())
        b.db.execute("UPDATE items SET normalized_sha=NULL WHERE stage='ACCEPTED'")
        b.db.commit()
        with pytest.raises(Exception,match='normalized eligibility'):
            b.run(data[3][0],Tokenizer())


def test_audit_fixture_without_evidence_cannot_replay(tmp_path):
    from training.token_shards import TokenShardStream
    from training.production_corpus import file_digest
    text='Synthetic diagnostic record, never real personal data. Contact: audit-person@example.invalid. password = "FAKE_CANARY_NOT_A_REAL_SECRET_123456". This document tests mandatory screening evidence, not detector accuracy.'
    data=setup(tmp_path,texts=[text,'Independent ocean measurements describe seasonal coastal changes.'])
    p=bundle(tmp_path,data)
    with builder(tmp_path,data) as b:
        result=b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=p)
        assert result['accepted_clean_train_tokens']==217
        out=tmp_path/'build/authorized'
        b.export_shards(out,intended_use='commercial_training',decision_receipts=p)
        prepared=json.loads((out/'prepared.json').read_bytes())
        before=file_digest(out/'prepared.json')
        generic=lambda:list(TokenShardStream(out/'shards',split='train',tokenizer_identity=Tokenizer.identity,prepared_corpus_identity=prepared['prepared_corpus_identity']))
        assert len(generic())==217
        item=b.db.execute("SELECT item FROM items WHERE stage='ACCEPTED'").fetchone()[0]
        b.path('screening',item,'.json').unlink()  # Exact old exploit: no screening evidence.
        with pytest.raises(ValueError,match='Screening'):
            b.replay_for_use(out,split='train',intended_use='commercial_training',decision_receipts=p)
        assert len(generic())==217
        assert file_digest(out/'prepared.json')==before
