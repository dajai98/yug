"""Historical export bytes must survive incompatible attempts unchanged."""
import json
import hashlib
import pytest
from tests.test_corpus_gate import setup, bundle, builder, Tokenizer
from training.production_acquisition import _Prepared, _TokenCache
from training.production_corpus import canonical, identity
from training.production_spec import seal
from training.token_shards import build_token_shards


def snapshot(path):
    return {str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}


def test_orphan_prepared_temporary_export_is_preserved(tmp_path):
    data=setup(tmp_path);receipts=bundle(tmp_path,data)
    with builder(tmp_path,data) as b:
        b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=receipts)
        out=tmp_path/'build/orphan-prepared';out.mkdir()
        temporary=out/'prepared.json.tmp'
        original=b'{"historical":"interrupted prepared publication"}'
        temporary.write_bytes(original)
        before=snapshot(out)
        digest=hashlib.sha256(original).hexdigest()
        modified=temporary.stat().st_mtime_ns
        with pytest.raises(Exception,match='Incomplete export'):
            b.export_shards(out,intended_use='commercial_training',decision_receipts=receipts)
        assert snapshot(out)==before
        assert temporary.exists() and temporary.read_bytes()==original
        assert hashlib.sha256(temporary.read_bytes()).hexdigest()==digest
        assert temporary.stat().st_mtime_ns==modified
        assert not (out/'prepared.json').exists()
        assert not (out/'admission.jsonl').exists()
        assert not (out/'shards').exists()


@pytest.mark.parametrize('relative',['admission.part','shards/request.json.tmp','shards/manifest.json.tmp'])
def test_other_orphan_publication_temporary_export_is_preserved(tmp_path,relative):
    data=setup(tmp_path);receipts=bundle(tmp_path,data)
    with builder(tmp_path,data) as b:
        b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=receipts)
        out=tmp_path/'build/orphan-publication';out.mkdir()
        temporary=out/relative;temporary.parent.mkdir(parents=True,exist_ok=True)
        temporary.write_bytes(b'interrupted publication sentinel')
        before=snapshot(out)
        with pytest.raises(Exception,match='Incomplete export'):
            b.export_shards(out,intended_use='commercial_training',decision_receipts=receipts)
        assert snapshot(out)==before
        assert not (out/'prepared.json').exists()
        assert not (out/'admission.jsonl').exists()


def test_historical_export_failure_preserves_bytes(tmp_path):
    data=setup(tmp_path);receipts=bundle(tmp_path,data)
    with builder(tmp_path,data) as b:
        b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=receipts)
        modern=tmp_path/'build/modern';b.export_shards(modern,intended_use='commercial_training',decision_receipts=receipts)
        old=tmp_path/'build/historical';old.mkdir()
        (old/'admission.jsonl').write_bytes((modern/'admission.jsonl').read_bytes())
        m=json.loads((modern/'prepared.json').read_bytes());m.pop('screening');m.pop('prepared_corpus_identity');m['prepared_corpus_identity']=identity(m)
        (old/'prepared.json').write_bytes(canonical(m))
        build_token_shards(_Prepared(b,m),_TokenCache(b),old/'shards')
        before=snapshot(old)
        with pytest.raises(Exception):b.export_shards(old,intended_use='commercial_training',decision_receipts=receipts)
        assert snapshot(old)==before


@pytest.mark.parametrize('damage',['prepared','shard','embedded','screening','admission_order','authorization','stale_result','malformed','partial','shard_bytes','request','orphan_admission'])
def test_incompatible_export_preserved(tmp_path,damage):
    data=setup(tmp_path);receipts=bundle(tmp_path,data)
    with builder(tmp_path,data) as b:
        b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=receipts)
        out=tmp_path/'build/export';b.export_shards(out,intended_use='commercial_training',decision_receipts=receipts)
        path=out/'prepared.json'
        if damage=='malformed':path.write_text('{')
        elif damage=='partial':(out/'shards/manifest.json').unlink()
        elif damage=='orphan_admission':path.unlink()
        elif damage=='admission_order':
            p=out/'admission.jsonl';p.write_bytes(p.read_bytes()+p.read_bytes())
        elif damage=='shard_bytes':
            m=json.loads((out/'shards/manifest.json').read_bytes());p=out/'shards'/m['splits']['train']['shards'][0]['path'];p.write_bytes(b'corrupt')
        elif damage=='stale_result':
            p=next((tmp_path/'build/screening').glob('*.json'));r=json.loads(p.read_bytes());r['detector']='stale';p.write_bytes(canonical(seal(r)))
        elif damage=='request':
            p=out/'shards/request.json';r=json.loads(p.read_bytes());r['shard_tokens']=1;p.write_bytes(canonical(r))
        elif damage in ('shard','embedded'):
            p=out/'shards/manifest.json';m=json.loads(p.read_bytes())
            if damage=='shard':m['split_identity']='bad'
            else:m['prepared_manifest']['counts']['train']=999
            m.pop('manifest_identity');m['manifest_identity']=identity(m);p.write_bytes(canonical(m))
        else:
            m=json.loads(path.read_bytes())
            if damage=='prepared':m['counts']['train']=999
            elif damage=='screening':m['screening']['results_sha256']='bad'
            else:m['use_authorization']['intended_use']='evaluation'
            m.pop('prepared_corpus_identity');m['prepared_corpus_identity']=identity(m);path.write_bytes(canonical(m))
        before=snapshot(out)
        with pytest.raises(Exception):b.export_shards(out,intended_use='commercial_training',decision_receipts=receipts)
        assert snapshot(out)==before


def test_fresh_and_compatible_export_no_rewrite(tmp_path):
    data=setup(tmp_path);receipts=bundle(tmp_path,data)
    with builder(tmp_path,data) as b:
        b.run(data[3][0],Tokenizer(),intended_use='commercial_training',decision_receipts=receipts)
        out=tmp_path/'build/export';first=b.export_shards(out,intended_use='commercial_training',decision_receipts=receipts)
        before=snapshot(out);times={str(p):p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}
        class NoEncode(Tokenizer):
            def encode(self,*args):pytest.fail('Unexpected retokenization')
        b.run(data[3][0],NoEncode(),fetch=lambda _:pytest.fail('Unexpected fetch'),
              intended_use='commercial_training',decision_receipts=receipts)
        second=b.export_shards(out,intended_use='commercial_training',decision_receipts=receipts)
        assert first==second and snapshot(out)==before
        assert times=={str(p):p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}
        replay=lambda:list(b.replay_for_use(out,split='train',intended_use='commercial_training',decision_receipts=receipts))
        assert replay()==replay() and len(replay())==619
