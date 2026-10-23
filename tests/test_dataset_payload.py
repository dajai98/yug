import hashlib
import io
import json
import pytest
from training import dataset_payload as p
from training.dataset_transport import payload_rows
from training.production_spec import SpecificationError

URL='https://huggingface.co/datasets/example/training/resolve/'+'a'*40+'/data.jsonl'


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data); self.status=status; self.headers=headers or {}


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(p,'CHUNK_BYTES',16)
    return b'{"text":"abcdefghijk"}\n{"text":"second"}\n'


def test_stream_resume_and_replay(tmp_path,small):
    sha=hashlib.sha256(small).hexdigest()
    with pytest.raises(SpecificationError):
        p.download(URL,tmp_path,len(small),sha,opener=lambda *a,**k:Response(small[:20]))
    receipt,offset=p.verify(tmp_path,complete=False)
    assert offset==16 and len(receipt['chunks'])==1
    def resume(req,timeout):
        assert req.headers['Range']=='bytes=16-'
        return Response(small[16:],206,{'Content-Range':f'bytes 16-{len(small)-1}/{len(small)}'})
    p.download(URL,tmp_path,len(small),sha,opener=resume)
    assert max(f.stat().st_size for f in tmp_path.glob('*.chunk'))<=16
    assert list(payload_rows(tmp_path,2,'data.jsonl'))==[{'text':'abcdefghijk'},{'text':'second'}]
    before=(tmp_path/'chunks.json').read_bytes()
    def forbidden(*a,**k):raise AssertionError('unexpected network')
    p.download(URL,tmp_path,len(small),sha,opener=forbidden)
    assert (tmp_path/'chunks.json').read_bytes()==before
    with io.BufferedReader(p.ChunkReader(tmp_path)) as stream:assert stream.read()==small


@pytest.mark.parametrize('fault',['hash','revision','chunk','manifest'])
def test_integrity(tmp_path,small,fault):
    sha=hashlib.sha256(small).hexdigest()
    if fault=='hash':
        with pytest.raises(SpecificationError):p.download(URL,tmp_path,len(small),'0'*64,opener=lambda *a,**k:Response(small))
        return
    p.download(URL,tmp_path,len(small),sha,opener=lambda *a,**k:Response(small))
    if fault=='revision':
        with pytest.raises(SpecificationError):p.download(URL.replace('a'*40,'b'*40),tmp_path,len(small),sha)
    else:
        (tmp_path/('00000000.chunk' if fault=='chunk' else 'chunks.json')).write_bytes(b'corrupt')
        with pytest.raises((ValueError,KeyError)):p.verify(tmp_path)


@pytest.mark.parametrize('status,headers',[(200,{}),(206,{'Content-Range':'wrong'})])
def test_no_silent_range_fallback(tmp_path,small,status,headers):
    sha=hashlib.sha256(small).hexdigest()
    with pytest.raises(SpecificationError):p.download(URL,tmp_path,len(small),sha,opener=lambda *a,**k:Response(small[:16]))
    with pytest.raises(SpecificationError):p.download(URL,tmp_path,len(small),sha,opener=lambda *a,**k:Response(small,status,headers))


def test_bounded_row_across_chunks(tmp_path,small,monkeypatch):
    from training import dataset_transport as t
    p.download(URL,tmp_path,len(small),hashlib.sha256(small).hexdigest(),opener=lambda *a,**k:Response(small))
    monkeypatch.setattr(t,'MAX_DOC',10)
    with pytest.raises(SpecificationError):list(payload_rows(tmp_path,1,'data.jsonl'))


def test_unpinned_and_oversized_rejected(tmp_path):
    for url,size in [(URL.replace('a'*40,'main'),1),(URL,p.MAX_PAYLOAD+1)]:
        with pytest.raises(SpecificationError):p.download(url,tmp_path,size,'0'*64)
    assert not list(tmp_path.iterdir())
