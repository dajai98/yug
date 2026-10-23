"""Verified range-resumable JSONL payloads stored in bounded local chunks."""
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

from training.production_corpus import atomic_json, atomic_replace, file_digest
from training.production_spec import need, seal, verify_seal

CHUNK_BYTES = 64 * 1024 * 1024
MAX_PAYLOAD = 1024 * 1024 * 1024


def _validate_url(url):
    need(isinstance(url, str) and re.fullmatch(
        r'https://huggingface\.co/datasets/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+/resolve/[0-9a-f]{40}/[A-Za-z0-9_./-]+\.jsonl', url)
        and '..' not in url.split('/'), 'Unpinned JSONL payload URL')


def verify(directory, *, complete=True):
    """Rehash retained bytes, including the original concatenated payload hash."""
    directory = Path(directory)
    receipt = json.loads((directory/'chunks.json').read_bytes())
    verify_seal(receipt)
    need(receipt['version'] == 'bounded-jsonl-chunks-v1', 'Unknown chunk format')
    _validate_url(receipt['url'])
    size = receipt['size']
    need(type(size) is int and 0 < size <= MAX_PAYLOAD, 'Invalid payload size')
    need(re.fullmatch('[0-9a-f]{64}', receipt['sha256']) is not None, 'Invalid payload hash')
    need(receipt['chunk_bytes'] == CHUNK_BYTES, 'Chunk bounds changed')
    digest = hashlib.sha256(); total = 0
    for i, entry in enumerate(receipt['chunks']):
        name = f'{i:08d}.chunk'
        expected = min(CHUNK_BYTES, size-total)
        need(expected > 0 and entry['name'] == name and entry['bytes'] == expected,
             'Invalid chunk order/size')
        path = directory/name
        need(path.stat().st_size == expected and file_digest(path) == entry['sha256'], 'Chunk corrupt')
        with path.open('rb') as stream:
            while block := stream.read(65536): digest.update(block)
        total += expected
    if total == size:
        need(digest.hexdigest() == receipt['sha256'], 'Whole payload hash mismatch')
    if complete: need(total == size, 'Incomplete chunked payload')
    return receipt, total


def download(url, directory, size, sha, *, opener=None):
    """Commit complete 64-MiB chunks; an interrupted chunk is fetched again."""
    _validate_url(url)
    need(type(size) is int and 0 < size <= MAX_PAYLOAD and re.fullmatch('[0-9a-f]{64}', sha), 'Invalid payload bounds/hash')
    directory = Path(directory); directory.mkdir(exist_ok=True)
    manifest = directory/'chunks.json'
    if manifest.exists():
        receipt, offset = verify(directory, complete=False)
        need((receipt['url'], receipt['size'], receipt['sha256']) == (url, size, sha), 'Payload binding changed')
    else:
        need(not any(directory.iterdir()), 'Unknown chunk directory')
        receipt = seal(dict(version='bounded-jsonl-chunks-v1', url=url, size=size,
                            sha256=sha, chunk_bytes=CHUNK_BYTES, chunks=[]))
        atomic_json(manifest, receipt); offset = 0
    if offset == size: return
    from training.dataset_transport import Redirect
    open_url = opener or urllib.request.build_opener(Redirect).open
    headers = {'Accept-Encoding': 'identity', 'User-Agent': 'yug-bounded-jsonl-v1'}
    if offset: headers['Range'] = f'bytes={offset}-'
    with open_url(urllib.request.Request(url, headers=headers), timeout=60) as response:
        if response.status == 206:
            need(response.headers.get('Content-Range') == f'bytes {offset}-{size-1}/{size}', 'Invalid chunk resume range')
        else:
            need(offset == 0 and response.status == 200, 'Server did not honor chunk resume range')
        need(response.headers.get('Content-Encoding', 'identity') == 'identity', 'Encoded payload response')
        while offset < size:
            count = min(CHUNK_BYTES, size-offset)
            name = f"{len(receipt['chunks']):08d}.chunk"
            target = directory/name; temp = directory/'pending.part'
            digest = hashlib.sha256(); written = 0
            with temp.open('wb') as stream:
                while written < count:
                    block = response.read(min(65536, count-written))
                    need(bool(block), 'Interrupted payload; resume from last committed chunk')
                    stream.write(block); digest.update(block); written += len(block)
            atomic_replace(temp, target)
            receipt = seal(dict(receipt, chunks=receipt['chunks'] + [dict(
                name=name, bytes=count, sha256='sha256:'+digest.hexdigest())]))
            atomic_json(manifest, receipt); offset += count
        need(not response.read(1), 'Payload exceeds pinned size')
    verify(directory)


class ChunkReader(io.RawIOBase):
    """Sequential bytes without concatenating the payload on disk or in RAM."""
    def __init__(self, directory):
        super().__init__()
        receipt, _ = verify(directory)
        self.paths = iter(Path(directory)/r['name'] for r in receipt['chunks'])
        self.current = None

    def readable(self): return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None: return 0
                self.current = path.open('rb')
            count = self.current.readinto(buffer)
            if count: return count
            self.current.close(); self.current = None

    def close(self):
        if self.current is not None: self.current.close()
        super().close()


def payload_digest(directory):
    directory = Path(directory)
    if (directory/'payload_chunks').exists():
        need(not (directory/'payload.parquet').exists(), 'Ambiguous payload storage')
        receipt, _ = verify(directory/'payload_chunks')
        return 'sha256:'+receipt['sha256']
    return file_digest(directory/'payload.parquet')
