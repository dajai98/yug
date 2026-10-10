"""Frozen bounded natural-data tokenizer comparison. No model training.

Reuses existing mini-corpus normalization, licensing and stratified split.
Five text families are a coverage proxy, NOT a replacement dataset allocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from training.experiments import representative_mini_corpus as mini
from training.experiments.preprocessing_scalability_probe import memory
from training.corpus_prep import RawDocument

FAMILIES = ('code', 'csharp_gamedev', 'general_english', 'nepali', 'software_docs')
# Byte ceilings derived from the five corresponding approved token shares only.
# Bytes are available before fitting; they are NOT token allocation percentages.
WEIGHTS = dict(zip(FAMILIES, (22, 13, 12, 8, 6)))
CRITERIA = {
    'fit_wall_limit_seconds': 600, 'peak_rss_limit_bytes': 512 * 1024**2,
    'encode_wall_limit_seconds': 180, 'vocab_targets': [4096, 8192],
    'actual_vocab_must_equal_target': True, 'independent_runs_per_target': 2,
    'identical_identity_serialization_and_encoded_output_required': True,
    'roundtrip_errors_allowed': 0,
    'validation_min_docs_per_family': 4,
    'validation_min_bytes_per_family': 10000,
    'minimum_validation_bytes_per_token_each_family': 2.0,
    'minimum_nepali_validation_characters_per_token': 1.0,
    '8192_min_overall_validation_token_reduction_fraction': 0.10,
    '8192_max_per_family_token_increase_fraction': 0.0,
    '8192_max_identifier_fragmentation_increase_fraction': 0.0,
    'threshold_status': 'Predeclared engineering screening thresholds, not validated task-quality thresholds',
    'final_recommendation_requires_representative_16_category_coverage': True,
    'fit_byte_ceiling': 300000,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return 'sha256:' + hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(canonical(value) + b'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    path.write_bytes(b''.join(canonical(r) + b'\n' for r in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def select_train(documents, cap=300000):
    """Whole documents in content-hash order within each fixed family ceiling."""
    selected = []
    for family in FAMILIES:
        used = 0
        ceiling = cap * WEIGHTS[family] // sum(WEIGHTS.values())
        for doc in sorted((d for d in documents if d.primary_domain == family),
                          key=lambda d: (d.content_hash, d.document_id)):
            size = len(doc.text.encode('utf-8'))
            if used + size <= ceiling:
                selected.append(doc)
                used += size
    return selected


def composition(rows):
    out = {}
    for family in FAMILIES:
        group = [r for r in rows if r['primary_domain'] == family]
        out[family] = dict(documents=len(group), characters=sum(len(r['text']) for r in group),
                           utf8_bytes=sum(len(r['text'].encode()) for r in group),
                           sources=dict(sorted(Counter(r['source_id'] for r in group).items())))
    return out


def validate_rows(train, validation):
    for split, rows in (('train', train), ('validation', validation)):
        if not rows:
            raise ValueError('Empty ' + split)
        ids, hashes = set(), set()
        for row in rows:
            if row['partition'] != split:
                raise ValueError('Wrong partition')
            h = hashlib.sha256(row['text'].encode()).hexdigest()
            if h != row['content_hash'] or not row.get('provenance'):
                raise ValueError('Corrupt text or missing provenance')
            if row['document_id'] in ids or h in hashes:
                raise ValueError('Duplicate document')
            ids.add(row['document_id'])
            hashes.add(h)
    for key in ('document_id', 'content_hash'):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError('Train/validation overlap: ' + key)


def acquire_and_freeze(root):
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'acceptance.json', CRITERIA)  # precedes acquisition and ALL fits
    raw, provenance, acquisition = [], {}, {}
    specs = [s.to_dict() for s in mini.build_source_registry() if s.status == 'ACCEPT']
    write_json(root / 'source_registry.json', specs)
    # Retain the English helper's otherwise-discarded attribution metadata.
    def english_fetch(n):
        docs, report = mini.acquire_common_pile_wikimedia_sample(n, max_records_scanned=1000)
        provenance.update(report.provenance)
        return docs, report
    try:
        print('Acquiring bounded English sample', flush=True)
        docs, acquisition['english'] = mini.acquire_english(acquire=english_fetch)
        raw.extend(docs)
        print('Acquiring 100 pinned, blob-verified source/documentation files', flush=True)
        docs, prov, errors = mini.acquire_github_files()
        provenance.update(prov)
        raw.extend(docs)
        acquisition['github_errors'] = errors
        if errors:
            raise ValueError('Pinned file acquisition incomplete')
        print('Acquiring bounded Nepali snapshot rows', flush=True)
        docs, prov, acquisition['nepali'] = mini.acquire_nepali_wikipedia()
        provenance.update(prov)
        raw.extend(docs)
        write_rows(root / 'raw.jsonl', [dict(asdict(d), provenance=provenance[d.document_id]) for d in raw])
        write_json(root / 'acquisition.json', acquisition)
        built = mini.build_corpus(raw)
        # Reconstruct the prior mini-corpus before subsampling; do not silently
        # move its validation documents into this fit if an upstream API drifted.
        previous = json.loads(Path('representative_mini_corpus_result.json').read_text(encoding='utf-8'))
        for cat in FAMILIES:
            old = previous['category_distribution'][cat]['accepted_content_hash']
            if built['per_category'][cat]['accepted_content_hash'] != old:
                raise ValueError('Historical accepted corpus changed: ' + cat)
        if previous['split'] != dict(seed=mini.SEED, val_fraction=mini.VAL_FRACTION,
                                      train_documents=len(built['train']), validation_documents=len(built['validation'])):
            raise ValueError('Historical split configuration mismatch')
        selected = select_train(built['train'], CRITERIA['fit_byte_ceiling'])
        def rows(docs, split):
            return [dict(asdict(d), partition=split, provenance=provenance[d.document_id]) for d in docs]
        train, val = rows(selected, 'train'), rows(built['validation'], 'validation')
        validate_rows(train, val)
        if any(not v['documents'] for v in composition(train).values()):
            raise ValueError('Missing required fit text family')
        write_rows(root / 'train.jsonl', train)
        write_rows(root / 'validation.jsonl', val)
        write_json(root / 'partition_index.json', {
            split: [dict(document_id=d.document_id, content_hash=d.content_hash) for d in built[split]]
            for split in ('train', 'validation')})
        files = ['acceptance.json', 'source_registry.json', 'raw.jsonl', 'acquisition.json',
                 'train.jsonl', 'validation.jsonl', 'partition_index.json']
        manifest = dict(schema='yug-tokenizer-fit-sample-v1',
            files={name: sha(root / name) for name in files},
            train_identity=identity(train), validation_identity=identity(val),
            composition={'train': composition(train), 'validation': composition(val)},
            selection='whole documents, family then content_hash/document_id order; byte ceilings 300000 * approved proxy weight / 61',
            proxy_weights=WEIGHTS, dataset_allocation_changed=False,
            split=dict(seed=mini.SEED, val_fraction=mini.VAL_FRACTION, method='existing mini.stratified_split before sample selection'),
            normalization='existing mini.normalize_document unchanged',
            prior_mini_corpus_accepted_hashes_match=True,
            historical_other_experiments_validation_membership='UNKNOWN: IDs not retained; this is not model evaluation',
            near_duplicate_exclusion='exact normalized hashes only; near-duplicate contamination not ruled out',
            representative_16_category_coverage=False,
            coverage_limitations=['Five text families, not all 16 approved categories',
                'No natural technical Nepali-English transfer or codeswitch sample',
                'No Blender/Unreal coverage; Unity covered only by third-party C# sources',
                'No representative C/C++/shader sample',
                'Byte proxy weights are not final-token proportions; omitted categories are not reassigned',
                'Existing source/license decisions reused, not a new per-file rights audit',
                'Nepali API revision unpinned upstream; retained bytes and hashes freeze this local sample'],
            code_sha256={p: sha(Path(p)) for p in (
                'yug/tokenizer/bpe.py', 'training/experiments/representative_mini_corpus.py',
                'training/experiments/tokenizer_fit_sample_gate.py', 'training/corpus_prep.py', 'training/split.py')})
        manifest['identity'] = identity(manifest)
        write_json(root / 'manifest.json', manifest)
        print(json.dumps({'manifest_identity':manifest['identity'], 'composition':manifest['composition']}), flush=True)
    except Exception as exc:
        write_json(root / 'acquisition_failure.json', dict(error=str(exc), acquired_documents=len(raw), acquisition=acquisition))
        raise


def load_sample(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    declared = manifest.pop('identity')
    if identity(manifest) != declared:
        raise ValueError('Manifest identity mismatch')
    manifest['identity'] = declared
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha(root / name) != expected:
            raise ValueError('Sample file checksum mismatch: ' + name)
    train, val = read_rows(root / 'train.jsonl'), read_rows(root / 'validation.jsonl')
    validate_rows(train, val)
    if identity(train) != manifest['train_identity'] or identity(val) != manifest['validation_identity']:
        raise ValueError('Partition identity mismatch')
    return manifest, train, val


def measure(tok, rows):
    groups, digest = {}, hashlib.sha256()
    for family in FAMILIES:
        docs = [r for r in rows if r['primary_domain'] == family]
        n = chars = size = words = 0
        identifiers = set()
        for r in docs:
            text = r['text']
            ids = tok.encode(text)
            if tok.decode(ids) != text or any(i < 0 or i >= tok.vocab_size for i in ids):
                raise ValueError('Roundtrip or token range failure')
            digest.update(canonical([r['document_id'], ids]))
            n += len(ids)
            chars += len(text)
            size += len(text.encode())
            words += len(text.split())
            if family in ('code', 'csharp_gamedev'):
                identifiers.update(re.findall(r'\b[A-Za-z_][A-Za-z_0-9]*\b', text))
        # Deterministic bounded identifier diagnostic, measured separately in
        # interpretation from contextual document-token/whitespace-unit ratios.
        selected = sorted(identifiers)[:1000]
        groups[family] = dict(documents=len(docs), characters=chars, utf8_bytes=size, tokens=n,
            bytes_per_token=size/n if n else None, chars_per_token=chars/n if n else None,
            tokens_per_whitespace_unit=n/words if words else None,
            identifier_count=len(selected),
            tokens_per_isolated_identifier=sum(len(tok.encode(x)) for x in selected)/len(selected) if selected else None)
    totals = {key:sum(g[key] for g in groups.values()) for key in ('documents','characters','utf8_bytes','tokens')}
    totals['bytes_per_token'] = totals['utf8_bytes']/totals['tokens']
    totals['chars_per_token'] = totals['characters']/totals['tokens']
    return dict(groups=groups, totals=totals, encoded_sha256=digest.hexdigest())


def worker(root, vocab, run):
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer
    manifest, train, val = load_sample(root)
    limits = json.loads((root / 'acceptance.json').read_text())
    if vocab not in limits['vocab_targets']:
        raise ValueError('Unapproved vocabulary target')
    output = root / f'run-{vocab}-{run}'
    output.mkdir(exist_ok=False)
    state = {'phase':'fit', 'phase_start':time.perf_counter()}
    done = threading.Event()
    def watch():
        while not done.wait(0.5):
            mem = memory()
            elapsed = time.perf_counter()-state['phase_start']
            phase = state['phase']
            write_json(output / 'progress.json', dict(phase=phase, elapsed_seconds=elapsed, memory=mem))
            reason = None
            if mem['peak_rss_bytes'] is None:
                reason = 'memory_unavailable'
            elif mem['peak_rss_bytes'] > limits['peak_rss_limit_bytes']:
                reason = 'memory_limit'
            elif elapsed > limits[phase+'_wall_limit_seconds']:
                reason = phase+'_time_limit'
            if reason:
                write_json(output / 'failure.json', dict(reason=reason, phase=phase, memory=mem, elapsed_seconds=elapsed))
                os._exit(42)
    threading.Thread(target=watch, daemon=True).start()
    start, cpu = time.perf_counter(), time.process_time()
    tok = ByteLevelBpeTokenizer.train([r['text'] for r in train], vocab)
    wall, cpu_seconds, mem = time.perf_counter()-start, time.process_time()-cpu, memory()
    if wall > limits['fit_wall_limit_seconds'] or mem['peak_rss_bytes'] > limits['peak_rss_limit_bytes']:
        raise ValueError('Fit acceptance limit exceeded')
    tok.save(output / 'tokenizer.json')
    if ByteLevelBpeTokenizer.load(output / 'tokenizer.json').identity != tok.identity:
        raise ValueError('Save/load identity failure')
    fit = dict(sample_identity=manifest['identity'], train_identity=manifest['train_identity'],
        target_vocab=vocab, actual_vocab=tok.vocab_size, merges_completed=len(tok.to_dict()['merges']),
        fit_wall_seconds=wall, fit_cpu_seconds=cpu_seconds, peak_rss_bytes=mem['peak_rss_bytes'],
        tokenizer_identity=tok.identity, tokenizer_file_sha256=sha(output / 'tokenizer.json'),
        vocabulary_complete=tok.vocab_size==vocab, hashseed=os.environ.get('PYTHONHASHSEED'))
    write_json(output / 'fit.json', fit)
    state.update(phase='encode', phase_start=time.perf_counter())
    start, cpu = time.perf_counter(), time.process_time()
    results = dict(fit=fit, train=measure(tok,train), validation=measure(tok,val))
    results.update(encode_measurement_wall_seconds=time.perf_counter()-start,
                   encode_measurement_cpu_seconds=time.process_time()-cpu,
                   process_peak_rss_bytes=memory()['peak_rss_bytes'])
    done.set()
    write_json(output / 'result.json', results)
    print(json.dumps(fit), flush=True)


def run_comparison(root):
    manifest, _, _ = load_sample(root)
    rows = []
    for repeat in range(2):
        for vocab in (4096,8192):
            env = dict(os.environ, PYTHONHASHSEED=str(repeat+1))
            command = [sys.executable, '-B', '-m', __spec__.name, 'worker', str(root), '--vocab', str(vocab), '--run',str(repeat)]
            try:
                result = subprocess.run(command, env=env, timeout=810, check=False)
                row = dict(vocab=vocab, repeat=repeat, returncode=result.returncode)
            except subprocess.TimeoutExpired:
                row = dict(vocab=vocab, repeat=repeat, returncode=None, error='whole_worker_timeout_810_seconds')
            rows.append(row)
            write_json(root / 'execution.json', rows)
            if row['returncode'] != 0:
                raise RuntimeError('Worker failed resource gate; see execution.json and run failure.json')
    for vocab in (4096,8192):
        a,b = [json.loads((root/f'run-{vocab}-{r}'/'result.json').read_text()) for r in range(2)]
        for key in ('tokenizer_identity','tokenizer_file_sha256','actual_vocab','train_identity'):
            if a['fit'][key] != b['fit'][key]:
                raise ValueError('Non-reproducible '+key)
        if a['train'] != b['train'] or a['validation'] != b['validation']:
            raise ValueError('Non-reproducible encoding')
    write_json(root/'reproducibility.json',dict(passed=True, runs=4, sample_identity=manifest['identity']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze','run','worker','verify'))
    p.add_argument('root', type=Path)
    p.add_argument('--vocab',type=int)
    p.add_argument('--run',type=int)
    args = p.parse_args()
    if args.action == 'freeze':
        acquire_and_freeze(args.root)
    elif args.action == 'run':
        run_comparison(args.root)
    elif args.action == 'worker':
        worker(args.root,args.vocab,args.run)
    else:
        print(load_sample(args.root)[0]['identity'])


if __name__ == '__main__':
    main()
