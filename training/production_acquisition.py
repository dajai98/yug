"""Bounded, resumable production admission coordinator; no training entry point.

The immutable plan is JSONL with per-item clearance. SQLite holds stage state,
source split groups, dedup features and quota counters. Dedup settles BEFORE
accounting. Export reuses token_shards with a verified cached-token adapter.
Network acquisition is opt-in through an explicit fetch callback; default input
is a bounded local file under the supplied repository root.
"""
from __future__ import annotations
from array import array
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from training.production_spec import (TOKENIZER, QUOTAS, CHECKPOINTS, need, validate,
    acquisition_blockers, path_allowed, seal, verify_seal, local_path, source_manifest, iter_manifest_files)
from training.production_corpus import (canonical, identity, file_digest, atomic_json, atomic_replace)
from training.corpus_prep import normalize_text, check_quality, QualityConfig
from training.experiments.representative_mini_corpus import normalize_structured_text
from training.production_dedup import DedupIndex, POLICY
from training.token_shards import _writer_lock, _pack, build_token_shards, TokenShardStream

MAX_ITEM_BYTES=POLICY['max_document_bytes']

def digest(data): return 'sha256:'+hashlib.sha256(data).hexdigest()

def item_key(record):
    return identity({k:record[k] for k in ('source_id','revision','path','sha256')})

def read_plan(path):
    with Path(path).open('rb') as stream:
        while line:=stream.readline(MAX_ITEM_BYTES+1):
            need(len(line)<=MAX_ITEM_BYTES,'Oversized plan record')
            if line.strip(): yield json.loads(line)

def validate_clearance(record,source,spec,root,approved_file=None):
    """File-specific, content-bound approval; source ACCEPT is necessary only."""
    verify_seal(record)
    from training.conditional_license import validate_approval
    validate_approval(record,source,root)
    need(not acquisition_blockers(source),'Source is not production-cleared: '+source['source_id'])
    need(record['source_id']==source['source_id'] and record['revision']==source['revision'],'Clearance source/revision mismatch')
    need(record['clearance_status']=='ACCEPT','Item clearance does not permit admission')
    need(record['owner'] in source['allowed_categories'],'Item budget owner not permitted')
    categories={c['id']:c for c in spec['categories']}
    need(record['subcategory'] in categories[record['owner']]['subcategories'],'Unknown item subcategory')
    need(record['split']==source['split'] and record['split_group']==source['split_group'],'Source split conflict')
    need(record['split_unit_kind'] in ('repository','project','work','domain','dataset_subsource','documentation_project','source'),'Unknown split unit')
    need(record['allow_rule'] in source['allow_paths'],'Allow rule not in source policy')
    need(record['deny_rules_checked']==source['deny_paths'],'Deny policy differs from clearance')
    need(path_allowed(record['path'],[record['allow_rule']],source['deny_paths']),'Item path excluded')
    need(type(record['bytes']) is int and 0<record['bytes']<=MAX_ITEM_BYTES,'Item byte bound exceeded')
    need(record['content_type'] in ('code','markdown','prose') and isinstance(record['language'],str),'Unknown content/language route')
    need(record['license_id'] and record['license_evidence'] and record['attribution'],'Missing item license evidence')
    for evidence in record['license_evidence']:
        need(file_digest(local_path(root,evidence['path']))==evidence['sha256'],'Item license evidence corrupt')
    # Existing source clearance binds the same immutable manifest. The item
    # license must be explicit in that manifest, including mixed-license repos.
    if approved_file is None:
        manifest=source_manifest(root,source)
        matches=[f for f in iter_manifest_files(root,manifest) if f['path']==record['path']]
        need(len(matches)==1,'Item absent/ambiguous in cleared file manifest')
        approved_file=matches[0]
    f=approved_file
    for field in ('sha256','bytes','owner','content_type','license_id'):
        need(f.get(field)==record[field],'Item '+field+' differs from cleared manifest')
    need(record['quality_review']=='ACCEPT','Item quality review not cleared')
    topic,form=record.get('three_d_topic'),record.get('three_d_form')
    need(bool(topic)==bool(form),'3D topic/form required together')
    if topic:
        topics={t['id']:t for t in spec['three_d']['topics']}
        need(topic in topics and topics[topic]['owner']==record['owner'],'Wrong 3D owner')
        need(form in spec['three_d']['form_targets'],'Unknown 3D form')
    return item_key(record)

class ProductionBuild:
    def __init__(self,directory,spec,registry,*,repository_root,reference_plan=None,diagnostic=False,prior_production=None):
        validate(spec,registry,repository_root)
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        self.root=Path(repository_root).resolve();self.spec=json.loads(canonical(spec));self.registry=json.loads(canonical(registry))
        need(reference_plan is not None or diagnostic,'Production requires pinned held-out reference inventory')
        self.reference_plan=Path(reference_plan).resolve() if reference_plan is not None else None
        if self.reference_plan:need(self.reference_plan.is_relative_to(self.root),'Reference inventory escapes root')
        self.diagnostic=diagnostic
        self.prior_production=Path(prior_production).resolve() if prior_production is not None else None
        self.sources={s['source_id']:s for s in self.registry['sources']}
        self.db=sqlite3.connect(self.directory/'state.sqlite')
        self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('PRAGMA cache_size=-16384')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS items(item TEXT PRIMARY KEY,record TEXT NOT NULL,
          stage TEXT NOT NULL,reason TEXT,normalized_sha TEXT,token_sha TEXT,
          token_count INTEGER,split TEXT NOT NULL,owner TEXT NOT NULL,source TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS item_stages ON items(stage,split,item);
        CREATE INDEX IF NOT EXISTS item_normalized ON items(normalized_sha,stage);
        CREATE TABLE IF NOT EXISTS approved_files(source TEXT,path TEXT,payload TEXT,PRIMARY KEY(source,path));
        CREATE TABLE IF NOT EXISTS splits(group_id TEXT PRIMARY KEY,split TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS totals(kind TEXT,key TEXT,tokens INTEGER NOT NULL,
          PRIMARY KEY(kind,key));
        CREATE TABLE IF NOT EXISTS milestones(tokens INTEGER PRIMARY KEY,report TEXT NOT NULL);
        ''')
        request=dict(spec=spec['identity'],registry=registry['identity'],tokenizer=TOKENIZER,dedup=identity(POLICY),coordinator='production-admission-v1',diagnostic=diagnostic,reference_identity=file_digest(self.reference_plan) if self.reference_plan else None)
        old=self.get('request')
        if old is not None:need(old==request,'Incompatible production build resume')
        else:self.put('request',request);self.put('state','PARTIAL');self.db.commit()
        self.dedup=DedupIndex(self.db)
        for name in ('raw','normalized','tokens','clearances','quarantine'):(self.directory/name).mkdir(exist_ok=True)

    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def get(self,key):
        row=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None
    def put(self,key,value):self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,canonical(value).decode()))
    def path(self,kind,item,suffix=''):
        need(item.startswith('sha256:') and len(item)==71 and all(c in '0123456789abcdef' for c in item[7:]),'Invalid item identity')
        return self.directory/kind/(item[7:]+suffix)

    def register_references(self):
        if self.reference_plan is None:return
        need(file_digest(self.reference_plan)==self.get('request')['reference_identity'],'Held-out reference inventory changed')
        if self.get('references_registered'):return
        count=0
        with self.db:
            for record in read_plan(self.reference_plan):
                need(record['provenance'] and record['reference_id'],'Reference provenance missing')
                text=normalize_structured_text(record['text']) if record['content_type'] in ('code','markdown') else normalize_text(record['text'])
                key='reference:'+identity(dict(inventory=self.get('request')['reference_identity'],reference_id=record['reference_id']))
                provenance={k:v for k,v in record.items() if k!='text'}
                self.dedup.add(key,text,record['content_type'],record['language'],'validation',provenance)
                count+=1
            need(count>0,'Production held-out inventory is empty')
            self.put('references_registered',count)

    def register(self,plan):
        plan_hash=file_digest(plan)
        old=self.get('plan_identity')
        need(old is None or old==plan_hash,'Plan changed; cannot append to a finalized dedup universe')
        if self.get('plan_registered'):return
        # One transaction: partial plan import cannot become an accepted subset.
        with self.db:
            for source in self.sources.values():
                if acquisition_blockers(source):continue
                for f in iter_manifest_files(self.root,source_manifest(self.root,source)):
                    self.db.execute('INSERT OR REPLACE INTO approved_files VALUES (?,?,?)',(source['source_id'],f['path'],canonical(f).decode()))
            for record in read_plan(plan):
                source=self.sources.get(record['source_id']);need(source is not None,'Unknown source')
                row=self.db.execute('SELECT payload FROM approved_files WHERE source=? AND path=?',(source['source_id'],record['path'])).fetchone()
                need(row is not None,'Item not in pinned inventory')
                item=validate_clearance(record,source,self.spec,self.root,json.loads(row[0]))
                stored=self.db.execute('SELECT record FROM items WHERE item=?',(item,)).fetchone()
                need(stored is None,'Duplicate item in acquisition plan')
                if stored is None:
                    self.db.execute('INSERT INTO items(item,record,stage,split,owner,source) VALUES (?,?,?,?,?,?)',
                        (item,canonical(record).decode(),'PENDING',record['split'],record['owner'],record['source_id']))
                group=self.db.execute('SELECT split FROM splits WHERE group_id=?',(record['split_group'],)).fetchone()
                need(group is None or group[0]==record['split'],'Source group split conflict')
                self.db.execute('INSERT OR IGNORE INTO splits VALUES (?,?)',(record['split_group'],record['split']))
                clearance=self.path('clearances',item,'.json')
                if clearance.exists():need(json.loads(clearance.read_bytes())==record,'Immutable clearance file changed')
                else:atomic_json(clearance,record)
            self.put('plan_identity',plan_hash);self.put('plan_registered',True)

    def register_prior_production(self):
        """Bind an owner-supplied complete inventory; seals prove integrity, not authority.

        An empty inventory must explicitly declare genesis. Never infer genesis
        from a missing file. Legacy completed builds may replay, not admit more.
        Prior databases and payloads are opened read-only; only this build's
        existing dedup index receives reference members (never token accounting).
        """
        bound=self.get('prior_production')
        pending=not self.get('plan_registered') or self.db.execute(
            "SELECT 1 FROM items WHERE stage IN ('PENDING','ACQUIRED','PREPARED') LIMIT 1").fetchone()
        if self.prior_production is None:
            need(bound is None,'Bound prior-production inventory required on resume')
            need(self.diagnostic or not pending,'Production admission requires explicit prior-production inventory')
            return
        need(not self.diagnostic,'Prior production cannot be attached to diagnostic builds')
        inventory=json.loads(self.prior_production.read_bytes());verify_seal(inventory)
        need(inventory.get('schema')=='yug-prior-production-v1' and inventory.get('complete') is True,
             'Incomplete/incompatible prior-production inventory')
        entries=inventory.get('builds')
        need(isinstance(entries,list) and type(inventory.get('genesis')) is bool
             and inventory['genesis']==(not entries),'Explicit genesis or prior builds required')
        need(bound is None or bound==inventory,'Prior-production inventory changed on resume')
        # Do not retroactively reinterpret previously accounted items.
        need(bound is not None or not self.db.execute("SELECT 1 FROM items WHERE stage='ACCEPTED' LIMIT 1").fetchone(),
             'Cannot attach history after accounting; replay legacy build without new admission')
        paths={}
        for entry in entries:
            path=local_path(self.root,entry['path'])
            need(path!=self.directory/'state.sqlite' and path not in paths,'Self/duplicate prior production build')
            need(file_digest(path)==entry['sha256'],'Prior production database hash mismatch')
            paths[path]=entry['sha256']
        self.db.execute('CREATE TABLE IF NOT EXISTS prior_items(item TEXT PRIMARY KEY,split TEXT NOT NULL)')
        with self.db:
            for path,expected in sorted(paths.items()):
                db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
                try:
                    need(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Corrupt prior production database')
                    meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM meta')}
                    req=meta['request']
                    need(req['tokenizer']==TOKENIZER and req['dedup']==identity(POLICY)
                         and req['diagnostic'] is False and meta.get('dedup_settled')
                         and meta.get('state') in ('COMPLETE','TARGET_REACHED','TARGET_BOUNDARY'),
                         'Incompatible/unfinished prior production build')
                    for ancestor in (meta.get('prior_production') or {}).get('builds',[]):
                        need(paths.get(local_path(self.root,ancestor['path']))==ancestor['sha256'],
                             'Prior-production inventory omits/changes an ancestor')
                    totals=Counter()
                    for item,encoded,nsha,tsha,count,split,owner in db.execute(
                        "SELECT item,record,normalized_sha,token_sha,token_count,split,owner FROM items WHERE stage='ACCEPTED' ORDER BY item"):
                        r=json.loads(encoded);verify_seal(r)
                        need(item_key(r)==item and split==r['split'] and owner==r['owner'],'Prior item identity mismatch')
                        need(json.loads((path.parent/'clearances'/(item[7:]+'.json')).read_bytes())==r,'Prior clearance changed')
                        raw=path.parent/'raw'/item[7:]; normalized=path.parent/'normalized'/item[7:]
                        tokens=path.parent/'tokens'/(item[7:]+'.bin')
                        need(file_digest(raw)==r['sha256'] and file_digest(normalized)==nsha
                             and file_digest(tokens)==tsha and tokens.stat().st_size==count*4,'Prior production payload corrupt')
                        text=raw.read_bytes().decode('utf-8-sig')
                        text=normalize_structured_text(text) if r['content_type'] in ('code','markdown') else normalize_text(text)
                        need(digest(text.encode('utf-8'))==nsha,'Prior normalization mismatch')
                        key='prior:'+identity(dict(database=expected,item=item))
                        self.dedup.add(key,text,r['content_type'],r['language'],split,r)
                        self.db.execute('INSERT OR IGNORE INTO prior_items VALUES (?,?)',(key,split))
                        group=self.db.execute('SELECT split FROM splits WHERE group_id=?',(r['split_group'],)).fetchone()
                        need(group is None or group[0]==split,'Prior source group split conflict')
                        self.db.execute('INSERT OR IGNORE INTO splits VALUES (?,?)',(r['split_group'],split))
                        totals[('category' if split=='train' else 'validation_category',owner)]+=count
                    actual={(k,v):n for k,v,n in db.execute("SELECT kind,key,tokens FROM totals WHERE kind IN ('category','validation_category')")}
                    need(dict(totals)==actual,'Prior production accounting mismatch')
                finally:db.close()
                need(file_digest(path)==expected,'Prior database changed during verification')
            self.put('prior_production',inventory)

    def prior_clusters(self):
        self.db.execute('CREATE TEMP TABLE IF NOT EXISTS prior_clusters(cluster TEXT,split TEXT,PRIMARY KEY(cluster,split))')
        self.db.execute('DELETE FROM prior_clusters')
        if self.get('prior_production') is not None:
            for item,split in self.db.execute('SELECT item,split FROM prior_items ORDER BY item'):
                self.db.execute('INSERT OR IGNORE INTO prior_clusters VALUES (?,?)',
                    (self.dedup.assignment(item)['cluster_id'],split))

    def verify(self):
        need(self.db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Corrupt production state database')
        for item,encoded,stage,nsha,tsha,count in self.db.execute('SELECT item,record,stage,normalized_sha,token_sha,token_count FROM items ORDER BY item'):
            record=json.loads(encoded)
            need(item_key(record)==item,'Item key corrupt');verify_seal(record)
            need(json.loads(self.path('clearances',item,'.json').read_bytes())==record,'Clearance changed')
            if stage!='PENDING':need(file_digest(self.path('raw',item))==record['sha256'],'Committed raw payload corrupt')
            if nsha:need(file_digest(self.path('normalized',item))==nsha,'Committed normalized payload corrupt')
            if tsha:
                need(file_digest(self.path('tokens',item,'.bin'))==tsha,'Committed tokens corrupt')
                need(self.path('tokens',item,'.bin').stat().st_size==count*4,'Committed token size corrupt')
        self.dedup.verify()
        self.db.execute('CREATE TEMP TABLE IF NOT EXISTS expected_totals(kind TEXT,key TEXT,tokens INTEGER,PRIMARY KEY(kind,key))')
        self.db.execute('DELETE FROM expected_totals')
        for item,encoded,count in self.db.execute("SELECT item,record,token_count FROM items WHERE stage='ACCEPTED'"):
            record=json.loads(encoded)
            need(self.dedup.assignment(item)['canonical_item']==item,'Noncanonical item accounted')
            for kind,key,_ in self._limits(record):
                self.db.execute('INSERT INTO expected_totals VALUES (?,?,?) ON CONFLICT(kind,key) DO UPDATE SET tokens=tokens+excluded.tokens',(kind,key,count))
        need(self.db.execute('SELECT * FROM totals EXCEPT SELECT * FROM expected_totals').fetchone() is None and self.db.execute('SELECT * FROM expected_totals EXCEPT SELECT * FROM totals').fetchone() is None,'Committed accounting totals corrupt')
        self.db.commit()

    def _acquire(self,item,record,fetch):
        final=self.path('raw',item);temp=final.with_suffix('.part')
        # Verified uncommitted complete file can be reused after a crash.
        if final.exists():
            if final.stat().st_size==record['bytes'] and file_digest(final)==record['sha256']:return
            atomic_replace(final,self.path('quarantine',item,'.raw.bad'))
        if fetch is None:
            handle=local_path(self.root,record['input_path']).open('rb')
            chunks=iter(lambda:handle.read(65536),b'')
        else:handle=None;chunks=fetch(record)
        try:
            size=0;sha=hashlib.sha256()
            with temp.open('wb') as out:
                for block in chunks:
                    need(isinstance(block,bytes) and len(block)<=MAX_ITEM_BYTES,'Invalid/unbounded acquisition chunk')
                    size+=len(block);need(size<=record['bytes'],'Download exceeds approved size')
                    sha.update(block);out.write(block)
                out.flush();os.fsync(out.fileno())
            need(size==record['bytes'] and 'sha256:'+sha.hexdigest()==record['sha256'],'Acquisition hash/size mismatch')
            atomic_replace(temp,final)
        finally:
            if handle:handle.close()
            if hasattr(chunks,'close'):chunks.close()

    def _prepare(self,item,record):
        raw=self.path('raw',item).read_bytes() # explicit 1 MiB admission bound
        try:text=raw.decode('utf-8-sig')
        except UnicodeDecodeError:return 'invalid_utf8',None
        if '\x00' in text:return 'binary',None
        if any(marker.casefold() in text[:4096].casefold() for marker in self.spec['filters']['generated_markers']):return 'generated',None
        normalized=normalize_structured_text(text) if record['content_type'] in ('code','markdown') else normalize_text(text)
        reason=check_quality(normalized,QualityConfig())
        if reason:return reason.value,None
        data=normalized.encode('utf-8');out=self.path('normalized',item);temp=out.with_suffix('.part')
        with temp.open('wb') as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
        atomic_replace(temp,out)
        self.dedup.add(item,normalized,record['content_type'],record['language'],record['split'],record)
        return None,digest(data)

    def _cached_tokens(self,item,record,tokenizer):
        target=self.path('tokens',item,'.bin');metadata=self.path('tokens',item,'.json')
        normalized=self.path('normalized',item);nsha=file_digest(normalized)
        if metadata.exists():
            m=json.loads(metadata.read_bytes());verify_seal(m)
            need(m['tokenizer_identity']==TOKENIZER and m['normalized_sha256']==nsha,'Token cache input changed')
            need(file_digest(target)==m['sha256'] and target.stat().st_size==m['count']*4,'Token cache corrupt')
            return m
        ids=tokenizer.encode(normalized.read_text(encoding='utf-8'))
        temp=target.with_suffix('.part');sha=hashlib.sha256()
        with temp.open('wb') as out:
            for block in _pack(ids,4096):out.write(block);sha.update(block)
            out.flush();os.fsync(out.fileno())
        atomic_replace(temp,target)
        m=seal(dict(tokenizer_identity=TOKENIZER,normalized_sha256=nsha,sha256='sha256:'+sha.hexdigest(),count=len(ids)))
        atomic_json(metadata,m);return m

    def _limits(self,record):
        s=self.sources[record['source_id']]
        limits=[('source',s['source_id'],s['target_cap_tokens']),
            ('source_owner',s['source_id']+':'+record['owner'],s['category_caps_tokens'][record['owner']])]
        for field in ('repository_group','domain_group'):limits.append((field,s[field],self.spec['group_caps'][field][s[field]]))
        if record['split']=='train':
            limits.append(('category',record['owner'],QUOTAS[record['owner']]))
            if record.get('three_d_topic'):
                topics={t['id']:t['target_tokens'] for t in self.spec['three_d']['topics']}
                limits.extend([('three_d_topic',record['three_d_topic'],topics[record['three_d_topic']]),('three_d_form',record['three_d_form'],self.spec['three_d']['form_targets'][record['three_d_form']])])
        else:limits.append(('validation_category',record['owner'],2_000_000_000))
        return limits

    def _account(self,item,record,count):
        limits=self._limits(record)
        for kind,key,limit in limits:
            current=self.db.execute('SELECT tokens FROM totals WHERE kind=? AND key=?',(kind,key)).fetchone()
            if (current[0] if current else 0)+count>limit:return kind+'_cap'
        for kind,key,_ in limits:
            self.db.execute('INSERT INTO totals VALUES (?,?,?) ON CONFLICT(kind,key) DO UPDATE SET tokens=tokens+excluded.tokens',(kind,key,count))
        return None

    def dry_run(self,plan):
        """Validate/register an immutable scope without acquiring or encoding.

        Held-out references are indexed, but planned payloads stay PENDING.
        Content/near-duplicate clearance evidence must precede this preflight;
        actual admission re-verifies downloaded bytes and runs global dedup.
        """
        with _writer_lock(self.directory/'coordinator'):
            try:
                self.register_references();self.register(plan);self.verify()
                need(self.db.execute("SELECT count(*) FROM items WHERE stage!='PENDING'").fetchone()[0]==0,'Dry run requires an unacquired scope')
                report=self.checkpoint('DRY_RUN')
                report=seal(dict(report,dry_run=True,downloaded_bytes=0,
                    scheduled_download_files=self.db.execute('SELECT count(*) FROM items').fetchone()[0],
                    scheduled_download_bytes=report['planned_raw_bytes'],encoded_tokens=0))
                atomic_json(self.directory/'dry_run.json',report)
                return report
            except Exception:
                self.db.rollback()
                raise

    def run(self,plan,tokenizer,*,fetch=None,max_items=None,target_tokens=None):
        need(target_tokens is None or (type(target_tokens) is int and 0<target_tokens<=sum(QUOTAS.values())),'Invalid target token ceiling')
        need(max_items is None or (type(max_items) is int and max_items>0),'max_items must be a positive integer or None')
        need(tokenizer.identity==TOKENIZER and tokenizer.vocab_size==4096,'Frozen tokenizer mismatch')
        with _writer_lock(self.directory/'coordinator'):
            try:
                self.register_prior_production()
                self.register_references();self.register(plan);self.verify();done=0
                for item,encoded,stage in self.db.execute("SELECT item,record,stage FROM items WHERE stage IN ('PENDING','ACQUIRED') ORDER BY item"):
                    if max_items is not None and done>=max_items:return self.checkpoint('PARTIAL')
                    record=json.loads(encoded)
                    if stage=='PENDING':
                        self._acquire(item,record,fetch)
                        with self.db:self.db.execute("UPDATE items SET stage='ACQUIRED' WHERE item=?",(item,))
                    with self.db:
                        reason,nsha=self._prepare(item,record)
                        self.db.execute('UPDATE items SET stage=?,reason=?,normalized_sha=? WHERE item=?',('FILTERED' if reason else 'PREPARED',reason,nsha,item))
                    done+=1
                self.put('dedup_settled',True);self.db.commit()
                self.prior_clusters()
                for item,encoded in self.db.execute("SELECT item,record FROM items WHERE stage='PREPARED' ORDER BY item"):
                    if max_items is not None and done>=max_items:return self.checkpoint('PARTIAL')
                    record=json.loads(encoded);assignment=self.dedup.assignment(item)
                    prior=list(self.db.execute('SELECT split FROM prior_clusters WHERE cluster=?',(assignment['cluster_id'],)))
                    if prior:
                        reason='prior_production_split_conflict' if any(s!=record['split'] for (s,) in prior) else 'prior_production_duplicate'
                        with self.db:self.db.execute("UPDATE items SET stage='DUPLICATE',reason=? WHERE item=?",(reason,item))
                    elif assignment['canonical_item']!=item:
                        with self.db:self.db.execute("UPDATE items SET stage='DUPLICATE',reason=? WHERE item=?",(canonical(assignment).decode(),item))
                    else:
                        need(assignment['split']==record['split'],'Cluster split conflict')
                        if target_tokens is not None and record['split']=='train':
                            current=self.db.execute("SELECT coalesce(sum(tokens),0) FROM totals WHERE kind='category'").fetchone()[0]
                            if current>=target_tokens:return self.checkpoint('TARGET_REACHED')
                        m=self._cached_tokens(item,record,tokenizer)
                        if target_tokens is not None and record['split']=='train' and current+m['count']>target_tokens:
                            return self.checkpoint('TARGET_BOUNDARY')
                        with self.db:
                            reason=self._account(item,record,m['count'])
                            self.db.execute('UPDATE items SET stage=?,reason=?,token_sha=?,token_count=? WHERE item=?',('QUOTA_REJECTED' if reason else 'ACCEPTED',reason,m['sha256'],m['count'],item))
                    done+=1
                    total=self.db.execute("SELECT coalesce(sum(tokens),0) FROM totals WHERE kind='category'").fetchone()[0]
                    for milestone in CHECKPOINTS:
                        if total>=milestone and not self.db.execute('SELECT 1 FROM milestones WHERE tokens=?',(milestone,)).fetchone():
                            self.db.execute('INSERT INTO milestones VALUES (?,?)',(milestone,canonical(self.progress('PARTIAL')).decode()));self.db.commit()
                return self.checkpoint('COMPLETE')
            except Exception as exc:
                self.db.rollback();self.put('last_error',type(exc).__name__+': '+str(exc));self.db.commit()
                self.checkpoint('FAILED');raise

    def progress(self,state):
        totals={(k,key):n for k,key,n in self.db.execute('SELECT kind,key,tokens FROM totals')}
        stages=dict(self.db.execute('SELECT stage,count(*) FROM items GROUP BY stage'))
        train={c:totals.get(('category',c),0) for c in QUOTAS}
        validation={c:totals.get(('validation_category',c),0) for c in QUOTAS}
        return seal(dict(schema='yug-production-build-progress-v1',state=state,request=self.get('request'),plan_identity=self.get('plan_identity'),
            stages=stages,accepted_clean_train_tokens=sum(train.values()),category_tokens=train,category_validation_tokens=validation,
            category_deficits={c:QUOTAS[c]-train[c] for c in QUOTAS},
            source_progress={sid:dict(tokens=totals.get(('source',sid),0),remaining_cap=s['target_cap_tokens']-totals.get(('source',sid),0)) for sid,s in self.sources.items()},
            rejected_reasons=dict(self.db.execute("SELECT reason,count(*) FROM items WHERE stage IN ('FILTERED','QUOTA_REJECTED') GROUP BY reason")),
            duplicate_documents=stages.get('DUPLICATE',0),source_diversity=self.db.execute("SELECT count(DISTINCT source) FROM items WHERE stage='ACCEPTED'").fetchone()[0],
            planned_raw_bytes=sum(json.loads(r[0])['bytes'] for r in self.db.execute('SELECT record FROM items')),
            verified_acquired_raw_bytes=sum(json.loads(r[0])['bytes'] for r in self.db.execute("SELECT record FROM items WHERE stage!='PENDING'")),
            duplication_rate=stages.get('DUPLICATE',0)/sum(stages.values()) if stages else None,
            three_d_topic_tokens={t['id']:totals.get(('three_d_topic',t['id']),0) for t in self.spec['three_d']['topics']},
            three_d_form_tokens={f:totals.get(('three_d_form',f),0) for f in self.spec['three_d']['form_targets']},
            license_status_counts=dict(self.db.execute("SELECT json_extract(record,'$.license_id'),count(*) FROM items WHERE stage='ACCEPTED' GROUP BY json_extract(record,'$.license_id')")),
            raw_tokens=None,rejected_raw_tokens=None,tokenizer_statistics='Only cleared canonical items encoded; raw/rejected token counts not fabricated',
            storage_bytes=sum(p.stat().st_size for p in self.directory.rglob('*') if p.is_file()),
            actual_corpus_complete=False,diagnostic=self.diagnostic,heldout_references=self.get('references_registered') or 0,plan_complete=state=='COMPLETE',last_error=self.get('last_error')))

    def checkpoint(self,state):
        self.put('state',state)
        if state!='FAILED':self.put('last_error',None)
        self.db.commit();report=self.progress(state);atomic_json(self.directory/'progress.json',report);return report

    def export_shards(self,output_directory=None):
        try:return self._export_shards(output_directory)
        except Exception as exc:
            self.db.rollback();self.put('last_error',type(exc).__name__+': '+str(exc));self.db.commit();self.checkpoint('FAILED');raise

    def _export_shards(self,output_directory=None):
        destination=Path(output_directory).resolve() if output_directory is not None else self.directory
        need(destination.is_relative_to(self.directory),'Export must stay inside build directory')
        destination.mkdir(parents=True,exist_ok=True)
        need(self.get('state') in ('COMPLETE','TARGET_REACHED','TARGET_BOUNDARY') and self.get('dedup_settled'),'Build not complete')
        with _writer_lock(self.directory/'coordinator'):
            self.verify()
            accepted=destination/'admission.jsonl'
            temp=accepted.with_suffix('.part')
            with temp.open('wb') as out:
                for item,record,stage,reason,nsha,tsha,count in self.db.execute('SELECT item,record,stage,reason,normalized_sha,token_sha,token_count FROM items ORDER BY item'):
                    out.write(canonical(dict(item=item,clearance=json.loads(record),stage=stage,reason=reason,normalized_sha256=nsha,token_sha256=tsha,tokens=count))+b'\n')
                out.flush();os.fsync(out.fileno())
            atomic_replace(temp,accepted)
            counts={split:self.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED' AND split=?",(split,)).fetchone()[0] for split in ('train','validation')}
            split_id=identity(dict(plan=self.get('plan_identity'),rule='source-group-v1-validation-priority-clusters',admission_hash=file_digest(accepted)))
            manifest=dict(schema_version='yug-admitted-production-v1',tokenizer_identity=TOKENIZER,split_identity=split_id,
                spec_identity=self.spec['identity'],registry_identity=self.registry['identity'],plan_identity=self.get('plan_identity'),diagnostic=self.diagnostic,reference_identity=self.get('request')['reference_identity'],
                dedup_policy=POLICY,admission_sha256=file_digest(accepted),counts=counts)
            manifest['prepared_corpus_identity']=identity(manifest)
            atomic_json(destination/'prepared.json',manifest)
            prepared=_Prepared(self,manifest);cache=_TokenCache(self)
            result=build_token_shards(prepared,cache,destination/'shards')
            for split in ('train','validation'):
                TokenShardStream(destination/'shards',split=split,tokenizer_identity=TOKENIZER,prepared_corpus_identity=prepared.identity)
            return result

class _Prepared:
    def __init__(self,build,manifest):
        self.build=build;self.manifest=manifest;self.identity=manifest['prepared_corpus_identity']
        self.index={'splits':{k:range(v) for k,v in manifest['counts'].items()}}
    def documents(self,split,start=0):
        for item, in self.build.db.execute("SELECT item FROM items WHERE stage='ACCEPTED' AND split=? ORDER BY item LIMIT -1 OFFSET ?",(split,start)):
            yield self.build.path('normalized',item).read_text(encoding='utf-8')

class _TokenCache:
    identity=TOKENIZER;vocab_size=4096
    def __init__(self,build):self.build=build
    def encode(self,text):
        row=self.build.db.execute("SELECT item,token_sha FROM items WHERE stage='ACCEPTED' AND normalized_sha=?",(digest(text.encode('utf-8')),)).fetchone()
        need(row is not None,'Unaccounted normalized payload')
        path=self.build.path('tokens',row[0],'.bin');data=path.read_bytes();need(digest(data)==row[1],'Cached token payload corrupt')
        ids=array('I');ids.frombytes(data)
        if sys.byteorder!='little':ids.byteswap()
        return ids
