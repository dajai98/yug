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
from training.corpus_prep import normalize_text, QualityConfig
from training.quality_filtering import check_quality, quality_policy
from training.experiments.representative_mini_corpus import normalize_structured_text
from training.production_dedup import DedupIndex, POLICY, features
from training.token_shards import _writer_lock, _pack, build_token_shards, TokenShardStream, DEFAULT_SHARD_TOKENS
from training import content_screening
from training.language_validation import POLICY as LANGUAGE_POLICY, validate_label, validate_language
from training.protected_references import read_inventory
from training.category_ownership import (POLICY_IDENTITY as OWNERSHIP_POLICY_IDENTITY,
    verify as verify_ownership, verify_approval, verify_structure)
from training.cumulative_accounting import CorpusLedger, build_identity
from training.notice_attribution import (FILENAME as NOTICE_FILENAME, build_notice,
    obligation as notice_obligation, reference as notice_reference,
    source_clearance, verify_file as verify_notice_file)
from training.approval_governance import Governance

MAX_ITEM_BYTES=POLICY['max_document_bytes']

def digest(data): return 'sha256:'+hashlib.sha256(data).hexdigest()

def item_key(record):
    return identity({k:record[k] for k in ('source_id','revision','path','sha256')})

def read_plan(path):
    with Path(path).open('rb') as stream:
        while line:=stream.readline(MAX_ITEM_BYTES+1):
            need(len(line)<=MAX_ITEM_BYTES,'Oversized plan record')
            if line.strip(): yield json.loads(line)

def validate_clearance(record,source,spec,root,approved_file=None,*,allow_legacy=False):
    """File-specific, content-bound approval; source ACCEPT is necessary only."""
    verify_seal(record)
    from training.conditional_license import validate_approval
    validate_approval(record,source,root)
    need(not acquisition_blockers(source),'Source is not production-cleared: '+source['source_id'])
    need(record['source_id']==source['source_id'] and record['revision']==source['revision'],'Clearance source/revision mismatch')
    need(record['clearance_status']=='ACCEPT','Item clearance does not permit admission')
    need(type(record.get('owner')) is str and record['owner'] in QUOTAS,
         'Missing or unsupported single budget owner')
    need(record['owner'] in source['allowed_categories'],'Item budget owner not permitted')
    categories={c['id']:c for c in spec['categories']}
    need(record['subcategory'] in categories[record['owner']]['subcategories'],'Unknown item subcategory')
    need(record['split']==source['split'] and record['split_group']==source['split_group'],'Source split conflict')
    need(record['split_unit_kind'] in ('repository','project','work','domain','dataset_subsource','documentation_project','source'),'Unknown split unit')
    need(record['allow_rule'] in source['allow_paths'],'Allow rule not in source policy')
    need(record['deny_rules_checked']==source['deny_paths'],'Deny policy differs from clearance')
    need(path_allowed(record['path'],[record['allow_rule']],source['deny_paths']),'Item path excluded')
    need(type(record['bytes']) is int and 0<record['bytes']<=MAX_ITEM_BYTES,'Item byte bound exceeded')
    validate_label(record.get('language'),record.get('content_type'))
    need(record['license_id'] and record['license_evidence'] and record['attribution'],'Missing item license evidence')
    grant=source_clearance(source,root)
    bound=record.get('source_clearance_identity')
    need(bound==grant['identity'] or (allow_legacy and bound is None),
         'Item source-clearance identity missing or changed')
    notice_obligation(record,source,grant,root,allow_legacy=allow_legacy and bound is None)
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
    verify_ownership(record,f,source,root)
    need(record['quality_review']=='ACCEPT','Item quality review not cleared')
    topic,form=record.get('three_d_topic'),record.get('three_d_form')
    need(bool(topic)==bool(form),'3D topic/form required together')
    if topic:
        topics={t['id']:t for t in spec['three_d']['topics']}
        need(topic in topics and topics[topic]['owner']==record['owner'],'Wrong 3D owner')
        need(form in spec['three_d']['form_targets'],'Unknown 3D form')
    return item_key(record)

class ProductionBuild:
    def __init__(self,directory,spec,registry,*,repository_root,reference_plan=None,diagnostic=False,prior_production=None,expected_plan_sha256=None,ownership_approval=None,expected_ownership_approval_sha256=None,corpus_ledger=None,expected_corpus_head=None,approved_historical_import_identity=None,approved_historical_manifest_identity=None,approved_historical_manifest_sha256=None,governance_log=None,expected_governance_sha256=None):
        validate(spec,registry,repository_root)
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        self.root=Path(repository_root).resolve();self.spec=json.loads(canonical(spec));self.registry=json.loads(canonical(registry))
        need(reference_plan is not None or diagnostic,'Production requires pinned held-out reference inventory')
        self.reference_plan=Path(reference_plan).resolve() if reference_plan is not None else None
        if self.reference_plan:need(self.reference_plan.is_relative_to(self.root),'Reference inventory escapes root')
        self.diagnostic=diagnostic
        need(expected_plan_sha256 is None or (isinstance(expected_plan_sha256,str) and
             expected_plan_sha256.startswith('sha256:') and len(expected_plan_sha256)==71 and
             all(c in '0123456789abcdef' for c in expected_plan_sha256[7:])),
             'Invalid external production plan identity')
        self.expected_plan_sha256=expected_plan_sha256
        self.ownership_approval=Path(ownership_approval).resolve() if ownership_approval is not None else None
        self.expected_ownership_approval_sha256=expected_ownership_approval_sha256
        self.governance_log=Path(governance_log).resolve() if governance_log is not None else None
        self.expected_governance_sha256=expected_governance_sha256
        need(diagnostic or self.governance_log is None or
             not self.governance_log.is_relative_to(self.directory),
             'Governance status must be external to mutable build state')
        if not diagnostic:
            self.verify_approval_authority()
        self.prior_production=Path(prior_production).resolve() if prior_production is not None else None
        self.corpus_ledger_path=Path(corpus_ledger).resolve() if corpus_ledger is not None else None
        self.expected_corpus_head=expected_corpus_head
        self.historical_approval=dict(
            approved_historical_import_identity=approved_historical_import_identity,
            approved_historical_manifest_identity=approved_historical_manifest_identity,
            approved_historical_manifest_sha256=approved_historical_manifest_sha256)
        self.corpus_ledger=None
        self.corpus_build_id=None
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
        request=dict(spec=spec['identity'],registry=registry['identity'],tokenizer=TOKENIZER,dedup=identity(POLICY),quality_policy=self.quality_policy_identity(),language_policy=identity(LANGUAGE_POLICY),ownership_policy=OWNERSHIP_POLICY_IDENTITY,plan_authority=expected_plan_sha256,ownership_approval_authority=expected_ownership_approval_sha256,ownership_approval_path=str(self.ownership_approval) if self.ownership_approval else None,coordinator='production-admission-v1',diagnostic=diagnostic,reference_identity=file_digest(self.reference_plan) if self.reference_plan else None)
        old=self.get('request')
        self._new_build=old is None
        if old is not None:need(old==request,'Incompatible production build resume')
        else:self.put('request',request);self.put('state','PARTIAL');self.db.commit()
        # Check compatibility before introducing new state into a historical DB.
        columns=[row[1] for row in self.db.execute('PRAGMA table_info(quality_decisions)')]
        expected=['item','policy_identity','normalized_sha','outcome','reason','decision_identity']
        need(not columns or columns==expected,
             'Incompatible quality decision evidence; fresh build required')
        self.db.execute('''CREATE TABLE IF NOT EXISTS quality_decisions(
          item TEXT PRIMARY KEY,policy_identity TEXT NOT NULL,
          normalized_sha TEXT,outcome TEXT NOT NULL,reason TEXT,
          decision_identity TEXT NOT NULL)''')
        self.db.commit()
        self.dedup=DedupIndex(self.db)
        for name in ('raw','normalized','tokens','clearances','quarantine','screening'):(self.directory/name).mkdir(exist_ok=True)

    def close(self):
        if self.corpus_ledger is not None:self.corpus_ledger.close()
        self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def get(self,key):
        row=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None
    def put(self,key,value):self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,canonical(value).decode()))
    def path(self,kind,item,suffix=''):
        need(item.startswith('sha256:') and len(item)==71 and all(c in '0123456789abcdef' for c in item[7:]),'Invalid item identity')
        return self.directory/kind/(item[7:]+suffix)

    def current_quality_policy(self):
        return quality_policy(QualityConfig(),self.spec['filters']['generated_markers'])

    def quality_policy_identity(self):
        return identity(self.current_quality_policy())

    def verify_approval_authority(self):
        if not self.diagnostic:
            verify_approval(self.ownership_approval,self.expected_ownership_approval_sha256,
                            self.expected_plan_sha256,self.directory)

    def require_current_approvals(self,plan=None,*,records=None,receipts=()):
        """Future admission/use requires the caller's current external status pin.

        Historical verify() deliberately does not require current approval.
        """
        if self.diagnostic:return None
        plan_path=Path(plan) if plan is not None else Path(self.get('plan_path'))
        plan_sha=file_digest(plan_path)
        need(plan_sha==self.expected_plan_sha256,'Plan differs from external approved identity')
        ownership=verify_approval(self.ownership_approval,
            self.expected_ownership_approval_sha256,plan_sha,self.directory)
        selected=tuple(records) if records is not None else None
        grants={}
        for record in read_plan(plan_path) if selected is None else selected:
            source=self.sources[record['source_id']]
            if source['source_id'] not in grants:
                grants[source['source_id']]=source_clearance(source,self.root)['identity']
        need(file_digest(plan_path)==plan_sha,'Plan changed during current approval check')
        # Perform slow artifact checks first, then evaluate every required status
        # from one freshly pinned governance byte snapshot.
        governance=Governance(self.governance_log,self.expected_governance_sha256)
        governance.require_current('production_plan',plan_sha,
                                   dict(plan_sha256=plan_sha))
        governance.require_current('ownership_approval',ownership,
                                   dict(plan_sha256=plan_sha))
        for record in read_plan(plan_path) if selected is None else selected:
            source=self.sources[record['source_id']]
            governance.require_current('source_clearance',grants[source['source_id']],
                dict(source_id=source['source_id'],revision=source['revision']))
            governance.require_current('file_clearance',record['identity'],
                dict(source_id=record['source_id'],revision=record['revision'],
                     path=record['path'],sha256=record['sha256']))
            if record.get('conditional_approval_identity') is not None:
                governance.require_current('conditional_approval',
                    record['conditional_approval_identity'],
                    dict(source_id=record['source_id'],revision=record['revision'],
                         license_id=record['license_id']))
        for receipt in receipts:
            governance.require_current('decision_receipt',receipt['identity'],
                dict(clearance_identity=receipt['binding']['clearance_identity']))
        if selected is None:
            # The full-plan pass performs file I/O after loading the status
            # snapshot. Require the same pinned bytes after that pass too.
            need(file_digest(plan_path)==plan_sha,'Plan changed during current approval check')
            fresh=Governance(self.governance_log,self.expected_governance_sha256)
            need(fresh.byte_sha256==governance.byte_sha256,
                 'Governance changed during current approval check')
            return fresh
        return governance

    def _legacy_obligation_record(self,record):
        """Old completed, externally pinned builds may be read, never newly admitted."""
        return (record.get('source_clearance_identity') is None and
                self.get('state') in ('COMPLETE','TARGET_REACHED','TARGET_BOUNDARY') and
                self.get('plan_registered') is True and
                self.get('plan_identity')==self.expected_plan_sha256 and
                self.get('corpus_build_id') is None)

    def record_quality(self,item,outcome,reason,normalized=None):
        normalized_sha=digest(normalized.encode('utf-8')) if normalized is not None else None
        decision=dict(item=item,policy_identity=self.quality_policy_identity(),
                      normalized_sha256=normalized_sha,outcome=outcome,reason=reason)
        previous=self.quality_decision(item)
        need(previous is None or all(previous[k]==v for k,v in decision.items()),
             'Quality decision changed; fresh build required')
        self.db.execute('INSERT OR REPLACE INTO quality_decisions VALUES (?,?,?,?,?,?)',
                        (item,decision['policy_identity'],normalized_sha,outcome,reason,identity(decision)))

    def quality_decision(self,item):
        row=self.db.execute('SELECT policy_identity,normalized_sha,outcome,reason,decision_identity FROM quality_decisions WHERE item=?',(item,)).fetchone()
        if row is None:return None
        decision=dict(item=item,policy_identity=row[0],normalized_sha256=row[1],outcome=row[2],reason=row[3])
        need(row[4]==identity(decision),'Quality decision identity mismatch')
        return dict(decision,decision_identity=row[4])

    def verify_quality_acceptance(self,item,record,normalized_sha):
        decision=self.quality_decision(item)
        need(decision is not None and decision['policy_identity']==self.quality_policy_identity() and
             decision['outcome']=='PASS' and decision['reason'] is None and
             decision['normalized_sha256']==normalized_sha,
             'Quality acceptance decision changed')
        raw=self.path('raw',item).read_bytes()
        need(digest(raw)==record['sha256'],'Committed raw payload corrupt')
        try:text=raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            need(False,'Quality accepted invalid UTF-8')
        need('\x00' not in text and not any(marker.casefold() in text.casefold()
             for marker in self.spec['filters']['generated_markers']),
             'Quality accepted ineligible raw content')
        normalized=normalize_structured_text(text) if record['content_type'] in ('code','markdown') else normalize_text(text)
        need(digest(normalized.encode('utf-8'))==normalized_sha,
             'Quality evaluated content changed')
        need(check_quality(normalized,QualityConfig(),content_type=record['content_type']) is None,
             'Quality decision does not match current policy')
        validate_language(record['language'],record['content_type'],normalized)

    def register_references(self):
        if self.reference_plan is None:return
        need(file_digest(self.reference_plan)==self.get('request')['reference_identity'],'Held-out reference inventory changed')
        if self.get('references_registered') is not None:
            self.verify_references()
            return
        count=0
        with self.db:
            for record in read_inventory(self.reference_plan):
                count+=1
                text=normalize_structured_text(record['text']) if record['content_type'] in ('code','markdown') else normalize_text(record['text'])
                key='reference:'+identity(dict(inventory=self.get('request')['reference_identity'],reference_id=record['reference_id']))
                provenance={k:v for k,v in record.items() if k!='text'}
                self.dedup.add(key,text,record['content_type'],record['language'],'validation',provenance)
            self.put('references_registered',count)
            self.verify_references()

    def verify_references(self):
        """Verify the pinned inventory and every indexed reference before use."""
        if self.reference_plan is None:return
        pinned=self.get('request')['reference_identity']
        need(file_digest(self.reference_plan)==pinned,'Held-out reference inventory changed')
        registered=self.get('references_registered')
        need(type(registered) is int and registered>0,'Protected references not registered')
        count=0
        for record in read_inventory(self.reference_plan):
            count+=1
            text=normalize_structured_text(record['text']) if record['content_type'] in ('code','markdown') else normalize_text(record['text'])
            key='reference:'+identity(dict(inventory=pinned,reference_id=record['reference_id']))
            provenance={k:v for k,v in record.items() if k!='text'}
            actual=self.db.execute('SELECT exact,split,provenance FROM dd WHERE item=?',(key,)).fetchone()
            need(actual==(digest(text.encode('utf-8')),'validation',canonical(provenance).decode()),
                 'Protected reference index missing or changed')
        need(file_digest(self.reference_plan)==pinned,'Held-out reference inventory changed')
        indexed=self.db.execute("SELECT count(*) FROM dd WHERE item LIKE 'reference:%'").fetchone()[0]
        need(count==registered and indexed==count,'Protected reference index count changed')

    def register(self,plan):
        self.verify_approval_authority()
        self.require_current_approvals(plan)
        registered=self.get('plan_registered')
        need(registered is True or (registered is None and self._new_build and
             self.get('plan_identity') is None and self.get('plan_path') is None and
             self.db.execute('SELECT 1 FROM items LIMIT 1').fetchone() is None),
             'Missing or invalid production plan registration')
        plan_hash=file_digest(plan)
        if self.expected_plan_sha256 is None:
            need(self.diagnostic,'Production admission requires an external approved plan SHA256')
            # Diagnostic builds may bind a plan for this process only. A later
            # process must supply the original digest to resume that build.
            need(not self.get('plan_registered'),'Diagnostic resume requires the original external plan SHA256')
            self.expected_plan_sha256=plan_hash
            request=self.get('request');request['plan_authority']=plan_hash
            self.put('request',request);self.db.commit()
        need(plan_hash==self.expected_plan_sha256,'Plan changed; differs from external approved identity')
        old=self.get('plan_identity')
        need(old is None or old==plan_hash,'Plan changed; cannot append to a finalized dedup universe')
        plan_path=str(Path(plan).resolve())
        need(self.get('plan_path') is None or self.get('plan_path')==plan_path,
             'Pinned production plan path changed')
        if registered is True:return
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
            self.put('plan_identity',plan_hash);self.put('plan_path',plan_path);self.put('plan_registered',True)

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
        self.verify_approval_authority()
        need(self.expected_plan_sha256 is not None and
             self.get('request').get('plan_authority')==self.expected_plan_sha256,
             'Missing or incompatible external production plan authority')
        need(self.get('request').get('language_policy')==identity(LANGUAGE_POLICY),
             'Language policy changed; fresh build required')
        need(self.get('request').get('ownership_policy')==OWNERSHIP_POLICY_IDENTITY,
             'Ownership policy changed; fresh build required')
        need(self.get('request')['quality_policy']==self.quality_policy_identity(),
             'Quality policy changed; fresh build required')
        need(self.db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Corrupt production state database')
        need(self.get('plan_registered') is True,'Missing or invalid production plan registration')
        plan_path=self.get('plan_path')
        need(isinstance(plan_path,str) and bool(plan_path), 'Missing pinned production plan path')
        need(self.get('plan_identity')==self.expected_plan_sha256,
             'Stored plan identity differs from external approved identity')
        need(file_digest(plan_path)==self.get('plan_identity'), 'Pinned production plan changed')
        planned=0
        for expected in read_plan(plan_path):
            verify_seal(expected)
            item=item_key(expected)
            actual=self.db.execute('SELECT record FROM items WHERE item=?',(item,)).fetchone()
            need(actual is not None and actual[0]==canonical(expected).decode(),
                 'Stored language/clearance differs from pinned production plan')
            planned+=1
        need(planned==self.db.execute('SELECT count(*) FROM items').fetchone()[0],
             'Stored production items differ from pinned plan')
        for item,encoded,stage,nsha,tsha,count in self.db.execute('SELECT item,record,stage,normalized_sha,token_sha,token_count FROM items ORDER BY item'):
            record=json.loads(encoded)
            need(item_key(record)==item,'Item key corrupt');verify_seal(record)
            row=self.db.execute('SELECT payload FROM approved_files WHERE source=? AND path=?',
                                (record['source_id'],record['path'])).fetchone()
            need(row is not None,'Approved ownership evidence missing')
            validate_clearance(record,self.sources[record['source_id']],self.spec,self.root,
                               json.loads(row[0]),allow_legacy=self._legacy_obligation_record(record))
            need(json.loads(self.path('clearances',item,'.json').read_bytes())==record,'Clearance changed')
            if stage!='PENDING':
                need(file_digest(self.path('raw',item))==record['sha256'],'Committed raw payload corrupt')
                verify_structure(self.path('raw',item).read_bytes(),record)
            if stage in ('PREPARED','ACCEPTED','DUPLICATE','QUOTA_REJECTED'):
                need(isinstance(nsha,str) and nsha,'Missing normalized eligibility identity')
            if stage not in ('PENDING','ACQUIRED'):
                decision=self.quality_decision(item)
                need(decision is not None and decision['policy_identity']==self.quality_policy_identity(),
                     'Missing or stale quality decision')
                if stage=='FILTERED':
                    need(decision['outcome'] in ('FILTERED','NOT_EVALUATED') and
                         decision['reason']==self.db.execute('SELECT reason FROM items WHERE item=?',(item,)).fetchone()[0],
                         'Quality rejection decision changed')
                    if decision['outcome']=='FILTERED':
                        raw=self.path('raw',item).read_bytes().decode('utf-8-sig')
                        normalized=normalize_structured_text(raw) if record['content_type'] in ('code','markdown') else normalize_text(raw)
                        result=check_quality(normalized,QualityConfig(),content_type=record['content_type'])
                        need(result is not None and result.value==decision['reason'],
                             'Quality rejection does not match current policy')
                else:
                    self.verify_quality_acceptance(item,record,nsha)
                if decision['normalized_sha256'] is not None:
                    raw=self.path('raw',item).read_bytes().decode('utf-8-sig')
                    expected=normalize_structured_text(raw) if record['content_type'] in ('code','markdown') else normalize_text(raw)
                    need(digest(expected.encode('utf-8'))==decision['normalized_sha256'],
                         'Quality evaluated content changed')
            if nsha:
                need(file_digest(self.path('normalized',item))==nsha,'Committed normalized payload corrupt')
                expected_nsha=record['ownership_decision'].get('normalized_sha256')
                need(expected_nsha is None or expected_nsha==nsha,
                     'Approved ownership normalization differs')
            if nsha:self._verify_screening(item,record)
            if stage in ('PREPARED','ACCEPTED','DUPLICATE','QUOTA_REJECTED'):
                indexed=self.db.execute('SELECT exact,split,provenance,shingles FROM dd WHERE item=?',(item,)).fetchone()
                need(indexed is not None and indexed[:3]==(nsha,record['split'],canonical(record).decode()),
                     'Dedup language/provenance differs from pinned clearance')
                # Exact duplicates deliberately store no feature blob. Other
                # entries must match the current language-specific feature route.
                if indexed[3]:
                    normalized=self.path('normalized',item).read_text(encoding='utf-8')
                    need(indexed[3]==b''.join(features(normalized,record['content_type'],record['language'])),
                         'Dedup features differ from content/language route')
                else:
                    earlier=self.db.execute('SELECT 1 FROM dd WHERE exact=? AND id<(SELECT id FROM dd WHERE item=?) LIMIT 1',(nsha,item)).fetchone()
                    need(earlier is not None or not features(self.path('normalized',item).read_text(encoding='utf-8'),record['content_type'],record['language']),
                         'Missing dedup features for language route')
            if tsha:
                need(file_digest(self.path('tokens',item,'.bin'))==tsha,'Committed tokens corrupt')
                need(self.path('tokens',item,'.bin').stat().st_size==count*4,'Committed token size corrupt')
        self.dedup.verify()
        need(self.db.execute('SELECT count(*) FROM quality_decisions').fetchone()[0]==
             self.db.execute("SELECT count(*) FROM items WHERE stage NOT IN ('PENDING','ACQUIRED')").fetchone()[0],
             'Quality decision set changed')
        self.db.execute('CREATE TEMP TABLE IF NOT EXISTS expected_totals(kind TEXT,key TEXT,tokens INTEGER,PRIMARY KEY(kind,key))')
        self.db.execute('DELETE FROM expected_totals')
        for item,encoded,count in self.db.execute("SELECT item,record,token_count FROM items WHERE stage='ACCEPTED'"):
            record=json.loads(encoded)
            need(self.dedup.assignment(item)['canonical_item']==item,'Noncanonical item accounted')
            for kind,key,_ in self._limits(record):
                self.db.execute('INSERT INTO expected_totals VALUES (?,?,?) ON CONFLICT(kind,key) DO UPDATE SET tokens=tokens+excluded.tokens',(kind,key,count))
        need(self.db.execute('SELECT * FROM totals EXCEPT SELECT * FROM expected_totals').fetchone() is None and self.db.execute('SELECT * FROM expected_totals EXCEPT SELECT * FROM totals').fetchone() is None,'Committed accounting totals corrupt')
        self.db.commit()

    def _verify_screening(self,item,record):
        # Recompute rather than trusting a resealable local result. Bound reads,
        # sanitized errors: scanner exceptions must never leak matched content.
        try:
            path=self.path('screening',item,'.json')
            need(path.stat().st_size<=1024*1024,'Oversized screening evidence')
            raw_path=self.path('raw',item);normal_path=self.path('normalized',item)
            need(raw_path.stat().st_size<=MAX_ITEM_BYTES and normal_path.stat().st_size<=MAX_ITEM_BYTES,'Screening input bound')
            raw=raw_path.read_bytes();normalized=normal_path.read_bytes().decode('utf-8')
            need(digest(raw)==record['sha256'],'Screening raw mismatch')
            decoded=raw.decode('utf-8-sig')
            expected=normalize_structured_text(decoded) if record['content_type'] in ('code','markdown') else normalize_text(decoded)
            need(normalized==expected,'Screening normalization mismatch')
            return content_screening.verify_result(json.loads(path.read_bytes()),raw,normalized,item)
        except Exception:
            raise ValueError('Screening verification failed; eligibility withheld') from None

    def _screening_identity(self):
        sha=hashlib.sha256()
        for item,encoded in self.db.execute("SELECT item,record FROM items WHERE stage='ACCEPTED' ORDER BY item"):
            sha.update(canonical([item,self._verify_screening(item,json.loads(encoded))])+b'\n')
        return dict(policy=content_screening.POLICY,detector=content_screening.DETECTOR,
                    results_sha256='sha256:'+sha.hexdigest())

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
        verify_structure(raw,record)
        try:text=raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            self.record_quality(item,'NOT_EVALUATED','invalid_utf8');return 'invalid_utf8',None
        if '\x00' in text:
            self.record_quality(item,'NOT_EVALUATED','binary');return 'binary',None
        if any(marker.casefold() in text.casefold() for marker in self.spec['filters']['generated_markers']):
            self.record_quality(item,'NOT_EVALUATED','generated');return 'generated',None
        normalized=normalize_structured_text(text) if record['content_type'] in ('code','markdown') else normalize_text(text)
        expected_nsha=record['ownership_decision'].get('normalized_sha256')
        need(expected_nsha is None or expected_nsha==digest(normalized.encode('utf-8')),
             'Approved ownership normalization differs')
        try:
            screening=content_screening.screen(raw,normalized,item)
            path=self.path('screening',item,'.json')
            if path.exists():
                need(path.stat().st_size<=1024*1024,'Oversized screening evidence')
                need(json.loads(path.read_bytes())==screening,'Existing screening evidence differs')
            else:atomic_json(path,screening)
        except Exception:
            raise ValueError('Screening failed; eligibility withheld') from None
        if screening['outcome']!='ACCEPT':
            reason='screening_'+screening['outcome']
            self.record_quality(item,'NOT_EVALUATED',reason,normalized);return reason,None
        validate_language(record['language'],record['content_type'],normalized)
        reason=check_quality(normalized,QualityConfig(),content_type=record['content_type'])
        if reason:
            self.record_quality(item,'FILTERED',reason.value,normalized);return reason.value,None
        self.record_quality(item,'PASS',None,normalized)
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

    def _claim_corpus(self):
        if self.diagnostic:return None
        if self.corpus_ledger_path is None and self.expected_corpus_head is None:
            # Completed pre-ledger builds can be replayed without new admission.
            pending=self.db.execute("SELECT 1 FROM items WHERE stage IN ('PENDING','ACQUIRED','PREPARED') LIMIT 1").fetchone()
            if self.get('state') in ('COMPLETE','TARGET_REACHED','TARGET_BOUNDARY') and not pending and self.get('corpus_build_id') is None:
                return 'LEGACY'
        need(self.corpus_ledger_path is not None and self.expected_corpus_head is not None,
             'New production admission requires corpus ledger and external trusted head')
        need(not self.corpus_ledger_path.is_relative_to(self.directory),
             'Shared corpus ledger cannot be inside one build directory')
        need(self.directory.is_relative_to(self.root),'Ledger-backed build must be inside repository root')
        if self.corpus_ledger is None:
            self.corpus_ledger=CorpusLedger(self.corpus_ledger_path,self.root,self.expected_corpus_head,
                                            **self.historical_approval)
        self.corpus_build_id=build_identity(self.get('request'),self.get('plan_identity'))
        old=self.get('corpus_build_id')
        need(old is None or old==self.corpus_build_id,'Local corpus build identity changed')
        state=self.corpus_ledger.claim(self.corpus_build_id,self.get('plan_identity'),
            self.directory.relative_to(self.root).as_posix(),self.get('prior_production'))
        self.put('corpus_build_id',self.corpus_build_id);self.db.commit()
        return state

    def _corpus_evidence(self):
        return [self._ledger_item_evidence(item,json.loads(encoded),nsha,tsha,count)
                for item,encoded,nsha,tsha,count in self.db.execute(
                    "SELECT item,record,normalized_sha,token_sha,token_count FROM items WHERE stage='ACCEPTED' ORDER BY item")]

    def _ledger_item_evidence(self,item,record,nsha,tsha,count):
        s=self.sources[record['source_id']]
        return dict(item=item,record=record,normalized_sha=nsha,token_sha=tsha,
            source_id=record['source_id'],owner=record['owner'],split=record['split'],
            repository_group=s['repository_group'],domain_group=s['domain_group'],
            source_locator=s['locator'],split_group=s['split_group'],
            three_d_topic=record.get('three_d_topic'),three_d_form=record.get('three_d_form'),
            tokens=count)

    def _finish(self,state,*,receipts=None):
        current_receipts=()
        if receipts is not None:
            current_receipts=tuple(receipts[json.loads(encoded)['identity']]
                for (encoded,) in self.db.execute("SELECT record FROM items WHERE stage='ACCEPTED'"))
        freshness_failed=False
        def require_fresh():
            nonlocal freshness_failed
            try:self.require_current_approvals(receipts=current_receipts)
            except Exception:
                freshness_failed=True
                raise
        try:
            require_fresh()
            report=self.checkpoint(state)
            if self.corpus_ledger is not None:
                self.verify()
                evidence=self._corpus_evidence()
                require_fresh()
                head=self.corpus_ledger.commit(self.corpus_build_id,self.get('plan_identity'),
                    evidence,state,before_commit=require_fresh)
                # This process just extended its externally trusted parent. Its own
                # returned child head is trusted for the immediately following export.
                self.corpus_ledger.expected_head=head
                self.expected_corpus_head=head
                self.put('corpus_commit_head',head);self.db.commit()
            return report
        except Exception:
            if freshness_failed:self._rollback_failed_governance()
            raise

    def _rollback_failed_governance(self):
        """Release rejected, uncommitted acceptance after any run failure."""
        if self.corpus_ledger is None or self.corpus_build_id is None:return
        # Never undo a committed ledger entry during local error cleanup.
        state=self.corpus_ledger.db.execute('SELECT state FROM builds WHERE build_id=?',
                                            (self.corpus_build_id,)).fetchone()
        if state!=('ACTIVE',):return
        self.corpus_ledger.rollback_failed_governance(self.corpus_build_id)
        with self.db:
            self.db.execute("UPDATE items SET stage='PREPARED',reason=NULL,token_sha=NULL,"
                            "token_count=NULL WHERE stage='ACCEPTED'")
            self.db.execute('DELETE FROM totals')
            self.db.execute('DELETE FROM milestones')
            self.put('corpus_build_id',None)
            self.put('corpus_commit_head',None)
        self.corpus_build_id=None

    def _verify_corpus_commit(self):
        bid=self.get('corpus_build_id')
        if bid is None:return  # Historical integrity-only/preparation path.
        need(self.corpus_ledger_path is not None and self.expected_corpus_head is not None,
             'Corpus ledger authority required for committed export or replay')
        if self.corpus_ledger is None:
            self.corpus_ledger=CorpusLedger(self.corpus_ledger_path,self.root,self.expected_corpus_head,
                                            **self.historical_approval)
        return self.corpus_ledger.require_committed(bid,self.get('plan_identity'),
                                                    self._corpus_evidence())

    def abort_corpus_reservation(self):
        """Explicit, irreversible release of this build's uncommitted usage."""
        need(not self.diagnostic and self.corpus_ledger_path is not None and
             self.expected_corpus_head is not None,'Corpus reservation authority required')
        if self.corpus_ledger is None:
            self.corpus_ledger=CorpusLedger(self.corpus_ledger_path,self.root,self.expected_corpus_head,
                                            **self.historical_approval)
        bid=self.get('corpus_build_id')
        need(bid is not None,'No corpus reservation to abort')
        self.corpus_ledger.abort(bid)
        self.checkpoint('ABORTED')

    def _account(self,item,record,count,*,intended_use='production_preparation',receipts=None):
        self.verify_approval_authority()
        verify_structure(self.path('raw',item).read_bytes(),record)
        stored=self.db.execute('SELECT record,normalized_sha FROM items WHERE item=?',(item,)).fetchone()
        need(stored is not None and stored[0]==canonical(record).decode(),
             'Accounting record differs from pinned item')
        normalized_sha=stored[1]
        row=self.db.execute('SELECT payload FROM approved_files WHERE source=? AND path=?',
                            (record['source_id'],record['path'])).fetchone()
        need(row is not None,'Approved ownership evidence missing')
        validate_clearance(record,self.sources[record['source_id']],self.spec,self.root,json.loads(row[0]))
        expected_nsha=record['ownership_decision'].get('normalized_sha256')
        need(expected_nsha is None or expected_nsha==normalized_sha,
             'Approved ownership normalization differs')
        self.verify_quality_acceptance(item,record,normalized_sha)
        self._verify_screening(item,record)
        limits=self._limits(record)
        for kind,key,limit in limits:
            current=self.db.execute('SELECT tokens FROM totals WHERE kind=? AND key=?',(kind,key)).fetchone()
            if (current[0] if current else 0)+count>limit:return kind+'_cap'
        current_receipts=()
        if receipts is not None:
            from training.corpus_gate import verify_receipt
            receipt=receipts.get(record['identity'])
            use='evaluation' if intended_use=='commercial_training' and record['split']=='validation' else intended_use
            verify_receipt(receipt,record,self.sources[record['source_id']],self.spec,self.root,use)
            current_receipts=(receipt,)
        evidence=None
        if self.corpus_ledger is not None:
            evidence=self._ledger_item_evidence(item,record,normalized_sha,
                file_digest(self.path('tokens',item,'.bin')),count)
        # This is the last authority check before the serialized reservation and
        # local accounting update. All applicable statuses share one snapshot.
        self.require_current_approvals(records=(record,),receipts=current_receipts)
        if self.corpus_ledger is not None:
            reason=self.corpus_ledger.reserve(self.corpus_build_id,evidence,limits,
                before_commit=lambda:self.require_current_approvals(
                    records=(record,),receipts=current_receipts))
            if reason:return reason
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

    def authorize_use(self,intended_use,decision_receipts,*,accepted_only=False):
        from training.corpus_gate import load_receipts, verify_receipt
        need(intended_use in ('production_preparation','evaluation','commercial_training'), 'Unsupported production intended use')
        if intended_use=='production_preparation' and decision_receipts is None:
            need(self.get('use_authorization') is None,'Bound use authorization required on resume')
            self.require_current_approvals()
            return None  # Legacy preparation only; never a training grant.
        need(not self.diagnostic,'Diagnostic builds cannot authorize production use')
        bundle,receipts=load_receipts(decision_receipts)
        request=dict(intended_use=intended_use,decision_bundle=bundle)
        old=self.get('use_authorization')
        need(old is None or old==request,'Use authorization changed on resume')
        query="SELECT record FROM items"+(" WHERE stage='ACCEPTED'" if accepted_only else '')+" ORDER BY item"
        current_receipts=[]
        for (encoded,) in self.db.execute(query):
            r=json.loads(encoded)
            validate_clearance(r,self.sources[r['source_id']],self.spec,self.root,
                               allow_legacy=self._legacy_obligation_record(r))
            # A training-corpus export also carries validation; authorize its use separately.
            use='evaluation' if intended_use=='commercial_training' and r['split']=='validation' else intended_use
            verify_receipt(receipts.get(r['identity']),r,self.sources[r['source_id']],self.spec,self.root,use)
            current_receipts.append(receipts[r['identity']])
        # Verify all applicable statuses from one fresh snapshot after receipt
        # evidence checks, including any change made during those checks.
        self.require_current_approvals(receipts=current_receipts)
        return request

    def require_historical_item_training_use(self,item,*,link_store,
            expected_link_store_sha256,material_paths,expected_identities,
            retention_candidate=None,expected_retention_identity=None,
            expected_retention_sha256=None,expected_link_head=None,
            expected_ledger_sha256=None):
        """Check a separately linked historical item; never adds it to this build.

        This is a prospective use check only. Normal export/replay and training
        authorization retain all of their existing independent gates.
        """
        need(not self.diagnostic and self.corpus_ledger_path is not None
             and self.expected_corpus_head is not None,
             'Production corpus head required for historical use')
        from training.historical_authority_link import (
            require_linked_historical_training_use,
            verify_retention_candidate,
        )
        verify_retention_candidate(
            self.root,retention_candidate,expected_retention_identity,
            expected_retention_sha256,link_store,expected_link_store_sha256,
            expected_link_head,self.corpus_ledger_path,expected_ledger_sha256,
            self.expected_corpus_head,material_paths,expected_identities)
        link=require_linked_historical_training_use(
            self.root,item,link_store,expected_link_store_sha256,
            self.corpus_ledger_path,self.expected_corpus_head,
            material_paths,expected_identities)
        return dict(status='VERIFIED_PROSPECTIVE_LINK_ONLY',authority='NONE',link=link)

    def run(self,plan,tokenizer,*,fetch=None,max_items=None,target_tokens=None,
            intended_use='production_preparation',decision_receipts=None):
        need(target_tokens is None or (type(target_tokens) is int and 0<target_tokens<=sum(QUOTAS.values())),'Invalid target token ceiling')
        need(max_items is None or (type(max_items) is int and max_items>0),'max_items must be a positive integer or None')
        need(tokenizer.identity==TOKENIZER and tokenizer.vocab_size==4096,'Frozen tokenizer mismatch')
        with _writer_lock(self.directory/'coordinator'):
            try:
                self.register_prior_production()
                self.register_references();self.register(plan);self.verify();done=0
                authorization=self.authorize_use(intended_use,decision_receipts)
                receipts=None
                if authorization is not None:
                    from training.corpus_gate import load_receipts
                    bundle,receipts=load_receipts(decision_receipts)
                    need(bundle==authorization['decision_bundle'],'Decision bundle changed during authorization')
                    self.put('use_authorization',authorization);self.db.commit()
                corpus_state=self._claim_corpus()
                if corpus_state=='COMMITTED':
                    terminal=self._verify_corpus_commit()
                    if terminal=='COMPLETE':
                        need(not self.db.execute("SELECT 1 FROM items WHERE stage IN ('PENDING','ACQUIRED','PREPARED') LIMIT 1").fetchone(),
                             'Committed complete build has unsettled items')
                    return self.checkpoint(terminal)
                if corpus_state=='LEGACY':return self.progress(self.get('state'))
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
                            if current>=target_tokens:return self._finish('TARGET_REACHED',receipts=receipts)
                        m=self._cached_tokens(item,record,tokenizer)
                        if target_tokens is not None and record['split']=='train' and current+m['count']>target_tokens:
                            return self._finish('TARGET_BOUNDARY',receipts=receipts)
                        with self.db:
                            reason=self._account(item,record,m['count'],intended_use=intended_use,receipts=receipts)
                            self.db.execute('UPDATE items SET stage=?,reason=?,token_sha=?,token_count=? WHERE item=?',('QUOTA_REJECTED' if reason else 'ACCEPTED',reason,m['sha256'],m['count'],item))
                    done+=1
                    total=self.db.execute("SELECT coalesce(sum(tokens),0) FROM totals WHERE kind='category'").fetchone()[0]
                    for milestone in CHECKPOINTS:
                        if total>=milestone and not self.db.execute('SELECT 1 FROM milestones WHERE tokens=?',(milestone,)).fetchone():
                            self.db.execute('INSERT INTO milestones VALUES (?,?)',(milestone,canonical(self.progress('PARTIAL')).decode()));self.db.commit()
                return self._finish('COMPLETE',receipts=receipts)
            except Exception as exc:
                self.db.rollback()
                # Provisional admission can precede failures other than a
                # governance rejection (for example, final integrity checks).
                self._rollback_failed_governance()
                self.put('last_error',type(exc).__name__+': '+str(exc));self.db.commit()
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

    def _expected_notice(self,admission_hash):
        rows=((item,json.loads(encoded)) for item,encoded in self.db.execute(
            "SELECT item,record FROM items WHERE stage='ACCEPTED' ORDER BY item"))
        return build_notice(rows,self.sources,self.root,plan_identity=self.get('plan_identity'),
                            admission_sha256=admission_hash,
                            legacy_record=self._legacy_obligation_record)

    def export_shards(self,output_directory=None,*,intended_use='production_preparation',decision_receipts=None):
        try:return self._export_shards(output_directory,intended_use=intended_use,decision_receipts=decision_receipts)
        except Exception as exc:
            self.db.rollback();self.put('last_error',type(exc).__name__+': '+str(exc));self.db.commit();self.checkpoint('FAILED');raise

    def _export_shards(self,output_directory=None,*,intended_use='production_preparation',decision_receipts=None):
        destination=Path(output_directory).resolve() if output_directory is not None else self.directory
        need(destination.is_relative_to(self.directory),'Export must stay inside build directory')
        destination.mkdir(parents=True,exist_ok=True)
        need(self.get('state') in ('COMPLETE','TARGET_REACHED','TARGET_BOUNDARY') and self.get('dedup_settled'),'Build not complete')
        with _writer_lock(self.directory/'coordinator'):
            self.verify_references()
            self.verify()
            self.require_current_approvals()
            self._verify_corpus_commit()
            authorization=self.authorize_use(intended_use,decision_receipts,accepted_only=True)
            if (destination/'prepared.json').exists():
                previous=json.loads((destination/'prepared.json').read_bytes())
                need(previous.get('use_authorization')==authorization,
                     'Export use differs; choose a new destination, never rewrite historical evidence')
            # Calculate the candidate before touching any existing export bytes.
            # Iterate twice to retain bounded memory without an export-side temp file.
            def admission_rows():
                for item,record,stage,reason,nsha,tsha,count in self.db.execute('SELECT item,record,stage,reason,normalized_sha,token_sha,token_count FROM items ORDER BY item'):
                    yield canonical(dict(item=item,clearance=json.loads(record),stage=stage,reason=reason,normalized_sha256=nsha,quality_decision=self.quality_decision(item),token_sha256=tsha,tokens=count))+b'\n'
            admission_digest=hashlib.sha256()
            for row in admission_rows():admission_digest.update(row)
            admission_hash='sha256:'+admission_digest.hexdigest()
            notice=self._expected_notice(admission_hash)
            accepted=destination/'admission.jsonl'
            counts={split:self.db.execute("SELECT count(*) FROM items WHERE stage='ACCEPTED' AND split=?",(split,)).fetchone()[0] for split in ('train','validation')}
            split_id=identity(dict(plan=self.get('plan_identity'),rule='source-group-v1-validation-priority-clusters',admission_hash=admission_hash))
            manifest=dict(schema_version='yug-admitted-production-v1',tokenizer_identity=TOKENIZER,split_identity=split_id,
                spec_identity=self.spec['identity'],registry_identity=self.registry['identity'],plan_identity=self.get('plan_identity'),diagnostic=self.diagnostic,reference_identity=self.get('request')['reference_identity'],
                dedup_policy=POLICY,quality_policy=self.current_quality_policy(),language_policy=LANGUAGE_POLICY,
                admission_sha256=admission_hash,notice_attribution=notice_reference(notice),counts=counts)
            if authorization is not None:manifest['use_authorization']=authorization
            manifest['screening']=self._screening_identity()
            manifest['prepared_corpus_identity']=identity(manifest)
            existing=[destination/'prepared.json',accepted,destination/'shards',destination/NOTICE_FILENAME,
                      accepted.with_suffix('.part'),destination/'prepared.json.tmp',
                      destination/(NOTICE_FILENAME+'.tmp')]
            if any(p.exists() for p in existing):
                # Completed exports are immutable. Incomplete or incompatible
                # publication requires a fresh destination, never in-place repair.
                need(all(p.exists() for p in existing[:4]) and not any(p.exists() for p in existing[4:]),
                     'Incomplete export; choose a fresh destination')
                need(canonical(json.loads(existing[0].read_bytes()))==canonical(manifest),
                     'Incompatible prepared manifest; choose a fresh destination')
                need(file_digest(accepted)==admission_hash,'Incompatible admission export')
                verify_notice_file(destination,manifest['notice_attribution'],expected=notice)
                result=None
                for split in ('train','validation'):
                    stream=TokenShardStream(destination/'shards',split=split,tokenizer_identity=TOKENIZER,
                                            prepared_corpus_identity=manifest['prepared_corpus_identity'])
                    result=stream.manifest
                    need(canonical(result.get('prepared_manifest'))==canonical(manifest),
                         'Embedded prepared manifest mismatch')
                request_keys=('schema_version','prepared_corpus_identity','tokenizer_identity','vocab_size',
                              'split_identity','shard_tokens','token_dtype','endianness','tokenization_semantics')
                need(result['vocab_size']==4096 and result['shard_tokens']==DEFAULT_SHARD_TOKENS,
                     'Incompatible export token configuration')
                need(json.loads((destination/'shards/request.json').read_bytes())=={k:result[k] for k in request_keys},
                     'Incompatible resume request')
                need(self.authorize_use(intended_use,decision_receipts,accepted_only=True)==authorization,
                     'Export authority changed during reuse')
                return result
            temp=accepted.with_suffix('.part')
            atomic_json(destination/NOTICE_FILENAME,notice)
            with temp.open('wb') as out:
                for row in admission_rows():out.write(row)
                out.flush();os.fsync(out.fileno())
            atomic_replace(temp,accepted)
            prepared=_Prepared(self,manifest);cache=_TokenCache(self)
            if self.diagnostic:
                # Diagnostic interruption recovery keeps its prior publication
                # order; diagnostics cannot authorize production use.
                atomic_json(destination/'prepared.json',manifest)
            result=build_token_shards(prepared,cache,destination/'shards')
            for split in ('train','validation'):
                TokenShardStream(destination/'shards',split=split,tokenizer_identity=TOKENIZER,prepared_corpus_identity=prepared.identity)
            need(self.authorize_use(intended_use,decision_receipts,accepted_only=True)==authorization,
                 'Export authority changed before publication')
            if not self.diagnostic:
                atomic_json(destination/'prepared.json',manifest)
            return result

    def replay_for_use(self,output_directory,*,split,intended_use,decision_receipts):
        """Authorized production consumer boundary; generic shard decoding is not approval."""
        destination=Path(output_directory).resolve()
        need(destination.is_relative_to(self.directory),'Replay must stay inside build directory')
        need(intended_use in ('commercial_training','evaluation'),'Explicit consumer use required')
        need(split=='validation' or intended_use=='commercial_training','Evaluation cannot consume training split')
        with _writer_lock(self.directory/'coordinator'):
            self.verify_references()
            self.verify()
            self.require_current_approvals()
            self._verify_corpus_commit()
            authorization=self.authorize_use(intended_use,decision_receipts,accepted_only=True)
            prepared=json.loads((destination/'prepared.json').read_bytes())
            body=dict(prepared)
            prepared_identity=body.pop('prepared_corpus_identity',None)
            need(prepared_identity==identity(body),'Local prepared manifest identity mismatch')
            need(prepared.get('use_authorization')==authorization,'Export not authorized for requested use')
            need(prepared.get('admission_sha256')==file_digest(destination/'admission.jsonl'),'Admission export corrupt')
            notice=self._expected_notice(prepared['admission_sha256'])
            verify_notice_file(destination,prepared.get('notice_attribution'),expected=notice)
            need(prepared.get('screening')==self._screening_identity(),'Export screening evidence missing/stale')
            need(prepared.get('quality_policy')==self.current_quality_policy(),
                 'Export quality policy missing/stale')
            need(canonical(prepared.get('language_policy'))==canonical(LANGUAGE_POLICY),
                 'Export language policy missing/stale')
            stream=TokenShardStream(destination/'shards',split=split,tokenizer_identity=TOKENIZER,
                                    prepared_corpus_identity=prepared_identity)
            need(canonical(prepared)==canonical(stream.manifest.get('prepared_manifest')),
                 'Local and shard-embedded prepared manifests disagree')
            need(self.authorize_use(intended_use,decision_receipts,accepted_only=True)==authorization,
                 'Replay authority changed before return')
            return _AuthorizedReplay(stream,self.governance_log,self.expected_governance_sha256)


class _AuthorizedReplay:
    """Check current pinned authority after each shard advance, before emission."""
    def __init__(self,stream,governance_log,expected_governance_sha256):
        self.stream=stream
        self.manifest=stream.manifest
        self.governance_log=governance_log
        self.expected_governance_sha256=expected_governance_sha256

    def __iter__(self):
        for token in self.stream:
            Governance(self.governance_log,self.expected_governance_sha256)
            yield token

    def sequences(self,sequence_length):
        from training.token_shards import _ShardSequences
        return _ShardSequences(self,sequence_length)

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
