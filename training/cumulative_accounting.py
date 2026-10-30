"""Single-machine, externally pinned corpus lineage and cumulative token ledger.

The ledger is operational state, not an approval: callers must supply its trusted
current head independently of build-local SQLite. One active build at a time is
intentional; it makes reservation, completion and crash recovery unambiguous.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sqlite3

from training.production_corpus import canonical, file_digest, identity
from training.production_spec import need, verify_seal


POLICY = 'yug-cumulative-accounting-v1'
GENESIS = identity(dict(policy=POLICY, genesis=True))


def build_identity(request, plan_identity):
    """Directory-independent identity of one approved production build."""
    return identity(dict(policy=POLICY, plan=plan_identity,
        spec=request['spec'], registry=request['registry'],
        tokenizer=request['tokenizer'], dedup=request['dedup'],
        quality_policy=request['quality_policy'],
        language_policy=request['language_policy'],
        ownership_policy=request['ownership_policy'],
        reference_identity=request['reference_identity'],
        ownership_approval_authority=request['ownership_approval_authority'],
        diagnostic=request['diagnostic']))


def _usage_rows(db, build_id):
    return [list(row) for row in db.execute(
        'SELECT item,kind,key,tokens,cap FROM usage WHERE build_id=? ORDER BY item,kind,key',
        (build_id,))]


def _evidence_rows(db, build_id):
    return [json.loads(row[0]) for row in db.execute(
        'SELECT evidence FROM item_evidence WHERE build_id=? ORDER BY item', (build_id,))]


def _expected_usage(evidence):
    count = evidence['tokens']
    need(type(count) is int and count > 0, 'Invalid committed corpus token count')
    record = evidence['record']
    need(record.get('source_id') == evidence['source_id']
         and record.get('owner') == evidence['owner']
         and record.get('split') == evidence['split']
         and record.get('split_group') == evidence['split_group'],
         'Corpus evidence source/owner/split mismatch')
    source = evidence['source_id']
    owner = evidence['owner']
    rows = [('source', source, count), ('source_owner', source + ':' + owner, count),
            ('repository_group', evidence['repository_group'], count),
            ('domain_group', evidence['domain_group'], count)]
    if evidence['split'] == 'train':
        rows.append(('category', owner, count))
        if evidence['three_d_topic']:
            rows.extend((('three_d_topic', evidence['three_d_topic'], count),
                         ('three_d_form', evidence['three_d_form'], count)))
    else:
        rows.append(('validation_category', owner, count))
    return sorted([list(row) for row in rows])


class CorpusLedger:
    def __init__(self, path, root, expected_head, *,
                 approved_historical_import_identity=None,
                 approved_historical_manifest_identity=None,
                 approved_historical_manifest_sha256=None,
                 governance_log=None, expected_governance_sha256=None):
        self.root = Path(root).resolve()
        self.path = Path(path).resolve()
        need(self.path.is_relative_to(self.root), 'Corpus ledger must be inside the repository root')
        need(isinstance(expected_head, str) and expected_head.startswith('sha256:')
             and len(expected_head) == 71, 'Trusted corpus head required')
        self.expected_head = expected_head
        pins = (approved_historical_import_identity,
                approved_historical_manifest_identity,
                approved_historical_manifest_sha256)
        need(all(pin is None for pin in pins) or all(pin is not None for pin in pins),
             'Incomplete external historical corpus approval pins')
        self.approved_historical_import_identity = approved_historical_import_identity
        self.approved_historical_manifest_identity = approved_historical_manifest_identity
        self.approved_historical_manifest_sha256 = approved_historical_manifest_sha256
        self.governance_log = governance_log
        self.expected_governance_sha256 = expected_governance_sha256
        if not self.path.exists():
            need(expected_head == GENESIS, 'Missing corpus ledger for trusted non-genesis head')
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        try:
            self._initialize()
        except Exception:
            self.db.close()
            raise

    def _initialize(self):
        self.db.execute('PRAGMA busy_timeout=30000')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS builds(
                build_id TEXT PRIMARY KEY,parent TEXT NOT NULL,plan TEXT NOT NULL,
                path_hint TEXT NOT NULL,state TEXT NOT NULL,commit_head TEXT);
            CREATE TABLE IF NOT EXISTS item_evidence(
                build_id TEXT NOT NULL,item TEXT NOT NULL,evidence TEXT NOT NULL,
                PRIMARY KEY(build_id,item));
            CREATE TABLE IF NOT EXISTS usage(
                build_id TEXT NOT NULL,item TEXT NOT NULL,kind TEXT NOT NULL,
                key TEXT NOT NULL,tokens INTEGER NOT NULL,cap INTEGER NOT NULL,
                PRIMARY KEY(build_id,item,kind,key));
            CREATE INDEX IF NOT EXISTS usage_keys ON usage(kind,key);
            CREATE TABLE IF NOT EXISTS totals(
                kind TEXT NOT NULL,key TEXT NOT NULL,tokens INTEGER NOT NULL,
                cap INTEGER NOT NULL,PRIMARY KEY(kind,key));
            CREATE TABLE IF NOT EXISTS commits(
                ordinal INTEGER PRIMARY KEY,build_id TEXT UNIQUE NOT NULL,
                parent TEXT NOT NULL,head TEXT UNIQUE NOT NULL,entry TEXT NOT NULL);
        ''')
        if self.db.execute('SELECT count(*) FROM meta').fetchone()[0] == 0:
            need(not self.db.execute('SELECT 1 FROM builds LIMIT 1').fetchone()
                 and not self.db.execute('SELECT 1 FROM commits LIMIT 1').fetchone(),
                 'Incomplete corpus ledger genesis')
            self.db.execute('INSERT INTO meta VALUES (?,?)', ('policy', POLICY))
            self.db.execute('INSERT INTO meta VALUES (?,?)', ('head', GENESIS))
            self.db.commit()
        self.verify_chain()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _meta(self, key):
        row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return row[0] if row else None

    @property
    def head(self):
        return self._meta('head')

    def verify_chain(self):
        need(self.db.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'Corrupt corpus ledger')
        need(self._meta('policy') == POLICY, 'Unsupported corpus ledger policy')
        parent = GENESIS
        seen = set()
        source_scopes = {}
        locator_scopes = {}
        historical_bindings = []
        for ordinal, bid, old, head, encoded in self.db.execute(
                'SELECT ordinal,build_id,parent,head,entry FROM commits ORDER BY ordinal'):
            entry = json.loads(encoded)
            need(ordinal == len(seen) + 1 and bid not in seen and old == parent,
                 'Corpus lineage gap or fork')
            need(entry.get('policy') == POLICY and entry.get('state') == 'COMMITTED'
                 and entry.get('build_id') == bid and entry.get('parent') == old
                 and entry.get('terminal_state') in ('COMPLETE','TARGET_REACHED','TARGET_BOUNDARY')
                 and identity(entry) == head, 'Corpus commit identity mismatch')
            row = self.db.execute('SELECT state,parent,plan,commit_head FROM builds WHERE build_id=?',
                                  (bid,)).fetchone()
            need(row == ('COMMITTED', old, entry['plan'], head),
                 'Corpus committed build metadata mismatch')
            if 'historical_import_identity' in entry:
                binding = (bid, entry['historical_import_identity'])
                self._verify_historical_manifest_binding(*binding)
                historical_bindings.append(binding)
            need(entry['usage_identity'] == identity(_usage_rows(self.db, bid))
                 and entry['evidence_identity'] == identity(_evidence_rows(self.db, bid)),
                 'Corpus committed usage/evidence mismatch')
            for evidence in _evidence_rows(self.db, bid):
                need(_item_limits(self.db, bid, evidence['item']) == _expected_usage(evidence),
                     'Corpus committed per-item usage differs from evidence')
                _check_scope(evidence, source_scopes, locator_scopes)
            seen.add(bid)
            parent = head
        need(parent == self.head, 'Corpus head differs from committed lineage')
        need({row[0] for row in self.db.execute(
            "SELECT build_id FROM builds WHERE state='COMMITTED'")} == seen,
            'Unchained committed build')
        need(self.db.execute("SELECT count(*) FROM builds WHERE state='ACTIVE'").fetchone()[0] <= 1,
             'Multiple active corpus reservations')
        need(not self.db.execute('SELECT 1 FROM usage WHERE build_id NOT IN '
            '(SELECT build_id FROM builds) LIMIT 1').fetchone(),
            'Orphan corpus usage')
        need(not self.db.execute('SELECT 1 FROM item_evidence WHERE build_id NOT IN '
            '(SELECT build_id FROM builds) LIMIT 1').fetchone(),
            'Orphan corpus item evidence')
        need(not self.db.execute('SELECT 1 FROM usage u LEFT JOIN item_evidence e '
            'ON u.build_id=e.build_id AND u.item=e.item WHERE e.item IS NULL LIMIT 1').fetchone(),
            'Corpus usage without item evidence')
        need(not self.db.execute("SELECT 1 FROM usage WHERE build_id IN "
            "(SELECT build_id FROM builds WHERE state='ABORTED') LIMIT 1").fetchone(),
            'Aborted corpus reservation still consumes capacity')
        for bid, in self.db.execute("SELECT build_id FROM builds WHERE state='ACTIVE'"):
            for evidence in _evidence_rows(self.db, bid):
                need(_item_limits(self.db, bid, evidence['item']) == _expected_usage(evidence),
                     'Active corpus usage differs from evidence')
                _check_scope(evidence, source_scopes, locator_scopes)
        counts = Counter()
        caps = {}
        for kind, key, tokens, cap in self.db.execute('SELECT kind,key,tokens,cap FROM usage'):
            need(type(tokens) is int and tokens > 0 and type(cap) is int and cap >= 0,
                 'Invalid corpus usage row')
            old = caps.setdefault((kind,key),cap)
            need(old == cap, 'Inconsistent corpus cap history')
            counts[(kind,key)] += tokens
        actual = {(kind,key):(tokens,cap) for kind,key,tokens,cap in
                  self.db.execute('SELECT kind,key,tokens,cap FROM totals')}
        need(actual == {key:(tokens,caps[key]) for key,tokens in counts.items()},
             'Corpus cached totals differ from item usage')
        self._source_scopes = source_scopes
        self._locator_scopes = locator_scopes
        self._historical_bindings = tuple(historical_bindings)
        self._verified_data_version = self.db.execute('PRAGMA data_version').fetchone()[0]
        return parent

    def _verify_historical_manifest_binding(self, build_id, import_identity):
        from training.historical_corpus_import import verify_approved_manifest

        need(self.approved_historical_import_identity == import_identity
             and self.approved_historical_manifest_identity is not None
             and self.approved_historical_manifest_sha256 is not None,
             'External historical corpus approval pins missing or mismatched')
        row = self.db.execute('SELECT path_hint FROM builds WHERE build_id=?',
                              (build_id,)).fetchone()
        need(row is not None, 'Historical manifest binding missing')
        verify_approved_manifest(self.root, self.root / row[0],
            self.approved_historical_manifest_identity,
            self.approved_historical_manifest_sha256)

    def _inventory_builds(self, inventory):
        need(inventory is not None and inventory.get('complete') is True,
             'Complete prior-production inventory required')
        verify_seal(inventory)
        need(inventory.get('schema') == 'yug-prior-production-v1' and
             type(inventory.get('genesis')) is bool and
             inventory['genesis'] == (not inventory.get('builds')),
             'Incompatible corpus prior inventory')
        ids = []
        imports = [(bid, set(x['state'] for x in json.loads(encoded)['historical_builds']))
                   for bid, encoded in self.db.execute('SELECT build_id,entry FROM commits')
                   if 'historical_builds' in json.loads(encoded)]
        historical_seen = {bid: set() for bid, _ in imports}
        for entry in inventory['builds']:
            path = (self.root / entry['path']).resolve()
            need(path.is_relative_to(self.root) and file_digest(path) == entry['sha256'],
                 'Prior corpus database identity changed')
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
                row = db.execute('SELECT value FROM meta WHERE key=?', ('corpus_build_id',)).fetchone()
                if row is None:
                    matching = [(bid, hashes) for bid, hashes in imports
                                if entry['sha256'] in hashes]
                    need(len(matching) == 1,
                         'Prior build lacks approved historical corpus import')
                    need(entry['sha256'] not in historical_seen[matching[0][0]],
                         'Duplicate historical prior build')
                    historical_seen[matching[0][0]].add(entry['sha256'])
                    continue
                bid = json.loads(row[0])
                expected = Counter()
                for _, kind, key, tokens, _ in _usage_rows(self.db, bid):
                    expected[(kind, key)] += tokens
                actual = {(kind, key): tokens for kind, key, tokens in db.execute(
                    'SELECT kind,key,tokens FROM totals')}
                need(dict(expected) == actual,
                     'Prior non-category or category accounting differs from accepted evidence')
                ids.append(bid)
        for bid, hashes in imports:
            need(historical_seen[bid] == hashes,
                 'Prior inventory omits retained historical corpus evidence')
            from training.historical_corpus_import import reconstruct
            hint = self.db.execute('SELECT path_hint FROM builds WHERE build_id=?',
                                   (bid,)).fetchone()[0]
            candidate = reconstruct(self.root, self.root / hint,
                approved_manifest_identity=self.approved_historical_manifest_identity,
                approved_manifest_sha256=self.approved_historical_manifest_sha256)
            encoded = self.db.execute('SELECT entry FROM commits WHERE build_id=?',
                                      (bid,)).fetchone()[0]
            historical_entry = json.loads(encoded)
            need(candidate['identity'] == historical_entry['historical_import_identity']
                 and candidate['items'] == _evidence_rows(self.db, bid)
                 and [list(row) for row in candidate['usage']] == _usage_rows(self.db, bid),
                 'Historical import evidence changed before new admission')
            ids.append(bid)
        need(len(ids) == len(set(ids)), 'Copied prior build counted twice')
        return set(ids)

    def import_historical(self, manifest_path, approved_import_identity):
        """Commit the complete retained pre-ledger corpus once, with an external pin."""
        from training.historical_corpus_import import reconstruct

        need(approved_import_identity == self.approved_historical_import_identity,
             'Externally approved historical import identity mismatch')
        candidate = reconstruct(self.root, manifest_path,
            approved_manifest_identity=self.approved_historical_manifest_identity,
            approved_manifest_sha256=self.approved_historical_manifest_sha256)
        need(approved_import_identity == candidate['identity'],
             'Externally approved historical import identity mismatch')
        from training.approval_governance import Governance
        def require_current_import():
            Governance(self.governance_log,self.expected_governance_sha256).require_current(
                'historical_import',candidate['identity'],
                dict(manifest_identity=self.approved_historical_manifest_identity,
                     manifest_sha256=self.approved_historical_manifest_sha256))
        require_current_import()
        bid = identity(dict(policy=POLICY, historical_import=candidate['identity']))
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.verify_chain()
            need(self.head == self.expected_head == GENESIS
                 and not self.db.execute('SELECT 1 FROM builds LIMIT 1').fetchone()
                 and not self.db.execute('SELECT 1 FROM commits LIMIT 1').fetchone(),
                 'Historical corpus import requires empty trusted genesis')
            need(Path(manifest_path).resolve().is_relative_to(self.root),
                 'Historical manifest escapes repository root')
            self.db.execute('INSERT INTO builds VALUES (?,?,?,?,?,NULL)',
                (bid, GENESIS, candidate['identity'],
                 Path(manifest_path).resolve().relative_to(self.root).as_posix(),
                 'COMMITTED'))
            for evidence in candidate['items']:
                self.db.execute('INSERT INTO item_evidence VALUES (?,?,?)',
                    (bid, evidence['item'], canonical(evidence).decode()))
            for item, kind, key, count, cap in candidate['usage']:
                self.db.execute('INSERT INTO usage VALUES (?,?,?,?,?,?)',
                    (bid, item, kind, key, count, cap))
                old = self.db.execute('SELECT tokens,cap FROM totals WHERE kind=? AND key=?',
                                      (kind, key)).fetchone()
                need(old is None or old[1] == cap, 'Historical quota cap inconsistent')
                self.db.execute('INSERT INTO totals VALUES (?,?,?,?) ON CONFLICT(kind,key) '
                    'DO UPDATE SET tokens=tokens+excluded.tokens',
                    (kind, key, count, cap))
            entry = dict(policy=POLICY, state='COMMITTED', build_id=bid,
                parent=GENESIS, plan=candidate['identity'], terminal_state='COMPLETE',
                historical_import_identity=candidate['identity'],
                historical_builds=candidate['proposal']['builds'],
                evidence_identity=identity(candidate['items']),
                usage_identity=identity(_usage_rows(self.db, bid)))
            need(reconstruct(self.root, manifest_path,
                approved_manifest_identity=self.approved_historical_manifest_identity,
                approved_manifest_sha256=self.approved_historical_manifest_sha256)['identity'] == candidate['identity'],
                 'Historical evidence changed during import')
            head = identity(entry)
            self.db.execute('INSERT INTO commits(build_id,parent,head,entry) VALUES (?,?,?,?)',
                            (bid, GENESIS, head, canonical(entry).decode()))
            self.db.execute('UPDATE builds SET commit_head=? WHERE build_id=?', (head, bid))
            self.db.execute('UPDATE meta SET value=? WHERE key=?', (head, 'head'))
            require_current_import()
            self.db.commit()
            self.expected_head = head
            return dict(build_id=bid, head=head, identity=candidate['identity'],
                        accepted_documents=len(candidate['items']),
                        training_tokens=candidate['proposal']['training_tokens'],
                        validation_tokens=candidate['proposal']['validation_tokens'])
        except Exception:
            self.db.rollback()
            raise

    def claim(self, build_id, plan, path_hint, inventory):
        """Atomically bind a build to the trusted head and complete prior lineage."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.verify_chain()
            row = self.db.execute('SELECT parent,plan,state FROM builds WHERE build_id=?',
                                  (build_id,)).fetchone()
            if row and row[2] == 'COMMITTED':
                need(self.head == self.expected_head, 'Stale or untrusted corpus head')
                need(row[1] == plan, 'Committed corpus build identity changed')
                self.db.commit()
                return 'COMMITTED'
            need(self.head == self.expected_head, 'Stale or untrusted corpus head')
            committed = {r[0] for r in self.db.execute('SELECT build_id FROM commits')}
            need(self._inventory_builds(inventory) == committed,
                 'Prior inventory omits or adds committed corpus history')
            active = self.db.execute("SELECT build_id FROM builds WHERE state='ACTIVE'").fetchone()
            need(active is None or active[0] == build_id,
                 'Another corpus build holds the reservation')
            if row:
                need(row == (self.head, plan, 'ACTIVE'), 'Corpus reservation changed')
            else:
                self.db.execute('INSERT INTO builds VALUES (?,?,?,?,?,NULL)',
                    (build_id, self.head, plan, path_hint, 'ACTIVE'))
            self.db.commit()
            return 'ACTIVE'
        except Exception:
            self.db.rollback()
            raise

    def reserve(self, build_id, evidence, limits, *, before_commit=None):
        """One serialized item reservation; retry of identical evidence is free."""
        item = evidence['item']
        count = evidence['tokens']
        need(type(count) is int and count > 0, 'Invalid corpus token reservation')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('PRAGMA data_version').fetchone()[0] != self._verified_data_version:
                self.verify_chain()
            # SQLite's data version does not cover externally pinned manifest bytes.
            for binding in self._historical_bindings:
                self._verify_historical_manifest_binding(*binding)
            row = self.db.execute('SELECT parent,state FROM builds WHERE build_id=?',
                                  (build_id,)).fetchone()
            need(row == (self.head, 'ACTIVE'), 'Inactive or stale corpus reservation')
            scopes = dict(self._source_scopes)
            locators = dict(self._locator_scopes)
            _check_scope(evidence, scopes, locators)
            previous = self.db.execute('SELECT evidence FROM item_evidence WHERE build_id=? AND item=?',
                                       (build_id, item)).fetchone()
            need(sorted([list((kind, key, count)) for kind, key, _ in limits]) ==
                 _expected_usage(evidence), 'Corpus usage dimensions differ from item evidence')
            if previous:
                need(previous[0] == canonical(evidence).decode() and
                     _item_limits(self.db, build_id, item) == _expected_usage(evidence),
                     'Repeated item reservation differs')
                if before_commit is not None:
                    before_commit()
                self.db.commit()
                return None
            for kind, key, limit in limits:
                need(type(limit) is int and limit >= 0, 'Invalid corpus cap')
                old = self.db.execute('SELECT tokens,cap FROM totals WHERE kind=? AND key=?',
                                      (kind,key)).fetchone()
                need(old is None or old[1] == limit,
                     'Corpus quota cap changed across builds')
                used = old[0] if old else 0
                if used + count > limit:
                    self.db.rollback()
                    return kind + '_cap'
            self.db.execute('INSERT INTO item_evidence VALUES (?,?,?)',
                            (build_id, item, canonical(evidence).decode()))
            for kind, key, limit in limits:
                self.db.execute('INSERT INTO usage VALUES (?,?,?,?,?,?)',
                                (build_id, item, kind, key, count, limit))
                self.db.execute('INSERT INTO totals VALUES (?,?,?,?) ON CONFLICT(kind,key) '
                    'DO UPDATE SET tokens=tokens+excluded.tokens',
                    (kind,key,count,limit))
            for binding in self._historical_bindings:
                self._verify_historical_manifest_binding(*binding)
            if before_commit is not None:
                before_commit()
            self.db.commit()
            self._source_scopes = scopes
            self._locator_scopes = locators
            return None
        except Exception:
            self.db.rollback()
            raise

    def commit(self, build_id, plan, evidence, terminal_state='COMPLETE', *, before_commit=None):
        """Seal a completed build into the append-only lineage, exactly once."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.verify_chain()
            row = self.db.execute('SELECT parent,plan,state,commit_head FROM builds WHERE build_id=?',
                                  (build_id,)).fetchone()
            need(row is not None and row[1] == plan, 'Missing corpus reservation')
            need(row[2] in ('ACTIVE', 'COMMITTED'), 'Inactive corpus reservation')
            need(identity(evidence) == identity(_evidence_rows(self.db, build_id)),
                 'Committed build differs from reserved item evidence')
            if row[2] == 'COMMITTED':
                if before_commit is not None:
                    before_commit()
                self.db.commit()
                return row[3]
            need(row[2] == 'ACTIVE' and row[0] == self.head == self.expected_head,
                 'Stale corpus commit parent')
            need(terminal_state in ('COMPLETE','TARGET_REACHED','TARGET_BOUNDARY'),
                 'Nonterminal corpus build cannot commit')
            entry = dict(policy=POLICY, state='COMMITTED', build_id=build_id,
                         parent=row[0], plan=plan, terminal_state=terminal_state,
                         evidence_identity=identity(evidence),
                         usage_identity=identity(_usage_rows(self.db, build_id)))
            head = identity(entry)
            self.db.execute('INSERT INTO commits(build_id,parent,head,entry) VALUES (?,?,?,?)',
                            (build_id, row[0], head, canonical(entry).decode()))
            self.db.execute('UPDATE builds SET state=?,commit_head=? WHERE build_id=?',
                            ('COMMITTED', head, build_id))
            self.db.execute('UPDATE meta SET value=? WHERE key=?', (head, 'head'))
            if before_commit is not None:
                before_commit()
            self.db.commit()
            return head
        except Exception:
            self.db.rollback()
            raise

    def require_committed(self, build_id, plan, evidence):
        self.verify_chain()
        need(self.head == self.expected_head, 'Stale or untrusted corpus head')
        row = self.db.execute('SELECT plan,state,commit_head FROM builds WHERE build_id=?',
                              (build_id,)).fetchone()
        need(row is not None and row[0] == plan and row[1] == 'COMMITTED'
             and row[2] is not None, 'Corpus build is not committed')
        need(identity(evidence) == identity(_evidence_rows(self.db, build_id)),
             'Local build differs from committed corpus evidence')
        entry = json.loads(self.db.execute('SELECT entry FROM commits WHERE build_id=?',
            (build_id,)).fetchone()[0])
        return entry['terminal_state']

    def abort(self, build_id):
        """Explicitly release an uncommitted build; never subtract committed usage."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.verify_chain()
            row = self.db.execute('SELECT state FROM builds WHERE build_id=?', (build_id,)).fetchone()
            need(row == ('ACTIVE',), 'Only an active corpus reservation can be aborted')
            for kind,key,tokens in self.db.execute(
                    'SELECT kind,key,sum(tokens) FROM usage WHERE build_id=? GROUP BY kind,key',
                    (build_id,)):
                self.db.execute('UPDATE totals SET tokens=tokens-? WHERE kind=? AND key=?',
                                (tokens,kind,key))
            self.db.execute('DELETE FROM totals WHERE tokens=0')
            self.db.execute('DELETE FROM usage WHERE build_id=?', (build_id,))
            self.db.execute('DELETE FROM item_evidence WHERE build_id=?', (build_id,))
            self.db.execute('UPDATE builds SET state=? WHERE build_id=?', ('ABORTED', build_id))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def rollback_failed_governance(self, build_id):
        """Discard only an uncommitted claim rejected by current governance.

        Unlike explicit abort, this leaves the deterministic build ID available
        for a clean retry after the external authority becomes valid again.
        """
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.verify_chain()
            row = self.db.execute('SELECT parent,state FROM builds WHERE build_id=?',
                                  (build_id,)).fetchone()
            if row is None:
                self.db.commit()
                return False
            need(row == (self.head, 'ACTIVE') and self.head == self.expected_head,
                 'Only an active uncommitted claim can be rolled back')
            for kind,key,tokens in self.db.execute(
                    'SELECT kind,key,sum(tokens) FROM usage WHERE build_id=? GROUP BY kind,key',
                    (build_id,)):
                self.db.execute('UPDATE totals SET tokens=tokens-? WHERE kind=? AND key=?',
                                (tokens,kind,key))
            self.db.execute('DELETE FROM totals WHERE tokens=0')
            self.db.execute('DELETE FROM usage WHERE build_id=?', (build_id,))
            self.db.execute('DELETE FROM item_evidence WHERE build_id=?', (build_id,))
            self.db.execute('DELETE FROM builds WHERE build_id=?', (build_id,))
            self.verify_chain()
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def reconstruction(self, build_paths=None):
        """Read-only recomputation from sealed lineage and retained build payloads."""
        self.verify_chain()
        need(self.head == self.expected_head, 'Trusted corpus head mismatch')
        totals = Counter()
        for bid, hint in self.db.execute("SELECT build_id,path_hint FROM builds WHERE state='COMMITTED'"):
            encoded = self.db.execute('SELECT entry FROM commits WHERE build_id=?',
                                      (bid,)).fetchone()[0]
            entry = json.loads(encoded)
            if 'historical_import_identity' in entry:
                from training.historical_corpus_import import reconstruct
                candidate = reconstruct(self.root, self.root / hint,
                    approved_manifest_identity=self.approved_historical_manifest_identity,
                    approved_manifest_sha256=self.approved_historical_manifest_sha256)
                need(candidate['identity'] == entry['historical_import_identity']
                     and candidate['items'] == _evidence_rows(self.db, bid)
                     and [list(row) for row in candidate['usage']] == _usage_rows(self.db, bid),
                     'Historical corpus reconstruction differs from approved import')
                for _, kind, key, count, _ in candidate['usage']:
                    totals[(kind, key)] += count
                continue
            path = Path((build_paths or {}).get(bid, self.root / hint)).resolve()
            need(path.is_relative_to(self.root), 'Corpus build evidence escapes root')
            with sqlite3.connect((path / 'state.sqlite').as_uri() + '?mode=ro', uri=True) as build:
                meta = {k: json.loads(v) for k, v in build.execute('SELECT key,value FROM meta')}
                need(meta.get('corpus_build_id') == bid and meta.get('state') in
                     ('COMPLETE', 'TARGET_REACHED', 'TARGET_BOUNDARY'), 'Committed build evidence missing')
                actual = []
                for item, encoded, nsha, tsha, count in build.execute(
                    "SELECT item,record,normalized_sha,token_sha,token_count FROM items WHERE stage='ACCEPTED' ORDER BY item"):
                    record = json.loads(encoded)
                    verify_seal(record)
                    need(file_digest(path / 'raw' / item[7:]) == record['sha256']
                         and file_digest(path / 'normalized' / item[7:]) == nsha
                         and file_digest(path / 'tokens' / (item[7:] + '.bin')) == tsha,
                         'Committed build payload changed')
                    stored = self.db.execute('SELECT evidence FROM item_evidence WHERE build_id=? AND item=?',
                                             (bid,item)).fetchone()
                    need(stored is not None,'Committed item absent from corpus ledger')
                    described = json.loads(stored[0])
                    need(described['record'] == record and described['normalized_sha'] == nsha
                         and described['token_sha'] == tsha and described['tokens'] == count,
                         'Committed item evidence differs from build')
                    actual.append(described)
                need(actual == _evidence_rows(self.db, bid), 'Committed build item evidence changed')
            for _, kind, key, count, _ in _usage_rows(self.db, bid):
                totals[(kind, key)] += count
        return dict(sorted((kind + ':' + key, count) for (kind, key), count in totals.items()))


def _item_limits(db, build_id, item):
    return [list(row) for row in db.execute(
        'SELECT kind,key,tokens FROM usage WHERE build_id=? AND item=? ORDER BY kind,key',
        (build_id, item))]


def _check_scope(evidence, by_source, by_locator):
    scope = (evidence['repository_group'], evidence['domain_group'],
             evidence['split_group'])
    source = evidence['source_id']
    locator = evidence['source_locator'].casefold().rstrip('/')
    need(by_source.setdefault(source, scope) == scope and
         by_locator.setdefault(locator, scope) == scope,
         'Corpus source concentration/split scope changed across builds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--approved-head',required=True,
                        help='Trusted head supplied outside mutable corpus state')
    parser.add_argument('--approved-historical-import-identity')
    parser.add_argument('--approved-historical-manifest-identity')
    parser.add_argument('--approved-historical-manifest-sha256')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--reconstruct',action='store_true')
    action.add_argument('--abort-build-id',help='Explicitly release an active build only')
    args = parser.parse_args()
    need(args.ledger.exists(), 'Corpus ledger does not exist')
    with CorpusLedger(args.ledger,args.root,args.approved_head,
                      approved_historical_import_identity=args.approved_historical_import_identity,
                      approved_historical_manifest_identity=args.approved_historical_manifest_identity,
                      approved_historical_manifest_sha256=args.approved_historical_manifest_sha256) as ledger:
        need(ledger.head == args.approved_head, 'Trusted corpus head mismatch')
        if args.reconstruct:
            print(canonical(dict(policy=POLICY,head=ledger.head,
                                 committed_usage=ledger.reconstruction())).decode())
        else:
            ledger.abort(args.abort_build_id)
            print(canonical(dict(policy=POLICY,head=ledger.head,
                                 aborted_build_id=args.abort_build_id)).decode())


if __name__ == '__main__':
    main()
