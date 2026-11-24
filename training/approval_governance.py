"""Externally pinned, append-only approval status history.

The log proves consistency with an operator-supplied current byte digest. It
does not authenticate that operator or alter the approval artifacts it names.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from training.production_corpus import canonical, file_digest
from training.production_spec import need, seal, verify_seal

SCHEMA = 'yug-approval-governance-event-v1'
POLICY = 'yug-approval-governance-v1'
TYPES = frozenset(('production_plan', 'ownership_approval', 'source_clearance',
                   'source_component', 'file_clearance', 'decision_receipt', 'conditional_approval',
                   'historical_import', 'source_revision', 'rights_evidence_status'))
REVISION_SCHEMA = 'yug-source-revision-activation-v1'
REVISION_FIELDS = frozenset(('schema', 'decision', 'source_id', 'locator',
    'prior_revision', 'revision', 'prior_clearance_identity', 'clearance_identity',
    'rights_dependencies', 'reviewer', 'decided_at', 'identity'))
REVISION_SCHEMA_V2 = 'yug-source-revision-activation-v2'
REVISION_FIELDS_V2 = REVISION_FIELDS | frozenset((
    'candidate_complete_identity', 'candidate_complete_sha256',
    'candidate_plan_sha256', 'expected_governance_sha256',
    'expected_governance_head'))
STATUSES = frozenset(('ACTIVE', 'SUPERSEDED', 'REVOKED'))
MAX_BYTES = 16 * 1024 * 1024
SHA = re.compile(r'sha256:[0-9a-f]{64}\Z')
UTC_TIME = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z')
FIELDS = frozenset(('schema', 'policy', 'approval_identity', 'approval_type',
                    'scope', 'previous_status', 'new_status', 'replaced_by',
                    'replacement_scope', 'transition_at', 'reason_ref',
                    'previous_event_identity', 'identity'))


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'Duplicate governance field')
        value[key] = item
    return value


def _scope(scope):
    need(type(scope) is dict and bool(scope) and all(type(k) is str and k and
         type(v) is str and v for k, v in scope.items()),
         'Invalid approval governance scope')
    return canonical(scope)


def _current_key(kind, scope, approval_identity):
    # Rights evidence is independently revocable. Other approval types retain
    # their existing exclusive scope semantics.
    key = (kind, _scope(scope))
    return (*key, approval_identity) if kind == 'rights_evidence_status' else key


def revision_scope(source_id, locator):
    """One governance scope across every revision of the same exact source."""
    parsed = urlsplit(locator) if type(locator) is str else None
    need(type(source_id) is str and re.fullmatch(r'[a-z0-9][a-z0-9-]*', source_id)
         and parsed is not None and parsed.scheme == 'https' and parsed.netloc and parsed.path.strip('/')
         and not parsed.username and not parsed.password and
         not parsed.query and not parsed.fragment and
         locator == locator.strip(),
         'Invalid stable source revision scope')
    return dict(source_id=source_id, locator=locator.casefold().rstrip('/'))


def verify_revision_activation(path, expected_sha256, prior_source, new_source,
                               prior_clearance_identity, clearance_identity, root):
    """Check an external activation decision; this never clears R2 for production.

    Rights entries are dependencies on existing evidence bytes and governance
    status, not grants of rights. A later transition validator must establish
    that the list is complete before an activated R2 can be used.
    """
    from training.production_spec import local_path
    need(type(expected_sha256) is str and SHA.fullmatch(expected_sha256),
         'External activation decision byte pin required')
    path = Path(path).resolve()
    root = Path(root).resolve()
    need(path.is_relative_to(root) and path.is_file() and
         file_digest(path) == expected_sha256,
         'Missing or changed externally pinned activation decision')
    raw = path.read_bytes()
    need(len(raw) <= 65536, 'Oversized activation decision')
    try:
        decision = json.loads(raw, object_pairs_hook=_pairs)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Malformed activation decision') from exc
    need(canonical(decision) == raw and set(decision) in (REVISION_FIELDS, REVISION_FIELDS_V2),
         'Noncanonical or incomplete activation decision')
    verify_seal(decision)
    scope = revision_scope(prior_source['source_id'], prior_source['locator'])
    need(revision_scope(new_source['source_id'], new_source['locator']) == scope and
         prior_source['locator'] == new_source['locator'] and
         decision['schema'] == (REVISION_SCHEMA_V2 if set(decision) == REVISION_FIELDS_V2
                                else REVISION_SCHEMA) and
         decision['decision'] == 'ACTIVATE_REVISION' and
         decision['source_id'] == scope['source_id'] and
         decision['locator'] == prior_source['locator'] and
         decision['prior_revision'] == prior_source['revision'] and
         decision['revision'] == new_source['revision'] and
         decision['prior_revision'] != decision['revision'] and
         decision['prior_clearance_identity'] == prior_clearance_identity and
         decision['clearance_identity'] == clearance_identity and
         all(type(x) is str and SHA.fullmatch(x) for x in
             (prior_clearance_identity, clearance_identity)) and
         type(decision['reviewer']) is str and bool(decision['reviewer'].strip()) and
         type(decision['decided_at']) is str and bool(decision['decided_at'].strip()),
         'Activation decision has wrong source, revision, or clearance scope')
    stable = ('repository_group', 'domain_group', 'split_group', 'split',
              'target_cap_tokens', 'category_caps_tokens', 'allowed_categories',
              'category', 'subcategory', 'license_id')
    need(all(prior_source[k] == new_source[k] for k in stable),
         'Revision activation changes source allocation or rights policy')
    dependencies = decision['rights_dependencies']
    need(type(dependencies) is list and bool(dependencies) and
         dependencies == sorted(dependencies, key=lambda x: x['path']) and
         all(type(x) is dict and set(x) == {'path', 'sha256'} and
             type(x['path']) is str and type(x['sha256']) is str and
             SHA.fullmatch(x['sha256']) for x in dependencies) and
         len({x['path'] for x in dependencies}) == len(dependencies),
         'Unknown or ambiguous activation rights dependencies')
    for dependency in dependencies:
        need(file_digest(local_path(root, dependency['path'])) == dependency['sha256'],
             'Activation rights evidence missing or changed')
    need(file_digest(path) == expected_sha256,
         'Activation decision changed during verification')
    return decision


_lock_state = threading.local()


@contextmanager
def governance_lock(path, *, timeout_seconds=0, writer=False):
    """One stable OS lock for governance writers and prospective consumers.

    A lock file is coordination metadata, never a second authority record. It
    must not be removed: removing it would permit two distinct lock inodes.
    Nested use in the same thread supports ledger calls inside a build lease.
    """
    path = Path(path).resolve()
    lock_path = path.with_name('.' + path.name + '.lock')
    held = getattr(_lock_state, 'held', None)
    if held is None:
        held = _lock_state.held = {}
    key = str(lock_path)
    if key in held:
        need(not writer, 'Governance writer cannot overtake an active consumer')
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return
    need(type(timeout_seconds) in (int, float) and 0 <= timeout_seconds <= 30,
         'Invalid governance drain timeout')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+b') as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b'0')
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                handle.seek(0)
                if sys.platform == 'win32':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise ValueError('Governance consumer or writer has not drained') from exc
                time.sleep(0.025)
        held[key] = 1
        try:
            yield
        finally:
            del held[key]
            handle.seek(0)
            if sys.platform == 'win32':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _replace_governance(path, data):
    """Replace one same-directory log after flushing the complete next image."""
    temporary = path.with_name(path.name + '.stage')
    if temporary.exists():
        need(temporary.read_bytes() == data,
             'Indeterminate interrupted governance staging')
    try:
        if not temporary.exists():
            with temporary.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        if sys.platform == 'win32':
            import ctypes
            move = ctypes.WinDLL('kernel32', use_last_error=True).MoveFileExW
            move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
            move.restype = ctypes.c_int
            moved = move(str(temporary), str(path), 0x1 | 0x8)
            if not moved:
                raise OSError(ctypes.get_last_error(), 'Governance log replacement failed')
        else:
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        # Preserve a staged file after an uncertain failure for explicit audit.
        pass


def event(*, approval_identity, approval_type, scope, previous_status,
          new_status, transition_at, reason_ref, previous_event_identity=None,
          replaced_by=None, replacement_scope=None):
    """Construct one event; replay validates its sequence and exact schema."""
    value = dict(schema=SCHEMA, policy=POLICY,
                 approval_identity=approval_identity, approval_type=approval_type,
                 scope=scope, previous_status=previous_status,
                 new_status=new_status, replaced_by=replaced_by,
                 replacement_scope=replacement_scope, transition_at=transition_at,
                 reason_ref=reason_ref,
                 previous_event_identity=previous_event_identity)
    _validate_fields(value)
    return seal(value)


def _validate_fields(value):
    need(type(value) is dict and set(value) in (FIELDS, FIELDS - {'identity'}),
         'Malformed governance event fields')
    need(value['schema'] == SCHEMA and value['policy'] == POLICY and
         value['approval_type'] in TYPES and type(value['approval_identity']) is str
         and SHA.fullmatch(value['approval_identity']), 'Unsupported governance approval')
    _scope(value['scope'])
    if value['approval_type'] in ('source_revision', 'rights_evidence_status'):
        scope = value['scope']
        need(set(scope) == {'source_id', 'locator'} and
             scope == revision_scope(scope['source_id'], scope['locator']),
             'Wrong stable source governance scope')
    need(value['previous_status'] in ('NONE', 'ACTIVE') and
         value['new_status'] in STATUSES, 'Invalid governance transition status')
    need(type(value['reason_ref']) is str and bool(value['reason_ref'].strip()),
         'Missing governance decision reference')
    need(type(value['transition_at']) is str and UTC_TIME.fullmatch(value['transition_at']),
         'Invalid governance timestamp')
    try:
        parsed = datetime.fromisoformat(value['transition_at'].replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('Invalid governance timestamp') from exc
    need(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
         'Governance timestamp must be UTC')
    previous = value['previous_event_identity']
    need(previous is None or (type(previous) is str and SHA.fullmatch(previous)),
         'Invalid previous governance identity')
    replacement = value['replaced_by']
    if value['new_status'] == 'SUPERSEDED':
        need(value['previous_status'] == 'ACTIVE' and type(replacement) is str
             and SHA.fullmatch(replacement) and replacement != value['approval_identity']
             and _scope(value['replacement_scope']) == _scope(value['scope']),
             'Invalid governance supersession scope')
    else:
        need(replacement is None and value['replacement_scope'] is None,
             'Unexpected governance replacement')


class Governance:
    """Read and replay the complete log every time current authority is needed."""
    def __init__(self, path, expected_sha256):
        need(path is not None and type(expected_sha256) is str and
             SHA.fullmatch(expected_sha256), 'Current external governance pin required')
        self.path = Path(path)
        try:
            with self.path.open('rb') as source:
                raw = source.read(MAX_BYTES + 1)
        except OSError as exc:
            raise ValueError('Governance log missing or unreadable') from exc
        need(len(raw) <= MAX_BYTES, 'Governance history exceeds bounded metadata limit')
        self.byte_sha256 = 'sha256:' + hashlib.sha256(raw).hexdigest()
        need(self.byte_sha256 == expected_sha256,
             'Governance log missing, stale or changed')
        self.states = {}
        self.current = {}
        self.events = []
        previous = None
        last_time = None
        for line in raw.splitlines(keepends=True):
            need(line.endswith(b'\n') and len(line) <= 16384,
                 'Incomplete or oversized governance event')
            try:
                value = json.loads(line[:-1], object_pairs_hook=_pairs)
            except (ValueError, UnicodeError) as exc:
                raise ValueError('Malformed governance event') from exc
            need(canonical(value) + b'\n' == line,
                 'Noncanonical governance event')
            _validate_fields(value)
            verify_seal(value)
            need(value['previous_event_identity'] == previous,
                 'Governance history order changed')
            when = value['transition_at']
            need(last_time is None or when >= last_time,
                 'Governance transition time moved backwards')
            last_time = when
            self._apply(value)
            self.events.append(value)
            previous = value['identity']
        need(bool(self.events), 'Empty governance history')
        self.head = previous

    def _apply(self, value):
        aid = value['approval_identity']
        kind = value['approval_type']
        scope = value['scope']
        key = _current_key(kind, scope, aid)
        if kind == 'source_revision':
            for state in self.states.values():
                if state['type'] != 'source_revision':
                    continue
                earlier = state['scope']
                need(earlier['source_id'] != scope['source_id'] and
                     earlier['locator'] != scope['locator'] or
                     _scope(earlier) == key[1],
                     'Conflicting source revision identity or locator alias')
        prior = self.states.get(aid)
        status = value['new_status']
        if value['previous_status'] == 'NONE':
            need(status == 'ACTIVE' and prior is None and key not in self.current,
                 'Unknown or conflicting governance registration')
            self.states[aid] = dict(type=kind, scope=scope, status='ACTIVE',
                                    replaced_by=None)
            self.current[key] = aid
            return
        need(prior is not None and prior['type'] == kind and
             _scope(prior['scope']) == key[1] and prior['status'] == 'ACTIVE' and
             self.current.get(key) == aid,
             'Unknown, stale or conflicting governance transition')
        need(status in ('REVOKED', 'SUPERSEDED'),
             'Invalid governance state change')
        del self.current[key]
        prior['status'] = status
        prior['replaced_by'] = value['replaced_by']
        if status == 'SUPERSEDED':
            replacement = value['replaced_by']
            need(replacement not in self.states,
                 'Duplicate or conflicting superseding approval')
            self.states[replacement] = dict(type=kind, scope=scope,
                                            status='ACTIVE', replaced_by=None)
            self.current[_current_key(kind, scope, replacement)] = replacement

    def require_current(self, approval_type, approval_identity, scope):
        key = _current_key(approval_type, scope, approval_identity)
        state = self.states.get(approval_identity)
        need(state is not None and state['type'] == approval_type and
              _scope(state['scope']) == key[1] and state['status'] == 'ACTIVE' and
             self.current.get(key) == approval_identity,
             'Approval revoked, superseded, unknown or wrong scope')
        return state

    def revision_activation_identity(self, source_id, locator):
        """Return current activation and whether this source ever entered the lifecycle."""
        scope = revision_scope(source_id, locator)
        key = ('source_revision', _scope(scope))
        seen = any(state['type'] == 'source_revision' and
                   _scope(state['scope']) == key[1] for state in self.states.values())
        return seen, self.current.get(key)

    def require_prospective_revision(self, source):
        """Legacy sources continue; transitioned sources await R2 lineage validation.

        A source-level event immediately retires old prospective plans. The
        later transition stage must supply and verify R2 clearance lineage
        before this gate can admit the newly active revision.
        """
        seen, _active = self.revision_activation_identity(
            source['source_id'], source['locator'])
        need(not seen and source.get('revision_activation') is None,
             'Source revision activation lacks verified transition clearance')

    def require_rights_dependencies(self, source_id, locator, dependencies, root):
        """Check explicit evidence status; no evidence-status event grants rights."""
        from training.production_spec import local_path
        scope = revision_scope(source_id, locator)
        need(type(dependencies) is list and bool(dependencies),
             'Unknown activation rights dependencies')
        seen = set()
        for ref in dependencies:
            need(type(ref) is dict and set(ref) == {'path', 'sha256'} and
                 type(ref['sha256']) is str and SHA.fullmatch(ref['sha256']) and
                 ref['sha256'] not in seen,
                 'Ambiguous activation rights dependencies')
            seen.add(ref['sha256'])
            need(file_digest(local_path(root, ref['path'])) == ref['sha256'],
                 'Activation rights evidence missing or changed')
            self.require_current('rights_evidence_status', ref['sha256'], scope)

    def require_activation_decision(self, path, expected_sha256, prior_source,
                                    new_source, prior_clearance_identity,
                                    clearance_identity, root):
        """Bind current status to pinned decision and current rights dependencies.

        This is a governance precondition, not a production clearance.
        """
        decision = verify_revision_activation(path, expected_sha256,
            prior_source, new_source, prior_clearance_identity,
            clearance_identity, root)
        scope = revision_scope(decision['source_id'], decision['locator'])
        self.require_current('source_revision', decision['identity'], scope)
        self.require_rights_dependencies(decision['source_id'],
            decision['locator'], decision['rights_dependencies'], root)
        return decision


def append_event(path, expected_sha256, value):
    """Replace the complete log with one more reviewed event; return its pin.

    The caller must publish/trust the returned pin outside mutable build state.
    An interrupted replacement leaves either the old or new complete log image.
    """
    path = Path(path).resolve()
    # A same-thread revocation can invalidate an active consumer and is caught
    # by its next current-pin check. An activation may never reenter a lease.
    with governance_lock(path, writer=value.get('approval_type') == 'source_revision'):
        return _append_event_locked(path, expected_sha256, value)


def _append_event_locked(path, expected_sha256, value, *, expected_new_sha256=None):
    """Caller holds governance_lock; compare-and-swap a complete log image."""
    if path.exists():
        history = Governance(path, expected_sha256)
        need(value.get('previous_event_identity') == history.head,
             'Stale governance transition')
    else:
        need(expected_sha256 is None and
             value.get('previous_event_identity') is None,
             'New governance history requires explicit genesis')
        history = object.__new__(Governance)
        history.states = {}
        history.current = {}
        history.events = []
    _validate_fields(value)
    verify_seal(value)
    need(not history.events or
         value['transition_at'] >= history.events[-1]['transition_at'],
         'Governance transition time moved backwards')
    history._apply(value)
    old = path.read_bytes() if path.exists() else b''
    data = old + canonical(value) + b'\n'
    need(len(data) <= MAX_BYTES,
         'Governance history exceeds bounded metadata limit')
    proposed = 'sha256:' + hashlib.sha256(data).hexdigest()
    need(expected_new_sha256 is None or proposed == expected_new_sha256,
         'Proposed governance log differs from independent expected digest')
    need((not path.exists() and expected_sha256 is None) or
         (path.exists() and file_digest(path) == expected_sha256),
         'Stale governance head before replacement')
    _replace_governance(path, data)
    need(file_digest(path) == proposed, 'Governance replacement identity changed')
    return proposed
