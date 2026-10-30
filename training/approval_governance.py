"""Externally pinned, append-only approval status history.

The log proves consistency with an operator-supplied current byte digest. It
does not authenticate that operator or alter the approval artifacts it names.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

from training.production_corpus import canonical, file_digest
from training.production_spec import need, seal, verify_seal

SCHEMA = 'yug-approval-governance-event-v1'
POLICY = 'yug-approval-governance-v1'
TYPES = frozenset(('production_plan', 'ownership_approval', 'source_clearance',
                   'file_clearance', 'decision_receipt', 'conditional_approval',
                   'historical_import'))
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
        key = (kind, _scope(scope))
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
            self.current[key] = replacement

    def require_current(self, approval_type, approval_identity, scope):
        key = (approval_type, _scope(scope))
        state = self.states.get(approval_identity)
        need(state is not None and state['type'] == approval_type and
             _scope(state['scope']) == key[1] and state['status'] == 'ACTIVE' and
             self.current.get(key) == approval_identity,
             'Approval revoked, superseded, unknown or wrong scope')
        return state


def append_event(path, expected_sha256, value):
    """Append one externally reviewed event; return the new byte pin.

    The caller must publish/trust the returned pin outside mutable build state.
    An interrupted append leaves an incomplete log that cannot authorize use.
    """
    path = Path(path)
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
    data = canonical(value) + b'\n'
    need((path.stat().st_size if path.exists() else 0) + len(data) <= MAX_BYTES,
         'Governance history exceeds bounded metadata limit')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    return file_digest(path)
