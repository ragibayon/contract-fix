"""Contract schema, frozen bundles, and observed-event admission (not proof)."""
import hashlib
import json
from collections import Counter
from pathlib import PurePosixPath
from .expressions import Expression, Unsupported, FUNCTIONS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def observation_digest(value):
    """Keep type and mapping order distinctions in bounded primitive observations."""
    def tagged(x):
        if type(x) is dict:
            return ['dict', [[tagged(k), tagged(v)] for k, v in x.items()]]
        if type(x) in (list, tuple):
            return [type(x).__name__, [tagged(v) for v in x]]
        return [type(x).__name__, x]
    return digest(tagged(value))


def validate_clause(c, parameters=None):
    required = {'id', 'target', 'role', 'when', 'ensure', 'evidence'}
    if not isinstance(c, dict) or not required <= set(c) or set(c) - required - {'observations'}:
        raise Unsupported('clause_keys')
    if not isinstance(c['id'], str) or not c['id']:
        raise Unsupported('clause_id')
    if c['role'] not in ('REPAIR', 'PRESERVE'):
        raise Unsupported('clause_role')
    t = c['target']
    if not isinstance(t, dict) or set(t) != {'file', 'symbol'}:
        raise Unsupported('target_keys')
    if not isinstance(t['file'], str) or not isinstance(t['symbol'], str):
        raise Unsupported('target_strings_required')
    if not all(isinstance(c[x], str) for x in ('when', 'ensure')):
        raise Unsupported('expression_strings_required')
    p = PurePosixPath(t['file'])
    if p.is_absolute() or '..' in p.parts or '\\' in str(p) or p.suffix != '.py':
        raise Unsupported('target_path')
    if not all(x.isidentifier() for x in t['symbol'].split('.')):
        raise Unsupported('target_symbol')
    if not isinstance(c['evidence'], str) or not c['evidence'].strip():
        raise Unsupported('evidence_required')
    if parameters is not None:
        if set(parameters) & ({'old', 'result', 'error'} | FUNCTIONS):
            raise Unsupported('reserved_parameter_name')
        from .observations import validate_bindings
        input_aliases, output_aliases = validate_bindings(c.get('observations', {}), parameters)
        guard = Expression(c['when'], set(parameters) | input_aliases, entry=True)
        check = Expression(c['ensure'], set(parameters) | input_aliases | output_aliases)
        if check.old_names & output_aliases:
            raise Unsupported('old_cannot_reference_output_observation')
        if c['when'].strip() == 'False' or c['ensure'].strip() == 'True':
            raise Unsupported('obvious_vacuity')
        return guard, check


def seal(clauses):
    if not clauses:
        raise Unsupported('empty_bundle')
    for c in clauses:
        validate_clause(c)
    if len({c['id'] for c in clauses}) != len(clauses):
        raise Unsupported('duplicate_clause_id')
    body = {'version': 'contractfix-ec/0.1', 'clauses': clauses}
    return {**body, 'sha256': digest(body)}


def verify(bundle):
    if not isinstance(bundle, dict) or set(bundle) != {'version', 'clauses', 'sha256'}:
        raise Unsupported('bundle_keys')
    expected = seal(bundle['clauses'])
    if bundle != expected:
        raise Unsupported('bundle_hash_or_version_mismatch')
    return bundle


def summarize(events, clauses, command_exit=0, infrastructure_error=None):
    states = Counter(e['state'] for e in events)
    ids = {e.get('clause_id') for e in events if e['state'] in ('SATISFIED', 'VIOLATED')}
    missing = sorted({c['id'] for c in clauses} - ids)
    if infrastructure_error or command_exit != 0 or states['CHECKER_ERROR'] or states['INCONCLUSIVE']:
        disposition = 'INCONCLUSIVE'
    elif states['VIOLATED']:
        disposition = 'REJECT'
    elif missing:
        disposition = 'INCONCLUSIVE'
    else:
        disposition = 'OBSERVED_PASS'
    return {'disposition': disposition, 'counts': dict(states), 'unexercised_clauses': missing,
            'command_exit': command_exit, 'infrastructure_error': infrastructure_error}


def admit_pair(bundle, base_report, candidate_report):
    """Gate a fixed bundle on matched primitive-input observations.

    Run/command IDs must match. This is deliberately conservative and is not a
    relational oracle for arbitrary objects, nondeterministic or concurrent code.
    """
    verify(bundle)
    for r in (base_report, candidate_report):
        if r.get('bundle_sha256') != bundle['sha256']:
            return {'decision': 'INCONCLUSIVE', 'reason': 'bundle_mismatch'}
    for field in ('repo_sha256', 'witness_sha256', 'runtime_sha256', 'edit_policy_sha256'):
        if not base_report.get(field) or base_report[field] != candidate_report.get(field):
            return {'decision': 'INCONCLUSIVE', 'reason': field + '_mismatch'}
    if base_report.get('executor_sha256') != candidate_report.get('executor_sha256'):
        return {'decision': 'INCONCLUSIVE', 'reason': 'executor_mismatch'}
    if base_report['summary'].get('infrastructure_error') or base_report['summary'].get('command_exit') != 0:
        return {'decision': 'INCONCLUSIVE', 'reason': 'base_infrastructure_failure'}
    if base_report.get('command') != candidate_report.get('command'):
        return {'decision': 'INCONCLUSIVE', 'reason': 'command_mismatch'}
    b, p = base_report['events'], candidate_report['events']
    if candidate_report['summary']['disposition'] != 'OBSERVED_PASS':
        return {'decision': candidate_report['summary']['disposition'], 'reason': 'candidate_not_clean'}
    if any(e['state'] in ('CHECKER_ERROR', 'INCONCLUSIVE') for e in b):
        return {'decision': 'INCONCLUSIVE', 'reason': 'base_monitor_error'}
    roles = {c['id']: c['role'] for c in bundle['clauses']}
    repairs = [e for e in b if e['state'] == 'VIOLATED' and roles.get(e['clause_id']) == 'REPAIR']
    if not repairs:
        return {'decision': 'INCONCLUSIVE', 'reason': 'no_repair_violation_witness'}
    if any(e['state'] == 'VIOLATED' and roles.get(e['clause_id']) == 'PRESERVE' for e in b):
        return {'decision': 'INCONCLUSIVE', 'reason': 'preserve_clause_conflicts_with_base'}
    key = lambda e: (e['clause_id'], e['test_id'], e.get('case_id'), e['input_sha256'])
    expected = [e for e in b if e['state'] in ('SATISFIED', 'VIOLATED')]
    if any(e.get('input_sha256') is None for e in expected):
        return {'decision': 'INCONCLUSIVE', 'reason': 'cannot_pair_nonprimitive_inputs'}
    available = Counter(key(e) for e in p if e['state'] == 'SATISFIED')
    for e in expected:
        k = key(e)
        if available[k] <= 0:
            return {'decision': 'INCONCLUSIVE', 'reason': 'lost_applicability_or_witness', 'clause_id': e['clause_id']}
        available[k] -= 1
    return {'decision': 'ADMIT_OBSERVED', 'reason': 'fixed_contracts_pass_matched_observations',
            'repair_witness_count': len(repairs), 'assurance': 'execution_bounded_not_proved'}
