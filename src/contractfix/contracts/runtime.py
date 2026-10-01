"""Observation-only monitors preserve target returns/exceptions and never re-invoke it."""
import functools
import inspect
import json
import os
from .core import observation_digest, validate_clause, verify
from .expressions import data, Unsupported
from .observations import environment


def _write(event):
    prefix = os.environ.get('CONTRACTFIX_EC_EVENTS')
    if prefix:
        try:
            with open(prefix + '.' + str(os.getpid()) + '.jsonl', 'a', encoding='utf-8') as output:
                output.write(json.dumps(event, allow_nan=False) + '\n')
        except (OSError, TypeError, ValueError):
            # A failed observer must not replace the target's outcome. The host
            # treats missing receipts as inconclusive rather than successful.
            pass


def _callers():
    root = os.environ.get('CONTRACTFIX_EC_REPO', '')
    callers = []
    frame = inspect.currentframe()
    try:
        frame = frame.f_back.f_back
        while frame and len(callers) < 8:
            filename = frame.f_code.co_filename
            if root and filename.startswith(root + os.sep):
                callers.append({'file': filename[len(root) + 1:],
                                'function': frame.f_code.co_name, 'line': frame.f_lineno})
            frame = frame.f_back
    finally:
        del frame
    return callers


def monitor(target_key):
    with open(os.environ['CONTRACTFIX_EC_BUNDLE'], encoding='utf-8') as stream:
        bundle = verify(json.load(stream))
    clauses = [c for c in bundle['clauses'] if c['target']['file'] + ':' + c['target']['symbol'] == target_key]

    def decorate(fn):
        signature = inspect.signature(fn)
        compiled = [(c, *validate_clause(c, set(signature.parameters))) for c in clauses]

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            common = {'target': target_key,
                      'test_id': os.environ.get('PYTEST_CURRENT_TEST', 'command'),
                      'case_id': os.environ.get('CONTRACTFIX_CASE_ID'),
                      'bundle_sha256': bundle['sha256']}
            _write(dict(common, state='TARGET_HIT', callers=_callers()))
            pending = []
            for clause, guard, check in compiled:
                bindings = clause.get('observations', {})
                event = dict(common, clause_id=clause['id'], role=clause['role'])
                try:
                    try:
                        input_sha = observation_digest(data(dict(bound.arguments)))
                        assurance = 'full_primitive_arguments'
                    except Unsupported:
                        inputs = {name for name, recipe in bindings.items() if recipe['root'] != 'result'}
                        case_sha = os.environ.get('CONTRACTFIX_CASE_SHA256')
                        if not inputs or not case_sha:
                            input_sha, assurance = None, 'unpairable'
                        else:
                            projected = environment(bound.arguments, inputs, bindings)
                            input_sha = observation_digest({'case': case_sha, 'projection': projected})
                            assurance = 'declared_projection_and_frozen_case_not_full_heap'
                    event.update(input_sha256=input_sha, input_assurance=assurance)
                    old_names = guard.old_names | check.old_names
                    old = environment(bound.arguments, old_names, bindings)
                    env = environment(bound.arguments, guard.names, bindings)
                    applies = guard.evaluate(dict(env, old=old))
                    if not applies:
                        _write(dict(event, state='NOT_APPLICABLE', reason='guard_false'))
                        continue
                    pending.append((event, check, old, bindings))
                except Exception as exc:
                    _write(dict(event, state='INCONCLUSIVE' if isinstance(exc, Unsupported) else 'CHECKER_ERROR',
                                reason='entry:' + type(exc).__name__ + ':' + str(exc)[:300]))
            result, error = None, None
            try:
                result = fn(*args, **kwargs)
                return result
            except BaseException as exc:
                typ = type(exc)
                error = typ.__module__ + '.' + typ.__qualname__
                raise
            finally:
                for event, check, old, bindings in pending:
                    try:
                        raw = dict(bound.arguments, result=result, error=error)
                        env = environment(raw, check.names, bindings)
                        ok = check.evaluate(dict(env, old=old))
                        _write(dict(event, state='SATISFIED' if ok else 'VIOLATED',
                                    reason='predicate_true' if ok else 'predicate_false',
                                    observed=env, old=old))
                    except Exception as exc:
                        _write(dict(event, state='INCONCLUSIVE' if isinstance(exc, Unsupported) else 'CHECKER_ERROR',
                                    reason='exit:' + type(exc).__name__ + ':' + str(exc)[:300]))
        return wrapped
    return decorate
