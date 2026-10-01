"""Explicit, bounded observation recipes; never call arbitrary properties or methods.

This module is copied into historical target environments and uses Python 3.8+
syntax with only standard-library dependencies. NumPy is imported only when an
explicit ndarray/shape adapter is requested in an environment that already has it.
"""
import inspect
import types
from .expressions import data, Unsupported, FUNCTIONS, MAX_ITEMS


def validate_bindings(bindings, parameters):
    if not isinstance(bindings, dict) or len(bindings) > 8:
        raise Unsupported('observation_bindings')
    inputs, outputs = set(), set()
    for name, recipe in bindings.items():
        if (not isinstance(name, str) or not name.isidentifier() or name.startswith('_')
                or name in set(parameters) | FUNCTIONS | {'old', 'result', 'error'}):
            raise Unsupported('observation_alias')
        if not isinstance(recipe, dict) or set(recipe) != {'root', 'path', 'adapter'}:
            raise Unsupported('observation_recipe')
        if recipe['root'] not in set(parameters) | {'result'}:
            raise Unsupported('observation_root')
        if (not isinstance(recipe['path'], list) or len(recipe['path']) > 4 or
                any(not isinstance(p, str) or not p.isidentifier() or p.startswith('__')
                    for p in recipe['path'])):
            raise Unsupported('observation_path')
        if recipe['adapter'] not in {'primitive', 'length', 'ndarray', 'shape'}:
            raise Unsupported('observation_adapter')
        (outputs if recipe['root'] == 'result' else inputs).add(name)
    return inputs, outputs


def project(raw, recipe):
    value = raw[recipe['root']]
    for field in recipe['path']:
        if type(value) is dict:
            value = value[field]
        else:
            # Static lookup does not execute descriptors or __getattr__.
            attribute = inspect.getattr_static(value, field)
            if isinstance(attribute, types.MemberDescriptorType):
                value = attribute.__get__(value, type(value))
            elif hasattr(type(attribute), '__get__'):
                raise Unsupported('property_or_descriptor_observation')
            else:
                value = attribute
    adapter = recipe['adapter']
    if adapter == 'primitive':
        return data(value)
    if adapter == 'length' and type(value) in (str, list, tuple, dict):
        return data(len(value))
    if adapter in ('ndarray', 'shape', 'length'):
        try:
            import numpy as np
        except ImportError as exc:
            raise Unsupported('numpy_adapter_unavailable') from exc
        if type(value) is not np.ndarray:
            raise Unsupported('adapter_requires_exact_numpy_ndarray')
        if adapter == 'shape':
            return data(tuple(int(x) for x in value.shape))
        if adapter == 'length':
            return data(len(value))
        if value.size > MAX_ITEMS or value.dtype.kind not in 'biufU':
            raise Unsupported('numpy_size_or_dtype')
        return data(value.tolist())
    raise Unsupported('unsupported_adapter')


def environment(raw, names, bindings):
    result = {}
    for name in names:
        recipe = bindings.get(name)
        # An exceptional exit has no return object to project. Preserve None so
        # guards such as `error is None and output_shape == ...` short-circuit.
        if recipe and recipe['root'] == 'result' and raw.get('error') is not None:
            result[name] = None
        else:
            result[name] = project(raw, recipe) if recipe else data(raw[name])
    return result
