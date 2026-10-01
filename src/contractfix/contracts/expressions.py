"""A bounded interpreter for a small, ordinary-Python expression subset.

No eval/exec, repository calls, properties, imports, or arbitrary objects.
This limits monitor side effects; it is NOT a sandbox for the target repository.
"""
import ast
import math
import operator

class Unsupported(ValueError):
    pass

MAX_ITEMS = 2048
MAX_STEPS = 20000
MAX_BITS = 4096
MAX_SOURCE = 2000


def data(value, depth=0, seen=None, budget=None):
    """Copy exact built-in data without invoking user-defined conversion hooks."""
    budget = [MAX_STEPS] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        raise Unsupported('snapshot_item_budget')
    if depth > 20:
        raise Unsupported('snapshot_depth_limit')
    t = type(value)
    if value is None or t in (bool, str):
        if t is str and len(value) > MAX_ITEMS:
            raise Unsupported('string_size_limit')
        return value
    if t is int:
        if value.bit_length() > MAX_BITS:
            raise Unsupported('integer_size_limit')
        return value
    if t is float:
        if not math.isfinite(value):
            raise Unsupported('nonfinite_float')
        return value
    if t not in (list, tuple, dict):
        raise Unsupported('unsupported_observation_type:' + t.__name__)
    if len(value) > MAX_ITEMS:
        raise Unsupported('collection_size_limit')
    seen = set() if seen is None else seen
    if id(value) in seen:
        raise Unsupported('cyclic_observation')
    seen.add(id(value))
    try:
        if t is dict:
            if any(type(k) is not str for k in value):
                raise Unsupported('unsupported_dict_key')
            return {data(k, depth + 1, seen, budget): data(v, depth + 1, seen, budget) for k, v in value.items()}
        out = [data(v, depth + 1, seen, budget) for v in value]
        return tuple(out) if t is tuple else out
    finally:
        seen.remove(id(value))


FUNCTIONS = {'len', 'abs', 'min', 'max', 'sum', 'all', 'any', 'sorted', 'range', 'isinstance'}
TYPE_NAMES = {'bool': bool, 'dict': dict, 'float': float, 'int': int,
              'list': list, 'str': str, 'tuple': tuple}
NODES = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Store,
         ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
         ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
         ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
         ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.Subscript, ast.Slice,
         ast.List, ast.Tuple, ast.Dict, ast.Call, ast.GeneratorExp,
         ast.ListComp, ast.comprehension, ast.IfExp)


class Expression:
    def __init__(self, source, parameters, entry=False):
        if not isinstance(source, str) or not source.strip() or len(source) > MAX_SOURCE:
            raise Unsupported('expression_length')
        self.source = source
        try:
            self.tree = ast.parse(source, mode='eval').body
        except SyntaxError as exc:
            raise Unsupported('syntax:' + str(exc)) from exc
        # Python 3.8 wraps subscription indices in ast.Index; normalize to 3.9+ form.
        class NormalizeIndex(ast.NodeTransformer):
            def visit_Index(self, node):
                return self.visit(node.value)
        self.tree = NormalizeIndex().visit(self.tree)
        nodes = list(ast.walk(self.tree))
        if len(nodes) > 300:
            raise Unsupported('expression_ast_limit')
        bound = {c.target.id for c in nodes if isinstance(c, ast.comprehension)
                 and isinstance(c.target, ast.Name)}
        if bound & (set(parameters) | FUNCTIONS | {'old', 'result', 'error'}):
            raise Unsupported('comprehension_variable_shadowing')
        allowed = set(parameters) | FUNCTIONS | set(TYPE_NAMES) | bound | {'old'}
        if not entry:
            allowed |= {'result', 'error'}
        self.names = set()
        self.old_names = set()
        for n in nodes:
            if not isinstance(n, NODES):
                raise Unsupported('unsupported_syntax:' + type(n).__name__)
            if isinstance(n, ast.Name):
                if n.id not in allowed or n.id.startswith('__'):
                    raise Unsupported('unknown_name:' + n.id)
                if isinstance(n.ctx, ast.Load) and n.id not in FUNCTIONS | bound | {'old'}:
                    self.names.add(n.id)
            if isinstance(n, ast.Call):
                if not isinstance(n.func, ast.Name) or n.func.id not in FUNCTIONS or n.keywords:
                    name = n.func.id if isinstance(n.func, ast.Name) else type(n.func).__name__
                    raise Unsupported('unapproved_call:' + name)
                if n.func.id == 'isinstance':
                    if len(n.args) != 2 or not self._supported_type_spec(n.args[1]):
                        raise Unsupported('isinstance_requires_builtin_type_literal')
            if isinstance(n, ast.comprehension):
                if not isinstance(n.target, ast.Name) or n.is_async:
                    raise Unsupported('unsupported_comprehension')
            if isinstance(n, ast.Dict) and any(k is None for k in n.keys):
                raise Unsupported('dict_unpack')
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == 'old':
                if not isinstance(n.slice, ast.Constant) or n.slice.value not in parameters:
                    raise Unsupported('old_requires_literal_parameter_name')
                self.old_names.add(n.slice.value)
            if isinstance(n, ast.Compare):
                operands = [n.left] + n.comparators
                for op, a, b in zip(n.ops, operands, operands[1:]):
                    if isinstance(op, (ast.Is, ast.IsNot)) and not any(
                        isinstance(x, ast.Constant) and x.value is None for x in (a, b)
                    ):
                        raise Unsupported('identity_only_supported_for_None')
        # Standalone old is disallowed: explicit observations avoid copying whole object graphs.
        parents = {id(child): parent for parent in nodes for child in ast.iter_child_nodes(parent)}
        for n in nodes:
            if isinstance(n, ast.Name) and n.id == 'old':
                p = parents.get(id(n))
                if not isinstance(p, ast.Subscript) or p.value is not n:
                    raise Unsupported('use_old_literal_parameter')

        type_nodes = {
            id(child)
            for call in nodes
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == 'isinstance' and len(call.args) == 2
            for child in ast.walk(call.args[1])
            if isinstance(child, ast.Name) and child.id in TYPE_NAMES
        }
        if any(isinstance(n, ast.Name) and n.id in TYPE_NAMES and id(n) not in type_nodes
               for n in nodes):
            raise Unsupported('builtin_type_only_allowed_in_isinstance')

    @staticmethod
    def _supported_type_spec(node):
        if isinstance(node, ast.Name):
            return node.id in TYPE_NAMES
        return (isinstance(node, ast.Tuple) and bool(node.elts)
                and all(isinstance(item, ast.Name) and item.id in TYPE_NAMES for item in node.elts))

    def evaluate(self, env):
        value = Evaluator(env).visit(self.tree)
        if type(value) is not bool:
            raise Unsupported('predicate_must_return_bool')
        return value


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.steps = MAX_STEPS

    def visit(self, n):
        self.steps -= 1
        if self.steps < 0:
            raise Unsupported('evaluation_step_limit')
        if isinstance(n, ast.Constant):
            return data(n.value)
        if isinstance(n, ast.Name):
            if n.id in TYPE_NAMES:
                return TYPE_NAMES[n.id]
            return self.env[n.id]
        if isinstance(n, ast.List):
            return data([self.visit(x) for x in n.elts])
        if isinstance(n, ast.Tuple):
            return data(tuple(self.visit(x) for x in n.elts))
        if isinstance(n, ast.Dict):
            return data({self.visit(k): self.visit(v) for k, v in zip(n.keys, n.values)})
        if isinstance(n, ast.BoolOp):
            result = None
            for x in n.values:
                result = self.visit(x)
                if isinstance(n.op, ast.And) and not result:
                    return result
                if isinstance(n.op, ast.Or) and result:
                    return result
            return result
        if isinstance(n, ast.UnaryOp):
            val = self.visit(n.operand)
            if isinstance(n.op, ast.Not):
                return not val
            if type(val) not in (int, float):
                raise Unsupported('numeric_unary_required')
            return data(-val if isinstance(n.op, ast.USub) else +val)
        if isinstance(n, ast.BinOp):
            a, b = self.visit(n.left), self.visit(n.right)
            if type(a) not in (int, float) or type(b) not in (int, float):
                raise Unsupported('arithmetic_requires_numbers')
            ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                   ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
            return data(ops[type(n.op)](a, b))
        if isinstance(n, ast.Compare):
            ops = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                   ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                   ast.Is: operator.is_, ast.IsNot: operator.is_not,
                   ast.In: lambda a,b: a in b, ast.NotIn: lambda a,b: a not in b}
            left = self.visit(n.left)
            for op, right in zip(n.ops, n.comparators):
                right = self.visit(right)
                if not ops[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return self.visit(n.body if self.visit(n.test) else n.orelse)
        if isinstance(n, ast.Slice):
            return slice(*(self.visit(x) if x is not None else None
                           for x in (n.lower, n.upper, n.step)))
        if isinstance(n, ast.Subscript):
            return self.visit(n.value)[self.visit(n.slice)]
        if isinstance(n, (ast.GeneratorExp, ast.ListComp)):
            # Generators are lazy: all()/any() retain Python short-circuit behavior.
            captured = dict(self.env)
            count = [0]
            def in_env(node, env):
                saved = self.env
                try:
                    self.env = env
                    return self.visit(node)
                finally:
                    self.env = saved
            def expand(i, env):
                if i == len(n.generators):
                    count[0] += 1
                    if count[0] > MAX_ITEMS:
                        raise Unsupported('comprehension_size_limit')
                    yield in_env(n.elt, env)
                    return
                gen = n.generators[i]
                sequence = in_env(gen.iter, env)
                if type(sequence) not in (list, tuple, dict, str, range):
                    raise Unsupported('unsupported_iterator')
                for value in sequence:
                    self.steps -= 1
                    if self.steps < 0:
                        raise Unsupported('evaluation_step_limit')
                    local = dict(env, **{gen.target.id: value})
                    if all(in_env(f, local) for f in gen.ifs):
                        yield from expand(i + 1, local)
            iterator = expand(0, captured)
            if isinstance(n, ast.GeneratorExp):
                return iterator
            return data(list(iterator))
        if isinstance(n, ast.Call):
            if n.func.id == 'isinstance':
                value = self.visit(n.args[0])
                spec = n.args[1]
                names = [spec.id] if isinstance(spec, ast.Name) else [item.id for item in spec.elts]
                # Exact built-in data checks avoid bool/int subclass surprises and
                # never consult target-defined classes or hooks.
                return type(value) in tuple(TYPE_NAMES[name] for name in names)
            args = [self.visit(x) for x in n.args]
            name = n.func.id
            if name == 'range':
                if not 1 <= len(args) <= 3 or any(type(v) is not int for v in args):
                    raise Unsupported('range_arguments')
                value = range(*args)
                if len(value) > MAX_ITEMS:
                    raise Unsupported('range_size_limit')
                return value
            funcs = {'len': len, 'abs': abs, 'min': min, 'max': max, 'sum': sum,
                     'all': all, 'any': any, 'sorted': sorted}
            return data(funcs[name](*args))
        raise Unsupported('unsupported_node')
