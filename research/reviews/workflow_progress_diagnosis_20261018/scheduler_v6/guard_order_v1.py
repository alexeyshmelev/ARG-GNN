"""V6-only scheduler clone: defer the unchanged fresh RAM check until eligible."""
from __future__ import annotations
import ast
import copy
import inspect
import textwrap
import types


def _need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def compile_scheduler_run(run_function):
    """Return a private run callable; never modify the original module or code.

    ROOT integrates this only into the new v6 core's _BASE_RUN binding. Full
    aggregate reservations stay first; the exact fresh host-RAM check follows
    GPU availability/lease and storage guards, immediately before materialize.
    """
    source = textwrap.dedent(inspect.getsource(run_function))
    tree = ast.parse(source)
    node = tree.body[0]
    _need(isinstance(node, ast.FunctionDef) and node.name == "run",
          "One original run function required")
    expected_test = ast.parse(
        "available_ram_bytes()-rss_reserved<task['max_rss_bytes']+plan['host_ram_margin_bytes']",
        mode="eval").body
    expected_body = ast.parse("entry['waiting_reason']='Host available RAM'\ncontinue").body
    matches = [(loop, at, item) for loop in ast.walk(node) if isinstance(loop, ast.For)
               for at, item in enumerate(loop.body)
               if isinstance(item, ast.If) and _same(item.test, expected_test)]
    _need(len(matches) == 1, "Exactly one unchanged original host RAM guard required")
    loop, ram_at, ram = matches[0]
    _need(len(ram.body) == len(expected_body)
          and all(_same(a, b) for a, b in zip(ram.body, expected_body)) and not ram.orelse,
          "Original host RAM rejection behavior must remain unchanged")
    storage_test = ast.parse(
        "projected>=min(plan['storage_ceiling_bytes'],980000000000)", mode="eval").body
    storage = [item for item in loop.body if isinstance(item, ast.If)
               and _same(item.test, storage_test)]
    lease = [item for item in loop.body if isinstance(item, ast.If)
             and 'with path.open' in ast.unparse(item)
             and 'GPU0 lease occupied' in ast.unparse(item)]
    _need(len(storage) == len(lease) == 1
          and ram_at < loop.body.index(lease[0]) < loop.body.index(storage[0]),
          "Original GPU lease and storage guard order required")
    storage_at = loop.body.index(storage[0])
    _need(storage_at + 1 < len(loop.body)
          and isinstance(loop.body[storage_at + 1], ast.Try)
          and "materialize(ref, plan, task, state)" in ast.unparse(loop.body[storage_at + 1]),
          "Original materialize must immediately follow storage admission")
    loop.body.pop(ram_at)
    loop.body.insert(loop.body.index(storage[0]) + 1, ram)
    namespace = dict(run_function.__globals__)
    exec(compile(ast.fix_missing_locations(tree),
                 run_function.__code__.co_filename + ":v6_guard_order", "exec"), namespace)
    compiled = namespace[node.name]
    result = types.FunctionType(compiled.__code__, run_function.__globals__,
                                run_function.__name__, run_function.__defaults__, run_function.__closure__)
    result.__kwdefaults__ = copy.copy(run_function.__kwdefaults__)
    result.__annotations__ = dict(run_function.__annotations__)
    result.__module__ = run_function.__module__
    return result
