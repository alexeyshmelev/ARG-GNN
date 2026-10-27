"""Create readable narrow successors; preserve every sealed v1 file."""
from pathlib import Path
import ast
import difflib

HERE = Path(__file__).parent
NAMES = ('aggregate_guard', 'contracts', 'worker', 'guardian', 'native_supervisor', 'test_contracts', 'test_native_binding')

def main():
    diffs = []
    for name in NAMES:
        old = HERE / (name + '_v1.py')
        new = HERE / (name + '_v2.py')
        before = old.read_text()
        after = before
        for module in NAMES:
            after = after.replace(module + '_v1', module + '_v2')
        if name == 'aggregate_guard':
            original = '        try:\n            self.stop_monitor()\n        finally:\n'
            corrected = ('        try:\n            self.stop_monitor()\n'
                '            # A final in-flight scan may fail while stop_monitor joins.\n'
                '            # Completion must observe that failure before releasing ownership.\n'
                '            if self._error is not None:\n'
                "                raise GuardFailure('Aggregate monitor failed during close: ' + str(self._error))\n"
                '        finally:\n')
            assert after.count(original) == 1
            after = after.replace(original, corrected)
        if name == 'contracts':
            assert after.count("HERE/'source_pins_v1.json'") == 1
            after = after.replace("HERE/'source_pins_v1.json'", "HERE/'source_pins_v2.json'")
        if name == 'test_contracts':
            after = after.replace('CONTRACTS_CPU_v1.json', 'CONTRACTS_CPU_v2.json')
        if name == 'test_native_binding':
            after = after.replace('native_binding_fixtures_v1', 'native_binding_fixtures_v2')
            after = after.replace('NATIVE_BINDING_CPU_v1.json', 'NATIVE_BINDING_CPU_v2.json')
        ast.parse(after, filename=str(new))
        with new.open('x') as stream:
            stream.write(after)
        diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile=old.name, tofile=new.name))
    with (HERE/'MONITOR_CLOSE_PLAINTEXT_v2.diff').open('x') as stream:
        stream.writelines(diffs)
    print('Created seven exclusive successor sources and readable delta; no worker invoked.')

if __name__ == '__main__':
    main()
