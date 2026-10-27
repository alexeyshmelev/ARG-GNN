"""Authenticate the local v5 adaptation and exercise it using tiny owned shells."""
import argparse
import ast
import json
from pathlib import Path
from unittest.mock import patch
from . import contracts_v2 as contracts, aggregate_guard_v2 as guard, native_supervisor_v2 as local
from research.efficiency_20261027.integration_candidate.european_short_development_val import native_supervisor_v5 as reviewed
from research.efficiency_20261027.execution import test_native_terminal_v5 as fixtures

HERE=Path(__file__).parent


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--fixture-root',default=str(HERE/'native_binding_fixtures_v2'))
    parser.add_argument('--receipt',default=str(HERE/'NATIVE_BINDING_CPU_v2.json'))
    args=parser.parse_args();base=Path(args.fixture_root);destination=Path(args.receipt)
    assert base.is_relative_to(HERE) and destination.is_relative_to(HERE)
    original=Path(reviewed.__file__).read_text()
    expected=original.replace('from .aggregate_guard_v3 import AggregateOwner, GuardFailure, METADATA',
        'from .aggregate_guard_v2 import AggregateOwner, GuardFailure, METADATA, RESERVATION, OPERATING')
    expected=expected.replace('from .short_plan_v1 import validate_native_config','from .contracts_v2 import validate_native_config')
    expected=expected.replace('storage_reservation_bytes=16*2**30, operating_threshold_bytes=12*2**30',
        'storage_reservation_bytes=RESERVATION, operating_threshold_bytes=OPERATING')
    expected=expected.replace("schema='short_native_owned_launch_v2'","schema='full_validation_native_owned_launch_v1'")
    assert Path(local.__file__).read_text()==expected
    assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(Path(local.__file__).read_text()))
    config,tools,chain=contracts.native_bindings()
    assert len(tools)==7 and local.validate_native_config(config)==config
    retained_ref,retained_bytes=contracts.retained_attempts()
    assert retained_bytes<=contracts.RETAINED_SHORT_RESERVATION
    base.mkdir();checks=[]
    with patch.object(fixtures,'AggregateOwner',guard.AggregateOwner),patch.object(fixtures,'GuardFailure',guard.GuardFailure):
        checks.append(fixtures.case(base,'immediate_exit',local,'printf immediate\\n\n','complete'))
        checks.append(fixtures.case(base,'stale_snapshot_rechecked',local,
            'printf ready > ready\nwhile [ ! -f release ]; do sleep 0.3; done\nprintf finished\\n\n',
            'complete',stale=True))
        checks.append(fixtures.case(base,'late_nonzero_adopted_exit',local,'(sleep 0.2; exit 9) &\nexit 0\n',
            'failed',defer_zombie_reap=True))
    for row in checks:
        terminal=contracts.read(row['terminal_ref'])
        assert terminal['storage_reservation_bytes']==contracts.RESERVATION
        assert terminal['operating_threshold_bytes']==contracts.OPERATING
    report=dict(schema='European_full_VAL_native_binding_CPU_v1',passed=True,
        exact_v5_lifecycle_source_with_only_local_imports_and_receipt_metadata_changes=True,
        current_installed_native_tool_hashes_verified=7,native_chain_review_ref=chain,
        retained_attempts_ref=retained_ref,retained_short_charged_bytes=retained_bytes,
        source_refs=[contracts.ref(Path(module.__file__)) for module in (local,reviewed,guard,fixtures,contracts)]
            +[contracts.ref(Path(__file__))],
        actual_tiny_shell_checks_passed=len(checks),checks=checks,native_tool_executions=0,
        simulation_model_GPU_executions=0,real_scientific_payloads_opened=False)
    with destination.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(passed=True,receipt=contracts.ref(destination),checks=len(checks))))


if __name__=='__main__':main()
