"""Tighten the existing owner limits before this isolated worker creates any owner."""
from research.efficiency_20261027.integration_candidate.european_short_development_val import aggregate_guard_v3 as aggregate
from .contracts_v3 import need,RESERVATION,OPERATING


def configure_retained_allowance(retained_bytes):
    # Exact stage classes and all scientific code remain unchanged. Their global
    # byte limits are reduced once in this fresh worker, before any monitor.
    need(type(retained_bytes) is int and 0<=retained_bytes<OPERATING,
         'Nonnegative real predecessor charge within the single reservation')
    need(aggregate.RESERVATION==RESERVATION and aggregate.OPERATING==OPERATING,
         'Retained byte allowance must be configured once before owner creation')
    aggregate.RESERVATION=RESERVATION-retained_bytes
    aggregate.OPERATING=OPERATING-retained_bytes
    return dict(retained_bytes=retained_bytes,attempt_reservation_bytes=aggregate.RESERVATION,
                attempt_operating_stop_bytes=aggregate.OPERATING,
                combined_reservation_bytes=RESERVATION,combined_operating_stop_bytes=OPERATING)
