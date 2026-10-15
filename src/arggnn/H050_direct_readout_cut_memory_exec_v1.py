"""Canonical H050 dispatch; frozen scientific producer, default inert."""

def main(argv=None):
    from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.producer_v1 import main as scientific_main
    return scientific_main(argv)

if __name__ == "__main__":
    raise SystemExit(main())
