"""Run the unchanged ARGMix stage CLI with genuine unlimited native waits."""
from arggnn.benchmarking.paper_argmix import paper_argmix_cli as cli
from arggnn.benchmarking.paper_argmix import paper_argmix_relate as native
from research.joint_evaluation_20261029.queue.argmix_native_wrapper import UnlimitedNativeSubprocess


def main():
    previous = native.subprocess
    native.subprocess = UnlimitedNativeSubprocess()
    try:
        cli.main()
    finally:
        native.subprocess = previous


if __name__ == '__main__':
    main()
