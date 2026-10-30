"""Initialize the admitted CUDA runtime before calling the unchanged native CLI."""
import sys
import torch
import setproctitle
from arggnn.benchmarking.paper_argmix.paper_argmix_train import validate_gpu_binding
from arggnn.benchmarking.paper_argmix import paper_argmix_cli


def main():
    if len(sys.argv) < 2 or sys.argv[1] != 'train':
        raise ValueError('This one-purpose operational wrapper accepts native train only')
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    validate_gpu_binding()
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly the admitted physical GPU must be visible')
    torch.cuda.init()
    paper_argmix_cli.main()


if __name__ == '__main__':
    main()
