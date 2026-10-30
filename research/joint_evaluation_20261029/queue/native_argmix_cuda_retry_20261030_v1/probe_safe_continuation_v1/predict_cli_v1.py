"""Name a native prediction process before calling its unchanged scientific CLI."""
import sys
import setproctitle


def main():
    if len(sys.argv) < 2 or sys.argv[1] != 'predict':
        raise ValueError('This operational wrapper accepts native predict only')
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    from arggnn.benchmarking.paper_argmix import paper_argmix_cli
    paper_argmix_cli.main()


if __name__ == '__main__':
    main()
