"""Set required visible process name, then execute the untouched author trainer."""
from pathlib import Path
import os
import runpy
import sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise RuntimeError('Only physical GPU0 may be visible')
    from setproctitle import setproctitle
    setproctitle('AI Auto Research Pipeline Test')
    script=Path(sys.argv[1]).resolve()
    if ROOT not in script.parents or script.name!='train_joint_model.py':
        raise RuntimeError('Owned untouched author train script required')
    sys.argv=sys.argv[1:]
    sys.path.insert(0,str(script.parent))
    runpy.run_path(str(script),run_name='__main__')
