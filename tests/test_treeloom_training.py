"""One fake numerical-loop check:600 updates, exact order and target rejection."""
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest

from arggnn.benchmarking import treeloom_training as training


class FullFreshTrainingLoopTest(unittest.TestCase):
    def test_full600_update_order_and_misbound_target_rejection(self):
        parents = tuple("fresh_train_%03d" % i for i in range(20))
        reference = {"path": "mock/fresh_registry.json", "sha256": "a"*64}
        class Registry:
            def ids(self, split):
                return parents if split == "train" else ()
            def split_for(self, parent):
                return "train" if parent in parents else "test"
            def epoch_order(self, epoch):
                order=list(parents)
                random.Random(70002+epoch).shuffle(order)
                return order
        registry=Registry()
        registry.reference=reference
        def ref(parent,kind):
            return {"path":"mock/"+parent+"/"+kind,"sha256":"b"*64}
        inputs={p:{k:ref(p,k)for k in (*training.fresh.INPUT_REFS,"truth_ref")}for p in parents}
        bundles={}
        metadata={}
        seals={}
        for p in parents:
            bundle=SimpleNamespace(parent_id=p,split="train",sequence_length_bp=30000000,
                focal_identity_sha256="c"*64,validate=lambda: None,
                **{k:inputs[p][k]for k in training.fresh.INPUT_REFS})
            bundles[p]=bundle
            metadata[p]=dict(parent_id=p,declared_split="train",parent_registry_ref=reference,
                all_native_trees_nodes_sites_samples_retained=True,private_truth_opened=False)
            seals[p]=dict(focal_identity_sha256=bundle.focal_identity_sha256,
                          **{k:inputs[p][k]for k in training.fresh.INPUT_REFS})
        barrier=dict(parent_registry_ref=reference,all20_observations_sealed_before_truth=True,
                     private_truth_opened=False,parents=seals)
        optimizer=SimpleNamespace(count=0)
        visits=[]
        class Runtime:
            def train_step(self,model,bundle,target,opt,meta,*,resource_check):
                visits.append(bundle.parent_id)
                opt.count+=1
                return dict(parent_id=bundle.parent_id,one_complete_parent_optimizer_update=True,
                            native_focals=10000,loss=.5,gradient_norm=.1)
        def target(parent,bundle,meta):
            receipt=dict(parent_id=parent,split="train",parent_registry_ref=reference,
                         truth_ref=inputs[parent]["truth_ref"],
                         focal_registry_ref=bundle.focal_registry_ref,
                         focal_identity_sha256=bundle.focal_identity_sha256,
                         records=10000,labels_in_observation_bank=False)
            return SimpleNamespace(parent_id=parent,focal_registry_ref=bundle.focal_registry_ref,
                                   focal_identity_sha256=bundle.focal_identity_sha256,receipt=receipt)
        saved=[]
        def checkpoint(epoch,steps,model,opt):
            saved.append((epoch,steps))
            return {"path":"mock/epoch_%02d.pt"%epoch,"sha256":"d"*64}
        kwargs=dict(registry=registry,runtime=Runtime(),model=object(),optimizer=optimizer,
                    train_inputs=inputs,observation_barrier=barrier,
                    load_parent=lambda p:(bundles[p],metadata[p]),join_targets=target,
                    checkpoint=checkpoint,event=lambda *args:None,resource_check=lambda *args:None,
                    optimizer_counts=lambda opt:[opt.count]*49)
        result=training.train_loop(**kwargs)
        expected=[p for epoch in range(1,31)for p in registry.epoch_order(epoch)]
        self.assertEqual(visits,expected)
        self.assertEqual(len(visits),600)
        self.assertEqual(result["complete_parent_updates"],600)
        self.assertEqual(saved,[(epoch,epoch*20)for epoch in training.PASSES])
        self.assertEqual(result["selected_epoch"],30)
        self.assertFalse(result["VAL_truth_read"])
        self.assertFalse(result["TEST_truth_read"])
        for defect in ("parent","truth","focal"):
            visits.clear();optimizer.count=0;saved.clear()
            def misbound(parent,bundle,meta):
                value=target(parent,bundle,meta)
                if defect=="parent":value.parent_id="held_out_parent"
                elif defect=="truth":value.receipt["truth_ref"]=ref("another_parent","truth_ref")
                else:value.focal_identity_sha256="e"*64
                return value
            kwargs["join_targets"]=misbound
            with self.assertRaises(training.c.ContractError):
                training.train_loop(**kwargs)
            self.assertEqual(optimizer.count,0)
            self.assertEqual(visits,[])
            self.assertEqual(saved,[])

    def test_import_isolation_in_fresh_interpreter(self):
        code = (
            "import sys; sys.path.insert(0, %r); " % str(Path(training.__file__).resolve().parents[2])
            + "from arggnn.benchmarking import treeloom_training; "
            "assert not any(name in sys.modules for name in "
            "('torch', 'numpy', 'torch_geometric'))"
        )
        subprocess.run([sys.executable, "-B", "-s", "-c", code], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))


if __name__ == "__main__":
    unittest.main()
