"""Tiny real-file lossless receipt gates and original scheduler compilation."""
import copy
import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from arggnn.benchmarking import contracts as c, workflow as original
from arggnn.benchmarking import retention_scheduler_v9 as repair


def ref(name):
    return dict(path="mock/"+name+".json",sha256="a"*64)


class RetentionSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.directory=Path(self.temporary.name)
        self.owner_scope=self.directory/"owner"
        self.output=self.directory/"retention"
        (self.owner_scope/"native").mkdir(parents=True)
        (self.output/"native").mkdir(parents=True)
        self.owner_id="AP__ancestralpaths_european_train_000__ap_infer"
        self.owner=dict(id=self.owner_id,resource="cpu",output_directory=str(self.owner_scope))
        self.task=dict(id=self.owner_id+"__lossless_archive_v9",resource="cpu",
            dependencies=[self.owner_id],factory_module=repair.ARCHIVE_MODULE,
            factory_function="factory",output_directory=str(self.output),
            remeasure_succeeded_tasks=[self.owner_id])
        self.entry=dict(status="succeeded",actual_controller_wait=0,owned_process_group_absent=True,
                        job_ref=ref("job"),result_ref=ref("envelope"))
        self.plan=dict(tasks=[self.owner,self.task])
        self.state=dict(tasks={self.owner_id:dict(status="succeeded",external=False)})
        content=b"owned synthetic intermediate\n"*8192
        raw=self.owner_scope/"native/pre_size.anc"
        archive=self.output/"native/pre_size.anc.retained-v9.gz"
        compressed=gzip.compress(content,mtime=0)
        archive.write_bytes(compressed)
        before=len(content)+4096
        after=max(len(compressed),archive.stat().st_blocks*512)+4096
        self.receipt=dict(schema=repair.ARCHIVE_SCHEMA,inference_ref=ref("inference"),
            actual_wait_ref=ref("actual_wait"),owner_scope=str(self.owner_scope),
            owner_task_id=self.owner_id,archive_output_directory=str(self.output/"native"),
            archives_inside_current_task_output=True,
            compressed_content_preserves_every_original_byte=True,
            final_observed_tree_and_inputs_unchanged=True,
            raw_path_replacement_only_after_verified_durable_archive=True,
            entries=[dict(original_path=str(raw),original_sha256=hashlib.sha256(content).hexdigest(),
                original_bytes=len(content),archive_path=str(archive),
                archive_sha256=hashlib.sha256(compressed).hexdigest(),archive_bytes=len(compressed),
                original_conservative_charge_bytes=before,archive_conservative_charge_bytes=after,
                decompressed_hash_and_size_verified=True,state="archive_committed",
                credited_bytes=before-after)],observed_saved_bytes=before-after)
        self.spec=dict(schema=repair.ARCHIVE_SCHEMA,owner_task_id=self.owner_id,
            owner_scope=str(self.owner_scope),output_directory=str(self.output/"native"),
            inference_ref=ref("inference"),actual_wait_ref=ref("actual_wait"))
        self.job=dict(task=copy.deepcopy(self.task),task_fingerprint=c.fingerprint(self.task),
            call=dict(module=repair.ARCHIVE_MODULE,function="run",spec=self.spec))
        self.envelope=dict(job_ref=ref("job"),
            value=dict(path=str(self.output/"native/restore_manifest.json"),sha256="b"*64))

    def validate(self):
        def read(reference):
            if reference==ref("job"):return self.job
            if reference==ref("envelope"):return self.envelope
            raise AssertionError("Unexpected receipt read")
        with mock.patch.object(repair.c,"read_ref",side_effect=read):
            return repair.validate_retirement(self.task,self.entry,self.receipt,self.plan,self.state)

    def test_verified_lossless_receipt_allows_original_current_scope_remeasurement(self):
        self.assertIsNone(self.validate())
        self.assertNotIn("successful_new_scratch_only",self.receipt)
        self.assertEqual(self.state["tasks"][self.owner_id]["status"],"succeeded")
        self.assertNotIn("retained_bytes",self.state["tasks"][self.owner_id])

    def test_unverified_or_misbound_receipts_are_rejected(self):
        original_receipt=copy.deepcopy(self.receipt)
        for key in ("compressed_content_preserves_every_original_byte",
                    "final_observed_tree_and_inputs_unchanged",
                    "raw_path_replacement_only_after_verified_durable_archive"):
            self.receipt=copy.deepcopy(original_receipt)
            self.receipt[key]=False
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError,"lossless retention"):self.validate()
        self.receipt=copy.deepcopy(original_receipt)
        self.receipt["entries"][0]["decompressed_hash_and_size_verified"]=False
        with self.assertRaisesRegex(ValueError,"Verified size"):self.validate()
        self.receipt=copy.deepcopy(original_receipt)
        self.receipt["observed_saved_bytes"]+=1
        with self.assertRaisesRegex(ValueError,"total"):self.validate()
        self.receipt=copy.deepcopy(original_receipt)
        self.receipt["inference_ref"]=ref("foreign_inference")
        with self.assertRaisesRegex(ValueError,"lossless retention"):self.validate()

    def test_actual_worker_wait_and_local_owner_are_required(self):
        self.entry["owned_process_group_absent"]=False
        with self.assertRaisesRegex(ValueError,"actual successful wait"):self.validate()
        self.entry["owned_process_group_absent"]=True
        self.state["tasks"][self.owner_id]["external"]=True
        with self.assertRaisesRegex(ValueError,"succeeded local"):self.validate()

    def test_payload_must_remain_in_separately_charged_archive_output(self):
        path=Path(self.receipt["entries"][0]["archive_path"])
        outside=self.directory/"foreign.gz"
        path.rename(outside)
        self.receipt["entries"][0]["archive_path"]=str(outside)
        with self.assertRaisesRegex(ValueError,"charged new output"):self.validate()

    def test_original_scratch_only_retirement_rule_is_preserved(self):
        task=dict(factory_module="arggnn.benchmarking.retention")
        self.assertIsNone(repair.validate_retirement(task,{},dict(successful_new_scratch_only=True),{},{}))
        with self.assertRaisesRegex(ValueError,"Actual retirement completion"):
            repair.validate_retirement(task,{},dict(successful_new_scratch_only=False),{},{})

    def test_actual_frozen_scheduler_compiles_without_mutation_and_uses_fresh_globals(self):
        source=Path(original.__file__)
        before=hashlib.sha256(source.read_bytes()).hexdigest()
        compiled=repair.compile_scheduler_run(original.run)
        self.assertIn("_validate_retirement_v9",compiled.__wrapped__.__code__.co_names)
        plan=dict(records_directory=str(self.directory/"records"),tasks=[dict(id="tiny",resource="cpu")])
        reference=ref("plan")
        with mock.patch.object(original,"load_plan",return_value=plan) as loader:
            observed=compiled(reference,execute_jobs=False)
        loader.assert_called_once_with(reference,approved=False)
        self.assertEqual(observed["plan_ref"],reference)
        self.assertEqual(observed["tasks"],{"tiny":dict(status="pending",external=False)})
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)


if __name__=="__main__":
    unittest.main()
