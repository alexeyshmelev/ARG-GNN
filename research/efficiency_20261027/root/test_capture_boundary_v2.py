"""CPU-only negative tests. No process signals, archive moves or live writes."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
from research.efficiency_20261027.root import capture_boundary_v2 as m


def fixtures():
    task={'call':{'module':m.MODULE},'max_rss_bytes':592*2**30,
          'source_refs':[{'path':'source.py','sha256':'fixed'}]}
    row={'status':'running','pid':m.PID,'process_start_ticks':m.TICKS,
         'attempt':m.ATTEMPT,'physical_gpu':3}
    state={'tasks':{m.TASK:row},'plan_revision':1,'plan_ref':{'path':'plan','sha256':'p'}}
    return {},state,task


class IdentityTests(unittest.TestCase):
    def call(self,parts):
        with patch.object(m.prior,'current',return_value=parts),patch.object(m.r,'ref',return_value={'path':'source.py','sha256':'fixed'}):
            return m.current('running')

    def test_original_identity_accepted(self):
        self.assertEqual(self.call(fixtures())[-1]['pid'],m.PID)

    def test_unrelated_or_reused_identity_rejected(self):
        for key,value in [('pid',m.PID+1),('process_start_ticks','1'),('attempt',6),('status','succeeded'),('physical_gpu',4)]:
            with self.subTest(key=key):
                parts=copy.deepcopy(fixtures())
                parts[1]['tasks'][m.TASK][key]=value
                with self.assertRaises(ValueError): self.call(parts)

    def test_changed_module_and_resource_rejected(self):
        for field in ['module','cap']:
            parts=copy.deepcopy(fixtures())
            if field=='module': parts[2]['call']['module']='unrelated.module'
            else: parts[2]['max_rss_bytes']+=1
            with self.assertRaises(ValueError): self.call(parts)

    def test_changed_source_rejected(self):
        with patch.object(m.prior,'current',return_value=fixtures()),patch.object(m.r,'ref',return_value={'path':'source.py','sha256':'changed'}):
            with self.assertRaises(ValueError): m.current('running')


class DispatchTests(unittest.TestCase):
    def test_record_before_single_signal_and_close(self):
        plan,state,task=fixtures();row=state['tasks'][m.TASK]
        events=[]
        with patch.object(m,'authenticate_release',return_value={'release':'fixed'}),patch.object(m,'review',return_value={'review':'fixed'}),patch.object(m,'current',return_value=(plan,state,task,row)),patch.object(m,'validate_live'),patch.object(m.Path,'exists',return_value=False),patch.object(m.processes,'_pidfd_open',return_value=77),patch.object(m.processes,'_pidfd_send',side_effect=lambda fd,sig:events.append(('signal',fd,sig))),patch.object(m.os,'close',side_effect=lambda fd:events.append(('close',fd))),patch.object(m,'save',side_effect=lambda path,value:events.append(('save',path.name)) or {'path':path.name}),patch('builtins.print'):
            m.stop(SimpleNamespace(release='release',review='review'))
        self.assertEqual([e[:2] for e in events],[('save','before_stop.json'),('save','stop_intent.json'),('signal',77),('close',77),('save','stop_sent.json')])

    def test_prior_intent_prevents_retry(self):
        plan,state,task=fixtures();row=state['tasks'][m.TASK]
        with patch.object(m,'authenticate_release'),patch.object(m,'review'),patch.object(m,'current',return_value=(plan,state,task,row)),patch.object(m,'validate_live'),patch.object(m.Path,'exists',return_value=True),patch.object(m.processes,'_pidfd_open') as opened,patch.object(m.processes,'_pidfd_send') as sent:
            with self.assertRaises(ValueError): m.stop(SimpleNamespace(release='r',review='v'))
            opened.assert_not_called();sent.assert_not_called()

    def test_revalidation_failure_never_signals_and_closes(self):
        plan,state,task=fixtures();row=state['tasks'][m.TASK]
        with patch.object(m,'authenticate_release'),patch.object(m,'review'),patch.object(m,'current',side_effect=[(plan,state,task,row),ValueError('identity changed')]),patch.object(m,'validate_live'),patch.object(m.Path,'exists',return_value=False),patch.object(m.processes,'_pidfd_open',return_value=77),patch.object(m.processes,'_pidfd_send') as sent,patch.object(m.os,'close') as closed,patch.object(m,'save') as saved:
            with self.assertRaises(ValueError): m.stop(SimpleNamespace(release='r',review='v'))
            sent.assert_not_called();saved.assert_not_called();closed.assert_called_once_with(77)


class DurableBoundaryTests(unittest.TestCase):
    def test_exact_latest_commit_and_negative_boundaries(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=m.r.ROOT/'.runtime/tmp',prefix='boundary-unit-') as directory:
            native=Path(directory)/'native'
            leaf=native/'exact_preprocessing_from_update241_optimization_attempt5'
            (leaf/'checkpoints').mkdir(parents=True)
            checkpoint=leaf/'checkpoints/latest.pt'
            checkpoint.write_bytes(b'fake checkpoint bytes, never loaded')
            metadata=leaf/'latest_checkpoint.json'
            task={'call':{'spec':{'resume_steps':241,'attempt_id':'optimization_attempt5'}}}
            value={'complete_parent_updates':267,'checkpoint_ref':m.r.ref(checkpoint)}
            metadata.write_text(json.dumps(value))
            with patch.object(m.prior,'NATIVE',native):
                self.assertEqual(m.validate_latest(task,checkpoint,267),m.r.ref(metadata))
                with self.assertRaises(ValueError): m.validate_latest(task,checkpoint,268)
                with self.assertRaises(ValueError): m.validate_latest(task,leaf/'wrong.pt',267)
                checkpoint.write_bytes(b'new checkpoint before metadata replacement')
                with self.assertRaises(ValueError): m.validate_latest(task,checkpoint,267)


if __name__=='__main__': unittest.main(verbosity=2)
