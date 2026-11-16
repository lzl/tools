import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

atom=load('forward_atom','atoms/forward_telegram_channel.py')
workflow=load('forward_workflow','workflows/forward_telegram_channel.py')

class ForwardTests(unittest.TestCase):
    def test_fixed_defaults(self):
        self.assertEqual((atom.REQUEST_SIZE,atom.INTERVAL),(30,108))

    def test_prefix_partial_and_mismatch(self):
        self.assertEqual(atom.check_prefix(['a'],['a','b']),1)
        self.assertEqual(atom.check_prefix([],['a']),0)
        with self.assertRaises(RuntimeError): atom.check_prefix(['b'],['a','b'])
        with self.assertRaises(RuntimeError): atom.check_prefix(['a','b'],['a'])

    def test_fingerprint_ignores_forward_dates_and_entities(self):
        a=SimpleNamespace(media=None,raw_text='hello')
        b=SimpleNamespace(media=None,raw_text='hello',fwd_from='different',entities=['auto'])
        self.assertEqual(atom.fingerprint(a),atom.fingerprint(b))
        b.raw_text='other'
        self.assertNotEqual(atom.fingerprint(a),atom.fingerprint(b))

    def test_atomic_state(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'progress.json'
            atom.save(p,{'pending':{'random_ids':[123]}})
            self.assertEqual(json.loads(p.read_text())['pending']['random_ids'],[123])
            self.assertFalse(p.with_suffix('.tmp').exists())

    def test_workflow_sequence_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(source_id=-1001,target_id=-1002,artifacts_root=Path(directory),verify_only=False,verify_after=True,recover_stale=False,import_legacy=None)
            with patch.object(workflow,'run_atom',side_effect=[{'status':'completed'},{'status':'verified'}]) as call:
                result=workflow.execute(args)
            self.assertEqual(call.call_count,2)
            self.assertEqual(call.call_args_list[0].args[0][-1],'forward')
            self.assertEqual(call.call_args_list[1].args[0][-1],'verify')
            self.assertTrue((Path(result['artifacts_dir'])/'summary.json').exists())

    def test_failure_stops_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(source_id=-1001,target_id=-1002,artifacts_root=Path(directory),verify_only=False,recover_stale=False,import_legacy=None)
            with patch.object(workflow,'run_atom',side_effect=RuntimeError('atom failed')) as call:
                with self.assertRaises(RuntimeError): workflow.execute(args)
            self.assertEqual(call.call_count,1)

    def test_completed_pair_invokes_incremental_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            pair=Path(directory)/'-1001_to_-1002';pair.mkdir()
            (pair/'progress.json').write_text('{"status":"completed"}')
            args=SimpleNamespace(source_id=-1001,target_id=-1002,artifacts_root=Path(directory),verify_only=False,verify_after=False,recover_stale=False,import_legacy=None)
            with patch.object(workflow,'run_atom',return_value={'status':'completed'}) as call:
                result=workflow.execute(args)
            self.assertEqual(call.call_count,1)
            self.assertEqual(result['status'],'completed')
            self.assertTrue(result['verification_skipped'])

if __name__=='__main__':unittest.main()
