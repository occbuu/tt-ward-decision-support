"""Offline checks. Synthetic predictions are temporary and are not research results."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import benchmark as b

class BenchmarkTests(unittest.TestCase):
    def test_http_payload_and_response(self):
        captured = {}
        response = {'model':'offline-fixture','choices':[{'message':{'content':'INFRA'},'finish_reason':'stop'}]}
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps(response).encode('utf-8')
        def open_request(req, timeout):
            captured.update(json.loads(req.data))
            return Response()
        config = {'endpoint':'http://localhost:8000/v1/chat/completions','model':'offline-fixture',
                  'request_options':{'temperature':0,'max_tokens':32}}
        with patch.object(b.urllib.request,'urlopen',side_effect=open_request):
            raw, body, attempts = b.request(config,[{'role':'user','content':'test response'}])
        self.assertEqual(raw,'INFRA')
        self.assertEqual(body,response)
        self.assertEqual(captured['model'],'offline-fixture')
        self.assertEqual(captured['temperature'],0)
        self.assertFalse(captured['stream'])
        self.assertEqual(len(attempts),1)

    def test_known_kappa_and_invalid(self):
        scores, _ = b.metrics(['A','A','B','B'], ['A','B','B','B'], ['A','B'])
        self.assertAlmostEqual(scores['accuracy'], .75)
        self.assertAlmostEqual(scores['cohen_kappa'], .5)
        self.assertAlmostEqual(scores['macro_f1'], 11/15)
        scores, _ = b.metrics(['A'], ['INVALID'], ['A','B'])
        self.assertEqual(scores['accuracy'], 0)
        self.assertEqual(scores['invalid_or_failed'], 1)

    def test_gold_never_in_prompt(self):
        items, frame = b.load_inputs()
        messages = b.prompt(items[0], frame, 'zero_shot', [])
        self.assertNotIn('gold_code', json.dumps(messages))
        self.assertEqual(len(items),130)
        self.assertEqual(len(frame),8)

    def test_overlap_rejected(self):
        items, frame = b.load_inputs()
        config = {'endpoint':'http://localhost:8000/v1/chat/completions','model':'offline-test',
                  'strategies':['few_shot'],'examples_file':'unused.csv','examples_provenance':'test'}
        example = dict(items[0],code='OTHER')
        with patch.object(b,'read_csv',return_value=[example]):
            with self.assertRaisesRegex(ValueError,'overlaps'):
                b.validate_config(config,items,frame)

    def test_full_roundtrip_resume_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory(dir=b.HERE) as temporary:
            root = Path(temporary)
            self.assertTrue(root.resolve().is_relative_to(b.HERE.resolve()))
            config = {'endpoint':'http://localhost:8000/v1/chat/completions', 'model':'OFFLINE_SYNTHETIC_TEST',
                      'strategies':['zero_shot'],'repeats':1,'delay_seconds':0,
                      'request_options':{'temperature':0,'max_tokens':32}}
            b.write_json(root/'config.json',config)
            args = SimpleNamespace(config=str(root/'config.json'),out=str(root/'output'))
            with patch.object(b,'request',return_value=('OTHER',{'test_fixture':True},[])) as request:
                with contextlib.redirect_stdout(io.StringIO()):
                    b.run(args)
                    b.run(args)
                self.assertEqual(request.call_count,130)
            evaluate_args = SimpleNamespace(out=args.out,bootstrap=20,seed=7)
            with contextlib.redirect_stdout(io.StringIO()):
                b.evaluate(evaluate_args)
            report = json.loads((root/'output/evaluation.json').read_text())
            gold = b.read_csv(b.HERE/'data/gold_labels.csv')
            expected = sum(r['gold_code']=='OTHER' for r in gold)/130
            self.assertAlmostEqual(report['runs'][0]['accuracy'],expected)
            records_path = root/'output/zero_shot_run1.jsonl'
            lines = records_path.read_text(encoding='utf-8').splitlines()
            records_path.write_text('\n'.join(lines[:-1])+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'incomplete'):
                b.evaluate(evaluate_args)

if __name__ == '__main__':
    unittest.main()
