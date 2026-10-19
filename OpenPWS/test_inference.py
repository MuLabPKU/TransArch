"""Protect experiment identity and continuation semantics during backend changes."""
import json
from pathlib import Path
import tempfile
import unittest

from inference import pending_rows


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output=Path(self.tmp.name)/'results.jsonl'
        self.rows=[{'id':'a'},{'id':'b'},{'id':'c'}]
        self.identity={'backend':'vllm','model':'/local/model',
                       'prompt_sha256':'inputs','engine_options':{'seed':42,'tensor_parallel_size':8}}

    def saved(self,rows):
        self.output.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.output.with_suffix('.provenance.json').write_text(json.dumps(self.identity))

    def test_resume_limit_applies_to_original_input_order(self):
        self.saved(self.rows[:1])
        self.assertEqual(pending_rows(self.rows,self.output,self.identity,2),self.rows[1:2])
        self.assertEqual(pending_rows(self.rows,self.output,self.identity),self.rows[1:])

    def test_changed_model_input_and_engine_rejected(self):
        self.saved(self.rows[:1])
        for key,value in [('model','/another/model'),('prompt_sha256','other'),
                          ('engine_options',{'seed':42,'tensor_parallel_size':4})]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Resume'):
                pending_rows(self.rows,self.output,{**self.identity,key:value})

    def test_duplicate_and_foreign_results_rejected(self):
        for rows in ([{'id':'a'},{'id':'a'}],[{'id':'unknown'}]):
            self.saved(rows)
            with self.assertRaisesRegex(ValueError,'Duplicate or unknown'):
                pending_rows(self.rows,self.output,self.identity)

    def test_duplicate_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate input'):
            pending_rows([{'id':'a'},{'id':'a'}],self.output,self.identity)


if __name__=='__main__':
    unittest.main()
