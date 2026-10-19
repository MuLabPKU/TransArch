"""Protect report integrity: no complete claims for partial or mislabeled outputs."""
import copy
import unittest

from paths import ROOT
from report import expected_ids, read_rows, validate


class ReportTests(unittest.TestCase):
    def test_published_code_evidence_is_complete(self):
        for name in ['v4', 'v41']:
            rows = read_rows(ROOT / f'reports/{name}_all_code.jsonl')
            self.assertTrue(validate(rows, 'all_code', name))
            self.assertFalse(validate(rows[:-1], 'all_code', name))

    def test_wrong_correctness_flag_rejected(self):
        rows = copy.deepcopy(read_rows(ROOT / 'reports/v4_all_code.jsonl'))
        rows[0]['correct'] = not rows[0]['correct']
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            validate(rows, 'all_code', 'v4')

    def test_niah_requires_all_bindings_and_residues(self):
        rows = []
        for b in range(256):
            for r in range(8):
                rows.append(dict(id=f'niah_b00_{b:03d}_r{r}', block=0, binding=b,
                                 residue=r, source_position=64000+r, gold='x', completion='x.', correct=True))
        self.assertEqual(len(expected_ids('niah', 'v41')), 20480)
        self.assertTrue(validate(rows, 'niah', 'v4'))
        self.assertFalse(validate(rows[:-1], 'niah', 'v4'))
        rows[0]['completion'] = ': x'
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            validate(rows, 'niah', 'v4')


if __name__ == '__main__':
    unittest.main()
