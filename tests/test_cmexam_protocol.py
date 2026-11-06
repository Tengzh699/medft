import json
import unittest

from medft.config import project_path, file_hash
from medft.data import read_jsonl
from scripts.prepare_cmexam import group_key, options_of, prompt_of, known_source_defect, NearStemIndex
from scripts.evaluate_cmexam import wilson, can_permute, permute_options
from scripts.paired_statistics import exact_paired_p


class CMExamProtocolTests(unittest.TestCase):
    def test_option_parser_rejects_ambiguous_labels(self):
        self.assertEqual(options_of('A 甲\nB 乙\nC 丙\nD 丁\nE 戊')['D'], '丁')
        for invalid in ['A 甲\nA 乙\nC 丙\nD 丁\nE 戊', 'A 甲\nB 甲\nC 丙\nD 丁\nE 戊', 'A 甲\nB 乙']:
            with self.assertRaises(ValueError):
                options_of(invalid)

    def test_prompt_does_not_include_gold_or_explanation(self):
        row = {'question': '题干', 'options': dict(zip('ABCDE', ['甲','乙','丙','丁','戊'])),
               'answer': 'SECRET_GOLD', 'explanation': 'SECRET_EXPLANATION'}
        prompt = prompt_of(row)
        self.assertNotIn('SECRET', prompt)
        self.assertIn('D 丁', prompt)

    def test_numeric_variants_share_stem(self):
        self.assertEqual(group_key('男性，34岁，值5.3'), group_key('男性，59岁，值6.1'))
        self.assertNotEqual(group_key('头痛两天'), group_key('腹痛两天'))

    def test_known_bad_source_examples_are_excluded(self):
        row = {'question':'选哪种润滑剂？', 'options':{'A':'接触角=36℃', 'B':'其他'}}
        self.assertEqual(known_source_defect(row), 'angle_mislabelled_as_temperature')
        row = {'question':'某题', 'options':{'A':'脓毒性休克','B':'感染性休克'}}
        self.assertEqual(known_source_defect(row), 'synonymous_options')

    def test_near_stem_guard_handles_small_changes_in_long_case(self):
        stem = group_key('患者反复右上腹隐痛已经多年近来出现发热恶心呕吐且腹部压痛明显最合适的检查是什么')
        index = NearStemIndex([stem])
        self.assertTrue(index.matches(stem))
        self.assertTrue(index.matches(stem+'呢'))
        self.assertFalse(index.matches(group_key('牙齿敏感应当提供哪些相关信息')))

    def test_interval_has_correct_extremes(self):
        low, high = wilson(0, 100)
        self.assertAlmostEqual(low, 0)
        self.assertLess(high, .04)
        low, high = wilson(100, 100)
        self.assertGreater(low, .96)
        self.assertAlmostEqual(high, 1)

    def test_exact_paired_test_against_hand_computable_counts(self):
        self.assertEqual(exact_paired_p(0,0), 1.0)
        self.assertEqual(exact_paired_p(10,10), 1.0)
        self.assertEqual(exact_paired_p(0,5), 1/16)
        self.assertEqual(exact_paired_p(15,5), 43400/1048576)
        self.assertEqual(exact_paired_p(5,15), exact_paired_p(15,5))
        self.assertEqual(exact_paired_p(0,1000), 2**-999)

    def test_permutation_preserves_answer_text_and_rejects_references(self):
        row = {'id':'example', 'question':'题干', 'options':dict(zip('ABCDE',['甲','乙','丙','丁','戊'])), 'answer':'D'}
        shuffled = permute_options(row, 'fixed')
        self.assertEqual(shuffled['options'][shuffled['answer']], row['options']['D'])
        self.assertEqual(row['answer'], 'D')
        self.assertEqual(shuffled, permute_options(row, 'fixed'))
        row['options']['E']='以上都是'
        self.assertFalse(can_permute(row))
        with self.assertRaises(ValueError):
            permute_options(row, 'fixed')

    def test_frozen_artifacts_and_split_isolation(self):
        prep = project_path('data/generated/v4-cmexam/preparation.json')
        if not prep.exists():
            self.skipTest('Prepare pinned public data first')
        protocol = json.loads(prep.read_text(encoding='utf-8'))
        train = read_jsonl(project_path('data/processed/v4-cmexam/train.jsonl'))
        validation = read_jsonl(project_path('data/processed/v4-cmexam/validation.jsonl'))
        public = [r for r in train if r['task']=='medical_exam']
        self.assertEqual(len(public), 2000)
        self.assertEqual({r['source_file'] for r in public}, {'train.csv'})
        self.assertFalse({r['case_id'] for r in train} & {r['case_id'] for r in validation})
        stems = {group_key(r['messages'][1]['content'].split('\n')[1][3:]) for r in public}
        for split in ['dev', 'test']:
            p = project_path(f'assets/cmexam_{split}_frozen.jsonl')
            self.assertEqual(file_hash(p), protocol[split+'_sha256'])
            cases = read_jsonl(p)
            self.assertEqual(len(cases), protocol[split+'_count'])
            self.assertFalse(stems & {r['stem_group'] for r in cases})


if __name__ == '__main__':
    unittest.main()
