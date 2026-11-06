import unittest
from decimal import Decimal

from scripts.build_complete_data import authored_rows, numerical_rows, freeze_holdout, group_complete_answers
from medft.data import case_groups, normalize
from scripts.finalize_base_replay import screen


class CompleteDataTests(unittest.TestCase):
    def test_actual_deterioration_has_action_without_more_questions(self):
        rows = [r for r in authored_rows() if r.get('clinical_state') in {'direct', 'change'}]
        self.assertEqual(len(rows), 24)
        for row in rows:
            answer = row['messages'][-1]['content']
            self.assertIn('立即', answer, row['id'])
            self.assertTrue(any(x in answer for x in ['120', '急诊', '110']), row['id'])
            self.assertNotIn('有无', answer, row['id'])

    def test_actual_and_hypothetical_risk_share_case_group_but_not_assertion(self):
        rows = [r for r in authored_rows() if r.get('clinical_state')]
        for group in case_groups(rows):
            self.assertEqual({r['clinical_state'] for r in group},
                             {'direct', 'change', 'hypothetical', 'stable', 'correction'})
            hypothetical = next(r for r in group if r['clinical_state'] == 'hypothetical')
            answer = hypothetical['messages'][-1]['content']
            self.assertIn('将来的假设', answer)
            self.assertIn('若将来真的出现', answer)
            self.assertNotIn('现在已经出现', answer)

    def test_portion_arithmetic_independently_matches_given_inputs(self):
        rows = [r for r in numerical_rows() if 'calculation' in r]
        for row in rows:
            data = row['calculation']
            expected = Decimal(data['per_100']) * Decimal(data['portion']) / Decimal(100)
            self.assertEqual(Decimal(data['result']), expected)
            self.assertIn('不是全天总量', row['messages'][-1]['content'])
        example = next(r for r in rows if r['id'] == 'complete-portion-0-2-60')
        self.assertEqual(Decimal(example['calculation']['result']), Decimal('228'))
        for row in rows:
            data=row['calculation']
            if data['unit']=='克':
                self.assertLessEqual(Decimal(data['per_100']),100)

    def test_report_direction_and_missing_fields_do_not_contradict_inputs(self):
        for row in numerical_rows():
            data = row.get('arithmetic')
            if not data:
                continue
            answer = row['messages'][-1]['content']
            if data['state'] == 'missing_unit':
                self.assertIn('单位未提供', answer)
                self.assertNotIn('本次单位已给出', answer)
            elif data['state'] == 'missing_range':
                self.assertIn('没有该机构的参考范围', answer)
            else:
                current, low, high = map(Decimal, [data['current'], data['low'], data['high']])
                relation = '低于' if current < low else '高于' if current > high else '处于'
                self.assertIn(relation + '这份模拟报告的参考范围', answer)

    def test_frozen_questions_are_unique_and_excluded_from_curriculum(self):
        held = freeze_holdout()
        questions = [normalize(r['messages'][0]['content']) for r in held]
        self.assertEqual(len(questions), len(set(questions)))
        train = {normalize(m['content']) for row in authored_rows()+numerical_rows()
                 for m in row['messages'] if m['role'] == 'user'}
        self.assertFalse(set(questions) & train)
        self.assertEqual(freeze_holdout(), held)

    def test_replay_screen_preserves_inputs_and_rejects_truncation_and_reviewed_errors(self):
        import copy
        good = {'id':'candidate-1','teacher_finished':True,
                'messages':[{'role':'assistant','content':'可以先记录实际发生的情况，并结合日期整理完整资料。'*8}]}
        truncated = copy.deepcopy(good)
        truncated.update(id='candidate-2',teacher_finished=False)
        flagged = copy.deepcopy(good)
        flagged['id']='candidate-3'
        candidates = [good,truncated,flagged]
        snapshot = copy.deepcopy(candidates)
        accepted,rejected=screen(candidates,{'candidate-3':'Contradicts supplied units.'})
        self.assertEqual([r['id'] for r in accepted],['candidate-1'])
        self.assertEqual({r['id'] for r in rejected},{'candidate-2','candidate-3'})
        self.assertEqual(candidates,snapshot)

    def test_identical_final_answers_link_case_families_across_tasks(self):
        rows=group_complete_answers(authored_rows()+numerical_rows())
        by_id={r['id']:r for r in rows}
        groups=case_groups(rows)
        where={r['id']:index for index,group in enumerate(groups) for r in group}
        self.assertEqual(where['complete-clinical-poison-direct'],where['complete-refusal-005'])
        unit_rows=[r for r in rows if r['id'].endswith('missing_unit')]
        self.assertGreater(len(unit_rows),2)
        self.assertEqual(len({where[r['id']] for r in unit_rows}),1)


if __name__ == '__main__':
    unittest.main()
