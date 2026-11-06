import re
import unittest
from decimal import Decimal

from medft.data import case_groups
from scripts.build_complete_data import group_complete_answers, fmt
from scripts.build_natural_curriculum import natural_authored, link_known_topics


class NaturalCurriculumTests(unittest.TestCase):
    def test_food_math_retains_result_working_and_scope_without_medication_boilerplate(self):
        for row in natural_authored():
            data = row.get('calculation')
            if not data:
                continue
            answer = row['messages'][-1]['content']
            expected = Decimal(data['per_100']) * Decimal(data['portion']) / Decimal(100)
            self.assertEqual(expected, Decimal(data['result']))
            self.assertIn(fmt(expected) + data['unit'], answer)
            self.assertIn('÷100', answer)
            self.assertIn('全天', answer)
            self.assertFalse(re.search('用药|处方|治疗', answer), row['id'])
        drink = next(r for r in natural_authored() if r['id'] == 'natural-extra-15')
        match = re.search(r'每(\d+)毫升标(\d+)千焦.*喝(\d+)毫升', drink['messages'][0]['content'])
        reference, energy, volume = map(Decimal, match.groups())
        self.assertIn(fmt(energy * volume / reference) + '千焦', drink['messages'][-1]['content'])

    def test_actual_urgent_states_offer_current_action_instead_of_stock_collection(self):
        rows = [r for r in natural_authored() if r.get('clinical_state') in {'direct', 'change'}]
        self.assertEqual(len(rows), 24)
        for row in rows:
            answer = row['messages'][-1]['content']
            self.assertRegex(answer, '立即|现在', row['id'])
            self.assertRegex(answer, '120|110|急诊', row['id'])
            self.assertNotIn('先说明年龄', answer, row['id'])
        vision = next(r for r in rows if r['id'] == 'complete-clinical-vision-direct')
        self.assertIn('眼科急诊', vision['messages'][-1]['content'])
        self.assertIn('不能先睡一觉', vision['messages'][-1]['content'])

    def test_related_paraphrases_and_cross_task_poisoning_stay_in_same_split_group(self):
        rows = group_complete_answers(link_known_topics(natural_authored()))
        owners = {r['id']: i for i, group in enumerate(case_groups(rows)) for r in group}
        for related in [
            ['complete-qa-014', 'complete-qa-015', 'natural-extra-12'],
            ['complete-qa-005', 'complete-qa-006', 'natural-extra-08'],
            ['complete-refusal-002', 'complete-refusal-007', 'natural-extra-14'],
            ['complete-clinical-poison-direct', 'complete-clinical-poison-correction', 'complete-refusal-005'],
        ]:
            self.assertEqual(len({owners[ident] for ident in related}), 1)


if __name__ == '__main__':
    unittest.main()
