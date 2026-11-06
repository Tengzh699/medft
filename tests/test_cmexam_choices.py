import copy
import unittest

from scripts.train_cmexam_choices import decision_feature, ChoiceCollator, choice_loss


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '|'.join(m['content'] for m in messages) + '|assistant:'

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class ChoiceObjectiveTests(unittest.TestCase):
    def test_answer_and_explanation_never_enter_prompt(self):
        row = {'task': 'medical_exam', 'gold_answer': 'B', 'messages': [
            {'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': 'SECRET explanation'}]}
        first = decision_feature(FakeTokenizer(), row, 100)
        row = copy.deepcopy(row)
        row['gold_answer'] = 'E'
        row['messages'][-1]['content'] = 'ENTIRELY DIFFERENT GOLD AND EXPLANATION'
        second = decision_feature(FakeTokenizer(), row, 100)
        self.assertEqual(first['input_ids'], second['input_ids'])
        self.assertEqual(first['labels'], 1)
        self.assertEqual(second['labels'], 4)

    def test_left_padding_leaves_answer_prefix_at_final_position(self):
        batch = ChoiceCollator(0)([
            {'input_ids': [5, 6], 'labels': 2}, {'input_ids': [7, 8, 9], 'labels': 4}])
        self.assertEqual(batch['input_ids'][:, -1].tolist(), [6, 9])
        self.assertEqual(batch['attention_mask'].sum(dim=1).tolist(), [2, 3])
        self.assertEqual(batch['labels'].tolist(), [2, 4])

    def test_loss_pushes_correct_choice_up(self):
        import torch
        logits = torch.zeros((2, 5), requires_grad=True)
        loss = choice_loss(logits, torch.tensor([1, 3]))
        self.assertAlmostEqual(loss.item(), 1.6094379, places=6)
        loss.backward()
        self.assertLess(logits.grad[0, 1], 0)
        self.assertGreater(logits.grad[0, 3], 0)
        self.assertLess(logits.grad[1, 3], 0)
        with self.assertRaises(ValueError):
            choice_loss(torch.zeros(2, 6), torch.tensor([1, 3]))


if __name__ == '__main__':
    unittest.main()
