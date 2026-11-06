import re
import unittest
from decimal import Decimal
import torch
from medft.config import ROOT
from medft.data import read_jsonl
from scripts.train_consultation_hard_preferences import preference_loss


class HardPreferenceTests(unittest.TestCase):
    def test_retinal_source_is_removed_from_validation(self):
        train=read_jsonl(ROOT/'data/generated/v5-1-consultation-hard/train.jsonl')
        val=read_jsonl(ROOT/'data/generated/v5-1-consultation-hard/validation.jsonl')
        holdout=read_jsonl(ROOT/'assets/consultation_v5_holdout_frozen.jsonl')
        source=lambda rows:{r['source'] for r in rows if r['source'].startswith('https:')}
        self.assertTrue(any('detached-retina' in s for s in source(train)))
        self.assertFalse(any('detached-retina' in s for s in source(val)))
        self.assertFalse(source(train)&source(val))
        self.assertFalse(source(train)&source(holdout))

    def test_numeric_preferences_agree_with_given_inclusive_interval(self):
        rows=[r for r in read_jsonl(ROOT/'data/generated/v5-1-consultation-hard/train.jsonl') if r['id'].startswith('v51-hard-number-')]
        self.assertEqual(len(rows),12)
        for row in rows:
            match=re.search(r'(\d+(?:\.\d+)?) [^，]+，本题给定范围(\d+(?:\.\d+)?)至(\d+(?:\.\d+)?)',row['prompt'][0]['content'])
            self.assertIsNotNone(match)
            value,lo,hi=map(Decimal,match.groups())
            relation='低于' if value<lo else '高于' if value>hi else '位于'
            self.assertIn(' '+relation+'本题',row['chosen'])
            self.assertNotIn(' '+relation+'本题',row['rejected'])

    def test_active_trainer_preference_gradient(self):
        logps=torch.tensor([-50.,-20.],requires_grad=True)
        loss,_=preference_loss(logps,logps.detach(),torch.tensor([25,10]))
        loss.backward()
        self.assertLess(logps.grad[0].item(),0)
        self.assertGreater(logps.grad[1].item(),0)


if __name__=='__main__':unittest.main()
