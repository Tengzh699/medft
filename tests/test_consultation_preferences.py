import unittest
import torch
from scripts.train_consultation_preferences import preference_loss,sequence_logps,PreferenceCollator


class PreferenceTests(unittest.TestCase):
    def test_gradient_prefers_chosen_and_opposes_rejected(self):
        logps=torch.tensor([-4.,-3.],requires_grad=True)
        loss,_=preference_loss(logps,logps.detach(),torch.tensor([2,2]))
        loss.backward()
        self.assertLess(logps.grad[0].item(),0)
        self.assertGreater(logps.grad[1].item(),0)

    def test_only_supervised_next_tokens_contribute(self):
        logits=torch.zeros(1,4,3,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,-100]])
        logps,counts=sequence_logps(logits,labels)
        (-logps.sum()).backward()
        self.assertEqual(counts.tolist(),[1])
        self.assertTrue(logits.grad[0,1].abs().sum()>0)
        self.assertEqual(logits.grad[0,0].abs().sum().item(),0)
        self.assertEqual(logits.grad[0,2:].abs().sum().item(),0)

    def test_pair_order_and_padding_mask(self):
        short=dict(input_ids=[1,2],attention_mask=[1,1],labels=[-100,2])
        long=dict(input_ids=[3,4,5],attention_mask=[1,1,1],labels=[-100,-100,5])
        batch=PreferenceCollator(0)([dict(chosen=short,rejected=long,reference_chosen=-1.,reference_rejected=-2.)])
        self.assertEqual(batch['reference_logps'].tolist(),[-1.,-2.])
        self.assertEqual(batch['labels'][0,:4].tolist(),[-100,2,-100,-100])

    def test_longer_chosen_answer_does_not_change_reference_zero_margin(self):
        # Different response lengths/logps must still have zero DPO advantage
        # when policy equals reference; the anchor intentionally remains CE.
        logps=torch.tensor([-40.,-5.])
        loss,metrics=preference_loss(logps,logps.clone(),torch.tensor([20,2]),sft_weight=0.)
        self.assertAlmostEqual(loss.item(),0.69314718,places=6)

    def test_actual_chat_template_masks_history_assistant(self):
        from medft.config import load_config
        from medft.models import load_tokenizer
        from scripts.train_consultation_preferences import encode_reply
        tokenizer=load_tokenizer(load_config('configs/v5_consultation.yaml'))
        prompt=[{'role':'system','content':'测试'}, {'role':'user','content':'之前的情况'},
                {'role':'assistant','content':'这段历史不应成为本条的监督目标'},
                {'role':'user','content':'现在情况变了'}]
        feature=encode_reply(tokenizer,prompt,'这是最终回答',512)
        prefix=tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True)
        self.assertEqual(feature['labels'][:len(prefix)],[-100]*len(prefix))
        self.assertEqual(feature['labels'][len(prefix):],feature['input_ids'][len(prefix):])


if __name__=='__main__':unittest.main()
