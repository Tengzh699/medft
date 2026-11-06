import math
import tempfile
import unittest


class TrainingStackTests(unittest.TestCase):
    def test_tiny_qwen_lora_gradients_and_evaluation_loss(self):
        """Exercise real Transformers/PEFT on CPU; no model weights are downloaded."""
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments

        from medft.formatting import AssistantOnlyCollator

        torch.manual_seed(42)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
            bos_token_id=1, eos_token_id=2, pad_token_id=0, use_sliding_window=False, sliding_window=None,
        ))
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]))
        features = [
            {"input_ids": [1, 5, 6, 7, 8, 2], "labels": [-100, -100, -100, 7, 8, 2]},
            {"input_ids": [1, 9, 10, 2], "labels": [-100, -100, 10, 2]},
        ]
        batch = AssistantOnlyCollator(0)(features)
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum().item() > 0 for parameter in model.parameters() if parameter.requires_grad))
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters() if parameter.requires_grad))

        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(
                output_dir=directory, use_cpu=True, report_to=[],
                label_names=["labels"], remove_unused_columns=False,
                dataloader_pin_memory=False, disable_tqdm=True,
            )
            trainer = Trainer(model=model, args=args, eval_dataset=features,
                              data_collator=AssistantOnlyCollator(0))
            self.assertTrue(math.isfinite(trainer.evaluate()["eval_loss"]))


if __name__ == "__main__":
    unittest.main()
