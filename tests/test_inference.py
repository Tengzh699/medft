import unittest
import warnings

from medft.inference import Generator


class InferenceTests(unittest.TestCase):
    def test_greedy_output_is_preserved_without_sampling_warnings(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        class TinyTokenizer:
            pad_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                return "test prompt"

            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[1, 5, 6]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

            def decode(self, tokens, **kwargs):
                return ",".join(str(token) for token in tokens.tolist())

        torch.manual_seed(42)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            use_sliding_window=False, sliding_window=None,
        )).eval()
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7
        model.generation_config.top_p = 0.8
        model.generation_config.top_k = 20
        generator = Generator.__new__(Generator)
        generator.config = {"generation": {"do_sample": False, "max_new_tokens": 2, "max_input_tokens": 32}}
        generator.tokenizer, generator.model = TinyTokenizer(), model

        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reference = model.generate(**generator.tokenizer("test prompt"), do_sample=False, max_new_tokens=2, pad_token_id=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = generator.generate([{"role": "user", "content": "test"}])

        self.assertEqual(actual, generator.tokenizer.decode(reference[0, 3:]))
        self.assertFalse(any("do_sample" in str(warning.message) for warning in caught))
        self.assertEqual(model.generation_config.temperature, 0.7)
        self.assertEqual(model.generation_config.top_p, 0.8)
        self.assertEqual(model.generation_config.top_k, 20)


if __name__ == "__main__":
    unittest.main()
