import unittest

from medft.formatting import AssistantOnlyCollator, SequenceTooLong, encode_conversation


class TinyTemplateTokenizer:
    """A prefix-consistent template for testing boundaries without downloads."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(f"<{message['role']}>\n{message['content']}<end>\n" for message in messages)
        if add_generation_prompt:
            text += "<assistant>\n"
        return [ord(char) for char in text] if tokenize else text


class FormattingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTemplateTokenizer()
        self.messages = [
            {"role": "system", "content": "规则"}, {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": "回答一"}, {"role": "user", "content": "问题二"},
            {"role": "assistant", "content": "回答二"},
        ]

    def test_all_assistant_turns_and_only_assistant_turns_supervised(self):
        feature = encode_conversation(self.tokenizer, self.messages, 1024)
        expected = set()
        for index in (2, 4):
            start = len(self.tokenizer.apply_chat_template(self.messages[:index], add_generation_prompt=True))
            end = len(self.tokenizer.apply_chat_template(self.messages[:index + 1]))
            expected.update(range(start, end))
        actual = {index for index, label in enumerate(feature["labels"]) if label != -100}
        self.assertEqual(expected, actual)
        for index in actual:
            self.assertEqual(feature["labels"][index], feature["input_ids"][index])

    def test_overlong_conversation_not_silently_truncated(self):
        with self.assertRaises(SequenceTooLong):
            encode_conversation(self.tokenizer, self.messages, 5)

    def test_non_prefix_template_rejected(self):
        class BrokenTokenizer(TinyTemplateTokenizer):
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
                tokens = super().apply_chat_template(messages, tokenize, add_generation_prompt)
                return [999] + tokens if add_generation_prompt else tokens

        with self.assertRaisesRegex(ValueError, "前缀一致性"):
            encode_conversation(BrokenTokenizer(), self.messages, 1024)

    def test_collator_preserves_labels_and_masks_padding(self):
        feature = encode_conversation(self.tokenizer, self.messages, 1024)
        shorter = encode_conversation(self.tokenizer, self.messages[:3], 1024)
        batch = AssistantOnlyCollator(0)([feature, shorter])
        size = len(shorter["input_ids"])
        self.assertEqual(batch["labels"][1, :size].tolist(), shorter["labels"])
        self.assertTrue((batch["labels"][1, size:] == -100).all().item())
        self.assertTrue((batch["attention_mask"][1, size:] == 0).all().item())
        self.assertEqual(batch["input_ids"].shape[1] % 8, 0)

    def test_collator_rejects_zero_supervision(self):
        with self.assertRaisesRegex(ValueError, "所有 labels"):
            AssistantOnlyCollator(0)([{"input_ids": [1, 2], "labels": [-100, -100]}])


if __name__ == "__main__":
    unittest.main()
