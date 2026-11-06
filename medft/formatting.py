from __future__ import annotations

from dataclasses import dataclass

from .data import validate_messages


class SequenceTooLong(ValueError):
    pass


def encode_conversation(tokenizer, messages: list[dict], max_length: int) -> dict:
    """Use the model template and supervise every assistant turn, including its EOS."""
    validate_messages(messages)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(full) > max_length:
        raise SequenceTooLong(f"对话共 {len(full)} tokens，超过 max_length={max_length}。")
    labels = [-100] * len(full)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True)
        completed = tokenizer.apply_chat_template(messages[:index + 1], tokenize=True, add_generation_prompt=False)
        if full[:len(prefix)] != prefix or full[:len(completed)] != completed:
            raise ValueError("当前聊天模板不满足前缀一致性，不能可靠屏蔽 labels。")
        if len(completed) <= len(prefix):
            raise ValueError("assistant 没有可训练 tokens。")
        labels[len(prefix):len(completed)] = full[len(prefix):len(completed)]
    if all(label == -100 for label in labels):
        raise ValueError("样本没有任何 assistant 监督信号。")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


@dataclass
class AssistantOnlyCollator:
    """Pad the already masked labels without rebuilding them from input_ids."""

    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict:
        import torch

        length = max(len(feature["input_ids"]) for feature in features)
        multiple = self.pad_to_multiple_of
        length = ((length + multiple - 1) // multiple) * multiple
        inputs = torch.full((len(features), length), self.pad_token_id, dtype=torch.long)
        masks = torch.zeros_like(inputs)
        labels = torch.full_like(inputs, -100)
        for index, feature in enumerate(features):
            size = len(feature["input_ids"])
            if size != len(feature["labels"]):
                raise ValueError("input_ids 与 labels 长度不一致。")
            if not any(label != -100 for label in feature["labels"]):
                raise ValueError("样本所有 labels 都被屏蔽。")
            inputs[index, :size] = torch.tensor(feature["input_ids"], dtype=torch.long)
            masks[index, :size] = 1
            labels[index, :size] = torch.tensor(feature["labels"], dtype=torch.long)
        return {"input_ids": inputs, "attention_mask": masks, "labels": labels}
