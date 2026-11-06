from __future__ import annotations

import json

from .config import project_path
from .models import load_quantized_model, load_tokenizer
from .safety import assess_input, finalize_output


class Generator:
    def __init__(self, config: dict, adapter: str | None = None):
        from transformers import set_seed

        set_seed(config["project"]["seed"])
        if adapter:
            adapter_path = project_path(adapter)
            if not (adapter_path / "adapter_config.json").is_file():
                raise FileNotFoundError(f"adapter 不存在或不完整：{adapter_path}")
        self.config = config
        self.tokenizer = load_tokenizer(config)
        self.model, self.manifest = load_quantized_model(config)
        if adapter:
            from peft import PeftModel

            run_path = adapter_path.parent / "run.json"
            if run_path.is_file():
                training_model = json.loads(run_path.read_text(encoding="utf-8"))["model"]
                if (training_model["repo_id"], training_model["resolved_revision"]) != (self.manifest["repo_id"], self.manifest["resolved_revision"]):
                    raise ValueError("adapter 使用的基座版本与当前配置不一致。")
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.model.eval()
        self.model.config.use_cache = True

    def generate(self, messages: list[dict]) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.config["generation"]["max_input_tokens"]:
            raise ValueError("对话超过输入长度限制，请输入 /reset 开始新对话。")
        inputs = {name: value.to(self.model.device) for name, value in inputs.items()}
        # The base model ships sampling defaults; they are unused in greedy decoding.
        sampling_kwargs = {} if self.config["generation"]["do_sample"] else {
            "temperature": None, "top_p": None, "top_k": None,
        }
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.config["generation"]["max_new_tokens"],
                                         do_sample=self.config["generation"]["do_sample"],
                                         pad_token_id=self.tokenizer.pad_token_id, **sampling_kwargs)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def chat(config: dict, adapter: str | None) -> None:
    generator = Generator(config, adapter)
    messages = [{"role": "system", "content": config["system_prompt"]}]
    print("模型已加载。输入 /reset 清空历史，/quit 退出。示例安全规则已开启。")
    while True:
        try:
            text = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text == "/quit":
            return
        if text == "/reset":
            messages = messages[:1]
            print("已清空历史。")
            continue
        if not text:
            continue
        current = {"role": "user", "content": text}
        decision = assess_input(text)
        if decision.response:
            answer = decision.response
            issues = []
        else:
            try:
                answer, issues = finalize_output(generator.generate(messages + [current]))
            except ValueError as error:
                print(error)
                continue
        messages.extend([current, {"role": "assistant", "content": answer}])
        print("\n助手：" + answer)
        if decision.human_review or issues:
            print("[复核标记：该样例需要专业人员复核；尚未接入人工处理系统。]")
