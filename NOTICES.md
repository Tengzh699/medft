# 第三方来源与许可说明

本项目参考公开教程，使用开源模型、训练框架和公开数据开展领域微调实验。代码、模型、数据和来源资料各自适用相应许可；本文件不将第三方内容改授其他许可，也不表示来源机构审核或背书。

## CMExam

- 上游：https://github.com/williamliujl/CMExam
- 使用的固定提交：`fadb22c89beb1b7115dc36460ba792eb96b7b972`。
- 引用：Liu et al., *Benchmarking Large Language Models on CMExam—A Comprehensive Chinese Medical Exam Dataset*, NeurIPS 2023；https://arxiv.org/abs/2306.03030。
- 上游LICENSE副本：[CMExam-LICENSE.txt](third_party/CMExam-LICENSE.txt)。上游README副本：[CMExam-README.md](third_party/CMExam-README.md)，同时保留其研究用途及禁止商业滥用等说明。
- `assets/cmexam_dev_frozen.jsonl`与`assets/cmexam_test_frozen.jsonl`是从上游记录筛选并重新排版为本项目JSONL的子集，新增了提示消息与分组信息，保留原始题干、选项、答案、解析和CSV记录序号。它们属于本项目修改后的数据表示，不能视为上游完整数据或医生审核金标。

## 华佗百科问答

- 上游：https://huggingface.co/datasets/FreedomIntelligence/huatuo_encyclopedia_qa
- 固定版本：`2989899dd5bed83cf9bd17cf9fff9889705436e2`。
- 数据卡声明Apache-2.0，介绍来源为中文维基百科和千问健康；具体来源条件仍须核对上游资料。
- `assets/v3_public_rewrites.json`包含工程改写、原文短摘录、来源行号、哈希和修改理由。改写未经过医学专业审核，原始完整语料未包含在仓库中。

## 公开健康信息与问诊教学对照

V5/V5.1的中文对话参考NHS和NIDDK等公开健康信息编写。来源页面URL保存在数据构建脚本及本地证据清单中。原始网页和媒体未上传；公开的中文对话与偏好对照属于教学改编，急救号码按中国120/110作本地化。

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

上述说明不表示NHS、NIDDK或其他来源机构对本项目、数据或模型提供审核或背书。图片、视频等第三方媒体不进入训练资产。

## 模型与框架

基模为[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)。模型权重和训练adapter不随本仓库分发；使用时应遵守模型及PyTorch、Transformers、PEFT、bitsandbytes等依赖的相应许可。

## 参考教程

- https://lcnipys9u0xh.feishu.cn/wiki/N9T2wtMzwiryVXklxHPcxlfNnIb
- https://lcnipys9u0xh.feishu.cn/wiki/LC0swKCUZiTBsdkVDaWcmkqqnGh

本项目根据这些教程的任务方向搭建并进一步迭代。教程来源及可公开复核范围见[教程证据说明](docs/TUTORIAL_EFFECT_EVIDENCE.md)。
