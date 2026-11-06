# MedFT：医疗领域微调与可信评测

基于 **Qwen2.5-7B-Instruct + QLoRA**，在单张RTX 3090 24GB上完成数据处理、模型训练、adapter推理和对照评测。项目面向中文健康信息与问诊流程研究，分别检查医学考试选项判断和自然语言问诊。

## 实测结果

| 任务 | 基模 | 微调 | 结论 |
| --- | ---: | ---: | --- |
| CMExam固定1000题 | 74.4% | **81.0%** | 受限A–E选择提高6.6个百分点 |
| 预定859题选项换序 | 75.09% | **80.33%** | 辅助检查提高5.24个百分点 |
| 最新32段问诊对照 | 基模较好4段 | 微调较好4段，24段相当 | 尚未证明整体优于基模 |

考试两项成绩来自同一个选定adapter。换序题与原题相关，不合并为1859个独立样本。问诊采用隐藏模型身份的工程复核，由编码助手完成，未经独立医生评价。项目没有将考试收益解释为诊断准确率，也没有将失败候选替换为默认问诊模型。

[公开实验摘要](reports/README.md) · [知识实验与局限](docs/CMEXAM_EXPERIMENT.md) · [问诊最新结果](docs/CONSULTATION_V5_1.md) · [项目总结与面试材料](docs/PROJECT_RESUME_INTERVIEW.md)

## 已实现

- **数据处理**：JSONL messages格式、基础正则脱敏、完全去重、病例关联分组、题干近似过滤、来源与哈希记录。
- **单卡训练**：4bit NF4与双重量化、PEFT LoRA、assistant-only监督、梯度检查点、梯度累积、验证选最优、早停与断点恢复。
- **专项目标**：完整回答SFT、医学五选项交叉熵、缓存参考概率的DPO加chosen-answer SFT。
- **推理与评测**：adapter加载、CLI多轮聊天、基模同条件对照、选项换序、逐题原始回答复核、数据与权重审计。
- **工程验证**：最新本地121项测试通过，覆盖mask、padding、偏好与选项梯度方向、数据隔离以及训练恢复等行为。

当前是本地实验工程。API、网页界面、RAG、真实医生审核路由和生产部署尚未实现。教学材料与示例安全规则不用于患者诊断或个体处方。

## 技术栈

Python 3.11、PyTorch 2.6.0/CUDA 12.4、Transformers、PEFT、Accelerate、bitsandbytes、Datasets/PyArrow、TensorBoard、YAML。开发环境为Windows、Anaconda、RTX 3090 24GB。依赖固定版本见[requirements.txt](requirements.txt)。

## 快速开始

在项目根目录运行。以下示例用于验证基础流程，自带50条未审核教学数据。

```powershell
conda create -n medft python=3.11 -y
conda activate medft
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
python -m medft check
```

准备数据、显式下载基模并做5步试跑：

```powershell
python -m medft build-data
python -m medft preview --split train --limit 1
python -m medft download
python -m medft train --max-steps 5
python -m medft chat --adapter outputs/v1-smoke/adapter
```

聊天输入`/reset`清空历史，`/quit`退出。首次下载需网络和足够磁盘空间；模型下载后记录固定提交，后续训练和推理从本地缓存加载。5步试跑只确认流程可运行。

原始模型输出对照：

```powershell
python -m medft evaluate --base --output-dir outputs/evaluations/demo-base
python -m medft evaluate --adapter outputs/v1-smoke/adapter --output-dir outputs/evaluations/demo-adapter
python -m medft compare --base-dir outputs/evaluations/demo-base --adapter-dir outputs/evaluations/demo-adapter --output outputs/evaluations/demo-comparison.json
```

`chat`使用示例安全规则；`evaluate`默认评测原始输出。`--guarded`用于单独检查应用规则，不能将其效果当作模型学习收益。关键词和多要点规则分数仅用于排错。

## 实验路线

| 阶段 | 工作与数据 | 观察 |
| --- | --- | --- |
| V1 | 50条试跑，进一步构建6000条、5600/400按组划分 | 行数与独立病例/知识量需要分别统计 |
| V2/V3 | 症状采集、分诊、检查、报告、拒答；数据质量、完整答案、能力回放与训练强度实验 | 多组loss下降却生成退化，保留失败记录 |
| V4 | CMExam公开源筛选与五选项决策训练 | 固定测试上取得知识选项收益，问诊未同步改善 |
| V5/V5.1 | 184组偏好对照扩展到220组，强调当前/假设/更正/恶化与关键行动 | 最终内容复核未证明问诊整体提升 |

V4.2使用2000道官方训练题、100道固定训练验证题，另有开发集用于选择方案。q/v层r8、alpha16，约252万可训练参数；学习率5e-5、有效batch8、2轮500步，按固定验证损失回载checkpoint-375。标准答案和解析不进入输入，只对下一token的A–E五个logit计算交叉熵。

V5.1使用220组原创教学偏好对照、18组固定验证，另留8类情境的32段对话。参考48页公开健康信息，未经医生标注。完整强度出现内容退化，预定0.5强度完成最终对照，4进4退24相当；未认定优于基模。

专用训练目标须使用对应脚本，不能用通用`medft train`代替：

```powershell
python scripts/train_cmexam_choices.py --config configs/v4_2_choices.yaml
python scripts/train_consultation_hard_preferences.py --config configs/v5_1_consultation_hard.yaml
```

以上两条需要先按各实验文档准备数据、模型和来源证据。已有输出目录会报错以保留历史。完整重建步骤见[CMExam实验](docs/CMEXAM_EXPERIMENT.md)、[问诊偏好数据卡](docs/CONSULTATION_V5.md)和[历史操作记录](docs/PROJECT_HISTORY.md)。

## 目录

```text
medft/        配置、数据、训练、推理、评测与示例规则
configs/      各阶段配置
assets/       教学种子、质量清单及固定评测子集
scripts/      数据构建、专项训练、评测与审计脚本
tests/        本地工程测试
docs/         设计、实验记录、失败分析及面试总结
reports/      可公开的轻量实验摘要与导出来源哈希
third_party/  第三方许可及来源README副本
```

仓库不包含模型权重、adapter、缓存、大型原始语料、生成训练集和完整本地执行输出；它们分别在`models/`、`.cache/`、`data/`、`outputs/`生成。历史文档会引用这些本地产物，GitHub公开摘要见[reports](reports/README.md)。

## 测试与复现边界

安装依赖后运行：

```powershell
python -m unittest discover -s tests -v
```

部分检查需要本地模型分词器或来源文件。测试验证工程约束，不代表临床安全。原始输出会受硬件、依赖、量化和批次设置影响；比较模型时需保持模型提交、提示词、生成参数及batch一致。

项目对病例变体与近似题干做隔离，但不能排除所有语义重叠和公开题进入基模预训练。最终1000题及32段问诊均已使用，后续不能再用它们调参并称为未见验收集。

## 数据与来源

- [CMExam](https://github.com/williamliujl/CMExam)：固定官方提交，保留来源、标签和研究用途说明。
- [华佗百科问答](https://huggingface.co/datasets/FreedomIntelligence/huatuo_encyclopedia_qa)：固定数据版本，增加主题筛选及工程剔除记录。
- NHS、NIDDK等公开健康信息：用于原创中文教学对照的行动边界参考，原始网页和媒体未随仓库发布。
- 参考两篇医疗微调教程的任务与流程，并对可见数据代码和评测表述进行复核；详情见[教程证据](docs/TUTORIAL_EFFECT_EVIDENCE.md)。

基础脱敏只覆盖手机号、18位身份证号和邮箱。全部教学材料和公开数据改写保留未经医学审核的状态。来源机构不对本项目背书。第三方许可、副本和改编说明见[NOTICES.md](NOTICES.md)。
