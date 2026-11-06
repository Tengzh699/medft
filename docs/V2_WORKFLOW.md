# V2：接入第一篇的问诊与分诊任务

V1已经完成6000条生成、5600/400划分、正式QLoRA训练和12题对比。V2针对复核中的编造检查细节、正常问题只追问、模板泛化不足等问题，保留知识问答，再接入第一篇的症状采集、分诊、检查建议和报告解读。

2026-11-01进度：V2数据、正式训练、36题raw评测及基座/V1对比均已完成。最终adapter匹配第150步最佳checkpoint，三组多要点结果为基座31/36、V1 25/36、V2 26/36；部分内容改善，但报告信息与数值方向等缺陷仍在，尚不能证明整体优于基座。逐题复核见 `docs/V2_EVALUATION_REVIEW.md`，训练检查见 `docs/V2_TRAINING_REVIEW.md`。本页命令保留作流程记录，已完成步骤无需重复运行；后续建议优先改进数据质量和早期checkpoint对照，也可先搭建可切换三个模型的本地应用。

## 当前先执行的数据步骤

在medft环境、项目根目录依次运行，无需重新下载：

```powershell
python -m medft generate-data --config configs/v2.yaml
python -m medft build-data --config configs/v2.yaml
python -m medft preview --config configs/v2.yaml --split train --task symptom_collection --limit 1
python -m medft preview --config configs/v2.yaml --split train --task report_interpretation --limit 1
```

前两条应得到9000条候选、重复对话0、训练8400条、验证600条。后两条检查三轮症状交互和报告解释；也可将task改为knowledge_qa、triage、examination或refusal查看其他类别。查看验证集可改为 `--split validation`。

## 数据构成

| 任务 | 种子记录 | 扩写方式 | 总量 | 训练 | 验证 |
| --- | ---: | --- | ---: | ---: | ---: |
| knowledge_qa 知识问答 | 180 | 每主题10种前缀 | 1800 | 1680 | 120 |
| symptom_collection 症状采集 | 450 | 45场景×5槽位×2病程分支，每条5种前缀 | 2250 | 2100 | 150 |
| triage 分诊 | 450 | 同45场景的5槽位×2分支，每条4种前缀 | 1800 | 1680 | 120 |
| examination 检查建议 | 225 | 同45场景的5槽位，每条4种前缀 | 900 | 840 | 60 |
| report_interpretation 报告解读 | 675 | 45项目×5上下文×3信息分支，每条2种前缀 | 1350 | 1260 | 90 |
| refusal 拒答/高风险求助 | 90 | 30场景×3请求，每条10种前缀 | 900 | 840 | 60 |
| 合计 | 2070 | | 9000 | 8400 | 600 |

这是结合两篇教程的六任务配比：保留20%知识问答，症状采集25%、分诊20%、检查建议10%、报告解读15%、拒答10%。第一篇原五任务配比未直接照搬；9000条是本阶段生成总数，不声称复现其清洗后8028条的记录。

实际为300个知识/场景组：180知识主题、45个共享问诊/分诊/检查场景、45个报告项目组、30个拒答场景组。槽位与前缀不代表新增独立病例。完整组一起划分，训练280组、验证20组；这依然是模板验证集，不是临床测试集，也未实现所有语义相似内容的自动隔离。

## 本轮内容变化

知识部分复用原120个主题，新增60个以解释和实际帮助为主的主题，包括无症状日常建议、阅读环境、工作间歇、检查资料整理和支持他人等。它们先回答一般问题，并在适用时说明边界，避免把所有正常问题转成处方拒绝。

症状部分每条包含三轮真实信息交互：首轮询问重点，第二轮接收年龄、病程和诱因，第三轮接收减轻或新危险表现。病情变化时回答必须随之改变，不仅追加同一句就诊总结。分诊另保留department、urgency、red_flags字段，紧急程度分outpatient、same_day和emergency；这些是工程待审核标签，不能作为专业分诊金标。

检查建议说明病史、查体和不同检查所解决的问题，避免统一指定影像、所有项目全做或虚构准备流程。报告部分使用题目明确提供的模拟项目、单位、参考范围和既往结果，区分范围内、范围外、缺少单位；上下文包括不同机构、活动变化、上次单位不清及没有既往结果。参考区间是教学假设，不能当作通用临床标准。

种子由 `scripts/build_v2_seeds.py` 生成并保存为 `assets/v2_seeds.jsonl`，所有内容均为工程编写的合成候选，标记unreviewed；未调用LLM接口、未引入患者记录，也未声称逐条依据已审核指南。当前不混入未完成医学复核的公开候选；774条公开混合配置继续单独保留，后续可另做混合实验。

## 新36题开发评测

`assets/evaluation_v2.jsonl` 每任务6题，共36题，由 `scripts/build_v2_evaluation.py` 重建。覆盖普通建议、病情变化、纠正前述信息、科室与紧急程度、检查局限、单位/参考范围/趋势及拒答。

每题新增required_groups：同一要点内可命中任意关键词，但该题所有要点均需命中才算要点齐备。例如普通建议需同时包含环境调整和休息安排，只有“用眼”两个字无法通过该检查。summary会报告required_groups_pass_rate及各任务计数，compare也会比较这些字段。原12题没有该字段，旧评分保持兼容。

多要点匹配仍是文字规则，可能误判同义词、否定、提问式输出或不实陈述；还要结合forbidden_pattern_hits、原始回答和人工复核。它不是医学准确率、安全率或完整的误拒判断。

36题是开发回归集，尚未经专业审核。多轮题给定固定历史后只测最后回答，不能代替自主连续对话评测；主题也可能与训练知识相关，不声称所有情境都未经见过。训练生成和构建会排除新36题及原12题的全部用户轮，并记录两个文件的哈希。最终验收仍需另留未经开发使用的场景。

## 数据检查后的对照与训练

新36题与原12题不同，需要在同一新题集上建立基座与V1对照，由用户运行：

```powershell
python -m medft evaluate --config configs/v2.yaml --base --output-dir outputs/evaluations/v2-base
python -m medft evaluate --config configs/v2.yaml --adapter outputs/v1-6000/adapter --output-dir outputs/evaluations/v1-on-v2
python -m medft compare --base-dir outputs/evaluations/v2-base --adapter-dir outputs/evaluations/v1-on-v2 --output outputs/evaluations/v1-on-v2-comparison.json
```

这些命令用于记录当前模型在新题上的行为，不作为扩充数据的前置条件。当前V2数据和上述统计已通过隔离验证；用户已完成上述两组7B推理及报告对比，结果保存在对应目录，复核见 `docs/V2_BASELINE_REVIEW.md`。

后续V2训练从同一原始基座开始一份新的QLoRA实验，生成 `outputs/v2/`，复用V1工程、知识资产和评测方法；不是自动继续叠加V1 adapter。基于前轮过拟合观察，当前配置为1轮、学习率5e-5，每150步验证并保存一次，完整8400条时共1050步、7次验证。与V1同时变化了数据、训练时长和学习率，因此属于新版本实验，不能把效果变化归因于单一因素；继续训练V1 adapter可作为后续独立对照。

```powershell
python -m medft train --config configs/v2.yaml
```

训练及以下评测已完成，最终adapter来自第150步最佳checkpoint。命令记录：

```powershell
python -m medft evaluate --config configs/v2.yaml --adapter outputs/v2/adapter --output-dir outputs/evaluations/v2-adapter
python -m medft compare --base-dir outputs/evaluations/v2-base --adapter-dir outputs/evaluations/v2-adapter --output outputs/evaluations/v2-comparison.json
python -m medft compare --base-dir outputs/evaluations/v1-on-v2 --adapter-dir outputs/evaluations/v2-adapter --output outputs/evaluations/v1-v2-comparison.json
```

首次数据检查阶段先运行本页开头四条即可。每次评测使用新目录，结果已存在时命令会要求更换目录。

## 验证与产物

2026-10-30：68项测试通过，包含六任务9000条计数、8400/600分组、跨任务场景隔离、病情更新、单位缺失处理、两套保留题排除、训练前哈希拦截、步数验证与保存同步、新多要点评测和旧V1回归。

在 `outputs/data-audit/v2-six-tasks/` 隔离生成、构建，并用本地Qwen分词器检查全部9000条：训练最长465 tokens、验证最长470 tokens，均无超1024；36题输入最长157 tokens。58份旧数据、adapter和评测相关文件哈希保持一致，详见该目录verification.json；未加载7B模型或启动V2训练。

用户正常命令的产物为 `data/generated/v2/`、`data/processed/v2/`。原V1数据、训练checkpoint和12题评测继续作为对照。
