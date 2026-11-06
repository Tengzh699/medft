# V2.1：信息一致性数据与较早验证

2026-11-01。沿用两篇教程的种子与模板路线，继续使用Qwen2.5-7B-Instruct、六任务和QLoRA。用户已完成数据构建及正式训练，第200步早停、最佳权重为第125步，产物核查通过。36题raw评测已完成，多要点27/36，基座31/36；存在部分改善及危险变化处理退步。详见 [训练检查](V2_1_TRAINING_REVIEW.md) 和 [评测复核](V2_1_EVALUATION_REVIEW.md)。

## 为什么继续改数据和训练

V2已经训练并评测：36题多要点检查为26/36，原基座为31/36。逐题复核发现已提供单位却说缺失、报告高低方向矛盾、补写未提供的病史等实际问题；部分失败则来自同义表达漏匹配。详见 [V2评测复核](V2_EVALUATION_REVIEW.md)。

V2每150步才验证，最早的checkpoint-150也是最低验证loss，后面持续上升。因此本次保持9000条规模，改进信息分支和可计算的检查，并把验证间隔缩到25步。新版本仍从同一原基座训练，不把V2 adapter当作起点。

## 数据构成

`scripts/build_v2_1_seeds.py` 可重建 `assets/v2_1_seeds.jsonl`，共7800条种子、330个知识/场景组。知识问答每条最多3个前缀，其余任务不再依靠前缀补足条数。

| 任务 | 种子 | 最终行数 | 本次内容变化 |
| --- | ---: | ---: | --- |
| 知识问答 | 600 | 1800 | 180条原有知识、210条模拟包装份量换算、210条已知/未知信息卡，各3种表述 |
| 症状采集 | 2250 | 2250 | 45场景×5组槽位×10种三轮分支：稳定、变化、缺信息、更正及假设风险等 |
| 分诊 | 1800 | 1800 | 同45场景×5组槽位×8种信息状态，信息不足时保留未确定 |
| 检查建议 | 900 | 900 | 同45场景×5组槽位×4分支，区分检查目的、全套检查、既往正常和急迫变化 |
| 报告解读 | 1350 | 1350 | 45个模拟项目×3组数值×10分支，区分单位、上下界、既往结果与方法 |
| 拒答 | 900 | 900 | 复用90条请求种子、30个组，每条10种请求框架，仍有答案复用 |

知识问答实际为普通知识540条、份量换算630条、信息卡630条，不能把1800条都当成不同医学知识。45个临床场景和45个报告项目沿用V2；新增的30个组来自模拟包装算术。9000条不是9000个独立病例，也不是新增7800条专业医学知识。

场景的信息卡、症状采集、分诊、检查任务共享 `case_id`；同一报告项目的数值分支、同一包装的份量变化也整组划分，不能跨训练和验证。

本次隔离构建实际得到：

| 任务 | 训练 | 验证 |
| --- | ---: | ---: |
| 知识问答 | 1680 | 120 |
| 症状采集 | 2100 | 150 |
| 分诊 | 1680 | 120 |
| 检查建议 | 840 | 60 |
| 报告解读 | 1260 | 90 |
| 拒答 | 840 | 60 |
| 合计 | 8400 | 600 |

训练314组，验证16组。与V2的280/20组不同；验证仍只有少数场景，且模板相似，不能只凭验证loss证明泛化。新旧版本的验证集内容也不同，不应直接把两版eval_loss的大小当作效果对比。

所有记录保留 `unreviewed`、来源和种子版本。本次没有新增公开源、调用LLM接口或使用患者记录；旧公开混合配置独立保留。模拟化验参考范围不是通用临床标准。

## 先执行这组数据命令

在已激活 `medft` 的项目根目录依次运行：

```powershell
python -m medft generate-data --config configs/v2_1.yaml
python -m medft build-data --config configs/v2_1.yaml
python -m medft audit-data --config configs/v2_1.yaml
python -m medft preview --config configs/v2_1.yaml --split train --task report_interpretation --limit 2
python -m medft preview --config configs/v2_1.yaml --split train --task symptom_collection --limit 1
```

1. `generate-data`：从新版种子生成候选，预期 `count: 9000`、`case_groups: 330`。
2. `build-data`：清洗并整组划分，预期 `duplicate_dialogues: 0`、训练8400、验证600。
3. `audit-data`：检查已构建数据、文件哈希及明示的一致性约束，预期 `checks_passed: true`。
4. 两条 `preview`：查看报告和三轮采集的真实内容，结合生成统计人工抽查。

`audit-data` 保存 `data/processed/v2-1/quality.json`：9000条检查版本，1350条报告核对本次数值、单位、参考范围、既往信息及方向，630条换算核对份量和结果；另检查指定的必需/禁止文本。已有单位与上次缺单位分别处理；相同方法未确认时不直接比较趋势；等于上下界判在范围内；下界为0的非负项目不制造负数。

检查只覆盖结构字段和指定表达。它不能识别所有自由文本矛盾、虚构病史或错误医学建议，也不是用来给模型生成结果自动判医学正确的评分器。

## 数据检查完成后再训练

```powershell
python -m medft train --config configs/v2_1.yaml
```

默认从原基座开始，最多1轮、1050次优化器更新，学习率仍为5e-5；batch size为1，累积8次后更新。每25次更新验证并保存一次，连续3次验证loss没有降低就停止。若第25步最好、50/75/100步都未改善，会在第100步触发早停；具体停止步数取决于实际loss。最多一轮不代表一定会完整训练1050步。

训练结束回载验证loss最低的已保存权重，再导出adapter。断点续训会恢复早停计数，`run.json` 新增 `selection`，记录实际步数、最佳checkpoint和是否提前停止。最佳验证loss不保证生成质量最好，仍需原始输出评测。

`--max-steps 5` 继续保留原有试跑含义：不做验证、checkpoint选择或早停，默认写入 `outputs/v2-1-smoke`。不要用它代替正式早停实验。

验证频率比V2高6倍，单独的验证开销会增加；早停也不保证总耗时一定缩短。本次同时修改了数据及checkpoint选择策略，结果变化不能单独归因于数据。若需区分原因，应另在原V2数据上仅改变验证/早停设置做控制实验。

## 36题评测记录

训练完成后，用户已在原36题做同条件raw评测，保留原基座、V1、V2对照。实际结果目录末尾有v，以下对应本次结果；再次评测应选择新目录。不要更改原36题、评分词组或历史预测来提高成绩，也不要把这些题的标准答案补进训练。现有36题已经用于开发观察，后续验收应另留未用于迭代的场景并审核答案。

```powershell
python -m medft evaluate --config configs/v2_1.yaml --adapter outputs/v2-1/adapter --output-dir outputs/evaluations/v2-1-adapterv
python -m medft compare --base-dir outputs/evaluations/v2-base --adapter-dir outputs/evaluations/v2-1-adapterv --output outputs/evaluations/v2-1-comparison.json
python -m medft compare --base-dir outputs/evaluations/v2-adapter --adapter-dir outputs/evaluations/v2-1-adapterv --output outputs/evaluations/v2-v2-1-comparison.json
```

已逐条阅读V2.1的单位、数值方向、病史引用、行动完整性和普通回答帮助程度，报告见 [评测复核](V2_1_EVALUATION_REVIEW.md)；关键词或多要点通过不等于医学准确率。

## 路径和本地验证

- 配置：`configs/v2_1.yaml`。
- 新候选：`data/generated/v2-1/`。
- 新训练/验证及质量记录：`data/processed/v2-1/`。
- 新训练输出：`outputs/v2-1/`。
- 编码助手的隔离审计：`outputs/data-audit/v2-1/verification.json`。

原V1/V2数据、adapter及评测结果保留。当前验证仅使用本地分词器、数据构建和微型CPU训练组件，没有代用户运行7B训练或推理。

79项自动化测试通过，含原V1/V2回归、9000条重建、错误字段/答案拦截，以及实际微型Qwen+LoRA的CPU早停、最佳权重回载和断点计数恢复。全量本地分词：训练最长357 tokens、验证最长352 tokens，均无超长；所有assistant轮次保留训练信号。新旧48题的全部用户轮无规范化精确文本重叠，旧文件哈希记录一致。这些检查不涵盖语义污染或专业医学正确性。

2026-11-01用户完成正式路径下的数据生成、构建、审计与预览。实际源文件及8400/600划分、quality.json哈希与上述隔离审计一致，基座仍为提交 `a09a35458c702b33eeacc393d103063234e8bc28`，训练预检通过，当时进入用户自行执行正式训练的步骤。

用户随后完成训练：八次验证最低loss为第125步1.023898，连续三次不改善后在第200步停止，Trainer计时约51分6秒。最终adapter与checkpoint-125权重哈希一致，早停记录、配置、数据及权重结构检查通过。记录见 [训练检查](V2_1_TRAINING_REVIEW.md)。

用户又完成36题评测及两份比较。V2.1关键词33/36、多要点27/36，V2为34/36与26/36，基座为34/36与31/36；第24题紧急行动遗漏等实际退步与同义漏匹配分别记录。原题、评分、预测和adapter保留，专业审核仍待完成；下一步优先风险变化及普通问答的独立回归，再推进应用。
