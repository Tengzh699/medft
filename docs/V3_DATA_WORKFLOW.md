# V3：数据筛选与训练策略对照

2026-11-06：A/B两组7B训练和36题raw评测已完成。两组均训练76步、导出第70步最佳已评测权重；共同验证loss为1.907570和1.762163，但多要点仅21/36和22/36，基座为31/36，尚不能证明新数据方案有稳定生成收益。记录见[V3训练复核](V3_TRAINING_REVIEW.md)及[V3评测复核](V3_EVALUATION_REVIEW.md)。C组低学习率实验尚未执行。当前目标是验证减少模板重复、补充完整回答和信息更新样例是否有帮助。既有36题继续作为开发回归；本阶段不改变其问题或评分。

## 本次实际数据

共766条未审核候选、373个来源病例组：训练614条/299组，验证152条/74组。病例组来自现有case_id、相同开场问题、公开备选问题和显式最终答案组；未完成全面语义去重，不能把这些组等同于独立临床病例。

| 任务 | 总数 | 训练 | 验证 |
| --- | ---: | ---: | ---: |
| 知识问答 | 237 | 190 | 47 |
| 症状采集 | 147 | 117 | 30 |
| 分诊 | 94 | 75 | 19 |
| 检查解释 | 96 | 77 | 19 |
| 报告解读 | 160 | 129 | 31 |
| 拒答/风险边界 | 32 | 26 | 6 |
| 合计 | 766 | 614 | 152 |

来源包括：

- 715条从V2.1种子中选取的记录：不加前缀扩写，临床场景每组选择一个槽位配置及少量信息分支；报告每组只取部分分支，移除相同任务的重复最终答案。保留这些记录原有的结构化一致性检查，原医学内容仍未获得专业审核。
- 36条新增工程教学样例，来自`assets/v3_behavior_seeds.jsonl`：普通解释、检查目的、未知信息、更正年龄、否定/假设、风险变化等。稳定分支和危险变化分支共享病例组。脚本为`scripts/build_v3_behavior_seeds.py`，未调用生成API。
- 15条公开来源的工程改编样例：保留原文、源行号、原答案哈希、摘录和修改原因。改编脚本为`scripts/build_v3_public_rewrites.py`，清单为`assets/v3_public_rewrites.json`。包含编辑补充的一般解释边界，不能称为上游原答案或医生金标。

配置中的target_mix只用于目标对照统计；实际比例以preparation.json及stats.json为准，不复制样本补齐数量。

## 公开源筛选的实际发现

沿用已下载的`FreedomIntelligence/huatuo_encyclopedia_qa`，版本`2989899dd5bed83cf9bd17cf9fff9889705436e2`。未新增下载；数据卡声明Apache-2.0。原文件SHA256为`d89c034a924b63c66f94e3a920b9872ed1a609d66d8ad79d8f747374521e9285`。

扫描362420行，范围筛选留下221个候选，既有26项质量剔除后剩195个。抽检发现“感冒发烧源于上呼吸道重度发炎”、缺失血压数字、未附来源的旧流行病学统计，以及拼接错误等问题。因此最终只针对16个有原文摘录的概念编写工程改编，其中15条通过本轮保守过滤。其余公开原答案没有直接加入训练。

另1条问句触发“ 一定会 ”的保守词规则而被排除，记录在selection_rejections.jsonl；该规则也可能排除合理提问，不能解释为医学错误判定。字符相似度检查同样只用于工程筛选。所有766条的review_status仍为unreviewed，没有专业医学审核。

第一轮自动筛选的906条候选及划分已保存在`outputs/data-audit/v3/initial-screening/`，用于记录收紧公开来源前的阶段产物，不用于本次训练。

## 三组实验

| 实验 | 配置 | 训练数据 | 学习率 | 输出 |
| --- | --- | --- | --- | --- |
| A：旧数据对照 | configs/v3_control.yaml | 从原V2.1数据等量抽样 | 5e-5 | outputs/v3-control |
| B：新数据 | configs/v3.yaml | 本次筛选、补充的数据 | 5e-5 | outputs/v3 |
| C：降低学习率 | configs/v3_low_lr.yaml | 与B完全相同 | 2e-5 | outputs/v3-low-lr |

A/B每个任务的训练行数完全相同，使用逐字节相同的152条验证集。对照抽样排除已留作验证的病例、开场/备选问题和相同最终答案，共排除1801条旧候选。旧数据的知识部分仍包含旧的信息卡、标签算术等分布，这属于本次数据方案的差异。

A/B评估整套数据选择与补充方案，不能归因到其中某一条过滤规则。训练监督tokens分别为41048和41212，行数和步数相同，但token预算并不完全相同。共用验证集来自本次选样分布，会影响checkpoint选择，仍需用生成评测检验收益。

B/C固定数据、验证、batch、LoRA及其余训练设置，改变学习率及输出路径。三组均从原始Qwen2.5-7B-Instruct重新建立LoRA；不从旧adapter继续训练。

统一设置为1轮上限、batch 1×累积8、LoRA r16/alpha32、每10步保存和验证、patience 3，恢复验证loss最低的checkpoint。按当前固定Transformers 4.51.3的调度计算，计划最大76步；验证发生在10、20、30、40、50、60、70步，早停可能缩短训练。最后未评测的几步不会自动被当成最佳checkpoint。

## 已执行的本地检查

- 完整89项测试通过，包括旧版本生成/分组流程、真实小型CPU Qwen+LoRA梯度与早停恢复、外部验证源变更拦截、公开改编清单溯源和幂等构建。
- 三组的614条训练、152条验证均通过真实本地分词器；无超长跳过。新数据最大352 tokens，对照最大347 tokens，均小于1024。
- 新旧保留评测的全部用户轮及公开备选问题没有规范化精确重叠。未验证全面语义独立性。
- 160条结构化报告、30条份量计算、36条行为记录等通过各自有界检查。这些检查只验证已有约束，不验证任意医学事实，也不约束模型生成一定遵守。
- 79个历史数据、评测、配置和权重文件哈希保持不变。记录见`outputs/data-audit/v3/verification.json`及同目录三个plan.json。

## 运行步骤

本机数据已构建，A/B训练、生成评测和比较均完成，接下来执行下方C组实验即可。若在另一个工作副本重建，使用以下命令；需要现有V2.1源数据及已下载的公开数据缓存：

```powershell
python -m medft curate-data --config configs/v3.yaml
python -m medft audit-data --config configs/v3.yaml
python -m medft plan-train --config configs/v3.yaml
```

curate-data同时准备A/B的源数据、清洗划分和共同验证集。相同输入可以重复执行；若目标目录已有不同数据，会要求为新实验选择独立目录，避免改变已经训练过的数据来源。公开改编清单变化也会触发重新准备要求。

plan-train只加载本地分词器，核算样本长度、监督token及计划步数；不加载7B权重，不启动训练。`train --max-steps`继续保留原来的smoke语义，不用于替代这里的正式实验。

首次训练A/B的命令如下；当前目录已完成，不需要重跑：

```powershell
python -m medft train --config configs/v3_control.yaml
python -m medft train --config configs/v3.yaml
```

训练后分别评测原始生成结果；当前A/B评测和比较已完成，以下命令保留用于复现，不需要重跑：

```powershell
python -m medft evaluate --config configs/v3_control.yaml --adapter outputs/v3-control/adapter --output-dir outputs/evaluations/v3-control
python -m medft evaluate --config configs/v3.yaml --adapter outputs/v3/adapter --output-dir outputs/evaluations/v3-adapter
python -m medft compare --base-dir outputs/evaluations/v3-control --adapter-dir outputs/evaluations/v3-adapter --output outputs/evaluations/v3-data-comparison.json
python -m medft compare --base-dir outputs/evaluations/v2-base --adapter-dir outputs/evaluations/v3-adapter --output outputs/evaluations/v3-base-comparison.json
```

接着使用同一数据检查C：

```powershell
python -m medft train --config configs/v3_low_lr.yaml
python -m medft evaluate --config configs/v3_low_lr.yaml --adapter outputs/v3-low-lr/adapter --output-dir outputs/evaluations/v3-low-lr
python -m medft compare --base-dir outputs/evaluations/v3-adapter --adapter-dir outputs/evaluations/v3-low-lr --output outputs/evaluations/v3-learning-rate-comparison.json
```

重点逐题检查正常解释是否完整、是否重复追问已有信息、出现危险变化是否立即调整行动、报告数值是否解释清楚。loss、关键词分数和回答变长均不能单独认定能力改善。现有36题已经参与过迭代，下一阶段仍需另留未用于修改数据的审核验收题，结合人工复核与自主多轮测试。

当前766条是受来源质量约束的小规模对照起点。后续增加公开源和病例数量时，应记录授权、原文、质量判断及样本组关系；先确认本轮生成行为是否改善，再把有效的数据方案扩展到更大的独立样本规模。
