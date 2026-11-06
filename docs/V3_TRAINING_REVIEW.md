# V3训练检查

2026-11-06检查：旧数据对照A与新数据B均正常完成计划训练，最终adapter均准确恢复第70步的最佳已评测权重。新数据在共同验证集上的loss较低；生成评测尚未执行，暂不能据此认定回答能力超过对照或基座。

后续更新：用户已完成A/B生成评测，多要点分别21/36和22/36，基座31/36；验证loss收益尚未转化为明确的整体回答改善。详见[V3评测复核](V3_EVALUATION_REVIEW.md)，下文保留训练检查时的状态和运行步骤。

## 实际结果

| 项目 | A：旧数据对照 | B：新数据 |
| --- | ---: | ---: |
| 输出目录 | outputs/v3-control | outputs/v3 |
| 实际完成时间（UTC） | 2026-11-03 08:20:11 | 2026-11-05 12:49:10 |
| 训练/验证可用记录 | 614 / 152 | 614 / 152 |
| 训练来源病例组 | 169 | 299 |
| 共同验证来源病例组 | 74 | 74 |
| 学习率 | 5e-5 | 5e-5 |
| 完成步数/计划步数 | 76 / 76 | 76 / 76 |
| Trainer耗时 | 758.17秒 | 745.31秒 |
| 全程平均训练loss | 1.9308 | 2.1888 |
| 最佳验证loss | 1.907570 | 1.762163 |
| 最佳已评测checkpoint | 70 | 70 |
| 是否早停 | 否 | 否 |

两组在10、20、30、40、50、60、70步验证，验证loss均逐次下降，本轮没有记录到验证loss反弹。第76步没有验证分数，因此第70步只能称为最佳已评测checkpoint。最终adapter与第70步权重的文件SHA256及392个张量均完全一致，且与第76步不同。

实际epoch为0.990228，与固定Transformers 4.51.3按614条、batch 1、累积8取整得到76步的计划相符。614条均通过分词检查、不因长度跳过；这不表示本轮已经完整遍历所有614条。

## 验证曲线

| 步数 | A验证loss | B验证loss |
| --- | ---: | ---: |
| 10 | 2.857599 | 2.842300 |
| 20 | 2.487463 | 2.450745 |
| 30 | 2.279845 | 2.174919 |
| 40 | 2.105002 | 1.978391 |
| 50 | 1.975649 | 1.853532 |
| 60 | 1.939238 | 1.782823 |
| 70 | 1.907570 | 1.762163 |

B的最佳验证loss比A低0.145407，约7.62%。两组使用逐字节相同的152条验证集、相同任务训练条数及其余训练设置；当前配置、源数据和划分均与run.json及训练前计划一致。训练监督tokens分别41048和41212，并非严格等token预算。

这是新数据方案对共同验证集拟合更好的迹象。共同验证来自本次筛选分布，两组各只运行一个随机种子；当前结果不能归因到某一条筛选规则，也不能从训练loss高低直接判断回答优劣。旧V2.1的1.023898来自不同验证集，不能直接与本次loss比较。

## 产物核对

本次仅执行CPU及文件检查，没有加载7B基座或运行新增生成：

- 两组均为正式运行，基座版本均为a09a35458c702b33eeacc393d103063234e8bc28；配置记录与当前文件相同。
- 数据预检通过：固定验证源、划分哈希、保留评测哈希、公开改编溯源和现有有界一致性检查均有效。
- 训练/验证分词统计与训练前plan一致；无超长跳过，验证监督tokens均10270。
- 两组adapter各含392个float32张量、40,370,176参数，全部为有限值；196个LoRA B张量均非零，符合已更新状态。训练计算使用FP16，adapter存为float32并不矛盾。
- 两组最终adapter准确匹配最佳已评测checkpoint；历史基座36题raw评测的基座身份、问题哈希、提示词、生成设置和种子与当前配置兼容，可复用。
- outputs/v3-low-lr尚无训练记录；两组V3尚无生成评测，目标评测目录未占用。

权重SHA256：

- A：139b872f560957648868f491e6b6c74c3f2b545c441131cd34a1458090a4e020
- B：1139b7943f426060c5e1d58258f14d7ba790e8eb55f7837695a1ef4e3f88c6af

详细机器记录见[verification.json](../outputs/training-audit/v3/verification.json)。本次没有修改源数据、训练配置、checkpoint、adapter、评测问题或评分规则。

## 下一步

由用户分别执行A/B的36题原始生成评测：

```powershell
python -m medft evaluate --config configs/v3_control.yaml --adapter outputs/v3-control/adapter --output-dir outputs/evaluations/v3-control
python -m medft evaluate --config configs/v3.yaml --adapter outputs/v3/adapter --output-dir outputs/evaluations/v3-adapter
```

生成完成后比较新旧数据，以及新数据与已有基座：

```powershell
python -m medft compare --base-dir outputs/evaluations/v3-control --adapter-dir outputs/evaluations/v3-adapter --output outputs/evaluations/v3-data-comparison.json
python -m medft compare --base-dir outputs/evaluations/v2-base --adapter-dir outputs/evaluations/v3-adapter --output outputs/evaluations/v3-base-comparison.json
```

保持现有36题及评分不变，复核全部回答；优先检查正常解释完整性、已知信息利用、危险变化时的行动调整及数值关系。规则命中不能代替内容审核，这36题已经用于开发迭代，也不能充当独立验收结果。完成A/B生成复核后，再按[V3数据步骤](V3_DATA_WORKFLOW.md)推进固定数据的低学习率实验。
