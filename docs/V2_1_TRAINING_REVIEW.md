# V2.1训练与早停检查

2026-11-01：用户报告训练完成。编码助手核查本地记录和权重文件，未启动新增7B训练或推理。

结论：训练在第200步正常早停，最终adapter与最佳已保存的checkpoint-125权重逐字节一致。随后已完成36题评测，多要点27/36；内容观察见 [V2.1评测复核](V2_1_EVALUATION_REVIEW.md)。

## 训练结果

依据：`outputs/v2-1/run.json`、`outputs/v2-1/checkpoints/checkpoint-200/trainer_state.json`、`outputs/v2-1/adapter/`。机器可读核查记录：`outputs/training-audit/v2-1/verification.json`。

| 项目 | 实际结果 |
| --- | --- |
| 基座 | Qwen/Qwen2.5-7B-Instruct |
| 基座提交 | a09a35458c702b33eeacc393d103063234e8bc28 |
| 数据 | V2.1六任务合成候选，仍为unreviewed |
| 训练/验证分词可用数 | 8400/600，超长跳过均0 |
| 监督token数 | 训练605576、验证43632 |
| 训练配置 | 从原基座做QLoRA，学习率5e-5，batch 1×累积8，最多1轮 |
| LoRA | r16、alpha32，392个张量、40,370,176个参数 |
| 实际更新步数 | 200/1050，epoch=0.190476 |
| 验证/保存间隔 | 每25步 |
| 早停 | 连续3次验证loss未降低，第200步停止 |
| 最佳已保存步数 | 125，eval_loss=1.0238983631 |
| Trainer计时 | 3066.0285秒，约51分6秒 |
| 平均train_loss | 1.1373279399 |
| 末次10步训练loss | 0.1753 |

训练记录时间为北京时间12:22:25至13:13:32。8400条是可用训练集规模；由于早停，实际没有完整处理一轮。按单卡batch 1、累积8计算，200次更新折合约1600条样本处理，不能把它描述为已完整训练8400条。

已核对当前配置与run.json一致，训练记录中的stats与当前源文件及split哈希匹配，数据预检和场景组隔离通过；分词可用数与监督token数匹配此前全量审计。当前基座manifest与训练记录一致，所有已记录训练loss、验证loss、梯度范数均为有限数值。adapter配置、张量数量、形状对应参数总数和文件长度核对通过。

## 八次验证与早停

| 步数 | eval_loss |
| ---: | ---: |
| 25 | 2.6046037674 |
| 50 | 1.6228088140 |
| 75 | 1.1327095032 |
| 100 | 1.0839509964 |
| 125 | **1.0238983631** |
| 150 | 1.0399428606 |
| 175 | 1.1159702539 |
| 200 | 1.1753023863 |

验证loss降到第125步后连续三次升高，同期训练loss继续下降，出现当前模板验证集上的过拟合迹象。最后trainer_state保存的早停计数为3、should_training_stop为true；第125步的计数为0，与配置及日志重算结果一致。第200步结束符合预设规则。

最终adapter权重SHA256：

```text
00376653bec66e320c4b6fef81f7d500b12a879d30496054e3bf6ff5c0360d74
```

该哈希与checkpoint-125一致；checkpoint-200为`c4a4f8f4b8defd7abeceb545caebd01949c46a4981fdbbd5bdec37e32b18b1b9`。因此最终推理使用第125步状态，训练过程总共执行200步。当前保留这两个checkpoint，符合最多保留2份并保留最佳与最后状态的设置。

第125步是八个验证时点中的最佳已保存状态，不能当作所有训练状态或生成质量的最优。V2.1验证集内容与V2不同，1.0239不能直接与旧V2的1.2926比较来证明模型更好；本次还同时修改了数据和验证/早停策略，效果变化也不能单独归因于其中一项。

## 36题原始输出评测（已完成）

已有基座`outputs/evaluations/v2-base/`和旧V2 `outputs/evaluations/v2-adapter/`的模型记录、题目哈希、system prompt、生成参数、seed、raw模式和36题数量均与当前配置相同，原摘要及预测文件哈希记录未变，可以复用。

用户已完成评测及两份比较，实际输出目录为 `outputs/evaluations/v2-1-adapterv`，比最初计划末尾多一个v。对应命令如下，再次评测应选择新的输出目录：

```powershell
python -m medft evaluate --config configs/v2_1.yaml --adapter outputs/v2-1/adapter --output-dir outputs/evaluations/v2-1-adapterv
python -m medft compare --base-dir outputs/evaluations/v2-base --adapter-dir outputs/evaluations/v2-1-adapterv --output outputs/evaluations/v2-1-comparison.json
python -m medft compare --base-dir outputs/evaluations/v2-adapter --adapter-dir outputs/evaluations/v2-1-adapterv --output outputs/evaluations/v2-v2-1-comparison.json
```

第一条加载第125步最佳adapter并回答36题；第二条与原基座比较；第三条与旧V2比较，其中base字段代表旧V2。

三组多要点分别为基座31/36、V2 26/36、V2.1 27/36。四组（含V1）条件和逐题分数重算一致，已逐题检查V2.1的普通解释、算术、已知/未知信息、病史引用、报告单位与高低方向及行动遗漏，并区分规则漏匹配。第24题危险变化处理有实际退步，不能由净增1题认定整体提升。36题已用于开发观察，分数不是临床准确率，也不能替代未参与迭代的场景验收或专业审核。
