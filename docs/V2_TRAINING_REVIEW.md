# V2正式训练检查

2026-11-01：用户报告V2已训练完成，编码助手读取本地训练产物进行核查。实际训练记录时间为北京时间2026-10-30 14:45:50至18:02:42。本轮仅检查文件、配置与日志，未启动额外7B训练或推理。

结论：六任务V2正式训练完整运行1轮、1050步，最终adapter与第150步的最佳已保存checkpoint逐字节一致。训练后段出现过拟合迹象；下一步评测已保存的最佳adapter，效果尚不能由loss判断。

## 训练记录

依据：`outputs/v2/run.json`、`outputs/v2/checkpoints/checkpoint-1050/trainer_state.json`及`outputs/v2/adapter/`。

| 项目 | 实际结果 |
| --- | --- |
| 基座 | Qwen/Qwen2.5-7B-Instruct |
| 基座提交版本 | a09a35458c702b33eeacc393d103063234e8bc28 |
| 数据 | 六任务合成教学候选，仍标记unreviewed |
| 训练/验证 | 8400/600条，超长跳过均0 |
| 监督token数 | 训练904617，验证66938 |
| 训练方式 | 单卡QLoRA，从原始基座训练，未叠加V1 adapter |
| LoRA | r16、alpha32，392个张量，共40,370,176个可训练参数 |
| 更新步数 | 1050，epoch=1.0，非smoke试跑 |
| 学习率/有效batch | 初始配置5e-5，batch 1×累积8 |
| Trainer计时 | 11811.2354秒，约3小时16分51秒 |
| 平均train_loss | 0.2541916576 |
| 末次10步训练loss | 0.0014 |

已确认当前配置与run.json保存的配置一致，当前数据预检通过，stats与训练记录一致，源文件与split哈希匹配，病例组不跨集合，两套保留题哈希未变。基座manifest与训练记录一致。全部训练/验证loss为有限数值，adapter文件头、张量形状对应参数总数和文件长度核对通过。

## 验证与最佳权重

| 步数 | epoch | eval_loss |
| ---: | ---: | ---: |
| 150 | 0.1429 | 1.2926341295 |
| 300 | 0.2857 | 1.4509122372 |
| 450 | 0.4286 | 1.6167542934 |
| 600 | 0.5714 | 1.8340430260 |
| 750 | 0.7143 | 2.0558190346 |
| 900 | 0.8571 | 2.1742618084 |
| 1050 | 1.0000 | 2.2103657722 |

验证loss在首次记录的第150步最低，之后六次持续升高；同期训练loss继续下降，末尾接近零，说明在当前按病例组隔离的模板验证集上有过拟合迹象。第150步的训练日志loss为0.3714。这些loss不是医学准确率，也不能与不同数据集上的V1验证loss直接比较。

Trainer记录best_model_checkpoint为`outputs/v2/checkpoints/checkpoint-150`，best_metric与七次验证最小值一致。最终`adapter/adapter_model.safetensors`的SHA256为：

```text
dca19c196f89d84ea608230d6585e1809dd1cf361a1ae4a40124560f1698f7be
```

该哈希与checkpoint-150权重相同，与checkpoint-1050不同，确认训练结束后正确加载并导出了最佳已保存权重。完整优化过程运行了1050步，但最终用于推理的权重来自第150步。该步是七个已评估时点中的最佳，不代表早于150步的所有状态均已比较，也不能证明优于未微调基座。

当前仅保留checkpoint-150和checkpoint-1050，符合save_total_limit=2同时保留最佳与最后checkpoint的行为；七次验证记录仍在最后checkpoint的trainer_state.json中。

## 下一步36题评测

已有`outputs/evaluations/v2-base/`及`outputs/evaluations/v1-on-v2/`的模型身份、system prompt、生成参数、seed、raw模式与题目哈希均与当前评测配置相同，可以复用。当前尚无`outputs/evaluations/v2-adapter/`，由用户依次执行：

```powershell
python -m medft evaluate --config configs/v2.yaml --adapter outputs/v2/adapter --output-dir outputs/evaluations/v2-adapter
python -m medft compare --base-dir outputs/evaluations/v2-base --adapter-dir outputs/evaluations/v2-adapter --output outputs/evaluations/v2-comparison.json
python -m medft compare --base-dir outputs/evaluations/v1-on-v2 --adapter-dir outputs/evaluations/v2-adapter --output outputs/evaluations/v1-v2-comparison.json
```

第一条加载第150步最佳adapter并回答36题；第二条与原基座比较；第三条与V1比较，其中报告的base字段代表V1。评测保留原始模型输出，不额外套用安全规则。

完成后重新计算并核对逐题规则，阅读普通问题解释、已知病史利用、报告数值和高风险行动等原始回答。优先回归`docs/V2_BASELINE_REVIEW.md`中的遗漏、虚构病史和数值矛盾，同时区分规则漏匹配。36题仍是开发回归集，分数不等于临床准确率或安全保证。无需在评测前增加训练轮数。
