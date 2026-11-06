# 6000条配置的正式训练记录

2026-10-29，用户完成 `python -m medft train --config configs/v1_6000.yaml`。本记录核对本地日志及文件；用户随后已完成最终adapter的12题原始输出评测，结果见 [逐题评测复核](V1_6000_EVALUATION_REVIEW.md)。

## 已核实的结果

- 基座：Qwen2.5-7B-Instruct，版本 `a09a35458c702b33eeacc393d103063234e8bc28`。
- 数据：训练5600条、验证400条，无超长跳过。实际文件哈希与训练时记录一致，病例组隔离检查通过。
- 训练：3轮、2100次优化器更新，`smoke_run: false`。
- Trainer记录用时17511.0785秒，约4小时52分；平均训练loss为0.0685563。
- LoRA：r=16、alpha=32，可训练参数40,370,176；导出文件含392个LoRA张量，参数数量匹配。
- 日志中未发现非有限浮点值。

| 轮次 | 步数 | 验证loss |
| --- | ---: | ---: |
| 1 | 700 | 2.907233 |
| 2 | 1400 | 3.250269 |
| 3 | 2100 | 3.334915 |

最终几步的训练loss约为0～0.0001（日志有舍入），而验证loss逐轮升高，表明存在明显过拟合迹象。当前模板变体较多，低训练loss不等于医疗回答更准确；验证集也只有11个完整场景/主题组，需要结合独立题目逐条检查实际回答。

## 最终adapter选择

`checkpoints/checkpoint-2100/trainer_state.json` 记录：

- `best_model_checkpoint` 指向 `checkpoint-700`。
- `best_global_step` 为700，`best_metric` 为2.9072329998016357。
- 最终 `adapter/adapter_model.safetensors` 与 `checkpoint-700/adapter_model.safetensors` 的SHA256完全一致。

因此，`outputs/v1-6000/adapter/` 保存的是本次验证loss最低的第1轮权重。无需手动改为最后一轮checkpoint，也无需因后两轮验证loss升高而立即重训。

## 已完成：原始输出评测

已核对已有 `outputs/evaluations/v1-base/summary.json` 与本轮条件一致：基座记录、独立12题哈希、system prompt、生成参数、seed和raw模式均匹配。可复用该基座报告。

用户已运行：

```powershell
python -m medft evaluate --config configs/v1_6000.yaml --adapter outputs/v1-6000/adapter --output-dir outputs/evaluations/v1-6000-adapter
python -m medft compare --base-dir outputs/evaluations/v1-base --adapter-dir outputs/evaluations/v1-6000-adapter --output outputs/evaluations/v1-6000-comparison.json
```

第一条重新加载最终adapter，对固定12题生成原始回答；第二条比较与基座的启发式指标。两组关键词命中均为12/12，禁止模式和空回答均为0。已逐题检查输出并记录腹超准备编造、日常用眼只追问等问题；不能把关键词命中率当作临床准确率，专业医学审核仍待完成。

如需手动体验对话：

```powershell
python -m medft chat --config configs/v1_6000.yaml --adapter outputs/v1-6000/adapter
```

chat开启示例安全规则，部分高风险问题由规则直接回答；评估微调本身时应以上述默认raw评测为依据。
