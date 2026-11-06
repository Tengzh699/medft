# 历史项目说明与操作记录

本文保留发布前README的历史记录。部分`outputs/`与`data/`链接指向复现实验后生成的本地产物；GitHub公开摘要见[reports](../reports/README.md)。

# 医疗问诊与分诊辅助微调项目

按「基础医疗助手 → 问诊与分诊增强 → 应用与评测完善」逐步迭代。V1 采用 **Qwen2.5-7B-Instruct + QLoRA**，复用两篇教程的思路，统一数据格式、训练和评测入口。

已完成V1、V2、V2.1、V3三组对照及V3.1–V3.4完整答案方案的真实训练评测。原36题基座31/36，V3.1–V3.4分别28、26、25、28题；新增29题基座18题，四组分别19、16、12、16题。尚未证明整体优于基座，数据仍为待审核候选。基模现环境重跑的逐题输出与原记录完全一致。

**最新实测：医学考试选项判断已取得改善，问诊整体目标仍未达到。** 从CMExam官方train筛选2000题，以选项决策损失训练V4.2。开发选定模型后，固定1000题最终测试：基模744/1000（74.4%），V4.2为810/1000（81.0%），提高6.6个百分点；预定859题换序检查从645到690题。相同选定模型的原36题问诊回归为30/36，低于基模31/36。较小强度虽最高33/36，但仍有实际危险建议，未设为默认问诊模型。完整结论、未修复错误和教程证据见[数据优化最终报告](DATA_OPTIMIZATION_REPORT.md)。113项本地测试及来源、最优权重、逐题分数审计通过。

## 优化过程：完整答案、能力回放与训练强度对照

2026-11-06重建321条候选：219条工程编写的完整解释、信息状态与可计算示例，加102条经过筛选的基模回答用于能力保留。按病例、完整开场及相同最终答案连接分组，实际272条训练、49条验证。保留原题与原评分，不把模板重复次数当作独立知识数量。数据与来源见[V3.1方案](V3_1_DATA_RECIPE.md)。

V3.1已真实训练并完成65题开发回归，逐题还发现猜单位、数值关系矛盾和风险行动错误，失败结果见[完整复核](V3_1_EVALUATION_REVIEW.md)。V3.2固定同一数据，测试1e-4、最多3轮，第90步早停并恢复checkpoint60；虽然验证loss降到0.988，生成效果没有改善。V3.3限制为q/v、r=8，仍退化。四组原始回答均已阅读，结果和实际错误见[小规模课程复核](SMALL_CURRICULUM_REVIEW.md)。

`configs/v3_4_natural_grouped.yaml`包含257条候选、209/48分组，155条具体场景加102条能力回放；减少同一数值模板的变体、去掉食品计算中无关的用药套话，按临床信息编写具体追问与行动。相关知识的明确近似问法也保守连接同组。已完成真实训练和评测，仍未达到替换基模的目标。

公开知识对照已完成：V4.1用2000道不同题干的官方训练题加209条自然场景、148条固定验证，完整解析SFT仍退化。V4.2仅使用其中2000/100道考试题，q/v层r8 QLoRA、5e-5、两轮500步，以五选项交叉熵训练；按固定训练验证损失回载checkpoint-375。实际数据ID、自然样本排除数量、源码和模型哈希均保留。它的专用入口是`python scripts/train_cmexam_choices.py --config configs/v4_2_choices.yaml`，通用`medft train`仍是完整回答SFT。已训练adapter在`outputs/v4-2-choices/adapter/`，用于专项研究，不作为已验证的问诊升级。数据条件、最终测试和评分边界见[CMExam实验](CMEXAM_EXPERIMENT.md)。最终测试已使用，后续迭代不能继续称其为未见验收集。

教程复核见[教程效果的证据与限制](TUTORIAL_EFFECT_EVIDENCE.md)：第二篇可见生成代码的6000条只有49种问答，随机划分的400条验证记录全部在训练中出现。较低验证loss、预期输出和验收目标均不能直接当作未见病例优于基模的证据。

## 历史结果：V2.1评测

V2.1第200步正常早停，最佳权重来自第125步。报告解读原规则从V2的4/6变为6/6，分诊3/6变为5/6；知识问答4/6降为3/6、症状采集4/6降为3/6、检查5/6降为4/6。第24题呕血伴症状迅速加重没有给出明确急诊/120渠道，属于实际退步；部分其他丢分来自同义词漏匹配。完整逐题观察见 [V2.1评测复核](V2_1_EVALUATION_REVIEW.md)。

该次实际评测保存在 `outputs/evaluations/v2-1-adapterv/`，两份比较为 `v2-1-comparison.json` 和 `v2-v2-1-comparison.json`。四组条件、逐题分数及报告重算一致，权重匹配checkpoint-125。

V2.1后转入V3数据与训练策略对照。原36题和评分继续保留；6/6等小样本规则分数不是医学正确率。训练核查见 [V2.1训练记录](V2_1_TRAINING_REVIEW.md)，数据构成及操作见 [V2.1步骤](V2_1_WORKFLOW.md)。

## 历史结果：V3数据质量与学习率实验

已构建766条未审核候选：715条筛选种子、36条新增行为样例、15条公开来源工程改编；训练614条、验证152条/74组。公开源抽检发现内容噪声后，采用绑定原文、行号、哈希、摘录和修改理由的小批改编，保留真实unreviewed状态。

`configs/v3_control.yaml`与`configs/v3.yaml`的训练条数和任务数量相同、验证集字节相同，先比较数据方案；`configs/v3_low_lr.yaml`复用新数据，单独检查降低学习率的效果。三组真实训练评测已完成，36题多要点分别21、22、20题，均低于基座31题；不能用验证loss改善代替生成效果结论。完整观察见[V3评测复核](V3_EVALUATION_REVIEW.md)，后续转入上面的完整答案方案。

完整数据来源、重建、预览、训练、评测与限制见 [V3数据与训练对照步骤](V3_DATA_WORKFLOW.md)。

## 已完成：第二篇的6000条数据与训练

已新增独立配置 `configs/v1_6000.yaml`：知识2400、问诊2400、拒答/高风险1200，共6000条，按完整病例组划为5600条训练和400条验证。新增90个知识主题及问诊、拒答槽位，实际为345条种子、170个主题/场景组；问法变体不算独立知识。无需先完成旧adapter评测，也不需要重新下载。

```powershell
python -m medft generate-data --config configs/v1_6000.yaml
python -m medft build-data --config configs/v1_6000.yaml
python -m medft preview --config configs/v1_6000.yaml --split train --limit 2
```

生成和划分输出到独立的 `data/generated/v1-6000/`、`data/processed/v1-6000/`，不另切test集；固定12题独立评测另计。全部仍为未审核教学候选。数据构成、实际分组比例及后续训练见 [6000条复现步骤](TUTORIAL6000.md)。

用户已完成该配置的3轮正式训练（2100步）。第1轮验证loss最低，最终adapter已核对与该轮checkpoint一致；后两轮验证loss升高。12题原始输出评测和基座对照已完成，关键词均12/12；逐题检查发现腹超准备编造、普通用眼问题只追问等，不能据此认定整体效果提升。训练结果及命令见 [6000条训练记录](V1_6000_TRAINING_REVIEW.md)，内容观察与后续要求见 [逐题评测复核](V1_6000_EVALUATION_REVIEW.md)。

## 已有的774条公开混合试跑

使用 `configs/v1_expanded.yaml`：70个独立教学种子生成600条模板样例，再抽样200条公开百科候选。当前质量清单剔除其中26条，重新生成后共774条，知识/问诊/拒答为294/320/160；40%/40%/20%保留为目标配比，报告记录实际配比。模板变体共享病例编号，多轮问诊包含追问、补充信息与就诊方向；后续V2再扩展第一篇的症状采集、分诊、检查建议和报告解读任务。

在已激活的 `medft` 环境、项目根目录依次运行：

```powershell
python -m medft download-data --config configs/v1_expanded.yaml --endpoint https://hf-mirror.com
python -m medft generate-data --config configs/v1_expanded.yaml
python -m medft build-data --config configs/v1_expanded.yaml
python -m medft preview --config configs/v1_expanded.yaml --split train --limit 2
python -m medft preview --config configs/v1_expanded.yaml --split train --origin public_encyclopedia --limit 5
```

分别负责下载公开源文件、组合候选语料、清洗并按病例划分、查看训练样例。`download-data` 下载数据集，原有的 `download` 下载基座模型。可访问Hugging Face官方站时，省略 `--endpoint`。首次会下载完整的上游训练文件，200条指质量清单应用前的抽样数量。

镜像若报 `FileMetadataError: Distant resource does not seem to be on huggingface.co`，给下载命令加 `--transport http`，使用带断点续传和上游哈希校验的HTTP方式。文件约604MB；中断后重复相同命令可续传。失败时会输出底层错误链，代理连接错误可按[数据扩充步骤](DATA_WORKFLOW.md)排查。

首次公开样例预览发现人物传记污染，已增加 `common-health-v2` 主题筛选与问题/答案共享主题检查，并导出 `public_review.jsonl`。已有公开源缓存时，只需重新运行 `generate-data`、`build-data` 和预览；构建与训练会拒绝旧筛选版本的数据。筛选通过仍为 `unreviewed` 候选，需要检查问答对应和医学事实。

后续预览还发现单位损坏、问答错配和不可靠医学表述，已在 `assets/huatuo_quality_exclusions.json` 记录26条工程初筛剔除决定，包含原始行号、答案哈希、理由及原文证据。生成时对固定抽样应用清单，按剩余数量报告；构建和训练会检查清单哈希。53项测试及真实缓存验证通过，当前可重建为617/80/77条。其余候选仍未经过专业医学审核，5步试跑用于验证新数据的训练流程。

当前用户已用774条候选完成扩展数据的5步试跑：训练耗时32.76秒，平均loss 2.7238，adapter保存在 `outputs/v1-expanded-smoke/adapter/`。重新加载评测可后续补做，命令见[数据扩充步骤](DATA_WORKFLOW.md)末尾；当前优先完成上面的6000条数据。

操作说明、来源、过滤规则、产物路径及离线选项见 [数据扩充步骤](DATA_WORKFLOW.md)。建议先检查生成统计和逐条样例，再训练。下文保留原50条样例的基础流程；新阶段各命令都应传入扩展配置。

## 当前环境

- Windows、Anaconda `medft` 环境、Python 3.11。
- RTX 3090 24GB；PyTorch 2.6.0+cu124。
- 项目固定依赖见 `requirements.txt`，配置见 `configs/v1.yaml`。
- 所有命令在项目根目录运行；代码以项目根目录解析配置中的相对路径。

```bat
conda activate medft
cd /d "C:\Users\Administrator\Documents\ChatGPT\medft"
```

## 第 1 步：检查环境

```bat
python -m medft check
```

检查 NumPy/PyArrow/Datasets 的实际数据转换及 GPU 4bit 运算，生成 `outputs/environment.json`。关键结果是 `ok: true`、`data_backend.ok: true`、GPU 名称正确、`4bit_test.finite: true`。

已有环境安装过依赖时不必重复安装。新环境参考：

```bat
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
```

从根目录使用 `python -m medft` 不需要额外安装本项目。若需要 `medft` 命令，可选执行 `python -m pip install -e . --no-deps`。

若出现 `pyarrow requires NumPy 2.0 or newer, found 1.26.4`，说明安装了与本项目 NumPy 版本不兼容的 PyArrow。项目已固定 `numpy==1.26.4`、`pyarrow==19.0.1`，在 medft 环境运行：

```powershell
python -m pip install "pyarrow==19.0.1"
python -m pip check
python -m medft check
```

`pip check` 只核对包声明的依赖，实际导入和二进制兼容性由项目环境检查进一步验证。修复后可继续原来的训练试跑命令。

## 第 2 步：构建并查看数据

```bat
python -m medft build-data
python -m medft preview --split train --limit 1
```

自带 `assets/v1_demo_cases.jsonl`：50 个合成教学病例，知识 20、问诊 20、拒答 10。问诊样例包含两轮用户与助手交互。全部标记 `review_status: unreviewed`，用于验证流程，不足以支持有效性或安全性结论。

这50条是本工程编写的教学样例，不含外部医疗数据。两篇教程的默认种子与模板合成流程、数据规模及来源区别见 `docs/DATA_SOURCES.md`；新的公开数据导入使用独立的扩展配置。

默认产物：

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `data/processed/v1/train.jsonl` | 40 | 更新模型参数 |
| `data/processed/v1/validation.jsonl` | 5 | 观察验证 loss、选择 checkpoint |
| `data/processed/v1/test.jsonl` | 5 | 留出的病例，供后续人工或扩展评测使用 |
| `data/processed/v1/stats.json` | — | 数据数量、任务分布、来源哈希和划分哈希 |

构建时先去除完全重复对话，再把同一 `case_id`、首轮问题相同或共享 `related_questions` 的样例归组后划分，避免这些变体跨集合。该规则不能自动识别所有语义近似病例，外部数据仍需正确标注病例组。

正则脱敏仅覆盖手机号、18位身份证号和邮箱，姓名、地址、病历叙述等需要进一步审核或专用脱敏工具。

独立评测题另存于 `assets/evaluation_v1.jsonl`，共12道，构建数据时检查所有用户轮次及备用问法是否与评测中的用户问题重复。该检查只识别规范化后的文字重复，语义改写仍需审核。先固定题目，再扩充训练数据。

## 第 3 步：准备基座

```bat
python -m medft download
```

模型权重约15GB，需保持网络可用并预留缓存和训练空间。下载默认使用 Hugging Face，可重复运行利用已完成的缓存。

- 权重缓存：`.cache/huggingface/`。
- 实际模型提交版本：`models/base_manifest.json`。
- 后续训练和推理按记录加载本地模型，不在这些命令里自动下载。
- 已有完整本地模型时，可在 YAML 设置 `model.local_path`；本地模型标记为未固定版本，需自行保持文件一致。
- 如模型缓存移到别的磁盘，需要同步修改配置并重新下载记录或设置 `model.local_path`。

## 第 4 步：先做 5 个优化步骤的试跑

```bat
python -m medft train --max-steps 5
```

默认输出到 `outputs/v1-smoke/`，与正式训练分开。`max-steps` 指优化器更新步数，配合8次梯度累积，实际会处理更多微批次。

训练会检查 LoRA 层名、数据记录及 system prompt；使用模型自带聊天模板；每一轮 assistant 内容及结束标记参与 loss，system、user、assistant角色前缀和 padding 的 labels 均屏蔽。自定义 collator 保留这些 labels。

超长对话会整条跳过并记录 id，不会静默截断成无监督信号的样本。正式训练前查看 `run.json` 的 `tokenization`，必要时调整长度或数据。默认长度1024、batch 1、累积8、LoRA r16/alpha32。

试跑只验证训练是否能完成；不能用它评价模型能力。

试跑完成后，先验证保存的adapter能够重新加载并生成回答：

```powershell
python -m medft chat --adapter outputs/v1-smoke/adapter
```

看到「模型已加载」后，输入「低密度脂蛋白胆固醇为什么受到关注？」测试一轮回答，再输入 `/reset` 清空历史，或 `/quit` 退出。聊天附带示例安全规则，这一步用于检查推理流程；效果比较使用第6步的原始输出评测。

配置 `do_sample: false` 时使用贪心解码，推理代码会清除继承自基座的 `temperature`、`top_p`、`top_k`，避免不生效的采样参数警告。开启采样时仍沿用基座默认采样参数。

训练参数已显式设置 `label_names=["labels"]`，确保PEFT模型验证时记录 `eval_loss`。当前固定版本的Transformers还可能提示SDPA不支持滑动窗口；所下载基座的 `use_sliding_window` 为 `false`，无需为这条提示安装FlashAttention。

## 第 5 步：正式训练 V1

扩充并审核数据、固定评测集后，重新执行 `build-data`，再训练：

```bat
python -m medft train
```

以每轮验证 loss 选择 checkpoint，最多保留两份。主要产物：

- `outputs/v1/adapter/`：LoRA adapter 和分词器。
- `outputs/v1/run.json`：完整配置、基座版本、数据哈希、样本筛选和训练指标。
- `outputs/v1/checkpoints/`：训练断点。
- `outputs/v1/logs/`：TensorBoard日志。

```bat
tensorboard --logdir outputs/v1/logs
```

已有训练目录时不会直接覆盖。可另设 `--output-dir outputs/v1-run2`；中断恢复用 `--resume outputs/v1/checkpoints/checkpoint-具体步数`，并保持原配置、数据和输出目录。

若 OOM，先将 YAML 中 `training.max_length` 从1024降至768，保持 batch为1并检查是否有超长样例；具体调整应依据报错和显存占用。

## 第 6 步：同条件比较基座与微调模型

当前试跑adapter可先用下列命令验证评测流程、收集错误；这次对照不足以判断微调带来稳定提升：

```powershell
python -m medft evaluate --base --output-dir outputs/evaluations/v1-base
python -m medft evaluate --adapter outputs/v1-smoke/adapter --output-dir outputs/evaluations/v1-smoke-adapter
python -m medft compare --base-dir outputs/evaluations/v1-base --adapter-dir outputs/evaluations/v1-smoke-adapter --output outputs/evaluations/v1-smoke-comparison.json
```

正式训练完成后，用正式adapter和新的输出目录再做同条件比较：

```bat
python -m medft evaluate --base --output-dir outputs/evaluations/v1-formal-base
python -m medft evaluate --adapter outputs/v1/adapter --output-dir outputs/evaluations/v1-adapter
python -m medft compare --base-dir outputs/evaluations/v1-formal-base --adapter-dir outputs/evaluations/v1-adapter
```

默认测原始模型输出。基座与adapter分别加载，适合单卡。输出包含逐题回答、规则统计及 `manual_review.csv`。对比时核对模型记录、题目哈希、system prompt、生成参数、种子和安全规则模式。

关键词命中、拒答关键词和求助渠道提及仅为诊断线索，不能当作医学准确率、正确拒答率或安全保证。12道题均为未审阅教学题。多轮题使用给定历史后生成末轮回答，完整的自主多轮问诊评测列入V2。

如单独检查应用安全层，可在评测命令加 `--guarded` 并使用新的输出目录。报告同时保留原始回答与处理后回答，防止把规则拦截效果误认为模型学习效果。

## 第 7 步：对话演示

```bat
python -m medft chat --base
python -m medft chat --adapter outputs/v1/adapter
```

输入 `/reset` 清空历史，`/quit` 退出。对话模式附带示例输入风险规则、输出复核和免责声明。规则并非经过临床验证的分诊器；「人工复核」当前只是标记，尚未接入人工工作台。

模型合并导出、API和界面将在后续阶段添加。日常开发保留原始基座和adapter即可，不必为了测试而先合并权重。

## 扩充训练数据

每行一个 JSON 对象，字段如下：

```json
{"id":"sample-001","case_id":"case-001","task":"consultation","source":"来源或授权记录标识","review_status":"unreviewed","messages":[{"role":"user","content":"用户问题"},{"role":"assistant","content":"候选回答"}]}
```

- `id` 唯一；同一病例的不同问法沿用同一个 `case_id`。
- `task` 为 `knowledge_qa`、`consultation`、`refusal`，V2再扩展。
- `messages` 支持多轮，user/assistant交替，以assistant结尾；system统一由配置注入。
- `source` 记录来源，可另加授权、采集时间和审核记录字段。`review_status: clinician_reviewed`仅在实际审核后填写。
- 默认目标配比40%/40%/20%用于对照统计，不会通过复制样本强行凑比例。
- 将配置中 `data.source` 改为新的JSONL路径，然后重新构建。

## 测试与后续计划

```bat
python -m unittest discover -s tests -v
```

测试使用临时目录、小型分词器替身和CPU张量，不下载模型。计划与阶段验收见 `docs/ROADMAP.md`。
