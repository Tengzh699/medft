# 按教程扩充V1数据

当前优先执行独立的 [6000条复现步骤](TUTORIAL6000.md)：`configs/v1_6000.yaml` 已支持6000条合成候选与5600/400划分，无需先完成旧adapter评测。下文保留774条公开混合数据的下载、构建和试跑流程。

本轮沿用第二篇的“种子 → 模板扩写 → 知识/问诊/拒答 → 清洗划分”路线，再补充公开百科知识。第一篇的症状采集、分诊、检查建议和报告解读将在V2细分；当前问诊数据先练习追问、利用补充信息和提供就诊方向。

## 候选数据构成

| 来源 | 知识问答 | 多轮问诊 | 拒答及高风险求助 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 本工程70个教学种子的模板变体 | 120 | 320 | 160 | 600 |
| 主题筛选后抽样的华佗百科候选 | 200 | 0 | 0 | 200 |
| 质量清单应用前的总量 | 320 | 320 | 160 | 800 |
| 当前质量清单剔除 | -26 | 0 | 0 | -26 |
| 当前实际候选总量 | 294 | 320 | 160 | 774 |

当前26条剔除决定记录在 `assets/huatuo_quality_exclusions.json`，由编码助手进行工程质量初筛，尚无专业医学审核。剔除后不补入未检查的条目，实际配比约为37.98%/41.34%/20.67%；所有保留样例仍为 `unreviewed`。

种子见 `assets/v1_seeds.jsonl`：知识30个、问诊20个、拒答20个，均为工程编写的合成教学候选，标记 `unreviewed`。知识种子涉及指标含义、检查准备、药物常识等；问诊包含普通症状、信息不足、报告缺失、补充信息后需要加急就医等情形；拒答包含处方剂量、危险自我处理、急症求助和报告造假等。

每个知识种子有4种提问前缀；每个拒答种子有8种前缀；每个问诊种子有8种前缀，并分别生成两轮和三轮交互，第三轮练习整理已给出的就诊信息。这些变体增加表达和对话长度，不代表新的独立医学知识。600条模板样例实际仍来自70个病例组，不能视为600个独立病例，也没有通过改写宣称达到教程的6000条知识规模。

公开部分每个答案组最多选一个问题，保留可用的其他问法为 `related_questions`。预计总计最多270个来源病例/答案组；这也不保证270个医学主题互不相似。

## 1. 下载公开源文件

```powershell
python -m medft download-data --config configs/v1_expanded.yaml --endpoint https://hf-mirror.com
```

- 数据仓库：[FreedomIntelligence/huatuo_encyclopedia_qa](https://huggingface.co/datasets/FreedomIntelligence/huatuo_encyclopedia_qa)。
- 固定提交：`2989899dd5bed83cf9bd17cf9fff9889705436e2`。
- 只下载 `train_datasets.jsonl` 和 `README.md`，不执行数据仓库脚本，不导入上游test/validation。
- 数据卡声明 `apache-2.0`；来源介绍包括中文维基百科疾病/药品条目和千问健康文章。该声明是仓库元数据，不代表医学内容已审核，也不代表已经逐条核查原始网页的权利。
- 文件缓存到 `.cache/datasets/`，版本、文件及数据卡哈希记录到 `data/raw/huatuo_encyclopedia/manifest.json`。

下载整个源文件后再本地筛选，抽样200条并不意味着只下载200条。可访问官方站时省略 `--endpoint`；下载失败可重跑以利用缓存。不需要再次下载已有基座模型或重新安装依赖。

### 镜像报FileMetadataError时

如果错误链包含 `FileMetadataError: Distant resource does not seem to be on huggingface.co`，表示Hub下载器没有从文件响应中取得必需的提交版本信息。2026-10-29用户在镜像下载时复现了此问题，不能仅根据外层 `LocalEntryNotFoundError` 判断文件不存在。

改用普通HTTP下载：

```powershell
python -m medft download-data --config configs/v1_expanded.yaml --endpoint https://hf-mirror.com --transport http
```

HTTP方式通过仓库API核对固定提交、文件大小和上游哈希，再用GET下载，不依赖文件HEAD响应中的Hub专用版本头。该训练文件为604103775字节（约576.1 MiB），下载期间会打印进度。缓存位于 `.cache/datasets/http/`；中断会保留 `.part`，重复同一命令可续传。完整文件经过大小和上游SHA-256校验后才写入正式缓存并生成数据来源记录；README使用Git blob SHA-1核验。

只有错误链明确包含 `ProxyError` 或代理连接拒绝，且当前网络允许直连时，再给HTTP命令加 `--no-proxy`。该选项仅让本次下载会话忽略环境变量和系统代理，不修改系统设置，TLS证书校验仍开启。镜像缺版本头本身不要求关闭代理或更新证书。

本地测试覆盖实际HTTP请求、续传、服务器忽略Range时重新下载、错误范围保留原文件、错误哈希拒绝入库及缓存复用；测试服务器只提供小型夹具。用户已完成真实604MB文件下载，后续修改筛选规则时直接使用本地缓存，无需重新下载。

## 2. 生成候选语料

```powershell
python -m medft generate-data --config configs/v1_expanded.yaml
```

实现位置：`medft/corpus.py`，公开主题规则在 `medft/public_filter.py`；生成设置在 `configs/v1_expanded.yaml` 的 `data_generation`。

首次预览发现“蒋敷是什么？”对应一段明代人物传记。旧版仅检查知识问句、长度和排除词，遗漏了主题归属；公开医疗数据仓库中也可能混有范围外条目。现使用 `common-health-v2` 规则重新筛选。

公开数据处理：

1. 校验源文件和数据卡哈希、版本、声明许可与上游train身份。
2. 支持 `questions` 中的嵌套列表，仅接收能确定为一个答案对应多个等价问法的记录。
3. 仅保留6～160字符的中文知识问题、80～700字符答案；按正则排除治疗处方、剂量、偏方、广告、个人信息模式及当前范围外的高风险内容。
4. 将本批限定为血糖、血压、常见检查指标及常见健康问题等明确主题。问题主体须符合这些概念，问题与答案须共享主题词；另行排除传记、植物、兽医、部分明显噪声。药品说明、食疗和诊断等问题暂不从这一公开批次导入，教学种子中的对应任务另行保留。
5. 按规范化答案与备用问法去重，排除与固定评测题及已生成首轮提问相同的问题。
6. 扫描上限由20000提高为400000行；固定版本的上游train共362420行，因此当前会扫描整个train文件。在合格候选中使用固定种子做蓄水池采样，选200条；数量不足会报错并保留原有生成文件。
7. 对这200条固定抽样应用质量剔除清单，核对原始行号及答案SHA-256，剔除26条后保留174条。不重新抽样补齐；报告分别记录抽样数、剔除数和最终导入数。

过滤是保守的范围筛选，主题词匹配不能判断答案是否正确回应问题，也不能验证医学事实或识别所有个人信息。正常科普也可能被排除，剩余公开问答仍统一标记 `unreviewed`，需要逐条抽查和专业复核。不要把被规则排除的数量等同于医学错误数量。

生成产物：

| 文件 | 用途 |
| --- | --- |
| `data/generated/v1-expanded/corpus.jsonl` | 合并后的候选，当前774条，尚未划分 |
| `data/generated/v1-expanded/public_review.jsonl` | 剔除后剩余的公开候选，当前174条，供逐条检查 |
| `data/generated/v1-expanded/preparation.json` | 任务与来源数量、实际配比、病例组数量、源哈希、筛选版本、主题分布、抽样/剔除/保留数量和剔除明细 |

每条公开数据保留仓库、提交版本、文件名、原始行号、许可声明、上游划分、`public_filter_version`、`health_topics` 和 `public_exclusions_sha256`，便于回溯。重复执行会重建同一路径的候选文件；相同源文件、代码、配置、剔除清单和种子产生相同语料，生成时间记录会变化。

已生成旧版数据或更新剔除清单时，依次重新运行 `generate-data` 和 `build-data`。构建与训练入口会拒绝旧筛选版本、未应用当前清单或含已排除答案的公开记录。版本和哈希用于检查处理流程，不能替代审核状态。

### 本轮内容检查的发现

| 预览中的问题 | 发现 | 处理 |
| --- | --- | --- |
| 湿疹和荨麻疹的症状 | 内容重复，单个风团持续时间的口径不清 | 剔除，待重写复核 |
| 急性腹泻症状 | 混入持续/间歇腹泻表现和笼统生活归因 | 剔除 |
| 血小板的结构 | 直径、厚度和体积单位写成m/m3 | 剔除整条，避免猜测补单位 |
| 尿常规检查价钱 | 未回答价格，且有未经限定的指标推断 | 剔除 |
| 慢性支气管炎患者的辅助检查 | 主题与问答基本对应，检查适用条件仍需专业核查 | 保留为待审核候选 |

另检查其余公开候选的问题和部分答案，发现OGTT“唯一诊断标准”、自行使用抗生素、缺失句子、TSH解释过度推断等，共记录26条剔除决定。此项检查没有为剩余174条逐句作医学事实核验，也没有生成医学金标。剔除清单只存原始证据及决定；源文件和原文保持可回溯，不将助手改写的答案冒充原始公开数据。

后续发现问题时，在清单追加原始行号、去除首尾空白后的原答案UTF-8 SHA-256、问题、理由编码、理由和原文证据，再重新生成、构建。清单限定于固定仓库版本；原始行号匹配而答案哈希不一致时停止生成并保留旧语料。

若暂时无法下载，可只验证模板生成：

```powershell
python -m medft generate-data --config configs/v1_expanded.yaml --synthetic-only
```

该命令生成600条，实际配比是20%知识、53.33%问诊、26.67%拒答，会在报告中明确记录。它使用相同候选输出路径；之后正常运行 `generate-data` 会加入通过当前清单的公开候选，当前重建为774条。不要把这两种数据当作同一个实验。

## 3. 清洗并划分

```powershell
python -m medft build-data --config configs/v1_expanded.yaml
```

清洗、脱敏模式匹配、完全重复对话去重后，按 `case_id`、相同首轮问题和公开备用问法归组，再按任务分层划分。一个种子的全部变体进入同一集合。所有用户轮次和备用问法都会与固定评测问题做规范化文字重复检查，语义重复仍需人工检查。

产物写入 `data/processed/v1-expanded/`：`train.jsonl`、`validation.jsonl`、`test.jsonl`、`stats.json`。目标比例80%/10%/10%按组计算；病例组大小不同，实际行数未必是640/80/80，以报告为准。统计包括任务、来源、审核状态、病例组、对话轮数和各任务助手回复平均字符数。

原有50条教学源、`configs/v1.yaml`、原训练输出及12道评测题保持独立，方便回归对比。`target_mix`是统计目标，不会靠重复复制样本强行配平。

## 4. 查看样例，再决定训练

```powershell
python -m medft preview --config configs/v1_expanded.yaml --split train --limit 2
python -m medft preview --config configs/v1_expanded.yaml --split train --origin public_encyclopedia --limit 5
Get-Content .\data\generated\v1-expanded\preparation.json
Get-Content .\data\processed\v1-expanded\stats.json
```

抽查不同任务及来源，重点看：是否属于健康主题、问题与答案是否对应、事实解释是否充分、问诊是否使用第二轮信息、是否过早指定检查或用药、紧急情形是否先求助、普通知识是否过度拒答。`--origin public_encyclopedia` 先筛选来源再限制条数；`--origin template` 查看模板数据。完整检查应打开 `public_review.jsonl`，预览前几条不代表整批质量。

审核后需要修改样例时，应修改种子或导入逻辑，再重新生成和构建，避免手改划分文件造成哈希不一致。实际完成专业审核后才填写审核状态和审核记录；不要仅通过改标签把候选标为已审核。

自动化测试使用本地小型公开数据格式夹具和实际70个种子，不下载完整语料或运行7B训练。另使用用户已下载的完整公开文件进行隔离验证；实际实验的筛选数量和内容分布，以重新生成的 `preparation.json` 为准。

2026-10-29本机检查：49项自动化测试通过，包含本次传记污染等回归案例。随后使用真实缓存，在 `outputs/data-audit/v2/` 隔离生成并构建：扫描362420行，221条符合当前保守规则，选200条；与600条模板合并后共800条、270个来源组，任务数量320/320/160，划分640/83/77。已确认原第19056行传记未入选。使用本地Qwen2.5分词器检查全部800条，最长656 tokens，1024长度设置下无超长跳过；记录见 `outputs/data-audit/v2/audit.json`。固定12题评测哈希与用户当前数据文件均未改变，未运行新一轮7B训练。此前仅模板的600条检查保留在 `outputs/data-preflight/`。这些检查验证处理流程、范围与长度，医学事实仍待复核。

随后按用户新预览加入质量剔除清单：53项测试通过，覆盖剔除不补齐、清单版本、答案哈希不匹配保留旧文件、旧数据要求重建。真实缓存在 `outputs/data-audit/quality-v1/` 隔离验证，共774条、244个来源组，任务294/320/160，划分617/80/77；全部分词最长656 tokens，无超长。也实际验证了旧split会在加载分词器和7B模型前被拒绝。原数据文件和固定评测哈希未改变，审计见 `outputs/data-audit/quality-v1/audit.json`。以上结果反映工程初筛及处理流程，剩余候选仍未通过专业医学审核。

数据检查后，后续试跑命令为：

```powershell
python -m medft train --config configs/v1_expanded.yaml --max-steps 5
```

输出为 `outputs/v1-expanded-smoke/`；5步只检查当前教学候选是否能完成训练、保存和重新加载，不评价医学效果。正式训练输出为 `outputs/v1-expanded/`，仍需内容复核和独立对照评测。本轮保持基座、system prompt和训练参数与原配置一致，便于分析数据变化。

2026-10-29用户已完成本轮5步试跑：617条训练、80条验证数据均未超长，平均loss 2.7238，训练耗时32.76秒。617指可用训练样本总数，本次5步配合batch 1、梯度累积8，约处理40条，epoch为0.0648。试跑关闭验证和中途checkpoint保存，最终仍保存adapter；没有 `eval_loss` 属于当前设置。`run.json` 与当前源数据及三个split的哈希匹配，adapter中392个LoRA张量合计40,370,176个参数。尚需实际加载和生成检查。

## 5. 评测扩展数据的试跑adapter

已有 `outputs/evaluations/v1-base/` 的基座结果，其模型记录、题目哈希、system prompt、生成参数、种子与本轮一致。复用该报告，评测新adapter并比较：

```powershell
python -m medft evaluate --config configs/v1_expanded.yaml --adapter outputs/v1-expanded-smoke/adapter --output-dir outputs/evaluations/v1-expanded-smoke-adapter
python -m medft compare --base-dir outputs/evaluations/v1-base --adapter-dir outputs/evaluations/v1-expanded-smoke-adapter --output outputs/evaluations/v1-expanded-smoke-comparison.json
```

第一条重新加载基座和新adapter，生成固定12题的原始回答、统计及人工复核表；第二条再次核对评测条件并生成对照报告。评测目录需为空，保留之前50条数据试跑的输出供追溯。对照结果用于发现空回答、过度拒答、事实错误和回答缩短等问题；5步数据量很少，关键词命中和loss不足以判断医疗效果，随后仍需人工阅读逐题回答。
