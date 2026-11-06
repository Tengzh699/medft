# 公开医学考试数据实验

本实验检验考试知识选项判断，另保留原36题咨询回归。考试分数不能替代问诊、自由生成解释或临床安全评价。

实验已完成。开发选定的V4.2完整强度，在首次最终固定1000题上由基模744题变为810题（74.4%→81.0%，+6.6个百分点），86进20退、精确配对p≈6.19e-11；预定换序859题从645变为690题（75.09%→80.33%，+5.24个百分点），67进22退、p≈1.90e-6。同一模型原36题为30/36，低于基模31/36。知识专项有实测改善，问诊整体目标未达到，未设为默认问诊模型。[最终报告](DATA_OPTIMIZATION_REPORT.md)包含全部强度结果、实际错误和数据残留问题。

## 来源与筛选

从[CMExam官方仓库](https://github.com/williamliujl/CMExam)的固定提交`fadb22c89beb1b7115dc36460ba792eb96b7b972`下载三份CSV。官方原始划分为54,497/6,811/6,811题。下载文件与官方Git blob及本地SHA256均核对；未使用非官方Hugging Face镜像作为训练源。

仓库LICENSE为Apache-2.0，README同时声明学术研究用途、禁止商业滥用、不用于患者诊断或个人医学能力认证。原始README、LICENSE和下载记录保存在`data/raw/cmexam/`，须同时保留这些来源条件。引用：Liu et al., *Benchmarking Large Language Models on CMExam—A Comprehensive Chinese Medical Exam Dataset*, NeurIPS 2023, [论文](https://arxiv.org/abs/2306.03030)。

`source_line`记录包含表头的CSV记录序号；选项字段内部有换行，因此它不是文本编辑器中的物理行号。审计脚本按CSV记录读取并逐条重建问题、选项、标签与解析。

只接受五个独立选项、单一A–E答案、无重复选项的记录。训练进一步要求解析35–450字、问题与选项长度受限、解析明确支持原标注答案，剔除解析中的缺图及外链标记，并用本地分词器检查长度。原检查不能保证题干媒体完整，末轮审计发现的漏项见文末。否定题的“X错，为本题正确答案”作为一种合法来源标记，不擅自反转答案。按预先固定的ID哈希顺序选取2,000题，不根据测试表现挑题。

抽查发现接触角误标为摄氏度、把脓毒性休克与感染性休克列为互斥选项等问题，新增通用筛除规则。筛除这些已知问题不代表已发现所有标注错误。解析及标准答案仍为公开数据标签，未经本项目医生审核，不称临床金标。

## 隔离与能力保留

训练仅取官方train。全部官方val/test的规范化题干及数字变体题干从训练候选排除，包括因选项格式未通过筛选的记录。每组只保留一条候选；仍无法排除语义近似或基模预训练已见这些公开题目。

另用规范化题干的字符三元组Jaccard相似度检查全部官方保留集，长度至少12、相似度至少0.90的近似题干排除。初版发现两条入选训练题达到阈值，已中止并保留`outputs/v4-cmexam/interrupted.json`；正式版为`v4-1-cmexam`，从原基模重新训练。对重建的2,000题再次独立扫描，该阈值下交集为0；更低相似度或语义层面的交集仍可能存在。开发题和最终测试题字节不变。

自然场景总计257条，保留原209条训练、48条验证，包括正常知识解释、多轮补充、急症行动、信息缺失、报告数值和拒答。其中102条筛选基模回放的86条在训练、16条在验证。它们仍是工程候选，不因混入公开数据而变成医学审核数据。

固定验证为官方val的100题加原自然场景验证48条，共148条；另从官方val保留100题只做生成/选项开发检查。最终官方test按固定ID哈希抽取1,000个不同题干，不使用测试准确率选模型或修改训练数据。

构建结果为训练2,209条、固定验证148条。训练共348,227个监督token；最长训练702 token、验证969 token，无超长丢弃。数据、来源行号、解析、选择标记和排除统计见`data/generated/v4-1-cmexam/preparation.json`。

## 训练和评价协议

从原Qwen2.5-7B-Instruct新建LoRA：4bit NF4、float16、r16/alpha32、七类投影层、学习率5e-5、1轮、有效batch8。计划276步，在92/184/276步验证并保存，只按固定验证loss选已保存checkpoint。它不继承已退化的旧adapter。

知识评价使用相同system prompt、考试题提示、左padding、batch8。输入只包括题干、选项及assistant前缀“答案：”，比较下一个token中A–E的logit并取最大值；检查分词边界，答案与解析不进入输入。不把关键词命中当答案正确，不把这种受限选择准确率称自由回答准确率。记录Wilson区间、逐题选项logit、配对改善/退步和McNemar精确检验。

辅助检查按固定种子交换选项位置；包含A–E引用、“以上”等位置相关文本的题目预先排除，避免换序改变题意。预先符合条件的开发题88道、最终测试题859道，基模和微调模型使用完全相同的置换。它与原序1,000题相关，不能将两者的样本数相加当成独立样本。

开发100题在两侧使用相同方法；完成开发选择后，才允许最终test一次基模/选中adapter对照。原36题继续原CLI单条raw生成和原评分，不加规则拦截、不改题；保留历史基模结果，并重新运行基模检查当前环境可复现性。各组真实结果须以完成后的报告为准，不能以本协议当作已经改善的证据。

在看到任何微调开发结果前，另记录可选的LoRA影响强度检查：1、0.5、0.25。它保持学到的A/B矩阵不变，仅将推理时的alpha同比缩小，公式为`W_base + scale × (alpha/r)BA`，不是再次训练或更换基模。先检查原强度；若知识改善伴随问诊退步，再在开发集检查较小强度。候选须在原36题至少通过31题、新100题严格多于基模82题，并阅读原始回答；可行候选优先知识开发分数，相同则取较小影响。最终test不能用于挑强度。没有候选通过时须报告目标未达到，不能把零更新的基模重新标成成功微调。协议保存在`outputs/optimization/cmexam-calibration-selection-protocol.json`。

环境没有SciPy，已在任何微调开发结果或最终测试运行前，将配对检验替换为标准库`math.comb`计算的等价精确双侧二项检验。保留原评测脚本，并以逐字替换核对只改变该统计计算，没有改动预测输入、模型调用、A–E评分或准确率。原冻结记录与补充记录分别保存在`cmexam-strict-pretraining-freeze.json`和`cmexam-statistics-amendment.json`；没有安装或更新环境包。108项完整本地测试通过。

## 复现

以下命令从项目根目录、medft环境运行。已有训练/评测目录时，须选新目录保留历史。

```powershell
python scripts/download_cmexam.py
python scripts/prepare_cmexam.py --strict-near-dedup
python -m unittest tests.test_cmexam_protocol -v
python scripts/evaluate_cmexam.py --config configs/v4_1_cmexam.yaml --base --split dev --output-dir outputs/evaluations/cmexam-strict-dev-base
python -m medft train --config configs/v4_1_cmexam.yaml
python scripts/evaluate_cmexam.py --config configs/v4_1_cmexam.yaml --adapter outputs/v4-1-cmexam/adapter --split dev --output-dir outputs/evaluations/cmexam-strict-dev-adapter --compare-base outputs/evaluations/cmexam-strict-dev-base
python -m medft evaluate --config configs/v4_1_cmexam.yaml --adapter outputs/v4-1-cmexam/adapter --output-dir outputs/evaluations/v4-1-cmexam-original
```

最终test命令应在开发模型已选定后运行。不要根据test表现反复选择checkpoint并继续称其独立测试。

## V4.2：只学习选项决策的受控实验

V4.1完整解析SFT在开发100题为82题，与基模相同，原36题则从31降到27；原始回答出现已知急症继续追问和血红蛋白历史值方向错误。缩小推理强度到0.5、0.25后，知识开发分别81、83题。多一题的开发差异不构成稳定改善证据。

因此另记录V4.2协议，仍在最终test未使用之前固定：只使用同一来源的2,000道官方训练题、100道固定训练验证题，排除209/48条自然场景。输入为题干、选项和assistant前缀“答案：”；标签单独存为A–E索引，绝不把答案或解析拼进输入。只对下一个token的五个选项logit计算交叉熵。它训练的是受限选择判断，不是自然语言解释或问诊能力。

从原基模新建q/v层、r8/alpha16的QLoRA，学习率5e-5、2轮500步、有效batch8，在125/250/375/500步保存并按固定100题的选项交叉熵选择checkpoint。开发集和最终测试集不用于挑checkpoint。仍须原36题至少31题通过并阅读危险情境回答，开发100题严格多于基模82题，才能作为候选。开发选择完毕后才允许一次最终test对照；不根据test结果继续挑模型。

专用训练入口如下。该实验采用不同损失，须使用此脚本；通用`medft train`仍是原完整回答SFT流程。

```powershell
python -m unittest tests.test_cmexam_choices -v
python scripts/train_cmexam_choices.py --config configs/v4_2_choices.yaml
```

预训练冻结、排除数量及选择约定见`outputs/optimization/cmexam-choice-pretraining-protocol.json`，训练保存实际参与记录ID。结果须在完成后报告，不能把目标或损失定义当作已证明的收益。

V4.2已真实完成500步，验证最低出现在375步（0.5215194），最后500步为0.5270456，因此导出checkpoint-375。开发100题基模82、新模型87，6进1退，精确McNemar p=0.125；后续已完成原36题内容回归和一次最终独立知识测试，结果见本文开头。实际参与数据为2000/100，配置引用的209/48自然记录被此专用脚本明确排除，排除数量和完整参与ID保存在run.json，不能按数据容器的2209/148误报本轮训练数量。

V4.2相对V4.1同时改变监督目标、自然场景混合、LoRA层/秩和训练轮数，是一套方案对照，不能据此把差异因果归结为其中单一因素。强度1、0.5、0.25的比较则保持同一训练权重，仅调整推理时更新幅度。

V4.2原36题回归：完整强度30题、0.5强度33题、0.25强度31题，基模31题。较小强度的知识开发均84题。两档较小强度却新增呼吸困难时平躺的建议，0.5还在危急值伴症状时引入可能延迟处理，未通过原文复核。因此没有问诊整体候选；按照前期失败策略，最终仅对开发知识最高的完整强度做专项考试检验，保留其30/36问诊退步。不能将不同强度的最优分数拼成一个模型的结果。首次最终推理前的决定见`cmexam-final-model-selection.json`；最终test不得用于再次选择。

## 最终检查与保留限制

主测试的准确率95% Wilson区间分别为71.61%–77.01%、78.45%–83.31%；换序辅助与主测试相关，不能相加样本量。全部测试答案仅用于评分，从未进入模型输入。最终113项完整本地测试通过，来源、冻结、最优权重、强度、逐题logit及配对统计核对记录在`outputs/optimization/cmexam-complete-final-integrity.json`。实际训练参与ID与最低验证损失选择另在`cmexam-choices-training-integrity.json`。

在最终模型已选定且test推理开始后，另做输入完整性审计：训练3条明确图片引用、开发1条缺表题，原解析过滤漏掉了这些题干问题。新增`scripts/audit_cmexam_media.py`和相应2项测试，区分明确图片引用与“心电图示ST段”“镜下表现”等文本。3条训练待隔离记录见`cmexam-explicit-media-training-audit.json`，本轮不删题改分、不换数据或权重。该检查用于后续冻结前的输入审核，不是本次成绩变更。

糖化血红蛋白的未计分旧问题仍回答空腹血糖代表过去2–3天，未修复自由回答事实错误；另一肾功能问题将肌酐偷换成肌酐清除率。保留原文，不把受限选择的改善扩展成自然问诊或医学安全结论。本轮最终题集已使用；进一步训练研究须另留验收集，不把它重新称为未见测试。
