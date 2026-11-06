# 公开实验摘要

这里保存本地已完成实验的轻量JSON副本。导出仅将项目目录的绝对路径替换为`<PROJECT_ROOT>`；未更改成绩、模型选择决定或失败结论。来源与导出文件哈希见[export-manifest.json](export-manifest.json)。

| 实验 | 基模 | 微调 | 范围 |
| --- | --- | --- | --- |
| CMExam固定1000题 | 74.4% | 81.0% | A–E受限选项判断，+6.6个百分点 |
| 预定859题换序 | 75.09% | 80.33% | 与原题相关的辅助检查 |
| 最新32段问诊内容复核 | 基模较好4段 | 微调较好4段，另24段相当 | 隐藏身份的工程复核，非医生评价 |

- [知识专项最终记录](knowledge-final-result.json)
- [知识专项完整性检查摘要](cmexam-final-integrity.json)
- [问诊最终记录](consultation-final-result.json)
- [问诊数据与权重审计摘要](consultation-data-audit.json)
- [问诊32段内容判断](consultation-content-review.json)
- [问诊测试前模型选择](consultation-selection-before-holdout.json)
- [问诊最终核对摘要](consultation-final-verification.json)
- [教程可见代码的重复审计](tutorial-repetition-audit.json)

训练权重、完整逐题logit、原始模型输出、原始网页及大型语料保留在本地`outputs/`、`models/`、`data/`和`.cache/`，不在Git仓库内。以上摘要不能替代全部原始证据；详细方法与局限见[知识实验](../docs/CMEXAM_EXPERIMENT.md)和[问诊实验](../docs/CONSULTATION_V5_1.md)。历史文档中的这些本地产物路径在复现后生成，公开摘要的入口以本页为准。

两项最终测试均已使用，后续不得继续将其当作未见测试进行调参。公开考试成绩不等于自由问诊或临床准确率。教学样本与工程复核未经本项目医生验证。
