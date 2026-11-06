from __future__ import annotations

import argparse
import json
import sys

from .config import load_config, project_path, split_names, write_json


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="医疗问诊 QLoRA 项目：数据、训练、对话与评测")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/v1.yaml", help="配置文件路径")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="检查依赖、CUDA 和实际 4bit 运算")
    check.add_argument("--output", default="outputs/environment.json")
    check.add_argument("--skip-4bit", action="store_true")
    data_download = commands.add_parser("download-data", parents=[common], help="显式下载固定版本的公开百科数据")
    data_download.add_argument("--endpoint", help="可选 Hugging Face 镜像地址")
    data_download.add_argument("--transport", choices=["hub", "http"], default="hub", help="下载方式；http支持独立断点续传")
    data_download.add_argument("--no-proxy", action="store_true", help="仅HTTP方式：本次会话不使用环境变量/系统代理")
    data_generate = commands.add_parser("generate-data", parents=[common], help="从种子和公开百科生成候选数据")
    data_generate.add_argument("--synthetic-only", action="store_true", help="仅生成模板数据，实际任务配比会变化")
    commands.add_parser("build-data", parents=[common], help="清洗、去重并按病例划分数据")
    commands.add_parser("curate-data", parents=[common], help="构建V3筛选数据及等量旧数据对照（不训练模型）")
    plan = commands.add_parser("plan-train", parents=[common], help="仅用本地分词器核算训练长度与步数，不加载模型权重")
    plan.add_argument("--output", help="可选计划JSON输出路径")
    commands.add_parser("audit-data", parents=[common], help="核对结构化数值与明确的信息一致性约束")
    preview = commands.add_parser("preview", parents=[common], help="查看生成的样本")
    preview.add_argument("--split", choices=["train", "validation", "test"], default="train")
    preview.add_argument("--limit", type=int, default=1)
    preview.add_argument("--origin", help="仅查看指定来源类别，例如public_encyclopedia或template")
    preview.add_argument("--task", help="仅查看指定任务，例如triage或report_interpretation")
    commands.add_parser("download", parents=[common], help="显式下载基座模型并记录实际版本")
    training = commands.add_parser("train", parents=[common], help="执行单 GPU QLoRA 训练")
    training.add_argument("--max-steps", type=int, help="仅训练指定步数；默认写入单独的 smoke 目录")
    training.add_argument("--output-dir")
    training.add_argument("--resume", help="已有 checkpoint 目录")
    for name, description in (("chat", "多轮对话，带示例安全规则"), ("evaluate", "使用独立题集评测")):
        command = commands.add_parser(name, parents=[common], help=description)
        selector = command.add_mutually_exclusive_group(required=True)
        selector.add_argument("--base", action="store_true", help="使用未微调基座")
        selector.add_argument("--adapter", help="使用指定 LoRA adapter")
        if name == "evaluate":
            command.add_argument("--output-dir", required=True, help="每次评测使用新的输出目录")
            command.add_argument("--guarded", action="store_true", help="另外应用安全规则；默认仅评测原始模型输出")
    compare = commands.add_parser("compare", help="核对评测条件并比较两份报告")
    compare.add_argument("--base-dir", required=True)
    compare.add_argument("--adapter-dir", required=True)
    compare.add_argument("--output", default="outputs/evaluations/comparison.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "check":
            from .environment import check_environment

            result = check_environment(test_4bit=not args.skip_4bit)
            write_json(project_path(args.output), result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ok"] else 1
        if args.command == "compare":
            from .evaluation import compare_reports

            result = compare_reports(args.base_dir, args.adapter_dir, args.output)
        else:
            config = load_config(args.config)
            if args.command == "download-data":
                from .corpus import download_public_data

                result = download_public_data(config, endpoint=args.endpoint, transport=args.transport, no_proxy=args.no_proxy)
            elif args.command == "generate-data":
                from .corpus import generate_data

                result = generate_data(config, synthetic_only=args.synthetic_only)
            elif args.command == "build-data":
                from .data import build_data

                result = build_data(config)
            elif args.command == "curate-data":
                from .curation import curate_data

                result = curate_data(config)
            elif args.command == "plan-train":
                from .training import plan_training

                result = plan_training(config)
                if args.output:
                    write_json(project_path(args.output), result)
            elif args.command == "audit-data":
                from .quality import audit_data

                result = audit_data(config)
            elif args.command == "preview":
                from .data import read_jsonl

                if args.limit < 1:
                    raise ValueError("limit 必须为正整数。")
                if args.split not in split_names(config["data"]):
                    raise ValueError(f"当前配置未启用{args.split}划分。")
                rows = read_jsonl(project_path(config["data"]["output_dir"]) / f"{args.split}.jsonl")
                if args.task is not None and args.task not in config["data"]["target_mix"]:
                    raise ValueError(f"当前配置没有任务：{args.task}")
                result = [row for row in rows if (args.origin is None or row.get("origin") == args.origin)
                          and (args.task is None or row["task"] == args.task)][:args.limit]
            elif args.command == "download":
                from .models import download_model

                result = download_model(config)
            elif args.command == "train":
                from .training import train

                result = train(config, max_steps=args.max_steps, output_dir=args.output_dir, resume=args.resume)
            elif args.command == "chat":
                from .inference import chat

                chat(config, args.adapter)
                return 0
            elif args.command == "evaluate":
                from .evaluation import evaluate

                result = evaluate(config, adapter=args.adapter, output_dir=args.output_dir, guarded=args.guarded)
            else:
                raise ValueError(f"未知命令：{args.command}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
