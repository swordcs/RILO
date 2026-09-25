import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

from .common import config, save_json


STAGES = ("embed", "extract", "aliases", "paths", "operators", "supervision", "proposal", "continuous", "query",
          "collect", "controller", "select", "retrieve", "answer", "path-eval")


def provenance(cfg):
    files = {}
    for path in sorted(Path(cfg["data"]).glob("*.json*")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        files[path.name] = digest.hexdigest()
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "faiss-cpu", "scikit-learn"):
        packages[name] = importlib.metadata.version(name)
    return {"config": cfg, "data_sha256": files, "packages": packages, "python": platform.python_version(),
            "cuda": torch.version.cuda, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def run_stage(name, cfg):
    from .analysis import evaluate_paths
    from .backends import embed_corpus, extract
    from .controller import collect, select_controller, train_controller
    from .evaluation import answer, retrieve
    from .paths import build_aliases, build_paths
    from .training import make_supervision, train_continuous, train_operators, train_proposal, train_query
    functions = {"embed": embed_corpus, "extract": extract, "aliases": build_aliases, "paths": build_paths,
                 "operators": train_operators, "supervision": make_supervision, "proposal": train_proposal,
                 "continuous": train_continuous, "query": train_query, "collect": collect,
                 "controller": train_controller, "select": select_controller, "retrieve": retrieve,
                 "answer": answer, "path-eval": evaluate_paths}
    if cfg.get("source_artifacts") and name not in {"embed", "retrieve", "answer"}:
        raise ValueError("Frozen transfer allows target indexing and evaluation only")
    if cfg["retrieval"]["budget"] < 1 or cfg["retrieval"]["budget"] > 3:
        raise ValueError("Supported search budgets are 1, 2, and 3")
    functions[name](cfg)
    destination = Path(cfg["output"] if name in {"retrieve", "answer", "path-eval"} else cfg["artifacts"])
    save_json(destination / "stage_configs" / (name + ".json"), cfg)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def freeze(args):
    from huggingface_hub import HfApi, snapshot_download
    cfg = config(args.config, args.set)
    api = HfApi()
    for name, revision in (("embedding_model", "embedding_revision"), ("language_model", "language_revision"), ("reader_model", "reader_revision")):
        if Path(cfg[name]).exists():
            raise ValueError("Freeze revisions before replacing Hub IDs with local model paths")
        cfg[revision] = api.model_info(cfg[name], revision=cfg[revision]).sha
        if args.download:
            snapshot_download(cfg[name], revision=cfg[revision])
    save_json(args.output, cfg)


def main():
    parser = argparse.ArgumentParser(prog="rilo")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare")
    data.add_argument("--dataset", choices=("2wiki", "hotpotqa", "musique", "bamboogle"), required=True)
    data.add_argument("--evaluation", required=True)
    data.add_argument("--train")
    data.add_argument("--corpus")
    data.add_argument("--eval-ids")
    data.add_argument("--output", required=True)
    data.add_argument("--count", type=int, default=1000)
    data.add_argument("--fit-count", type=int, default=900)
    data.add_argument("--tune-count", type=int, default=100)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--hop-counts")
    scale = commands.add_parser("scale")
    scale.add_argument("--base", required=True)
    scale.add_argument("--snapshot", required=True)
    scale.add_argument("--size", type=int, required=True)
    scale.add_argument("--output", required=True)
    for name in ("run", *STAGES, "freeze-models", "manifest", "experiments"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/2wiki.json")
        command.add_argument("--set", action="append", default=[])
        if name == "run":
            command.add_argument("--stages", nargs="+", choices=STAGES)
        if name == "freeze-models":
            command.add_argument("--output", required=True)
            command.add_argument("--download", action="store_true")
        if name == "experiments":
            command.add_argument("--suite", required=True, choices=("main", "codes", "depth", "budget", "stopping", "routing", "endpoints", "merge", "matched96", "pipelines", "route_noise", "planning"))
            command.add_argument("--output", required=True)
            command.add_argument("--artifacts", default="artifacts")
            command.add_argument("--runs", default="runs")
            command.add_argument("--execute", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--inputs", nargs="+", required=True)
    report.add_argument("--output", required=True)
    paired = commands.add_parser("paired")
    paired.add_argument("--first", nargs="+", required=True)
    paired.add_argument("--second", nargs="+", required=True)
    paired.add_argument("--metric", choices=("recall", "complete", "f1", "calls"), default="complete")
    paired.add_argument("--samples", type=int, default=10000)
    paired.add_argument("--seed", type=int, default=42)
    paired.add_argument("--output", required=True)
    strata = commands.add_parser("reader-strata")
    strata.add_argument("--first", required=True)
    strata.add_argument("--second", required=True)
    strata.add_argument("--output", required=True)
    transfer = commands.add_parser("transfer-config")
    transfer.add_argument("--source-config", required=True)
    transfer.add_argument("--target-config", required=True)
    transfer.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare(args)
    elif args.command == "scale":
        from .data import scale_corpus
        scale_corpus(args)
    elif args.command in {"report", "paired", "reader-strata"}:
        from .analysis import paired as paired_analysis, reader_strata, report as make_report
        {"report": make_report, "paired": paired_analysis, "reader-strata": reader_strata}[args.command](args)
    elif args.command == "experiments":
        from .experiments import generate
        generate(args)
    elif args.command == "freeze-models":
        freeze(args)
    elif args.command == "transfer-config":
        source, target = config(args.source_config), config(args.target_config)
        for key in ("dataset", "data", "output"):
            source[key] = target[key]
        source["source_artifacts"] = source["artifacts"]
        source["artifacts"] = target["artifacts"]
        save_json(args.output, source)
    else:
        cfg = config(args.config, args.set)
        if args.command == "manifest":
            save_json(Path(cfg["output"]) / "manifest.json", provenance(cfg))
        elif args.command == "run":
            from .experiments import default_stages
            save_json(Path(cfg["output"]) / "manifest.json", provenance(cfg))
            for stage in args.stages or default_stages(cfg):
                run_stage(stage, cfg)
        else:
            run_stage(args.command, cfg)
