import copy
import subprocess
import sys
from pathlib import Path

from .common import config, save_json


FIT_STAGES = ["embed", "extract", "paths", "operators", "supervision", "proposal", "query", "collect", "controller", "select"]


def default_stages(cfg):
    if cfg.get("source_artifacts"):
        return ["embed", "retrieve", "answer"]
    if cfg["variant"] in {"dense", "hybrid", "hyde"} or cfg.get("native_adapter"):
        return ["embed", "retrieve", "answer"]
    stages = list(FIT_STAGES)
    if cfg["variant"] == "continuous":
        stages[stages.index("proposal")] = "continuous"
    if cfg["variant"] == "direct":
        stages.remove("proposal")
    if cfg["controller"]["mode"] == "fixed":
        stages = [stage for stage in stages if stage not in {"collect", "controller", "select"}]
    return stages + ["retrieve", "answer"]


def generate(args):
    base = config(args.config, args.set)
    destination = Path(args.output).resolve()
    cases = []
    if args.suite == "main":
        cases = [(variant, {"variant": variant}) for variant in ("rilo", "direct", "continuous", "dense", "hybrid", "hyde")]
    elif args.suite == "codes":
        cases = [(f"K{count}", {"operator.codes": count}) for count in (4, 8, 16, 32)]
    elif args.suite == "depth":
        cases = [(f"H{depth}", {"operator.depth": depth}) for depth in (1, 2, 3)]
    elif args.suite == "budget":
        cases = [(f"B{budget}", {"retrieval.budget": budget}) for budget in (1, 2, 3)]
    elif args.suite == "stopping":
        cases = [(mode, {"controller.mode": mode}) for mode in ("adaptive", "fixed_stopping", "fixed")]
    elif args.suite == "routing":
        cases = [(variant, {"variant": variant}) for variant in ("rilo", "random", "clustered", "surface")]
    elif args.suite == "endpoints":
        cases = [(variant, {"variant": variant}) for variant in ("rilo", "text_only", "endpoint_only", "endpoint_shuffle")]
    elif args.suite == "merge":
        cases = [(mode, {"retrieval.merge_mode": mode}) for mode in ("endpoint", "text", "none")]
    elif args.suite == "matched96":
        cases = [(variant, {"variant": variant, "query.output_tokens": 96}) for variant in ("rilo", "direct", "continuous")]
    elif args.suite == "pipelines":
        cases = [(f"seed{seed}", {"seed": seed}) for seed in (42, 43, 44, 45, 46)]
    elif args.suite == "route_noise":
        cases = [(f"noise{noise}", {"operator.route_noise": noise}) for noise in (0.0, 0.1, 0.2, 0.4)]
    elif args.suite == "planning":
        cases = [("full", {}), ("immediate", {"controller.mode": "immediate"}), ("no_replan", {"retrieval.replan": False})]
    manifest = []
    inference_only = args.suite in {"budget", "stopping", "merge"}
    for label, settings in cases:
        cfg = copy.deepcopy(base)
        for key, value in settings.items():
            target = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        if not inference_only:
            cfg["artifacts"] = str(Path(args.artifacts) / cfg["dataset"] / args.suite / label)
        seeds = (11, 23, 37) if args.suite in {"main", "matched96"} and cfg["variant"] not in {"dense", "hybrid", "hyde"} else (cfg["controller_seed"],)
        for i, seed in enumerate(seeds):
            current = copy.deepcopy(cfg)
            current["controller_seed"] = seed
            current["output"] = str(Path(args.runs) / cfg["dataset"] / args.suite / label / str(seed))
            path = destination / (label + "_" + str(seed) + ".json")
            save_json(path, current)
            stages = ["retrieve", "answer"] if inference_only else default_stages(current) if i == 0 else ["controller", "select", "retrieve", "answer"]
            command = [sys.executable, "-m", "rilo", "run", "--config", str(path), "--stages", *stages]
            manifest.append({"label": label, "controller_seed": seed, "config": str(path), "stages": stages, "command": command})
    save_json(destination / "manifest.json", manifest)
    if args.execute:
        for row in manifest:
            subprocess.run(row["command"], check=True)
    else:
        print(destination / "manifest.json")
