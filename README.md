<meta name="robots" content="noindex, nofollow, noarchive" />

# Rilo

### Latent Relational Operators for Multi-Hop Retrieval

Rilo composes learned relational operators into multi-hop retrieval plans, generates search queries, and selects actions with a budget-aware controller.

[Quick Start](#quick-start) | [Data](#data) | [OpenIE](#openie) | [Training](#training) | [Experiments](#experiments) | [Evaluation](#evaluation) | [Repository](#repository)

## Quick Start

### Installation

Use Python 3.11 and a CUDA-compatible PyTorch environment.

```bash
cd /path/to/code_repo
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Models

| Role | Model | Training |
|:--|:--|:--|
| Passage and query encoder | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Frozen |
| Extraction and query generation | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Frozen extraction; LoRA query adaptation |
| Answer reader | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Frozen, non-thinking decoding |

Resolve model revisions to commit IDs and download the snapshots:

```bash
rilo freeze-models --config configs/2wiki.json --output configs/2wiki.locked.json --download
```

Model settings also accept local snapshot paths.

## Data

### Dataset Preparation

```bash
rilo prepare --dataset 2wiki --evaluation raw/2wiki/dev.json --train raw/2wiki/train.json --eval-ids raw/2wiki/eval1000_ids.json --output data/2wiki
rilo prepare --dataset hotpotqa --evaluation raw/hotpotqa/dev.json --train raw/hotpotqa/train.json --eval-ids raw/hotpotqa/eval1000_ids.json --output data/hotpotqa
rilo prepare --dataset musique --evaluation raw/musique/dev.jsonl --train raw/musique/train.jsonl --count 1000 --hop-counts 2:450,3:350,4:200 --output data/musique
rilo prepare --dataset bamboogle --evaluation raw/bamboogle.csv --corpus raw/bamboogle_corpus.jsonl --count 125 --output data/bamboogle
```

`--eval-ids` loads a fixed evaluation set; otherwise, preparation samples with seed 42. `--corpus` supplies a passage collection in place of the selected questions' contexts. Training questions are grouped by normalized text into disjoint fit/tune/evaluation sets. The default fit/tune sizes are 900/100, configurable through `--fit-count` and `--tune-count`.

### Format

Each dataset directory contains `corpus.jsonl`, `fit.jsonl`, `tune.jsonl`, and `eval.jsonl`.

Passage record:

```json
{"id":"chunk-<md5>","title":"Page title","text":"Passage text"}
```

Question record:

```json
{"id":"2wiki:question-id","question":"Question text","answers":["Answer","Alias"],"support_ids":["chunk-<md5>"],"hop":2}
```

Passage IDs follow:

```python
identity = "chunk-" + hashlib.md5((title + "\n" + text).encode("utf-8")).hexdigest()
```

`support_ids` references corpus passage IDs. Answer-only datasets use `support_ids: null`.

## OpenIE

OpenIE uses the BHR entity-extraction prompts and entity-conditioned RDF triple extraction. Both stages use greedy, non-thinking decoding with a 2,048-token output limit.

```bash
rilo extract --config configs/2wiki.locked.json
rilo extract --config configs/2wiki.locked.json --set openie_cache=/path/to/openie_results_ner_Qwen_Qwen3-8B.json
```

Extraction resumes from saved passages. `openie_cache` imports BHR caches by matching passage text to canonical IDs. Directed paths join triples by normalized entity identity; `aliases` exports auxiliary neighbors at cosine similarity 0.8 or above.

## Training

### Pipeline Stages

```bash
rilo run --config configs/2wiki.locked.json
```

Or run selected stages:

```bash
rilo run --config configs/2wiki.locked.json --stages embed extract paths operators
rilo run --config configs/2wiki.locked.json --stages supervision proposal query
rilo run --config configs/2wiki.locked.json --stages collect controller select
rilo run --config configs/2wiki.locked.json --stages retrieve answer
rilo path-eval --config configs/2wiki.locked.json
```

Stages write to `artifacts` and `output`. Re-running a stage replaces its outputs; OpenIE resumes its cache. Operators use path-validation checkpoint selection, controllers use tuning-set utility minus cost, and proposal/query models use the final checkpoint.

### Device Placement

```bash
rilo run --config configs/2wiki.locked.json --set device=cuda:0 --set embedding_device=cuda:1 --set query_device=cuda:2 --stages supervision proposal query collect controller select retrieve
```

Devices default to `cuda:0`. The reader uses `device_map=auto` or an external endpoint:

```bash
rilo answer --config configs/2wiki.locked.json --set reader_url=http://127.0.0.1:8000/v1
```

The endpoint accepts token-ID prompts at `/completions` using the configured reader model and tokenizer. Authentication uses `RILO_API_KEY`.

### Default Settings

| Component | Settings |
|:--|:--|
| Operators | K=16, hidden width 256, depth 3, 5 epochs, batch 512 |
| Operator optimizer | AdamW, learning rate 3e-4, weight decay 1e-3, gradient clipping 1 |
| Operator objective | Endpoint retrieval + 0.5 distillation + 0.03 balance + 0.002 entropy |
| Query LoRA | Rank 16, alpha 32, q/v projections, 3,000 updates |
| Query lengths | Input 2,048 tokens; training target 96; generation 32 |
| Planning | Up to 5 pivots, beam width 4, up to 12 plans, merge threshold 0.98 |
| Retrieval | Budget 3 including initialization, top 20 per search, RRF 60, top 10 evaluation |
| Controller | Up to 24,000 additional collection searches, 2,000 updates, 10 target refreshes |
| Utility | Mean of support recall and support completeness, search cost 0.02 |
| Reader | Top 10 passages, complete chat input capped at 4,096 tokens |

## Experiments

```bash
rilo experiments --config configs/2wiki.locked.json --suite main --output experiments/2wiki/main
rilo experiments --config configs/2wiki.locked.json --suite matched96 --output experiments/2wiki/matched96
rilo experiments --config configs/2wiki.locked.json --suite codes --output experiments/2wiki/codes
```

`experiments` writes configurations and a command manifest. Add `--execute` to run them.

| Suite | Comparison | Schedule |
|:--|:--|:--|
| `main` | Rilo, Direct, Continuous, Dense, Hybrid, HyDE-style | Controller seeds 11/23/37 reuse each trained method's collection |
| `matched96` | Three supervision-matched methods, 96-token generation | Separate training, three controller seeds |
| `codes` | K=4/8/16/32 | Train each setting |
| `depth` | H=1/2/3 | Train each setting |
| `routing` | Learned, random, clustered, surface | Train each setting |
| `route_noise` | 0/0.1/0.2/0.4 | Train each setting |
| `endpoints` | Full, text-only, endpoint-only, shuffled | Train each setting |
| `planning` | Full, immediate reward, no replanning | Train each setting |
| `budget` | B=1/2/3 | Evaluate existing checkpoints |
| `stopping` | Adaptive, fixed-stopping, proposal-only fixed | Evaluate existing checkpoints |
| `merge` | Endpoint, text, none | Evaluate existing checkpoints |
| `pipelines` | Seeds 42 through 46 | Five complete training pipelines |

`fixed` selects by proposal score; `fixed_stopping` uses Q scores and consumes the full budget.

### Frozen Transfer

```bash
rilo transfer-config --source-config configs/2wiki.locked.json --target-config configs/hotpotqa.json --output configs/2wiki_to_hotpotqa.json
rilo run --config configs/2wiki_to_hotpotqa.json
```

Transfer reuses source checkpoints and settings with a target index. Bamboogle uses a 2Wiki source checkpoint.

### Corpus Scaling

```bash
rilo scale --base data/2wiki --snapshot raw/wikipedia.jsonl --size 100000 --output data/2wiki_100k
rilo run --config configs/2wiki.locked.json --set data=data/2wiki_100k --set artifacts=artifacts/2wiki_100k --set source_artifacts=artifacts/2wiki/rilo/42 --set output=runs/2wiki_100k
```

Scaling adds distinct passages in snapshot order while retaining the base corpus and questions. Retrieval uses exact FAISS CPU inner-product search.

### External Baselines

Set `native_adapter=package.module:Factory` to load an external retriever. The factory receives the configuration and exposes `index.lookup` and `retrieve(question)`. Results include `ranking`, `rankings`, `searches`, `retrieval_seconds`, `generation_calls`, `input_tokens`, and `output_tokens`. Dense and sparse index calls count separately.

## Evaluation

### Outputs

| Output | Contents |
|:--|:--|
| `retrieval.jsonl` | Rankings, candidate queries, selected actions, search counts, tokens, latency |
| `results.jsonl` | Answers, metrics, reader-visible text spans |
| `summary.json` | Aggregate retrieval and answer metrics |
| `manifest.json` | Configuration, data hashes, environment |

R@10 and SC@10 are measured before reader truncation. Answers use normalized exact match and token F1. Retrieval latency includes planning and query generation; reader latency is measured separately.

### Reports

```bash
rilo report --inputs runs/2wiki/main/rilo/11/summary.json runs/2wiki/main/rilo/23/summary.json runs/2wiki/main/rilo/37/summary.json --output reports/main
rilo paired --first runs/rilo/results.jsonl --second runs/continuous/results.jsonl --metric complete --samples 10000 --output reports/paired_sc.json
rilo reader-strata --first runs/rilo/results.jsonl --second runs/continuous/results.jsonl --output reports/reader_strata.json
```

Reports export JSON, CSV, Markdown, and PDF. Multi-seed summaries use sample standard deviation; paired bootstrap resamples question-level means across runs.

## Repository

| File | Purpose |
|:--|:--|
| `src/rilo/openie.py` | BHR extraction prompts and response parsing |
| `src/rilo/backends.py` | Frozen models, dense index, extraction cache |
| `src/rilo/data.py` | Dataset preparation and corpus scaling |
| `src/rilo/paths.py` | Directed paths and auxiliary alias candidates |
| `src/rilo/models.py` | Operators, proposals, continuous plans, value model |
| `src/rilo/training.py` | Operator, proposal, and query training |
| `src/rilo/query.py` | LoRA query realization and soft prefixes |
| `src/rilo/runtime.py` | Online planning and retrieval |
| `src/rilo/controller.py` | Transition collection, fitted Q, checkpoint selection |
| `src/rilo/evaluation.py` | Baseline interfaces, reading, and metrics |
| `src/rilo/analysis.py` | Path evaluation, statistics, and reports |
| `src/rilo/experiments.py` | Experiment configuration matrices |
| `src/rilo/cli.py` | CLI, model revision locking, run manifests |
