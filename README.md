# Retrieval-VM-bench

PoC benchmark for iterative retrieval tuning with an LLM.

- data source: MS MARCO passages/questions
- evaluation: hidden split vs TF-IDF baseline
- optimization loop: task prompt -> code update -> evaluation -> retry

## Note

This repository is intended for prompt-testing experiments.
Service code was AI-generated to speed up testing of human-designed prompts.
It requires human review and further engineering before production use.

## How it works

1. `src/run_llm_loop.py` prepares data (if needed) and loads prompts.
2. The model updates `workspace/solution.py` (full code or search/replace edits).
3. `src/evaluator.py` evaluates baseline and candidate retrieval on hidden queries.
4. If the result is not good enough, the loop retries up to `--max-iters`.

Pass condition on hidden split:
- `mrr@10` >= baseline + `0.05`
- `ndcg@10` >= baseline + `0.05`

## Requirements

- Python `>=3.14` (as defined in `pyproject.toml`)
- `uv` package manager
- OpenAI API key in `.env`

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Create `.env` from `.env.example`:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Run

Basic run:

```bash
uv run python src/run_llm_loop.py --model gpt-4.1-mini --max-iters 3
```

Custom output solution path:

```bash
uv run python src/run_llm_loop.py --solution-path workspace/solution.py
```

The script prints iteration-by-iteration JSON and final status to stdout.

## Output and artifacts

- generated/updated retriever code: `workspace/solution.py`
- local dataset split files: `dataset/corpus.json`, `dataset/queries_public.json`, `dataset/qrels_public.json`, `dataset/queries_hidden.json`, `dataset/qrels_hidden.json`
- evaluation output includes:
  - `status` (`passed` or `failed`)
  - `baseline_score` (`mrr@10`, `ndcg@10`)
  - `search_score` (`mrr@10`, `ndcg@10`)
  - examples of successful and failed queries

## Important files

- `src/run_llm_loop.py` - main loop entrypoint
- `src/evaluator.py` - evaluator and pass/fail logic
- `src/prepare_ms_marco_data.py` - builds local MS MARCO dataset in `dataset`
- `prompts/retrieval_task.md` - first-step coding task prompt
- `prompts/judge_prompt.md` - feedback/refinement prompt
- `.env.example` - required environment variables
