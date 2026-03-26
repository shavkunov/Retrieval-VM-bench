from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from evaluator import evaluate_solution

DATA_DIR = Path("dataset")


def starter_solution() -> str:
    return """import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_base_vectorizer = None
_base_doc_ids = None
_base_matrix = None

def _init_baseline(corpus: dict[str, str]):
    global _base_vectorizer, _base_doc_ids, _base_matrix
    _base_doc_ids = list(corpus.keys())
    docs = [corpus[doc_id] for doc_id in _base_doc_ids]
    _base_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        smooth_idf=True,
        max_features=80000,
        min_df=1,
        max_df=0.95,
    )
    _base_matrix = _base_vectorizer.fit_transform(docs)

def baseline_retrieve(query: str, top_k: int, corpus: dict[str, str]) -> list[str]:
    global _base_vectorizer, _base_doc_ids, _base_matrix
    if _base_vectorizer is None or _base_doc_ids is None or _base_matrix is None:
        _init_baseline(corpus)
    query_vec = _base_vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _base_matrix).flatten()
    order = np.argsort(-scores)
    out = []
    seen = set()
    for idx in order:
        doc_id = _base_doc_ids[idx]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
        if len(out) == top_k:
            break
    if len(out) < top_k:
        for doc_id in _base_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
            if len(out) == top_k:
                break
    return out

def retrieve(query: str, top_k: int, corpus: dict[str, str]) -> list[str]:
    return baseline_retrieve(query, top_k, corpus)
"""


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_data() -> None:
    if (DATA_DIR / "corpus.json").exists():
        return
    print("Data not found, preparing MS MARCO split...")
    subprocess.check_call([sys.executable, "src/prepare_ms_marco_data.py"])


def extract_code(text: str) -> str:
    match = re.search(
        r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip() + "\n"
    return ""


def extract_edit_blocks(text: str) -> list[tuple[str, str]]:
    edits = []
    blocks = re.findall(r"```edit\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        if "SEARCH" not in block or "REPLACE" not in block:
            continue
        before, after = block.split("REPLACE", 1)
        search = before.split("SEARCH", 1)[1].strip("\n")
        replace = after.strip("\n")
        if search:
            edits.append((search, replace))
    return edits


def apply_edits(current_solution: str, edits: list[tuple[str, str]]) -> tuple[str, int]:
    updated = current_solution
    applied = 0
    for search, replace in edits:
        count = updated.count(search)
        if count != 1:
            continue
        updated = updated.replace(search, replace, 1)
        applied += 1
    return updated, applied


def extract_tool_commands(text: str) -> list[str]:
    commands = []
    for block in re.findall(r"```tool\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        for line in block.splitlines():
            cmd = line.strip()
            if cmd:
                commands.append(cmd)
    return commands


def _safe_package_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", name))


def _clamp_slice(limit: int, offset: int) -> tuple[int, int]:
    limit = max(1, min(limit, 30))
    offset = max(0, offset)
    return limit, offset


def _cmd_to_corpus_slice(cmd: str) -> tuple[int, int] | None:
    if cmd.startswith("read_corpus"):
        parts = cmd.split()
        if len(parts) == 2 and parts[1].isdigit():
            return _clamp_slice(int(parts[1]), 0)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            return _clamp_slice(int(parts[1]), int(parts[2]))
        return None

    if cmd.startswith("{") and cmd.endswith("}"):
        try:
            payload = json.loads(cmd)
        except json.JSONDecodeError:
            return None
        if payload.get("tool") != "read_corpus":
            return None
        limit = int(payload.get("limit", 8))
        offset = int(payload.get("offset", 0))
        return _clamp_slice(limit, offset)
    return None


def read_corpus_slice(limit: int, offset: int) -> str:
    corpus = load_json(DATA_DIR / "corpus.json")
    items = list(corpus.items())
    chunk = dict(items[offset : offset + limit])
    payload = {"offset": offset, "limit": limit, "rows": len(chunk), "data": chunk}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _cmd_to_uv_args(cmd: str) -> list[str] | None:
    if cmd.startswith("uv add "):
        args = shlex.split(cmd)
        if len(args) >= 3 and all(_safe_package_name(x) for x in args[2:]):
            return args
        return None

    if cmd.startswith("install_package "):
        package = cmd[len("install_package ") :].strip()
        if _safe_package_name(package):
            return ["uv", "add", package]
        return None

    if cmd.startswith("{") and cmd.endswith("}"):
        try:
            payload = json.loads(cmd)
        except json.JSONDecodeError:
            return None
        if payload.get("tool") != "install_package":
            return None
        package = str(payload.get("package", "")).strip()
        if _safe_package_name(package):
            return ["uv", "add", package]
    return None


def run_tool_commands(text: str) -> None:
    commands = extract_tool_commands(text)
    for cmd in commands:
        args = _cmd_to_uv_args(cmd)
        if args is None:
            continue
        subprocess.check_call(args)


def run_read_tools(text: str) -> str:
    outputs = []
    commands = extract_tool_commands(text)
    for cmd in commands:
        corpus_slice = _cmd_to_corpus_slice(cmd)
        if corpus_slice is None:
            continue
        limit, offset = corpus_slice
        outputs.append(read_corpus_slice(limit=limit, offset=offset))
    return "\n\n".join(outputs)


def ensure_solution_template(solution_path: Path) -> None:
    required_signature = "def retrieve(query: str, top_k: int, corpus: dict[str, str])"
    if not solution_path.exists():
        solution_path.write_text(starter_solution(), encoding="utf-8")
        return
    content = solution_path.read_text(encoding="utf-8")
    if "def baseline_retrieve(" not in content or required_signature not in content:
        solution_path.write_text(starter_solution(), encoding="utf-8")


def merge_with_baseline(current_solution: str, new_code: str) -> str:
    if "def baseline_retrieve(" in new_code:
        return new_code
    if "def baseline_retrieve(" in current_solution and "def retrieve(" in new_code:
        return current_solution + "\n\n" + new_code
    return new_code


def build_public_snapshot() -> str:
    corpus = load_json(DATA_DIR / "corpus.json")
    queries = load_json(DATA_DIR / "queries_public.json")
    qrels = load_json(DATA_DIR / "qrels_public.json")
    payload = {
        "corpus_sample": dict(list(corpus.items())[:8]),
        "queries_sample": dict(list(queries.items())[:20]),
        "qrels_sample": {k: v for k, v in list(qrels.items())[:20]},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_solver_step(
    llm: ChatOpenAI, prompt_text: str, public_snapshot: str, current_solution: str
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write practical retrieval code. Prefer editing existing file with exact search/replace patches.",
            ),
            (
                "human",
                "{task_prompt}\n\nPublic data snapshot:\n{public_snapshot}\n\nCurrent solution.py:\n```python\n{solution}\n```\n\nPreferred output format:\n```edit\nSEARCH\n<exact old snippet>\nREPLACE\n<new snippet>\n```\nYou may return multiple edit blocks. If needed, you can still return full python file in one ```python``` block.",
            ),
        ]
    )
    response = llm.invoke(
        prompt.format_messages(
            task_prompt=prompt_text,
            public_snapshot=public_snapshot,
            solution=current_solution,
        )
    )
    return response.content if isinstance(response.content, str) else str(response.content)


def run_feedback_step(
    llm: ChatOpenAI,
    prompt_text: str,
    current_solution: str,
    evaluator_output: dict,
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You fix retrieval code based on evaluation feedback."),
            (
                "human",
                "{task_prompt}\n\nCurrent solution:\n```python\n{solution}\n```\n\nEvaluator output:\n```json\n{evaluator_output}\n```\n\nPreferred output format:\n```edit\nSEARCH\n<exact old snippet>\nREPLACE\n<new snippet>\n```\nYou may return multiple edit blocks. If needed, you can still return full python file in one ```python``` block.",
            ),
        ]
    )
    response = llm.invoke(
        prompt.format_messages(
            task_prompt=prompt_text,
            solution=current_solution,
            evaluator_output=json.dumps(evaluator_output, ensure_ascii=False, indent=2),
        )
    )
    return response.content if isinstance(response.content, str) else str(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--solution-path", default="workspace/solution.py")
    args = parser.parse_args()

    load_dotenv()
    llm = ChatOpenAI(model=args.model, temperature=0.2)
    ensure_data()

    solution_path = Path(args.solution_path)
    ensure_solution_template(solution_path)
    retrieval_task = Path("prompts/retrieval_task.md").read_text(encoding="utf-8")
    judge_prompt = Path("prompts/judge_prompt.md").read_text(encoding="utf-8")
    public_snapshot = build_public_snapshot()

    current_solution = solution_path.read_text(encoding="utf-8")
    response_text = run_solver_step(
        llm=llm,
        prompt_text=retrieval_task,
        public_snapshot=public_snapshot,
        current_solution=current_solution,
    )
    run_tool_commands(response_text)
    read_output = run_read_tools(response_text)
    if read_output:
        response_text = run_solver_step(
            llm=llm,
            prompt_text=retrieval_task + "\n\nTool output (read_corpus):\n" + read_output,
            public_snapshot=public_snapshot,
            current_solution=current_solution,
        )
        run_tool_commands(response_text)
    edits = extract_edit_blocks(response_text)
    patched, applied = apply_edits(current_solution, edits)
    if applied > 0:
        solution_path.write_text(patched, encoding="utf-8")
    else:
        code = extract_code(response_text)
        if code.strip():
            code = merge_with_baseline(current_solution=current_solution, new_code=code)
            solution_path.write_text(code, encoding="utf-8")

    result = evaluate_solution(solution_path)
    print("\n=== Iteration 1 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    for i in range(2, args.max_iters + 1):
        if result.get("status") == "passed":
            break
        current_solution = solution_path.read_text(encoding="utf-8")
        response_text = run_feedback_step(
            llm=llm,
            prompt_text=judge_prompt,
            current_solution=current_solution,
            evaluator_output=result,
        )
        run_tool_commands(response_text)
        read_output = run_read_tools(response_text)
        if read_output:
            response_text = run_feedback_step(
                llm=llm,
                prompt_text=judge_prompt + "\n\nTool output (read_corpus):\n" + read_output,
                current_solution=current_solution,
                evaluator_output=result,
            )
            run_tool_commands(response_text)
        edits = extract_edit_blocks(response_text)
        patched, applied = apply_edits(current_solution, edits)
        if applied > 0:
            solution_path.write_text(patched, encoding="utf-8")
        else:
            code = extract_code(response_text)
            if code.strip():
                code = merge_with_baseline(current_solution=current_solution, new_code=code)
                solution_path.write_text(code, encoding="utf-8")
        result = evaluate_solution(solution_path)
        print(f"\n=== Iteration {i} ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== Final status ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
