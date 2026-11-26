#!/usr/bin/env python3
import os
import re
import json
import glob
import ast
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
#from inferenceKit.dataset.basedataset import clean_latex
#from inferenceKit.dataset.grader import math_equal
# ----------------------------
# Config
# ----------------------------

DEFAULT_BENCHMARK = "../benchmark/math_test500_dataset.json"
DEFAULT_RESULTS_ROOT = "../results/llama-3b"
TOPK_LIST = [1,2,3,4,5,6,7,8,9,10]


# ----------------------------
# Helpers
# ----------------------------

UUID_LIKE_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer(s: Optional[str]) -> str:
    """Cheap normalization for math-string comparisons."""
    if s is None:
        return ""

    s = str(s).strip()

    # Remove common LaTeX wrappers / spacing differences
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\,", "")
    s = s.replace("\\!", "")
    s = s.replace(" ", "")

    # Normalize repeated whitespace just in case before removing all spaces
    s = re.sub(r"\s+", "", s)

    # Strip enclosing $ if present
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1]

    return s


def is_obviously_garbage_answer(ans) -> bool:
    # allow either string candidate or dict candidate
    if isinstance(ans, dict):
        ans = ans.get("answer", "")  # or "ans" depending on your schema
    if ans is None:
        return True

    ans = str(ans).strip()
    if not ans:
        return True
    if UUID_LIKE_RE.match(ans):
        return True
    return False

def answers_match(pred: str, gold: str) -> bool:
    #from inferenceKit.dataset.basedataset import clean_latex
    #from inferenceKit.dataset.grader import math_equal
    if pred is None:
        return False
    pred = str(pred).strip()
    gold = str(gold).strip()

    if is_obviously_garbage_answer(pred):
        return False

    # SSDP-style cleaning + robust math equivalence
    #try:
        #pred_c = clean_latex(pred)
        #gold_c = clean_latex(gold)
        #return bool(math_equal(pred_c, gold_c))
    #except Exception:
        # If their sympy/latex parsing throws, fall back to cheap compare
    return normalize_answer(pred) == normalize_answer(gold)

def find_trace_dirs(job_id: str, results_root: Path) -> List[Path]:
    """
    Find all *_traces directories matching the job_id under results_root.
    """
    pattern = str(results_root / "**" / f"{job_id}*.gov_traces")
    matches = sorted(glob.glob(pattern, recursive=True))

    if not matches:
        raise FileNotFoundError(
            f"No trace directory found for job_id={job_id} under {results_root}"
        )

    return [Path(m) for m in matches]
def infer_method_from_trace_dir(trace_dir: Path) -> str:
    s = str(trace_dir)
    if "DPTS_" in s or "/DPTS_" in s:
        return "DPTS"
    if "SSDP_" in s or "/SSDP_" in s:
        return "SSDP"
    return "UNKNOWN"


def infer_model_from_trace_dir(trace_dir: Path) -> str:
    """
    Infer model tag expected in runlog basename, e.g. llama-3b or llama-8b.
    Adjust if your trace dir naming differs.
    """
    s = str(trace_dir).lower()

    if "llama-3b" in s or "llama_3b" in s or "llama-3.2-3b" in s:
        return "llama-3b"
    if "llama-8b" in s or "llama_8b" in s or "llama-3.1-8b" in s:
        return "llama-8b"

    # fallback: try immediate parent names too
    for part in trace_dir.parts:
        p = part.lower()
        if "llama-3b" in p or "llama_3b" in p:
            return "llama-3b"
        if "llama-8b" in p or "llama_8b" in p:
            return "llama-8b"

    return "UNKNOWN"


def find_runlog_for_trace(job_id: str, trace_dir: Path) -> Optional[Path]:
    """
    Match the correct runlog for this trace_dir using:
      - PBS job id
      - method (DPTS/SSDP)
      - model (llama-3b/llama-8b)

    Assumes runlogs live at: <repo>/scripts/runlogs
    where <repo> is the SSDP repo root.
    """
    method = infer_method_from_trace_dir(trace_dir)
    model = infer_model_from_trace_dir(trace_dir)

    repo_root = Path(__file__).resolve().parent.parent
    runlogs_dir = repo_root / "scripts" / "runlogs"

    if not runlogs_dir.exists():
        return None

    candidates = sorted(runlogs_dir.glob(f"*{job_id}*.log"))
    if not candidates:
        return None

    filtered = []
    for p in candidates:
        name = p.name.lower()
        if method.lower() not in name:
            continue
        if model != "UNKNOWN" and model.lower() not in name:
            continue
        filtered.append(p)

    if len(filtered) == 1:
        return filtered[0]

    if len(filtered) > 1:
        # usually rank0/rank1/etc. if duplicates somehow remain,
        # pick the largest file because the evaluation line is often there
        filtered.sort(key=lambda x: x.stat().st_size, reverse=True)
        return filtered[0]

    return None


def parse_evaluation_result_from_log(log_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse the LAST occurrence of:
      Evaluation Result: {'accuracy': 0.502, ...}
    """
    eval_line_re = re.compile(r"Evaluation Result:\s*(\{.*\})")
    last_match = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = eval_line_re.search(line)
            if m:
                last_match = m.group(1).strip()

    if last_match is None:
        return None

    try:
        # safer than json.loads because the log uses Python dict repr with single quotes
        return ast.literal_eval(last_match)
    except Exception:
        return None

def build_gold_map(benchmark_path: Path) -> Dict[int, str]:
    data = load_json(benchmark_path)
    gold = {}
    for row in data:
        idx = int(row["index"])
        gold[idx] = row["label"]
    return gold


def read_candidates(candidates_path: Path) -> Dict[int, List[str]]:
    rows = load_jsonl(candidates_path)
    out: Dict[int, List[str]] = {}

    for row in rows:
        if row.get("event") != "candidates":
            continue
        pid = int(row["problem_id"])
        cands_raw = row.get("candidates", [])

        # Keep only dict candidates (the new schema). If older schema, wrap into dicts.
        cands_dicts = []
        for c in cands_raw:
            if isinstance(c, dict):
                cands_dicts.append(c)
            else:
                # old schema: string answer only
                cands_dicts.append({"k": 10**9, "answer": c})

        # Filter garbage based on the answer field
        cands_dicts = [c for c in cands_dicts if not is_obviously_garbage_answer(c.get("answer", ""))]

        # Sort by k if present
        cands_dicts.sort(key=lambda c: c.get("k", 10**9))

        # Return ordered list of answers
        out[pid] = [c.get("answer", "") for c in cands_dicts]

    return out

def read_trace_summary(trace_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Extract per-problem summary stats from search_stop events.

    Returns:
        problem_id -> {
            total_generated_tokens,
            tokens_kept_in_final_path,
            rollouts_done,
            rollout_iters_target,
            stop_reason
        }
    """
    rows = load_jsonl(trace_path)
    out: Dict[int, Dict[str, Any]] = {}

    for row in rows:
        if row.get("event") != "search_stop":
            continue

        meta = row.get("meta", {}) or {}
        pid = meta.get("problem_id")
        if pid is None:
            continue
        pid = int(pid)

        # Prefer the "full" search_stop row that contains total_generated_tokens
        if "total_generated_tokens" in row:
            out[pid] = {
                "total_generated_tokens": int(row.get("total_generated_tokens", 0)),
                "tokens_kept_in_final_path": int(row.get("tokens_kept_in_final_path", 0)),
                "rollouts_done": int(row.get("rollouts_done", 0)),
                "rollout_iters_target": int(row.get("rollout_iters_target", 0)),
                "stop_reason": str(row.get("stop_reason", "")),
            }
        else:
            # If only the short stop row exists, keep at least stop_reason
            if pid not in out:
                out[pid] = {
                    "total_generated_tokens": 0,
                    "tokens_kept_in_final_path": 0,
                    "rollouts_done": 0,
                    "rollout_iters_target": 0,
                    "stop_reason": str(row.get("stop_reason", "")),
                }

    return out

def summarize_one_job(
    job_id: str,
    trace_dir: Path,
    benchmark_path: Path,
) -> Dict[str, Any]:
    gold_map = build_gold_map(benchmark_path)
    method = infer_method_from_trace_dir(trace_dir)
    model = infer_model_from_trace_dir(trace_dir)
    runlog_path = find_runlog_for_trace(job_id, trace_dir)
    evaluation_result = parse_evaluation_result_from_log(runlog_path) if runlog_path else None

    candidates_file = trace_dir / "candidates_all_problems.jsonl"
    if not candidates_file.exists():
        raise FileNotFoundError(f"Missing {candidates_file}")

    trace_candidates = [
        trace_dir / "dpts_trace_all_problems.jsonl",
        trace_dir / "ssdp_trace_all_problems.jsonl",
    ]
    trace_file = None
    for p in trace_candidates:
        if p.exists():
            trace_file = p
            break

    if trace_file is None:
        raise FileNotFoundError(
            f"Missing dpts_trace_all_problems.jsonl or ssdp_trace_all_problems.jsonl in {trace_dir}"
        )

    cand_map = read_candidates(candidates_file)
    trace_map = read_trace_summary(trace_file)

    problem_ids = sorted(set(cand_map.keys()) | set(trace_map.keys()))

    rows = []
    acc_counts = {k: 0 for k in TOPK_LIST}
    stop_reason_counts: Dict[str, int] = {}
    for pid in problem_ids:
        gold = gold_map.get(pid, "")
        cands = cand_map.get(pid, [])
        tinfo = trace_map.get(pid, {})

        stop_reason = tinfo.get("stop_reason", "")
        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1

        candidate_matches = [int(answers_match(pred, gold)) for pred in cands]

        first_match_rank = None
        for idx, m in enumerate(candidate_matches, start=1):
            if m:
                first_match_rank = idx
                break

        row = {
            "problem_id": pid,
            "gold": gold,
            "num_candidates": len(cands),
            "candidates": cands,
            "candidate_matches_gold": candidate_matches,
            "first_match_rank": first_match_rank,
            "total_generated_tokens": int(tinfo.get("total_generated_tokens", 0)),
            "tokens_kept_in_final_path": int(tinfo.get("tokens_kept_in_final_path", 0)),
            "rollouts_done": int(tinfo.get("rollouts_done", 0)),
            "rollout_iters_target": int(tinfo.get("rollout_iters_target", 0)),
            "stop_reason": stop_reason,
        }

        for k in TOPK_LIST:
            hit = any(candidate_matches[:k])
            row[f"first_{k}_correct"] = int(hit)
            acc_counts[k] += int(hit)

        rows.append(row)
    n = len(rows)
    avg_tokens = (sum(r["total_generated_tokens"] for r in rows) / n) if n else 0.0
    avg_tokens_kept = (sum(r["tokens_kept_in_final_path"] for r in rows) / n) if n else 0.0
    avg_rollouts_done = (sum(r["rollouts_done"] for r in rows) / n) if n else 0.0
    budget_stop_count = stop_reason_counts.get("budget_tokens", 0)
    early_stop_count = stop_reason_counts.get("early_stop_t_star", 0)

    acc = {f"acc@{k}": (acc_counts[k] / n if n else 0.0) for k in TOPK_LIST}

    summary = {
        "job_id": job_id,
        "method": method,
        "model": model,
        "trace_dir": str(trace_dir),
        "trace_file": trace_file.name,
        "runlog_file": str(runlog_path) if runlog_path else None,
        "evaluation_result": evaluation_result,
        "reported_accuracy": (
            float(evaluation_result["accuracy"])
            if evaluation_result and "accuracy" in evaluation_result
            else None
        ),
        "reported_total_samples": (
            int(evaluation_result["total_samples"])
            if evaluation_result and "total_samples" in evaluation_result
            else None
        ),
        "reported_correct_samples": (
            int(evaluation_result["correct_samples"])
            if evaluation_result and "correct_samples" in evaluation_result
            else None
        ),
        "reported_no_match_samples": (
            int(evaluation_result["no_match_samples"])
            if evaluation_result and "no_match_samples" in evaluation_result
            else None
        ),
        "num_problems": n,
        "avg_total_generated_tokens": avg_tokens,
        "avg_tokens_kept_in_final_path": avg_tokens_kept,
        "avg_rollouts_done": avg_rollouts_done,
        "budget_stop_count": budget_stop_count,
        "budget_stop_fraction": (budget_stop_count / n if n else 0.0),
        "early_stop_t_star_count": early_stop_count,
        "early_stop_t_star_fraction": (early_stop_count / n if n else 0.0),
        "stop_reason_counts": stop_reason_counts,
        **acc,
        "per_problem": rows,
    }
    return summary

def write_outputs(trace_dir: Path, summary: Dict[str, Any]) -> None:
    json_path = trace_dir / "job_summary_topk_tokens.json"
    txt_path = trace_dir / "job_summary_topk_tokens.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Job ID: {summary['job_id']}\n")
        f.write(f"Method: {summary['method']}\n")
        f.write(f"Trace dir: {summary['trace_dir']}\n")
        f.write(f"Trace file: {summary['trace_file']}\n")
        f.write(f"Runlog file: {summary.get('runlog_file')}\n")
        f.write(f"Model: {summary.get('model')}\n")

        if summary.get("evaluation_result") is not None:
            f.write(
                "Reported evaluation result: "
                f"{summary['evaluation_result']}\n"
            )
            f.write(f"Reported accuracy: {summary.get('reported_accuracy'):.4f}\n")
            f.write(f"Reported total_samples: {summary.get('reported_total_samples')}\n")
            f.write(f"Reported correct_samples: {summary.get('reported_correct_samples')}\n")
            f.write(f"Reported no_match_samples: {summary.get('reported_no_match_samples')}\n")
        else:
            f.write("Reported evaluation result: NOT FOUND\n")
        f.write(f"Problems: {summary['num_problems']}\n")
        f.write(f"Average total generated tokens/problem: {summary['avg_total_generated_tokens']:.2f}\n")
        f.write(f"Average tokens kept in final path/problem: {summary['avg_tokens_kept_in_final_path']:.2f}\n")
        f.write(f"Average rollouts done/problem: {summary['avg_rollouts_done']:.2f}\n")
        f.write(f"budget_tokens stop count: {summary['budget_stop_count']}\n")
        f.write(f"budget_tokens stop fraction: {summary['budget_stop_fraction']:.4f}\n")
        f.write(f"early_stop_t_star count: {summary['early_stop_t_star_count']}\n")
        f.write(f"early_stop_t_star fraction: {summary['early_stop_t_star_fraction']:.4f}\n")

        f.write("\nStop reason counts:\n")
        for k, v in sorted(summary["stop_reason_counts"].items()):
            f.write(f"  {k}: {v}\n")

        f.write("\nTop-k candidate accuracy:\n")
        for k in TOPK_LIST:
            f.write(f"  acc@{k}: {summary[f'acc@{k}']:.4f}\n")

        f.write("\nPer-problem:\n")
        header = (
            "problem_id\tnum_candidates\ttotal_generated_tokens\t"
            "tokens_kept_in_final_path\trollouts_done\trollout_iters_target\tstop_reason\t"
            + "\t".join([f"first_{k}_correct" for k in TOPK_LIST])
            + "\tgold\n"
        )
        f.write(header)

        f.write("\n\nDetailed candidates by problem:\n")

        for row in summary["per_problem"]:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Problem {row['problem_id']}\n")
            f.write(f"Gold: {row['gold']}\n")
            f.write(f"Num candidates: {row['num_candidates']}\n")
            f.write(f"First match rank: {row['first_match_rank']}\n")
            f.write(f"Total generated tokens: {row['total_generated_tokens']}\n")
            f.write(f"Tokens kept in final path: {row['tokens_kept_in_final_path']}\n")
            f.write(f"Rollouts done: {row['rollouts_done']}\n")
            f.write(f"Stop reason: {row['stop_reason']}\n")
            f.write("Candidates:\n")

            for i, (cand, match) in enumerate(
                zip(row["candidates"], row["candidate_matches_gold"]), start=1
            ):
                f.write(f"  [{i}] match={match} :: {cand}\n")
def main():
    parser = argparse.ArgumentParser(
        description="Summarize DPTS/SSDP job trace directories by job_id."
    )
    parser.add_argument(
        "--job-ids",
        nargs="+",
        required=True,
        help="One or more PBS job IDs, e.g. 6950854 6951001 6951002",
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root results directory (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help=f"Benchmark JSON file (default: {DEFAULT_BENCHMARK})",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    benchmark_path = Path(args.benchmark).resolve()

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    all_jobs_summary = []
    for job_id in args.job_ids:
        trace_dirs = find_trace_dirs(job_id, results_root)

        for trace_dir in trace_dirs:
            summary = summarize_one_job(job_id, trace_dir, benchmark_path)
            write_outputs(trace_dir, summary)
            all_jobs_summary.append(summary)

            print(f"[OK] Wrote summary files in: {trace_dir}")
    # Also write one combined file at results root for convenience
    combined_path = results_root / "combined_job_summaries.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(all_jobs_summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote combined summary: {combined_path}")


if __name__ == "__main__":
    main()
