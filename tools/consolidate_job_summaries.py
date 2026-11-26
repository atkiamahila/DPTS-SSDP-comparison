#!/usr/bin/env python3
import json
import glob
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TOPK_KEEP = [1, 2, 3, 4, 5, 6,7,8,9, 10]

# -------------------------------------------------
# Explicit job_id -> budget mapping from your notes
# -------------------------------------------------

JOB_TO_BUDGET = {
    "math": {
        "6957015": 3000,
        "6960340": 5000,
        "6961480": 8000,
        "6962936": 10000,
    },
    "gsm": {
        "6956560": 3000,   # general GSM 3000
        "6956717": 3000,   # special: 8b DPTS only
        "6958666": 5000,
        "6957664": 8000,
        "6958465": 10000,
    },
}

DATASETS = ["math", "gsm"]
MODELS = ["llama-3b", "llama-8b"]
DEFAULT_BUDGETS = [3000, 5000, 8000, 10000]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_summary_files(results_root: Path) -> List[Path]:
    pattern = str(results_root / "**" / "job_summary_topk_tokens.json")
    return sorted(Path(p) for p in glob.glob(pattern, recursive=True))


def infer_model(summary: Dict[str, Any], summary_path: Path) -> str:
    for key in ["model", "model_name", "model_tag"]:
        v = summary.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()

    text = f"{summary_path} || {summary.get('trace_dir', '')}".lower()

    if "llama-3b" in text or "llama_3b" in text:
        return "llama-3b"
    if "llama-8b" in text or "llama_8b" in text:
        return "llama-8b"

    return "unknown"


def infer_dataset(summary: Dict[str, Any], summary_path: Path) -> str:
    for key in ["dataset", "data", "benchmark_name"]:
        v = summary.get(key)
        if isinstance(v, str) and v.strip():
            vv = v.strip().lower()
            if "gsm" in vv:
                return "gsm"
            if "math" in vv:
                return "math"

    text = f"{summary_path} || {summary.get('trace_dir', '')}".lower()

    if "gsm8k" in text or "gsm" in text:
        return "gsm"
    if "math" in text:
        return "math"

    return "unknown"


def get_reported_accuracy(summary: Dict[str, Any]) -> Optional[float]:
    if isinstance(summary.get("reported_accuracy"), (int, float)):
        return float(summary["reported_accuracy"])

    ev = summary.get("evaluation_result")
    if isinstance(ev, dict) and isinstance(ev.get("accuracy"), (int, float)):
        return float(ev["accuracy"])

    return None


def get_topk(summary: Dict[str, Any], k: int) -> Optional[float]:
    v = summary.get(f"acc@{k}")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def get_oom_info(summary: Dict[str, Any]) -> Tuple[bool, int]:
    src = summary.get("stop_reason_counts", {})
    if not isinstance(src, dict):
        return False, 0

    raw = src.get("oom_error", 0)
    try:
        cnt = int(raw)
    except Exception:
        cnt = 0
    return cnt > 0, cnt


def format_pct(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:.2f}%"


def format_num(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:,.0f}"


def make_topk_list_str(summary: Dict[str, Any]) -> str:
    vals = []
    for k in TOPK_KEEP:
        v = get_topk(summary, k)
        vals.append("NA" if v is None else f"{100.0 * v:.2f}")
    return "[" + ", ".join(vals) + "]"


def pick_best_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = sorted(
        candidates,
        key=lambda x: (
            x["reported_accuracy"] is not None,
            int(x.get("num_problems") or 0),
            x.get("avg_total_generated_tokens") is not None,
        ),
        reverse=True,
    )
    return candidates[0]


def collect_rows(results_root: Path, budgets: List[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    skipped = []

    for sf in find_summary_files(results_root):
        try:
            summary = load_json(sf)
        except Exception as e:
            skipped.append({"summary_path": str(sf), "reason": f"json_load_failed: {e}"})
            continue

        job_id = str(summary.get("job_id", "")).strip()
        if not job_id:
            skipped.append({"summary_path": str(sf), "reason": "missing_job_id"})
            continue

        dataset = infer_dataset(summary, sf)
        if dataset not in JOB_TO_BUDGET:
            skipped.append({"summary_path": str(sf), "reason": f"unknown_dataset:{dataset}"})
            continue

        budget_map = JOB_TO_BUDGET[dataset]
        if job_id not in budget_map:
            skipped.append({"summary_path": str(sf), "reason": f"job_id_not_in_{dataset}_mapping"})
            continue

        budget = budget_map[job_id]
        if budget not in budgets:
            skipped.append({"summary_path": str(sf), "reason": f"budget_not_requested:{budget}"})
            continue

        model = infer_model(summary, sf)
        method = str(summary.get("method", "UNKNOWN")).strip().upper()

        has_oom, oom_count = get_oom_info(summary)

        row = {
            "summary_path": sf,
            "trace_dir": summary.get("trace_dir"),
            "job_id": job_id,
            "dataset": dataset,
            "model": model,
            "budget": budget,
            "method": method,
            "reported_accuracy": get_reported_accuracy(summary),
            "avg_total_generated_tokens": summary.get("avg_total_generated_tokens"),
            "num_problems": summary.get("num_problems"),
            "topk_list_str": make_topk_list_str(summary),
            "has_oom_error": has_oom,
            "oom_error_count": oom_count,
        }
        rows.append(row)

    grouped: Dict[Tuple[str, str, int, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["dataset"], r["model"], r["budget"], r["method"])
        grouped.setdefault(key, []).append(r)

    final_rows = [pick_best_candidate(cands) for cands in grouped.values()]
    final_rows = sorted(final_rows, key=lambda x: (x["dataset"], x["model"], x["budget"], x["method"]))

    return final_rows, skipped


def render_section(rows: List[Dict[str, Any]], dataset: str, model: str, budgets: List[int]) -> str:
    out = []
    out.append("=" * 100)
    out.append(f"Dataset: {dataset} | Model: {model}")
    out.append("=" * 100)
    out.append("")
    out.append("| Budget | Method | Reported Accuracy | Avg. Tokens | Top-k Accuracy [1,2,3,4,5,6,10] | OOM Error? | OOM Count | Job ID |")
    out.append("|---:|---|---:|---:|---|---|---:|---:|")

    section_rows = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
    section_rows = sorted(section_rows, key=lambda x: (x["budget"], x["method"]))

    if not section_rows:
        out.append("| - | - | NA | NA | NA | NA | NA | NA |")
        out.append("")
        return "\n".join(out)

    for budget in budgets:
        subset = [r for r in section_rows if r["budget"] == budget]
        subset = sorted(subset, key=lambda x: x["method"])

        if not subset:
            out.append(f"| {budget} | - | NA | NA | NA | NA | NA | NA |")
            continue

        for r in subset:
            out.append(
                f"| {r['budget']} | {r['method']} | {format_pct(r['reported_accuracy'])} | "
                f"{format_num(r['avg_total_generated_tokens'])} | {r['topk_list_str']} | "
                f"{'yes' if r['has_oom_error'] else 'no'} | {r['oom_error_count']} | {r['job_id']} |"
            )

    out.append("")
    return "\n".join(out)


def render_full_report(rows: List[Dict[str, Any]], budgets: List[int]) -> str:
    parts = []
    parts.append("# Consolidated Results Across All Datasets and Models")
    parts.append("")
    parts.append(f"Budgets included: {budgets}")
    parts.append("")

    for dataset in DATASETS:
        for model in MODELS:
            parts.append(render_section(rows, dataset, model, budgets))

    parts.append("=" * 100)
    parts.append("OOM Summary")
    parts.append("=" * 100)

    any_oom = False
    for dataset in DATASETS:
        for model in MODELS:
            subset = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
            subset = [r for r in subset if r["has_oom_error"]]
            if subset:
                any_oom = True
                parts.append(f"- {dataset} / {model}:")
                for r in sorted(subset, key=lambda x: (x["budget"], x["method"])):
                    parts.append(
                        f"  budget={r['budget']}, method={r['method']}, "
                        f"oom_count={r['oom_error_count']}, job_id={r['job_id']}"
                    )

    if not any_oom:
        parts.append("No summaries reported oom_error.")

    parts.append("")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Write one combined consolidated report across both datasets and both models."
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="Root containing job_summary_topk_tokens.json files, e.g. ./results",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
        help="Budgets to include, default: 3000 5000 8000 10000",
    )
    parser.add_argument(
        "--out-name",
        default="consolidated_all_results",
        help="Base filename to write in tools/, without extension",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    script_dir = Path(__file__).resolve().parent

    rows, skipped = collect_rows(results_root, args.budgets)

    out_txt = script_dir / f"{args.out_name}.txt"
    out_md = script_dir / f"{args.out_name}.md"
    out_json = script_dir / f"{args.out_name}.json"
    out_skipped = script_dir / f"{args.out_name}_skipped.json"

    report_text = render_full_report(rows, args.budgets)

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(report_text)

    with out_md.open("w", encoding="utf-8") as f:
        f.write(report_text)

    json_ready = []
    for r in rows:
        d = dict(r)
        d["summary_path"] = str(d["summary_path"])
        json_ready.append(d)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)

    with out_skipped.open("w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote: {out_txt}")
    print(f"[OK] Wrote: {out_md}")
    print(f"[OK] Wrote: {out_json}")
    print(f"[OK] Wrote: {out_skipped}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
