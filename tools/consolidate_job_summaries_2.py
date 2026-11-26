#!/usr/bin/env python3
import re
import ast
import json
import glob
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TOPK_LIST = list(range(1, 11))
BUDGETS = [3000, 5000, 8000, 10000]
DATASETS = ["math", "gsm"]
MODELS = ["llama-3b", "llama-8b"]

# -------------------------------------------------
# Explicit mapping from your notes
# -------------------------------------------------
JOB_TO_BUDGET = {
    "math": {
        "6957015": 3000,
        "6960340": 5000,
        "6961480": 8000,
        "6962936": 10000,
    },
    "gsm": {
        "6956560": 3000,
        "6956717": 3000,   # special 8b DPTS 3000
        "6958666": 5000,
        "6957664": 8000,
        "7100414": 8000,
        "6958465": 10000,
        "7098282": 10000,
    },
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_summary_files(results_root: Path) -> List[Path]:
    return sorted(Path(p) for p in glob.glob(
        str(results_root / "**" / "job_summary_topk_tokens.json"),
        recursive=True
    ))


def extract_job_id(summary: Dict[str, Any], summary_path: Path) -> Optional[str]:
    if summary.get("job_id"):
        return str(summary["job_id"]).strip()

    text = f"{summary_path} {summary.get('trace_dir', '')}"
    m = re.search(r"(\d+)\.polaris-pbs", text)
    if m:
        return m.group(1)

    m = re.search(r"/(\d+)[^/]*_traces", text)
    if m:
        return m.group(1)

    return None


def infer_dataset_from_job(job_id: str) -> str:
    for dataset, mp in JOB_TO_BUDGET.items():
        if job_id in mp:
            return dataset
    return "unknown"


def infer_budget_from_job(dataset: str, job_id: str) -> Optional[int]:
    return JOB_TO_BUDGET.get(dataset, {}).get(job_id)


def infer_model_from_summary(summary: Dict[str, Any], summary_path: Path) -> str:
    if summary.get("model"):
        m = str(summary["model"]).lower()
        if "3b" in m:
            return "llama-3b"
        if "8b" in m:
            return "llama-8b"

    text = f"{summary_path} {summary.get('trace_dir', '')}".lower()
    if "llama-3b" in text or "llama_3b" in text:
        return "llama-3b"
    if "llama-8b" in text or "llama_8b" in text:
        return "llama-8b"

    return "unknown"


def infer_method_from_summary(summary: Dict[str, Any], summary_path: Path) -> str:
    if summary.get("method"):
        return str(summary["method"]).upper()

    text = f"{summary_path} {summary.get('trace_dir', '')}".lower()
    if "dpts" in text:
        return "DPTS"
    if "ssdp" in text:
        return "SSDP"

    return "UNKNOWN"


def find_runlog_for_summary(summary: Dict[str, Any], summary_path: Path, repo_root: Path) -> Optional[Path]:
    job_id = extract_job_id(summary, summary_path)
    if not job_id:
        return None

    runlogs_dir = repo_root / "scripts" / "runlogs"
    candidates = sorted(runlogs_dir.glob(f"*{job_id}*.log"))
    if not candidates:
        return None

    model = infer_model_from_summary(summary, summary_path)
    method = infer_method_from_summary(summary, summary_path)

    filtered = []
    for p in candidates:
        name = p.name.lower()
        if method.lower() not in name:
            continue
        if model != "unknown" and model.lower() not in name:
            continue
        filtered.append(p)

    if len(filtered) == 1:
        return filtered[0]

    if len(filtered) > 1:
        filtered.sort(key=lambda x: x.stat().st_size, reverse=True)
        return filtered[0]

    candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
    return candidates[0]


def parse_evaluation_result_from_log(log_path: Path, chunk_size: int = 256 * 1024) -> Optional[Dict[str, Any]]:
    eval_re = re.compile(r"Evaluation Result:\s*(\{.*?\})")

    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            read_size = min(chunk_size, file_size)
            f.seek(-read_size, 2)
            data = f.read(read_size)
    except Exception:
        return None

    text = data.decode("utf-8", errors="replace")
    matches = eval_re.findall(text)
    if not matches:
        return None

    try:
        return ast.literal_eval(matches[-1])
    except Exception:
        return None


def get_acc(summary: Dict[str, Any], k: int) -> Optional[float]:
    v = summary.get(f"acc@{k}")
    return float(v) if isinstance(v, (int, float)) else None


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{float(x):,.0f}"


def topk_list_str(summary: Dict[str, Any]) -> str:
    vals = []
    for k in TOPK_LIST:
        v = get_acc(summary, k)
        vals.append("NA" if v is None else f"{100.0 * v:.2f}")
    return "[" + ", ".join(vals) + "]"


def clean_stop_reason_counts(summary: Dict[str, Any]) -> Dict[str, int]:
    raw = summary.get("stop_reason_counts", {})
    if not isinstance(raw, dict):
        return {}

    out = {}
    for k, v in raw.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            out[str(k)] = 0
    return out


def stop_reasons_str(stop_counts: Dict[str, int]) -> str:
    if not stop_counts:
        return "{}"
    items = [f"{k}:{v}" for k, v in sorted(stop_counts.items())]
    return "{" + ", ".join(items) + "}"


def build_row(summary_path: Path, summary: Dict[str, Any], repo_root: Path) -> Optional[Dict[str, Any]]:
    job_id = extract_job_id(summary, summary_path)
    if not job_id:
        return None

    dataset = infer_dataset_from_job(job_id)
    budget = infer_budget_from_job(dataset, job_id)
    model = infer_model_from_summary(summary, summary_path)
    method = infer_method_from_summary(summary, summary_path)

    runlog_path = find_runlog_for_summary(summary, summary_path, repo_root)
    ev = parse_evaluation_result_from_log(runlog_path) if runlog_path else None

    reported_accuracy = None
    total_samples = None
    correct_samples = None
    no_match_samples = None

    if isinstance(ev, dict):
        if isinstance(ev.get("accuracy"), (int, float)):
            reported_accuracy = float(ev["accuracy"])
        if isinstance(ev.get("total_samples"), (int, float)):
            total_samples = int(ev["total_samples"])
        if isinstance(ev.get("correct_samples"), (int, float)):
            correct_samples = int(ev["correct_samples"])
        if isinstance(ev.get("no_match_samples"), (int, float)):
            no_match_samples = int(ev["no_match_samples"])

    if reported_accuracy is None and isinstance(summary.get("reported_accuracy"), (int, float)):
        reported_accuracy = float(summary["reported_accuracy"])

    stop_counts = clean_stop_reason_counts(summary)
    oom_error = stop_counts.get("oom_error", 0)
    raw_budget_tokens = stop_counts.get("budget_tokens", 0)
    budget_tokens_plus_oom = raw_budget_tokens + oom_error

    # report budget_tokens as budget_tokens + oom_error
    stop_counts_reported = dict(stop_counts)
    stop_counts_reported["budget_tokens"] = budget_tokens_plus_oom

    return {
        "summary_path": str(summary_path),
        "trace_dir": summary.get("trace_dir"),
        "runlog_path": str(runlog_path) if runlog_path else None,
        "job_id": job_id,
        "dataset": dataset,
        "model": model,
        "method": method,
        "budget": budget,
        "reported_accuracy": reported_accuracy,
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "no_match_samples": no_match_samples,
        "num_problems": summary.get("num_problems"),
        "avg_total_generated_tokens": summary.get("avg_total_generated_tokens"),
        "avg_tokens_kept_in_final_path": summary.get("avg_tokens_kept_in_final_path"),
        "avg_rollouts_done": summary.get("avg_rollouts_done"),
        "raw_budget_tokens_count": raw_budget_tokens,
        "oom_error_count": oom_error,
        "budget_tokens_count": budget_tokens_plus_oom,
        "all_stop_reason_counts": stop_counts_reported,
        "all_stop_reason_counts_raw": stop_counts,
        "all_stop_reasons_str": stop_reasons_str(stop_counts_reported),
        "topk_list": topk_list_str(summary),
        **{f"acc@{k}": get_acc(summary, k) for k in TOPK_LIST},
    }


def collect_rows(results_root: Path, repo_root: Path) -> List[Dict[str, Any]]:
    rows = []

    for sf in find_summary_files(results_root):
        try:
            summary = load_json(sf)
        except Exception as e:
            print(f"[WARN] Could not load {sf}: {e}")
            continue

        row = build_row(sf, summary, repo_root)
        if row is not None:
            rows.append(row)

    grouped: Dict[Tuple[Any, Any, Any, Any, Any], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["dataset"], r["model"], r["budget"], r["method"], r["job_id"])
        grouped.setdefault(key, []).append(r)

    final_rows = []
    for _, items in grouped.items():
        items.sort(
            key=lambda r: (
                r["reported_accuracy"] is not None,
                r["runlog_path"] is not None,
                int(r.get("num_problems") or 0),
            ),
            reverse=True,
        )
        final_rows.append(items[0])

    return sorted(final_rows, key=lambda r: (r["dataset"], r["model"], r["budget"] or 0, r["method"]))


def render_section(rows: List[Dict[str, Any]], dataset: str, model: str) -> str:
    section = [
        r for r in rows
        if r["dataset"] == dataset
        and r["model"] == model
        and r["budget"] in BUDGETS
    ]
    section.sort(key=lambda r: (r["budget"], r["method"]))

    out = []
    out.append("=" * 140)
    out.append(f"Dataset: {dataset} | Model: {model}")
    out.append("=" * 140)
    out.append("")
    out.append(
        "| Budget | Method | Reported Accuracy | Avg. Tokens | Top-k Accuracy [1..10] | "
        "No Match | budget_tokens | oom_error | All Stop Reasons | Job ID |"
    )
    out.append(
        "|---:|---|---:|---:|---|---:|---:|---:|---|---:|"
    )

    if not section:
        out.append("| - | - | NA | NA | NA | NA | NA | NA | NA | NA |")
        out.append("")
        return "\n".join(out)

    for r in section:
        out.append(
            f"| {r['budget']} | {r['method']} | {fmt_pct(r['reported_accuracy'])} | "
            f"{fmt_num(r.get('avg_total_generated_tokens'))} | {r['topk_list']} | "
            f"{r['no_match_samples'] if r['no_match_samples'] is not None else 'NA'} | "
            f"{r['budget_tokens_count']} | "
            f"{r['oom_error_count']} | "
            f"{r['all_stop_reasons_str']} | "
            f"{r['job_id']} |"
        )

    out.append("")
    return "\n".join(out)


def render_report(rows: List[Dict[str, Any]]) -> str:
    out = []
    out.append("# Consolidated Job Summaries")
    out.append("")
    out.append("Notes:")
    out.append("- Dataset and budget come from the explicit job-id mapping.")
    out.append("- Model and method come from the summary/path tag-style inference.")
    out.append("- Reported Accuracy and No Match are parsed from the matching runlog Evaluation Result line.")
    out.append("- Reported budget_tokens column is budget_tokens + oom_error.")
    out.append("- All Stop Reasons preserves every stop reason found in stop_reason_counts.")
    out.append("")

    for dataset in DATASETS:
        for model in MODELS:
            out.append(render_section(rows, dataset, model))

    out.append("=" * 140)
    out.append("OOM Summary")
    out.append("=" * 140)

    oom_rows = [r for r in rows if int(r.get("oom_error_count") or 0) > 0]
    if not oom_rows:
        out.append("No summaries reported oom_error.")
    else:
        for r in sorted(oom_rows, key=lambda x: (x["dataset"], x["model"], x["budget"] or 0, x["method"])):
            out.append(
                f"- dataset={r['dataset']}, model={r['model']}, budget={r['budget']}, "
                f"method={r['method']}, oom_error={r['oom_error_count']}, job_id={r['job_id']}"
            )

    out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="./results",
        help="Root containing job_summary_topk_tokens.json files.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="SSDP repo root. Used to find scripts/runlogs.",
    )
    parser.add_argument(
        "--out-name",
        default="consolidated_all_results",
        help="Base output filename written to tools/.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    results_root = Path(args.results_root).resolve()
    tools_dir = repo_root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(results_root, repo_root)
    report = render_report(rows)

    out_txt = tools_dir / f"{args.out_name}.txt"
    out_md = tools_dir / f"{args.out_name}.md"
    out_json = tools_dir / f"{args.out_name}.json"

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(report)

    with out_md.open("w", encoding="utf-8") as f:
        f.write(report)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote: {out_txt}")
    print(f"[OK] Wrote: {out_md}")
    print(f"[OK] Wrote: {out_json}")
    print()
    print(report)


if __name__ == "__main__":
    main()
