#!/usr/bin/env python3
"""
aggregate_ssdp_chunks.py

Aggregates the filesystem-queue chunk results written by the modified main.py.

Layout it reads (from main.py):
  RESULTS/<MODEL>/b<BUDGET>/chunks/<cid>/results-<key>.json
  RESULTS/<MODEL>/b<BUDGET>/chunks/<cid>/chunk_meta.json
  QDIR/done/<cid>            (completion markers)
  QDIR/chunks.txt            (cid budget start end)

For each budget it:
  - checks which chunks are marked done vs. present vs. missing
  - concatenates all results-<key>.json across chunks (deduped by index)
  - runs dataset.evaluate_results() per voting key
  - reports mean/median token usage (auto-detects the token field)
  - reports oracle candidate acc@1..acc@10 from candidates_all_problems.jsonl

Usage:
  cd /lus/grand/projects/VeloC/atkia/ssdp/SSDP
  python scripts/aggregate_ssdp_chunks.py \
      --results results \
      --model llama-3b \
      --qdir results/_queue/llama-3b_math \
      --data math \
      --budgets 3000 5000 8000 10000
"""
import argparse
import glob
import json
import os
import statistics
import sys
import re

TOPK_LIST = list(range(1, 11))
UUID_LIKE_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def detect_token_count(record):
    """Best-effort extraction of a per-sample token count from a result record.
    Returns an int or None. Checks common field names, then falls back to
    counting whitespace tokens of the response so we always report *something*."""
    for key in ("total_tokens", "num_tokens", "token_usage", "tokens",
                "output_tokens", "generated_tokens", "n_tokens"):
        if key in record and isinstance(record[key], (int, float)):
            return int(record[key])
    # Nested under a 'usage'-style dict?
    for key in ("usage", "meta", "stats"):
        sub = record.get(key)
        if isinstance(sub, dict):
            for tk in ("total_tokens", "num_tokens", "output_tokens", "tokens"):
                if tk in sub and isinstance(sub[tk], (int, float)):
                    return int(sub[tk])
    return None


def is_obviously_garbage_answer(ans):
    if isinstance(ans, dict):
        ans = ans.get("answer", "")
    if ans is None:
        return True
    ans = str(ans).strip()
    if not ans:
        return True
    return bool(UUID_LIKE_RE.match(ans))


def load_candidate_answers(candidates_path):
    """
    Read candidates_all_problems.jsonl and return ordered candidate answers:
        problem_id -> [answer_1, answer_2, ...]

    New-schema candidates are sorted by their `k` field. Old-schema string
    candidates retain file order. If multiple candidate events exist for a
    problem, the last one wins (it should be the most complete snapshot).
    """
    by_problem = {}
    json_errors = 0

    try:
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    json_errors += 1
                    continue

                if row.get("event") != "candidates":
                    continue

                problem_id = row.get("problem_id")
                if problem_id is None:
                    continue
                try:
                    problem_id = int(problem_id)
                except (TypeError, ValueError):
                    continue

                raw = row.get("candidates", []) or []
                ordered = []
                for pos, cand in enumerate(raw):
                    if isinstance(cand, dict):
                        answer = cand.get("answer", "")
                        k = cand.get("k")
                        try:
                            order = int(k)
                        except (TypeError, ValueError):
                            order = 10**9 + pos
                    else:
                        answer = cand
                        order = 10**9 + pos

                    if is_obviously_garbage_answer(answer):
                        continue
                    ordered.append((order, pos, str(answer)))

                ordered.sort(key=lambda x: (x[0], x[1]))
                by_problem[problem_id] = [x[2] for x in ordered]

    except OSError as ex:
        print(f"[WARN] Could not read candidates {candidates_path}: {ex}")
        return {}

    if json_errors:
        print(f"[WARN] {candidates_path}: skipped {json_errors} malformed JSON lines")

    return by_problem


def build_gold_map(dataset):
    """Build problem_id -> gold answer from the already-loaded dataset."""
    gold = {}
    for row in dataset.data:
        try:
            pid = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        gold[pid] = str(row.get("answer", ""))
    return gold


def candidate_matches_gold(pred, gold, clean_latex, math_equal):
    """Use the SSDP math grader; fall back to normalized string equality."""
    if is_obviously_garbage_answer(pred):
        return False

    pred = str(pred).strip()
    gold = str(gold).strip()
    try:
        return bool(math_equal(clean_latex(pred), clean_latex(gold)))
    except Exception:
        norm = lambda x: re.sub(r"\s+", "", str(x).replace("\\left", "").replace("\\right", ""))
        return norm(pred) == norm(gold)


def compute_topk_stats(problem_ids, candidate_by_problem, gold_map, clean_latex, math_equal):
    """
    Oracle candidate accuracy:
      acc@k = P(at least one correct answer occurs among first k candidates).

    Denominator is the merged result set for the budget. Therefore a problem
    with zero/missing candidates counts as a miss, rather than disappearing.
    """
    problem_ids = sorted(problem_ids)
    n = len(problem_ids)
    hit_counts = {k: 0 for k in TOPK_LIST}
    with_candidates = 0
    missing_gold = []

    for pid in problem_ids:
        if pid not in gold_map:
            missing_gold.append(pid)
            continue

        cands = candidate_by_problem.get(pid, [])
        if cands:
            with_candidates += 1

        matches = [
            candidate_matches_gold(c, gold_map[pid], clean_latex, math_equal)
            for c in cands
        ]
        for k in TOPK_LIST:
            if any(matches[:k]):
                hit_counts[k] += 1

    return {
        "n": n,
        "with_candidates": with_candidates,
        "missing_candidates": n - with_candidates,
        "missing_gold_ids": missing_gold,
        "hits": hit_counts,
        "accuracy": {k: (hit_counts[k] / n if n else 0.0) for k in TOPK_LIST},
    }


def load_trace_summaries(trace_path):
    """
    Read dpts_trace_all_problems.jsonl and retain the final search_stop event
    containing total_generated_tokens for each problem.

    Returns:
        {
            problem_id: {
                "total_generated_tokens": int,
                "rollouts_done": int or None,
                "stop_reason": str or None,
            }
        }
    """
    by_problem = {}
    json_errors = 0

    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    json_errors += 1
                    continue

                if row.get("event") != "search_stop":
                    continue

                # Ignore intermediate search_stop rows that do not contain
                # the final per-problem accounting.
                if not isinstance(row.get("total_generated_tokens"), (int, float)):
                    continue

                meta = row.get("meta") or {}
                problem_id = meta.get("problem_id")

                if problem_id is None:
                    continue

                try:
                    problem_id = int(problem_id)
                except (TypeError, ValueError):
                    continue

                by_problem[problem_id] = {
                    "total_generated_tokens": int(row["total_generated_tokens"]),
                    "rollouts_done": (
                        int(row["rollouts_done"])
                        if isinstance(row.get("rollouts_done"), (int, float))
                        else None
                    ),
                    "stop_reason": row.get("stop_reason"),
                }

    except OSError as ex:
        print(f"[WARN] Could not read trace {trace_path}: {ex}")
        return {}

    if json_errors:
        print(
            f"[WARN] {trace_path}: skipped {json_errors} malformed JSON lines"
        )

    return by_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="RESULTS root (e.g. results)")
    ap.add_argument("--model", required=True, help="MODEL tag (e.g. llama-3b)")
    ap.add_argument("--qdir", required=True, help="queue dir with done/ and chunks.txt")
    ap.add_argument("--data", required=True, help="dataset key (e.g. math)")
    ap.add_argument("--budgets", nargs="+", type=int, required=True)
    ap.add_argument("--expected-total", type=int, default=500,
                    help="expected problems per budget (default 500)")
    args = ap.parse_args()

    # Import the dataset for evaluation (must run from repo root).
    sys.path.insert(0, os.getcwd())
    from inferenceKit.data_config import supported_dataset
    from inferenceKit.dataset.basedataset import clean_latex
    from inferenceKit.dataset.grader import math_equal
    dataset = supported_dataset[args.data]()
    gold_map = build_gold_map(dataset)

    # Parse chunks.txt to know the full expected chunk set per budget.
    chunks_txt = os.path.join(args.qdir, "chunks.txt")
    expected_by_budget = {}
    if os.path.exists(chunks_txt):
        with open(chunks_txt) as f:
            for ln in f:
                if not ln.strip():
                    continue
                cid, budget, s, e = ln.split()
                expected_by_budget.setdefault(int(budget), []).append(cid)
    else:
        print(f"[WARN] {chunks_txt} not found - can't verify completeness, "
              f"will aggregate whatever chunk dirs exist.")

    done_dir = os.path.join(args.qdir, "done")
    done_cids = set(os.listdir(done_dir)) if os.path.isdir(done_dir) else set()

    print("=" * 90)
    print(f"SSDP chunk aggregation | model={args.model} data={args.data}")
    print("=" * 90)
    final_rows = []

    for budget in args.budgets:
        chunk_root = os.path.join(args.results, args.model, f"b{budget}", "chunks")
        if not os.path.isdir(chunk_root):
            print(f"\n[b{budget}] no chunk dir at {chunk_root} - skipping.")
            continue

        chunk_dirs = sorted(glob.glob(os.path.join(chunk_root, "*")))
        expected_cids = expected_by_budget.get(budget, [os.path.basename(d) for d in chunk_dirs])

        # Completion accounting.
        present = {os.path.basename(d) for d in chunk_dirs}
        done_here = {c for c in expected_cids if c in done_cids}
        missing = [c for c in expected_cids if c not in present]
        not_done = [c for c in expected_cids if c not in done_cids]

        # Gather results across chunks, keyed by voting method, deduped by index.
        by_key = {}   # key -> {index: record}

        # Final trace summary keyed by problem_id.
        trace_by_problem = {}

        # Ordered candidate answers keyed by problem_id, used for acc@1..acc@10.
        candidate_by_problem = {}

        chunks_with_results = 0
        chunks_with_trace = 0
        chunks_with_candidates = 0
        for d in chunk_dirs:

            trace_path = os.path.join(d, "dpts_trace_all_problems.jsonl")
            if os.path.isfile(trace_path):
                chunks_with_trace += 1
                chunk_trace = load_trace_summaries(trace_path)

                trace_by_problem.update(chunk_trace)

            candidates_path = os.path.join(d, "candidates_all_problems.jsonl")
            if os.path.isfile(candidates_path):
                chunks_with_candidates += 1
                candidate_by_problem.update(load_candidate_answers(candidates_path))

            rfiles = glob.glob(os.path.join(d, "results-*.json"))
            if rfiles:
                chunks_with_results += 1
            for rf in rfiles:
                key = os.path.basename(rf)[len("results-"):-len(".json")]
                try:
                    recs = load_json(rf)
                except Exception as ex:
                    print(f"[b{budget}] could not read {rf}: {ex}")
                    continue
                bucket = by_key.setdefault(key, {})
                for r in recs:
                    bucket[r["index"]] = r

        print(
                f"\n[b{budget}] chunks: expected={len(expected_cids)} "
                f"present={len(present)} done_marked={len(done_here)} "
                f"with_results={chunks_with_results} "
                f"with_trace={chunks_with_trace} "
                f"with_candidates={chunks_with_candidates}"
            )
        if missing:
            print(f"[b{budget}]   MISSING chunk dirs ({len(missing)}): {missing[:10]}"
                  + (" ..." if len(missing) > 10 else ""))
        if not_done:
            print(f"[b{budget}]   NOT done-marked ({len(not_done)}): {not_done[:10]}"
                  + (" ..." if len(not_done) > 10 else ""))

        if not by_key:
            print(f"[b{budget}]   no results-*.json found - nothing to score.")
            continue



        print(f"[b{budget}] {'voting_key':<16} {'n':>4} {'acc':>7} {'correct':>8} {'no_match':>9}")

        majority_eval = None
        for key in sorted(by_key):
            merged = sorted(by_key[key].values(), key=lambda x: x["index"])
            try:
                ev = dataset.evaluate_results(merged)
                print(f"[b{budget}] {key:<16} {ev['total_samples']:>4} "
                      f"{ev['accuracy']:>7.4f} {ev['correct_samples']:>8} "
                      f"{ev['no_match_samples']:>9}")

                if key == "majority_vote":
                    majority_eval = ev
            except Exception as ex:
                print(f"[b{budget}] {key:<16} eval FAILED: {ex}")

        any_key = next(iter(by_key))
        result_problem_ids = set(by_key[any_key].keys())

        topk_stats = None
        if candidate_by_problem:
            topk_stats = compute_topk_stats(
                result_problem_ids, candidate_by_problem, gold_map, clean_latex, math_equal
            )
            print(
                f"[b{budget}] candidate coverage: "
                f"{topk_stats['with_candidates']}/{topk_stats['n']} problems "
                f"have >=1 candidate"
            )
            print(f"[b{budget}] top-k candidate accuracy (oracle hit in first k):")
            for k in TOPK_LIST:
                hits = topk_stats["hits"][k]
                acc = topk_stats["accuracy"][k]
                print(f"[b{budget}]   acc@{k:>2}: {acc:.4f} ({hits}/{topk_stats['n']})")

            if topk_stats["missing_candidates"]:
                print(
                    f"[b{budget}] WARNING: {topk_stats['missing_candidates']} result samples "
                    f"have no candidate record; they count as misses in acc@k"
                )
            if topk_stats["missing_gold_ids"]:
                print(
                    f"[b{budget}] WARNING: missing gold answer for "
                    f"{len(topk_stats['missing_gold_ids'])} result samples: "
                    f"{topk_stats['missing_gold_ids'][:10]}"
                )
        else:
            print(
                f"[b{budget}] top-k candidate accuracy unavailable: "
                f"no candidates_all_problems.jsonl files found"
            )

        matched_trace_rows = {
                pid: trace_by_problem[pid]
                for pid in result_problem_ids
                if pid in trace_by_problem
            }
        token_counts = [
                row["total_generated_tokens"]
                for row in matched_trace_rows.values()
            ]
        rollout_counts = [
                row["rollouts_done"]
                for row in matched_trace_rows.values()
                if row["rollouts_done"] is not None
            ]
        stop_reason_counts = {}
        for row in matched_trace_rows.values():
            reason = row.get("stop_reason") or "unknown"
            stop_reason_counts[reason] = stop_reason_counts.get(reason, 0) + 1

        missing_token_ids = sorted(result_problem_ids - set(matched_trace_rows))
        if token_counts:
            print(
                    f"[b{budget}] total_generated_tokens: "
                    f"n={len(token_counts)}/{len(result_problem_ids)} "
                    f"mean={statistics.mean(token_counts):.1f} "
                    f"median={statistics.median(token_counts):.1f} "
                    f"min={min(token_counts)} "
                    f"max={max(token_counts)}"
                )
        else:
            print(
                    f"[b{budget}] total_generated_tokens: "
                    f"no final token summaries found in chunk traces"
                )

        if majority_eval is not None:
            final_rows.append({
                "budget": budget,
                "accuracy": majority_eval["accuracy"],
                "avg_tokens": statistics.mean(token_counts) if token_counts else None,
                "total_samples": majority_eval["total_samples"],
                "topk_accuracy": (topk_stats["accuracy"] if topk_stats is not None else None),
                "topk_hits": (topk_stats["hits"] if topk_stats is not None else None),
            })
        if rollout_counts:
            print(
                    f"[b{budget}] rollouts_done: "
                    f"n={len(rollout_counts)} "
                    f"mean={statistics.mean(rollout_counts):.2f} "
                    f"median={statistics.median(rollout_counts):.2f}"
                )

        print(
            f"[b{budget}] trace stop reasons: "
            f"{dict(sorted(stop_reason_counts.items()))}"
        )

        if missing_token_ids:
            print(
                f"[b{budget}] WARNING: missing token trace for "
                f"{len(missing_token_ids)} result samples: "
                f"{missing_token_ids[:10]}"
                + (" ..." if len(missing_token_ids) > 10 else "")
            )

    print()
    topk_headers = " | ".join([f"Top{k}" for k in TOPK_LIST])
    print(f"| Budget | Reported Accuracy | Avg. Tokens | total_samples | {topk_headers} |")
    print("|---:|---:|---:|---:|" + "---:|" * len(TOPK_LIST))

    for row in final_rows:
        avg_tokens = (
            f"{row['avg_tokens']:.1f}"
            if row["avg_tokens"] is not None
            else "NA"
        )

        if row["topk_accuracy"] is None:
            topk_cells = " | ".join(["NA"] * len(TOPK_LIST))
        else:
            topk_cells = " | ".join(
                f"{100.0 * row['topk_accuracy'][k]:.2f}%" for k in TOPK_LIST
            )

        print(
            f"| {row['budget']} "
            f"| {100.0 * row['accuracy']:.2f}% "
            f"| {avg_tokens} "
            f"| {row['total_samples']} "
            f"| {topk_cells} |"
        )

    print("\n" + "=" * 90)
    print("Done. Any budget showing MISSING or NOT done-marked chunks is incomplete -")
    print("re-qsub the PBS job to finish remaining chunks before trusting its numbers.")
    print("=" * 90)


if __name__ == "__main__":
    main()
