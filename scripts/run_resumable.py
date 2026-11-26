"""
run_resumable.py

Generalized, reusable replacement for continue_experiment.py.
Given a --work-dir/--exp-name pair, it:
  1. Looks for the latest RESUME_{key}_{i}.json (or starts fresh if none exists).
  2. Computes which dataset indices are NOT yet completed.
  3. Runs inference on only the remaining indices.
  4. On success or crash, inference() itself writes the next RESUME_{key}_{i}.json
     (this is existing behavior in inferenceKit/inference.py - unchanged).

This lets a PBS job be killed/timed out and simply resubmitted: each resubmission
finds the latest RESUME file and picks up where it left off.

Usage mirrors main.py plus --data:
  python scripts/run_resumable.py \
      --data math500 \
      --model llama-3b \
      --reward_model qwen_prm-7b \
      -c configs/inference/DPTS.json \
      --work-dir results \
      --exp-name qwen_prm_dpts_math500
"""
import argparse
import glob
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from inferenceKit import utils
from inferenceKit.model_config import create_llm, create_prm, create_embedding_model
from inferenceKit.data_config import supported_dataset
from inferenceKit.inference import inference
from inferenceKit.models import InferenceConfig, DefaultInferenceModel, VLLMInferenceModel


class RemainingDataset(torch.utils.data.Dataset):
    """Wraps a full dataset, exposing only the given remaining indices."""
    def __init__(self, full_dataset, remaining_indices):
        self.full_dataset = full_dataset
        self.remaining_indices = remaining_indices

    def __len__(self):
        return len(self.remaining_indices)

    def __getitem__(self, idx):
        return self.full_dataset[self.remaining_indices[idx]]


def find_latest_resume_file(output_dir):
    """
    inference() writes RESUME_{key}_{i}.json with increasing i on each crash/interrupt.
    Find the highest-i file for any key; if multiple keys exist (e.g. multiple voting
    methods), the caller should pass --resume_key to disambiguate. Default: 'last_max'.
    """
    candidates = glob.glob(os.path.join(output_dir, "RESUME_*_*.json"))
    if not candidates:
        return None
    # Sort by modification time, latest first.
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--reward_model', type=str, default=None)
    parser.add_argument('--reward_model_path', type=str, default=None)
    parser.add_argument('--embedding_model', type=str, default=None)
    parser.add_argument('--embedding_model_path', type=str, default=None)
    parser.add_argument('--enable_clustering', action='store_true', default=False)
    parser.add_argument('--clustering_threshold', type=float, default=None)
    parser.add_argument('--clustering_method', type=str, default=None)
    parser.add_argument('--vllm', action='store_true', default=False)
    parser.add_argument('--work-dir', type=str, default='./outputs')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--dtype', type=str, default="float32")
    parser.add_argument('--flash-attn', action='store_true', default=False)
    parser.add_argument('--shard', action='store_true', default=False)
    parser.add_argument('--shard-start', type=int, default=None,
                         help='First dataset index (0-based, inclusive) this worker owns')
    parser.add_argument('--shard-end', type=int, default=None,
                         help='Last dataset index (0-based, exclusive) this worker owns')
    args = parser.parse_args()
    print(f"[RESUMABLE DEBUG] model={args.model!r} reward_model={args.reward_model!r} data={args.data!r}")

    accelerator = Accelerator()
    accelerator.even_batches = False
    logger = utils.get_logger('RESUMABLE')

    device_map = "auto" if (args.shard or args.vllm) else None
    dtype = getattr(torch, args.dtype)

    inference_config = InferenceConfig.from_json(args.config) if args.config else InferenceConfig()
    if hasattr(inference_config.config, 'cot_method'):
        inference_config.config.cot_method = inference_config.config.cot_method.lower()

    output_dir = os.path.join(args.work_dir, args.exp_name or "default_exp")
    os.makedirs(output_dir, exist_ok=True)

    # Persist config once, so future resumes / continue_experiment-style scripts can find it.
    config_snapshot = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_snapshot) and accelerator.is_main_process:
        utils.dump(inference_config.config.to_dict() if hasattr(inference_config.config, "to_dict") else vars(inference_config.config), config_snapshot)

    dataset = supported_dataset[args.data]()
    total_samples = len(dataset)

    shard_start = args.shard_start if args.shard_start is not None else 0
    shard_end = args.shard_end if args.shard_end is not None else total_samples
    shard_end = min(shard_end, total_samples)
    if shard_start >= shard_end:
        logger.info(f"Empty shard [{shard_start}, {shard_end}) - nothing to do.")
        sys.exit(0)

    # Prefer the persistent merged results-*.json this runner writes on clean finish;
    # fall back to crash-only RESUME_*.json; else start fresh.
    completed_indices = set()
    merged_candidates = glob.glob(os.path.join(output_dir, "results-*.json"))
    if merged_candidates:
        merged_candidates.sort(key=os.path.getmtime, reverse=True)
        logger.info(f"Found existing merged results file: {merged_candidates[0]}")
        with open(merged_candidates[0], "r") as f:
            completed_samples = json.load(f)
        completed_indices = {sample["index"] for sample in completed_samples}
    else:
        resume_file = find_latest_resume_file(output_dir)
        if resume_file is not None:
            logger.info(f"Found existing crash-resume file: {resume_file}")
            with open(resume_file, 'r') as f:
                completed_samples = json.load(f)
            completed_indices = {sample["index"] for sample in completed_samples}
        else:
            logger.info("No existing progress file found - starting fresh.")

    # +1 because dataset indices are 1-based in this codebase's result records
    # (see continue_experiment.py's `i + 1 not in completed_indices` convention).
    remaining_indices = [
        i for i in range(shard_start, shard_end)
        if (i + 1) not in completed_indices
    ]
    logger.info(
        f"Shard [{shard_start}, {shard_end}) | "
        f"Completed: {len(completed_indices)} | Remaining in shard: {len(remaining_indices)}"
    )

    if len(remaining_indices) == 0:
        logger.info("All samples already completed. Nothing to do.")
        sys.exit(0)

    remaining_dataset = RemainingDataset(dataset, remaining_indices)
    dataloader = DataLoader(remaining_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    generation_model, args.model_path = create_llm(args.model, args.model_path, args.vllm)
    generation_model = generation_model(
        model_path=args.model_path, device=accelerator.device,
        device_map=device_map, dtype=dtype, flash_attn=args.flash_attn,
    )

    reward_model = None
    if args.reward_model is not None:
        reward_model_class, args.reward_model_path = create_prm(args.reward_model, args.reward_model_path)
        reward_model = reward_model_class(
            model_path=args.reward_model_path, device=accelerator.device,
            device_map=device_map, dtype=dtype, flash_attn=args.flash_attn,
        )

    embedding_model = None
    if args.embedding_model is not None:
        embedding_model_class, embedding_model_path = create_embedding_model(args.embedding_model, args.embedding_model_path)
        embedding_model = embedding_model_class(model_path=embedding_model_path, device=accelerator.device)

    if args.vllm:
        model = VLLMInferenceModel(generation_model, reward_model, inference_config, device=accelerator.device, embedding_model=embedding_model)
    else:
        model = DefaultInferenceModel(generation_model, reward_model, inference_config, device=accelerator.device, embedding_model=embedding_model)

    dataloader = accelerator.prepare(dataloader)

    # inference() returns the per-key results for the samples processed THIS run
    # (i.e. only the remaining shard indices). On a clean finish it does NOT persist
    # them itself - it only writes timing_info.json / clustering_metrics.json - so we
    # must gather + merge + dump here, mirroring the tail of main.py's main().
    results = inference(model, dataloader, accelerator, output_dir)

    for k, res in results.items():
        all_results_this_run = accelerator.gather_for_metrics(res, True)

        if accelerator.is_main_process:
            # Merge with any previously-completed samples for this exp-name/shard.
            merged_path = os.path.join(output_dir, f"results-{k}.json")
            prior = []
            if os.path.exists(merged_path):
                with open(merged_path, "r") as f:
                    prior = json.load(f)

            by_index = {r["index"]: r for r in prior}
            for r in all_results_this_run:
                by_index[r["index"]] = r
            merged = sorted(by_index.values(), key=lambda x: x["index"])

            utils.dump(merged, merged_path)
            logger.info(f"Wrote {len(merged)} merged results to {merged_path}")

            # Evaluate on whatever is completed so far for this shard.
            try:
                eval_result = dataset.evaluate_results(merged)
                eval_path = os.path.join(output_dir, "evaluation_results.json")
                existing_eval = {}
                if os.path.exists(eval_path):
                    with open(eval_path, "r") as f:
                        existing_eval = json.load(f)
                existing_eval[k] = eval_result
                utils.dump(existing_eval, eval_path)
                logger.info(f"Wrote evaluation for key '{k}' to {eval_path}")
            except Exception as e:
                logger.info(f"Could not evaluate results for key '{k}': {e}")

    # Write a DONE marker if this shard is fully complete, so a clean finish leaves
    # persistent proof of completion (inference() only writes RESUME_* on a CRASH,
    # so without this a cleanly-finished shard would be re-run from scratch on the
    # next resubmission).
    if accelerator.is_main_process:
        # Recompute completion against the merged results file.
        done_for_shard = True
        for k in results.keys():
            merged_path = os.path.join(output_dir, f"results-{k}.json")
            if os.path.exists(merged_path):
                with open(merged_path, "r") as f:
                    merged = json.load(f)
                completed_now = {r["index"] for r in merged}
                still_missing = [
                    i for i in range(shard_start, shard_end)
                    if (i + 1) not in completed_now
                ]
                if still_missing:
                    done_for_shard = False
            else:
                done_for_shard = False

        if done_for_shard:
            with open(os.path.join(output_dir, "SHARD_DONE"), "w") as f:
                f.write(f"shard [{shard_start}, {shard_end}) complete\n")
            logger.info("Shard fully complete - wrote SHARD_DONE marker.")

    logger.info("Batch of remaining samples finished. Resubmit this job to continue if not all done.")


if __name__ == "__main__":
    main()
