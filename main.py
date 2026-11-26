import argparse
import os
import datetime
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from inferenceKit import utils
from inferenceKit.model_config import create_llm, create_prm, create_embedding_model
from inferenceKit.data_config import supported_dataset
from inferenceKit.inference import inference
from inferenceKit.models import InferenceConfig, DefaultInferenceModel, VLLMInferenceModel

import torch.nn.functional as F
import random
import numpy as np
import time
import glob
from torch.utils.data import Subset

def parse_args():
    parser = argparse.ArgumentParser()
    SUPPRESS = argparse.SUPPRESS
    
    parser.add_argument('-c', '--config', default=None, type=str, metavar='FILE', help='Json Config File specifying arguments')
    
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model',type=str, required=True)
    parser.add_argument('--model_path',type=str, default=None)
    parser.add_argument('--reward_model', type=str, default=None)
    parser.add_argument('--reward_model_path', type=str, default=None)
    
    # Embedding model arguments for clustering
    parser.add_argument('--embedding_model', type=str, default=None, help='Embedding model for node clustering')
    parser.add_argument('--embedding_model_path', type=str, default=None, help='Path to embedding model')
    parser.add_argument('--enable_clustering', action='store_true', default=False, help='Enable embedding-based node clustering')
    parser.add_argument('--clustering_threshold', type=float, default=None, help='Similarity threshold for clustering (0.0-1.0)')
    parser.add_argument('--clustering_method', type=str, default=None, help='Clustering method: cosine_similarity, kmeans, hierarchical')
    
    parser.add_argument('--vllm', action='store_true', default=False, help='Enable vllm')
    
    parser.add_argument('--cot_method', type=str, default=SUPPRESS)
    parser.add_argument('--step_method', type=str, default=SUPPRESS)
    parser.add_argument('--voting_method', type=str, default=SUPPRESS)

    parser.add_argument('--max_step_time', type=int, default=SUPPRESS)
    
    # Safe optimization arguments (no threading, no parameter changes)
    parser.add_argument('--fast_mode', action='store_true', default=False, help='Use fast mode with reduced parameters')

    parser.add_argument('--work-dir',type=str, default='./outputs')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--trace-log-dir', type=str, default=None, 
                        help='Directory for trace logs (default: {output_dir}/traces)')
    
    parser.add_argument('--dtype', type=str, default="float32")
    parser.add_argument('--flash-attn', action='store_true', default=False, help='Enable Flash Attention')
    parser.add_argument('--shard', action='store_true', default=False, help='Big Model Sharded Inference')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug Mode')
    parser.add_argument('--resume', type=str, default=None, help='Resume work-dir Experiment')
    parser.add_argument('--token-budget', type=int, default=SUPPRESS,
                        help='Per-problem generation/tree token budget')
    parser.add_argument('--seed',type=int,default=1029,help='Base random seed for Python, NumPy, and Torch')
    
    args = parser.parse_args()
    return args


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    parsed_args = parse_args()
    run_seed = int(parsed_args.seed)
    seed_all(run_seed)
    print(f"[SEED] run_seed={run_seed}", flush=True)
    args = vars(parsed_args)
    args.pop("seed", None)
    accelerator = Accelerator(device_placement=False)

    gen_device = torch.device("cuda:0")
    aux_device = torch.device("cuda:1")
    # --- save trace_dir argument now (args is a dict) ---
    trace_log_dir_arg = args.pop("trace_log_dir", None)
    if trace_log_dir_arg:
        print(f"[TRACE] trace enabled via CLI, dir={trace_log_dir_arg}")

    accelerator.even_batches = False
    logger = utils.get_logger('MAIN')
    args_shard = args.pop("shard")
    device_map = None
    dtype = getattr(torch, args.pop("dtype"))
    flash_attn = args.pop("flash_attn")
    
    # Extract optimization arguments
    fast_mode = args.pop("fast_mode", False)
    
    # Check if resuming before processing config
    resume_path = args.pop("resume", None)
    is_resume = resume_path is not None

    config_file = args.pop("config", None)
    
    # Handle resume case
    if is_resume:
        # Load config from the resume directory
        config_path = os.path.join(resume_path, "config.json")
        if os.path.exists(config_path):
            inference_config = InferenceConfig.from_json(config_path)
        else:
            inference_config = InferenceConfig()
    else:
        inference_config = InferenceConfig.from_json(config_file) if config_file else InferenceConfig()
        # Only update config with command line arguments that were actually provided (not defaults)
        # Filter out clustering-related defaults that would override config file settings
        clustering_defaults = {
            'embedding_model': None,
            'embedding_model_path': None, 
            'enable_clustering': False,
            'clustering_threshold': None,
            'clustering_method': None
        }
        provided_args = {k: v for k, v in args.items() if k not in clustering_defaults or v != clustering_defaults[k]}
        utils.update_single_config(inference_config.config, copy=False, **provided_args)
    
    args = inference_config.config
    
    # Only access cot_method if it exists (not when resuming)
    if hasattr(args, 'cot_method'):
        args.cot_method = args.cot_method.lower()
    
    # Apply ONLY safe optimizations that don't break tensor dimensions
    '''
    if fast_mode:
        print("Applying safe fast mode optimizations...")
        # Only reduce max_step_time and max_rollout, keep tree_width/num_beams the same
        if hasattr(args, 'max_step_time') and args.max_step_time > 60:
            print(f"Reducing max_step_time from {args.max_step_time} to 60")
            args.max_step_time = 60
        
        if hasattr(args, 'max_rollout') and args.max_rollout > 10:
            print(f"Reducing max_rollout from {args.max_rollout} to 10")
            args.max_rollout = 10
    '''
    print(args)
        
    reward_model = None
    if args.reward_model is not None:
        reward_model, args.reward_model_path = create_prm(args.reward_model, args.reward_model_path)
        reward_model = reward_model(
                model_path=args.reward_model_path,
                device=aux_device,
                device_map={"": aux_device},
                dtype=dtype,
                flash_attn=flash_attn
            )

    # Initialize embedding model for clustering
    embedding_model = None
    if hasattr(args, 'embedding_model') and args.embedding_model is not None and hasattr(args, 'enable_clustering') and args.enable_clustering:
        embedding_model_path = getattr(args, 'embedding_model_path', None)
        embedding_model_class, embedding_model_path = create_embedding_model(args.embedding_model, embedding_model_path)
        embedding_model = embedding_model_class(
                model_path=embedding_model_path,
                device=aux_device
                )
        print(f"Initialized embedding model: {args.embedding_model} with path: {embedding_model_path}")
    
    generation_model, args.model_path = create_llm(args.model, args.model_path, args.vllm)
    generation_model = generation_model(
        model_path=args.model_path,
        device=gen_device,
        device_map={"": gen_device},
        dtype=dtype,
        flash_attn=flash_attn
    )
    if args.vllm:
        model = VLLMInferenceModel(generation_model, reward_model, inference_config, device=accelerator.device, embedding_model=embedding_model)
    else:
        model = DefaultInferenceModel(generation_model, reward_model, inference_config, device=accelerator.device, embedding_model=embedding_model)
    model.eval()
    import glob
    from torch.utils.data import Subset

    TOKEN_BUDGET_FIELD = "max_total_tokens"   # <-- set to the real config key you grepped

    base_dataset = supported_dataset[args.data]()

    # queue + identity passed in from the launcher
    QDIR       = os.environ["SSDP_QDIR"]                 # shared control dir on Lustre
    RANK       = int(os.environ.get("SSDP_WORKER_RANK", "0"))
    RESULTS    = os.environ.get("SSDP_RESULTS", "./results")
    MODEL_TAG  = args.model
    DEADLINE_S = float(os.environ.get("SSDP_DEADLINE_S", "3300"))   # stop claiming after ~55 min
    MIN_LEFT_S = float(os.environ.get("SSDP_MIN_LEFT_S", "1200"))   # don't start unless >=20 min left
    t0 = time.time()

    claim_dir = os.path.join(QDIR, "claim")
    done_dir  = os.path.join(QDIR, "done")

    def is_done(cid):    return os.path.exists(os.path.join(done_dir, cid))
    def mark_done(cid):  open(os.path.join(done_dir, cid), "w").close()
    def try_claim(cid):
        try:
            os.mkdir(os.path.join(claim_dir, cid));  return True
        except FileExistsError:
            return False

    with open(os.path.join(QDIR, "chunks.txt")) as f:
        chunks = [ln.split() for ln in f if ln.strip()]   # cid budget start end
    # rotate start point per rank to cut claim contention
    if chunks:
        off = RANK % len(chunks)
        chunks = chunks[off:] + chunks[:off]

    print(f"[worker {RANK}] {len(chunks)} chunks visible")
    for cid, budget, s, e in chunks:
        s, e, budget = int(s), int(e), int(budget)
        if is_done(cid):                       continue
        if (time.time() - t0) > DEADLINE_S:    break
        if (DEADLINE_S - (time.time() - t0)) < MIN_LEFT_S:  break
        if not try_claim(cid):                 continue
        if is_done(cid):                       continue   # race double-check

        trace_dir = os.path.join(RESULTS, MODEL_TAG, f"b{budget}", "chunks", cid)
        # fresh dir so a re-run of a previously-interrupted chunk can't append dupes
        os.system(f'rm -rf "{trace_dir}"'); os.makedirs(trace_dir, exist_ok=True)

        setattr(model.inference_config.config, TOKEN_BUDGET_FIELD, budget)

        subset = Subset(base_dataset, list(range(s, e)))
        dl = DataLoader(dataset=subset, shuffle=False, batch_size=1, collate_fn=lambda x: x[0])
        dl = accelerator.prepare(dl)

        print(f"[worker {RANK}] RUN {cid} budget={budget} idx[{s}:{e}] -> {trace_dir}", flush=True)
        c_start = time.time()
        results = inference(model, dl, accelerator, trace_dir, trace_log_dir=trace_dir)

        for k, res in results.items():
            all_results = accelerator.gather_for_metrics(res, True)
            all_results = sorted(all_results, key=lambda x: x["index"])
            utils.dump(all_results, os.path.join(trace_dir, f"results-{k}.json"))
        utils.dump({"cid": cid, "budget": budget, "start": s, "end": e,
                    "seconds": time.time() - c_start,
                    "peak_gb": torch.cuda.max_memory_allocated() / 1024**3},
                   os.path.join(trace_dir, "chunk_meta.json"))
        mark_done(cid)
        print(f"[worker {RANK}] DONE {cid} in {time.time()-c_start:.1f}s", flush=True)

    print(f"[worker {RANK}] exiting after {time.time()-t0:.1f}s", flush=True)
    

if __name__ == '__main__':
    main()
