"""
calibrated_neighborhood_attack.py

A thin orchestration layer that runs the *calibrated neighborhood MIA*
by reusing your existing project utilities in `run_mia_unified.py`
(and dataset helpers in `custom_datasets.py`).

Key idea:
If `--ref_model` is provided, `run_mia_unified.get_ll` already returns
Δ log-likelihood = LL_target(x) - LL_ref(x). Therefore the standard
neighborhood gap
    (LL(x) - mean LL(neighbors))
becomes
    [LL_T(x) - E LL_T(neigh)] - [LL_R(x) - E LL_R(neigh)],
which is exactly the difficulty-calibrated neighborhood score.

This script wires up models, dataset, neighbor generation, and evaluation
using the existing functions, and saves ROC/PR metrics to JSON.
"""

from __future__ import annotations
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import transformers

# Import your project module
import run_mia_unified as rm

# --- HOTFIX v2: robust dataset loader for XSum/CNN-DailyMail (short texts) ---
import datasets as _hf_datasets

def _smart_text_dataset(dataset, key, train=True, cache_dir=None):
    """
    Robust loader for a single-text-column dataset.
    - For XSum requests, fall back to cnn_dailymail (short 'highlights' field).
    - For everything else, delegate to datasets.load_dataset as-is.
    """
    ds_name = str(dataset).lower()
    split = "train" if train else "validation"

    if ds_name in {"xsum", "edinburghnlp/xsum"}:
        print("[SPV-MIA] XSum JSON files are not available; "
              "falling back to cnn_dailymail v3.0.0 (field: 'highlights').")
        ds = _hf_datasets.load_dataset(
            "cnn_dailymail", "3.0.0", split=f"{split}[:10000]", cache_dir=cache_dir
        )
        # Use short summaries to avoid 512-token overflows
        fallback_col = "highlights"
        col = fallback_col if fallback_col in ds.column_names else ds.column_names[0]
        return ds[col]

    # Default: original path (works for datasets with built-in loaders)
    ds = _hf_datasets.load_dataset(
        dataset, split=f"{split}[:10000]", cache_dir=cache_dir
    )
    if key in ds.column_names:
        return ds[key]
    for cand in ("text", "document", "article", "highlights"):
        if cand in ds.column_names:
            return ds[cand]
    return ds[ds.column_names[0]]

# monkey-patch
rm.generate_data = _smart_text_dataset
# --- END HOTFIX ---



def build_args(ns: argparse.Namespace) -> SimpleNamespace:
    """Create a SimpleNamespace that mimics rm.args for all attributes accessed inside rm.* functions."""
    return SimpleNamespace(
        # dataset selection
        dataset_member=ns.dataset_member,
        dataset_member_key=ns.dataset_member_key,
        dataset_nonmember=ns.dataset_nonmember,
        dataset_nonmember_key=ns.dataset_nonmember_key,
        n_samples=ns.n_samples,
        batch_size=ns.batch_size,
        cache_dir=ns.cache_dir,
        # perturbation/neighborhood
        pct_words_masked=ns.pct_words_masked,
        span_length=ns.span_length,
        n_perturbation_rounds=ns.n_perturbation_rounds,
        buffer_size=ns.buffer_size,
        chunk_size=ns.chunk_size,
        ceil_pct=ns.ceil_pct,
        # models (target/ref + mask-filling)
        base_model_name=ns.base_model_name,
        ref_model=ns.ref_model,
        revision=ns.revision,
        mask_filling_model_name=ns.mask_filling_model_name,
        # NEW: pre-perturbation knobs expected by run_mia_unified.generate_samples
        pre_perturb_pct=ns.pre_perturb_pct,
        pre_perturb_span_length=ns.pre_perturb_span_length,
        # generation knobs for fill model
        do_top_k=ns.do_top_k,
        do_top_p=ns.do_top_p,
        temperature=ns.temperature,
        mask_top_p=ns.mask_top_p,
        # mixed-precision / memory
        int8=ns.int8,
        half=ns.half,
        # misc flags used in rm
        random_fills=ns.random_fills,
        random_fills_tokens=False,
        baselines_only=False,
        skip_baselines=False,
        tok_by_tok=False,
        # OpenAI (unused here)
        openai_model=None,
        openai_key=None,
        max_tries=ns.max_tries,
        max_length=ns.max_length,
    )


def init_models_and_tokenizers(args: SimpleNamespace):
    """Instantiate target, reference, and mask-filling objects as globals in the rm module, mirroring rm.__main__."""
    # Device + cache
    rm.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    rm.cache_dir = args.cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)

    # Tokenizer for token counting used inside rm
    rm.GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir)

    # Load target model/tokenizer as module globals
    rm.base_model, rm.base_tokenizer = rm.load_base_model_and_tokenizer(args.base_model_name)
    rm.load_base_model()

    # Optional reference model (enables calibrated Δ log-likelihoods inside rm.get_ll)
    if args.ref_model:
        rm.ref_model, rm.ref_tokenizer = rm.load_base_model_and_tokenizer(args.ref_model)
        rm.load_ref_model()

    # Mask-filling model + tokenizers for neighborhood generation
    # (replicates rm.__main__ logic succinctly)
    if not args.random_fills:
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)

        # Load mask-filling model (e.g., T5)
        print(f"Loading mask filling model {args.mask_filling_model_name}…")
        rm.mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            args.mask_filling_model_name, cache_dir=args.cache_dir, **int8_kwargs, **half_kwargs
        )
        try:
            n_positions = rm.mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
    else:
        n_positions = 512

    rm.preproc_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "t5-small", model_max_length=512, cache_dir=args.cache_dir
    )
    rm.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.mask_filling_model_name, model_max_length=n_positions, cache_dir=args.cache_dir
    )

    # Globals used in rm.*
    rm.mask_filling_model_name = args.mask_filling_model_name


def prepare_data(args: SimpleNamespace):
    """Build rm.data = {"member": [...], "nonmember": [...]} using rm.generate_data + rm.generate_samples."""
    print(f"Loading dataset {args.dataset_member} (member) and {args.dataset_nonmember} (nonmember)…")
    data_member = rm.generate_data(args.dataset_member, args.dataset_member_key)
    data_nonmember = rm.generate_data(args.dataset_nonmember, args.dataset_nonmember_key, train=False)

    data, seq_lens, n_samples = rm.generate_samples(
        data_member[: args.n_samples], data_nonmember[: args.n_samples], batch_size=args.batch_size
    )

    rm.data = data
    rm.n_perturbation_rounds = args.n_perturbation_rounds
    return n_samples


def run_attack(args: SimpleNamespace):
    """Compute calibrated neighborhood scores via rm’s existing machinery and write metrics to disk."""
    outputs = []
    for n_pert in args.n_perturbations:
        # Neighbor generation + scoring (uses ΔLL if ref_model is set)
        results = rm.get_perturbation_results(
            span_length=args.span_length, n_perturbations=n_pert, n_samples=args.n_samples
        )
        # Convert to decision scores (difference or z-normalized)
        out = rm.run_perturbation_experiment(
            results,
            criterion=args.criterion,
            span_length=args.span_length,
            n_perturbations=n_pert,
            n_samples=args.n_samples,
        )
        out["name"] = f"calibrated_neigh_{args.criterion}_n{n_pert}" if args.ref_model else f"neigh_{args.criterion}_n{n_pert}"
        outputs.append(out)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved results to {args.save_path}")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Datasets
    p.add_argument("--dataset_member", type=str, default="xsum")
    p.add_argument("--dataset_member_key", type=str, default="document")
    p.add_argument("--dataset_nonmember", type=str, default="xsum")
    p.add_argument("--dataset_nonmember_key", type=str, default="document")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cache_dir", type=str, default="/trunk/model-hub")
    # Neighborhood (mask filling)
    p.add_argument("--pct_words_masked", type=float, default=0.3)
    p.add_argument("--span_length", type=int, default=2)
    p.add_argument("--n_perturbations", type=str, default="1,10")  # comma-separated
    p.add_argument("--n_perturbation_rounds", type=int, default=1)
    p.add_argument("--buffer_size", type=int, default=1)
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--ceil_pct", action="store_true")
    # Target/Reference models
    p.add_argument("--base_model_name", type=str, default="gpt2-medium")
    p.add_argument("--ref_model", type=str, default=None)
    p.add_argument("--revision", type=str, default="main")
    # Mask-filling model
    p.add_argument("--mask_filling_model_name", type=str, default="t5-3b")
    p.add_argument("--do_top_k", action="store_true")
    p.add_argument("--do_top_p", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--mask_top_p", type=float, default=1.0)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--half", action="store_true")
    p.add_argument("--random_fills", action="store_true")
    # Criterion and output
    p.add_argument("--criterion", type=str, choices=["d", "z"], default="z")
    p.add_argument("--save_path", type=str, default="results/calibrated_neigh.json")
    p.add_argument("--max_tries", type=int, default=100)
    p.add_argument("--max_length", type=int, default=None)
    # ... inside parse_cli()
    p.add_argument("--pre_perturb_pct", type=float, default=0.0)
    p.add_argument("--pre_perturb_span_length", type=int, default=5)

    return p.parse_args()


def main():
    ns = parse_cli()
    # Parse list of perturbations
    n_pert_list = [int(x.strip()) for x in ns.n_perturbations.split(",") if x.strip()]

    # Build arg namespace for rm and mirror a few globals
    rm.args = build_args(ns)
    rm.args.n_perturbations = n_pert_list  # keep parity with rm naming style

    # --- SAFETY NET: ensure fields exist for generate_samples() ---
    for k, v in {
        "pre_perturb_pct": 0.0,
        "pre_perturb_span_length": 5,
    }.items():
        if not hasattr(rm.args, k):
            setattr(rm.args, k, v)
    # -------------------------------------------------------------


    init_models_and_tokenizers(rm.args)
    _ = prepare_data(rm.args)

    # Carry parsed list into rm.args for our loop
    rm.args.n_perturbations = n_pert_list
    rm.args.criterion = ns.criterion
    rm.args.save_path = ns.save_path

    # Attach to a lightweight container for convenience in run_attack
    exec_args = SimpleNamespace(**rm.args.__dict__)
    exec_args.n_perturbations = n_pert_list
    exec_args.criterion = ns.criterion
    exec_args.save_path = ns.save_path

    run_attack(exec_args)


if __name__ == "__main__":
    main()
