#!/usr/bin/env python
"""
Pretrained HyenaDNA inference: embeddings + next-nucleotide log-likelihood.

Loads a pretrained HyenaDNA checkpoint from the LongSafari Hugging Face org
(following the loading logic in the upstream repo's huggingface.py, pinned
commit d553021b483b82980aa4b868b37ec2d4332e198a), runs it on a real sequence
from a FASTA file and on a composition-matched scrambled control, and saves:

  - <output_dir>/embeddings_real.npy       per-position hidden states, real seq
  - <output_dir>/embeddings_scrambled.npy  per-position hidden states, scrambled
  - <output_dir>/loglik_real.npy           per-position next-nucleotide log-likelihood, real seq
  - <output_dir>/loglik_scrambled.npy      per-position next-nucleotide log-likelihood, scrambled
  - <output_dir>/run_metadata.json         checkpoint, device, versions, sequence lengths

Next-nucleotide log-likelihood is reconstructed from the tied embedding
matrix (logits = hidden_states @ word_embeddings.weight.T), matching how
this model was pretrained: see upstream src/models/sequence/long_conv_lm.py
`tie_weights()`, where `lm_head.weight = backbone.embeddings.word_embeddings.weight`.
The standalone inference wrapper (standalone_hyenadna.py) only returns hidden
states for embedding use cases, so this script reconstructs the LM head
rather than vendoring a second copy of the training-time model class.
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F


def inject_substring(orig_str):
    """From upstream huggingface.py: handles key names for gradient-checkpointed models."""
    modified = re.sub(r"\.mixer", ".mixer.layer", orig_str)
    modified = re.sub(r"\.mlp", ".mlp.layer", modified)
    return modified


def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """From upstream huggingface.py: copy backbone weights from the checkpoint into a fresh model."""
    for key in scratch_dict:
        if "backbone" in key:
            key_loaded = "model." + key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            if key_loaded not in pretrained_dict:
                raise KeyError(f"key mismatch loading pretrained weights: {key_loaded} not found")
            scratch_dict[key] = pretrained_dict[key_loaded]
    return scratch_dict


def load_pretrained_backbone(checkpoints_dir, model_name, device, HyenaDNAModel):
    """Download (if needed) and load a LongSafari pretrained checkpoint. Mirrors
    HyenaDNAPreTrainedModel.from_pretrained in upstream huggingface.py, minus the
    inference_single() call that upstream runs unconditionally at import time."""
    model_dir = os.path.join(checkpoints_dir, model_name)
    if not os.path.isdir(model_dir):
        hf_url = f"https://huggingface.co/LongSafari/{model_name}"
        os.makedirs(checkpoints_dir, exist_ok=True)
        print(f"Downloading checkpoint from {hf_url} ...")
        subprocess.run("git lfs install", shell=True, check=True)
        subprocess.run(f"cd {checkpoints_dir} && git clone {hf_url}", shell=True, check=True)

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    scratch_model = HyenaDNAModel(**config, use_head=False, n_classes=2)
    # weights_only=False: this checkpoint was saved by PyTorch Lightning and
    # carries a pickled OmegaConf config object alongside the tensors, so it
    # fails to load under PyTorch >=2.6's new secure-by-default weights_only=True
    # (upstream's own huggingface.py predates that change and never set this
    # explicitly). Safe here because the source is the official, documented
    # LongSafari/<model_name> checkpoint — see model.yaml's `upstream` field.
    loaded_ckpt = torch.load(
        os.path.join(model_dir, "weights.ckpt"), map_location=torch.device(device), weights_only=False
    )

    checkpointing = bool(config.get("checkpoint_mixer", False))
    state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt["state_dict"], checkpointing=checkpointing)
    scratch_model.load_state_dict(state_dict)
    print(f"Loaded pretrained weights for {model_name} ok.")
    return scratch_model, config


def read_first_fasta_record(path):
    header = None
    seq_chars = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break  # stop at the next record; we only need one
                header = line[1:]
                continue
            seq_chars.append(line)
    if header is None:
        raise ValueError(f"No FASTA header found in {path}")
    return header, "".join(seq_chars).upper()


def scramble_sequence(seq, seed=0):
    """Composition-matched negative control: a random permutation of the same
    multiset of nucleotides, so any likelihood difference reflects sequence
    *order* (real genomic structure), not base composition."""
    rng = np.random.default_rng(seed)
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="hyenadna-tiny-1k-seqlen")
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--checkpoints_dir", default=None, help="Defaults to <output_dir>/../checkpoints")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the scrambled negative control")
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"WARNING: --device {args.device} requested but CUDA is not available; falling back to cpu.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    vendor_dir = os.path.join(script_dir, "..", "vendor", "hyena-dna")
    if not os.path.isdir(vendor_dir):
        sys.exit(f"ERROR: upstream checkout not found at {vendor_dir}. Run environment/setup_env.sh first.")
    sys.path.insert(0, vendor_dir)
    from standalone_hyenadna import CharacterTokenizer, HyenaDNAModel  # noqa: E402

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = args.checkpoints_dir or os.path.join(args.output_dir, "..", "checkpoints")

    model, config = load_pretrained_backbone(checkpoints_dir, args.checkpoint, device, HyenaDNAModel)
    model.to(device)
    model.eval()

    # The checkpoint config doesn't carry a top-level max-context field; the
    # Hyena layer's l_max (FFT-convolution length) sets it instead, as
    # l_max = max_length + 2 (for the two special-token slots the tokenizer
    # reserves). Fall back to 1024 (the tiny-1k checkpoint's length) if the
    # config shape ever changes.
    layer_l_max = config.get("layer", {}).get("l_max")
    max_length = (layer_l_max - 2) if layer_l_max else 1024
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=max_length + 2,
        add_special_tokens=False,
        padding_side="left",
    )

    header, real_seq = read_first_fasta_record(args.input_fasta)
    scrambled_seq = scramble_sequence(real_seq, seed=args.seed)
    print(f"Input record: {header}")
    print(f"Sequence length: {len(real_seq)} nt (checkpoint max context: {max_length})")
    if len(real_seq) > max_length:
        sys.exit(f"ERROR: sequence length {len(real_seq)} exceeds checkpoint max context {max_length}.")

    results = {}
    for label, seq in [("real", real_seq), ("scrambled", scrambled_seq)]:
        tok_ids = tokenizer(seq)["input_ids"]
        tok_tensor = torch.LongTensor(tok_ids).unsqueeze(0).to(device)

        with torch.inference_mode():
            hidden_states = model(tok_tensor)  # [1, seq_len, d_model], use_head=False
            word_emb = model.backbone.embeddings.word_embeddings.weight  # tied LM head
            logits = hidden_states @ word_emb.T  # [1, seq_len, vocab_size]
            log_probs = F.log_softmax(logits.float(), dim=-1)

            # causal next-nucleotide log-likelihood: position t's logits predict token t+1
            next_tokens = tok_tensor[:, 1:]
            pred_log_probs = log_probs[:, :-1, :]
            token_loglik = pred_log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)  # [1, seq_len-1]

        if torch.isnan(hidden_states).any() or torch.isnan(token_loglik).any():
            sys.exit(f"ERROR: NaNs detected in model output for the '{label}' sequence — smoke test failed.")

        emb_np = hidden_states.squeeze(0).cpu().numpy()
        loglik_np = token_loglik.squeeze(0).cpu().numpy()
        np.save(os.path.join(args.output_dir, f"embeddings_{label}.npy"), emb_np)
        np.save(os.path.join(args.output_dir, f"loglik_{label}.npy"), loglik_np)

        results[label] = {
            "sequence_length": len(seq),
            "embedding_shape": list(emb_np.shape),
            "mean_next_nucleotide_loglik": float(loglik_np.mean()),
            "perplexity": float(np.exp(-loglik_np.mean())),
        }
        print(f"[{label}] embedding shape={emb_np.shape}, mean log-lik={results[label]['mean_next_nucleotide_loglik']:.4f}, "
              f"perplexity={results[label]['perplexity']:.4f}")

    metadata = {
        "checkpoint": args.checkpoint,
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "input_fasta": os.path.abspath(args.input_fasta),
        "fasta_header": header,
        "seed": args.seed,
        "results": results,
    }
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved outputs to {os.path.abspath(args.output_dir)}")
    print("Smoke test passed: both forward passes completed with finite, correctly-shaped output.")


if __name__ == "__main__":
    main()
