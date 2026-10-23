#!/usr/bin/env python
"""
Validation for the HyenaDNA inference recipe: a smoke test plus one
substantive scientific check.

Smoke test
----------
All expected output files exist, have the documented shapes, and contain
only finite values (no NaN/Inf) — i.e. the forward pass actually ran and
didn't silently produce garbage.

Substantive validation check
-----------------------------
HyenaDNA was pretrained with a causal next-nucleotide objective on real
human genomic sequence. If the pretrained weights actually learned
something about real genomic/transcript structure (rather than being
equivalent to a random projection), the model should assign a REAL human
mRNA sequence a *lower* next-nucleotide perplexity (higher likelihood) than
a random shuffle of the exact same nucleotides (same A/C/G/T composition,
scrambled order). This is a standard sanity check for autoregressive
genomic language models: order-dependent structure (codon periodicity,
k-mer biases, splice/regulatory motifs) is exactly what a real-vs-scrambled
likelihood gap is evidence of.

This is not a claim of state-of-the-art accuracy — it is a check that the
loaded checkpoint behaves like a genomic language model at all, which is
the right bar for "did inference reproduce correctly," not "does this match
a paper benchmark table."
"""

import argparse
import json
import os
import sys

import numpy as np


def smoke_test(results_dir, metadata):
    problems = []
    for label in ("real", "scrambled"):
        emb_path = os.path.join(results_dir, f"embeddings_{label}.npy")
        ll_path = os.path.join(results_dir, f"loglik_{label}.npy")

        if not os.path.exists(emb_path):
            problems.append(f"missing {emb_path}")
            continue
        if not os.path.exists(ll_path):
            problems.append(f"missing {ll_path}")
            continue

        emb = np.load(emb_path)
        ll = np.load(ll_path)

        expected_shape = metadata["results"][label]["embedding_shape"]
        if list(emb.shape) != expected_shape:
            problems.append(f"{label}: embedding shape {list(emb.shape)} != recorded {expected_shape}")

        if not np.isfinite(emb).all():
            problems.append(f"{label}: embeddings contain NaN/Inf")
        if not np.isfinite(ll).all():
            problems.append(f"{label}: next-nucleotide log-likelihoods contain NaN/Inf")

        if ll.shape[0] != emb.shape[0] - 1:
            problems.append(
                f"{label}: expected {emb.shape[0] - 1} log-likelihood values (seq_len - 1 for causal shift), got {ll.shape[0]}"
            )

    return problems


def substantive_check(metadata, margin_nats=0.0):
    real = metadata["results"]["real"]
    scrambled = metadata["results"]["scrambled"]

    real_ll = real["mean_next_nucleotide_loglik"]
    scrambled_ll = scrambled["mean_next_nucleotide_loglik"]
    gap = real_ll - scrambled_ll  # nats; expect > 0 (real sequence has higher mean log-likelihood)

    passed = gap > margin_nats
    return {
        "check": "real sequence should have higher mean next-nucleotide log-likelihood (lower perplexity) than its scrambled control",
        "real_mean_loglik_nats": real_ll,
        "scrambled_mean_loglik_nats": scrambled_ll,
        "real_perplexity": real["perplexity"],
        "scrambled_perplexity": scrambled["perplexity"],
        "loglik_gap_nats": gap,
        "margin_required_nats": margin_nats,
        "passed": passed,
        "interpretation": (
            "PASS: the pretrained model finds the real human mRNA sequence more "
            "predictable (lower perplexity) than a composition-matched random "
            "shuffle, consistent with the checkpoint having learned real genomic "
            "sequence structure during pretraining."
            if passed else
            "FAIL: the model did not assign lower perplexity to the real sequence "
            "than to the scrambled control. This suggests a loading or inference "
            "bug (e.g. wrong checkpoint, broken weight tying, or a tokenization "
            "mismatch) rather than a genuine model limitation, since even a lightly "
            "trained genomic LM should show some real-vs-scrambled gap on a coding mRNA."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--margin_nats", type=float, default=0.0,
                         help="Minimum required (real - scrambled) mean log-likelihood gap, in nats, to count as passing.")
    args = parser.parse_args()

    metadata_path = os.path.join(args.results_dir, "run_metadata.json")
    if not os.path.exists(metadata_path):
        sys.exit(f"ERROR: {metadata_path} not found — run scripts/run_inference.py first.")
    with open(metadata_path) as f:
        metadata = json.load(f)

    smoke_problems = smoke_test(args.results_dir, metadata)
    substantive = substantive_check(metadata, margin_nats=args.margin_nats)

    report = {
        "smoke_test": {
            "passed": len(smoke_problems) == 0,
            "problems": smoke_problems,
        },
        "substantive_validation": substantive,
        "checkpoint": metadata["checkpoint"],
        "device": metadata["device"],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print("== Smoke test ==")
    if smoke_problems:
        for p in smoke_problems:
            print(f"  FAIL: {p}")
    else:
        print("  PASS: all output files present, correctly shaped, and finite.")

    print("\n== Substantive validation ==")
    print(f"  real sequence:      mean log-lik = {substantive['real_mean_loglik_nats']:.4f} nats, "
          f"perplexity = {substantive['real_perplexity']:.4f}")
    print(f"  scrambled control:  mean log-lik = {substantive['scrambled_mean_loglik_nats']:.4f} nats, "
          f"perplexity = {substantive['scrambled_perplexity']:.4f}")
    print(f"  gap (real - scrambled): {substantive['loglik_gap_nats']:.4f} nats "
          f"({'PASS' if substantive['passed'] else 'FAIL'})")
    print(f"  {substantive['interpretation']}")
    print(f"\nFull report written to {os.path.abspath(args.report)}")

    if smoke_problems or not substantive["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
