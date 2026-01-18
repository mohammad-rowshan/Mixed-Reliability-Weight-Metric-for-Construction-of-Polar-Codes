# Mixed Reliability–Weight Metric for Construction of Polar Codes

This repository implements a **mixed reliability–weight metric** for the construction of polar(-like) codes over real BPSK–AWGN channels.

It accompanies and is based on the paper:

> **Mohammad Rowshan and Vlad-Florin Dragoi**,  
> *Mixed Reliability–Weight Metric for Construction of Polar Codes*,  
> arXiv:2601.10376, 2026.  
> https://arxiv.org/abs/2601.10376

The goal is to design, for a given block length \(N\) and dimension \(K\), two information sets:

- **`I_rel`** – a *baseline polar code* constructed purely from synthetic-channel reliabilities (Gaussian Approximation on AWGN),
- **`I_mixed`** – a *mixed-metric polar code* that jointly optimizes:
  - the synthetic-channel error probability \(P_e(i)\), and
  - a distance-related term derived from a truncated union bound at the minimum distance \(w_{\min}\).

The main entry point is the function:

```python
generate_mixed_order_polar(N, K, alpha, EbN0_dB, restrict_degree_to_rel=True)
