# Weight-Space Engineering

[![Verify Lab 01 (Zero Autograd)](https://github.com/Impius-Lab/weight-space-engineering/actions/workflows/ci.yml/badge.svg)](https://github.com/Impius-Lab/weight-space-engineering/actions/workflows/ci.yml)
[![License: MIT (Sample Code)](https://img.shields.io/badge/Code_License-MIT-blue.svg)](LICENSE)
[![ISBN: 979-8-1447-3586-5](https://img.shields.io/badge/ISBN-979--8--1447--3586--5-amber.svg)](https://weight-space-engineering.weight-space-engineering.workers.dev/)

**Official Public Companion Repository for *Weight-Space Engineering: Compiling, Editing, and Merging Neural Network Weights Without Gradient Descent* (1st Edition) by Aguentil H.**

- **Read Chapter 1 Free Online + Full 40-Paper Bibliography:** <https://weight-space-engineering.weight-space-engineering.workers.dev/chapter-1/>
- **Download the Complete DRM-Free Handbook (94-Page Print-Typeset PDF + Dark-Mode EPUB 3) & Full 8-Lab Executable Python Suite (USD 49):** <https://whop.com/weight-space-engineering/weight-space-engineering-handbook-python-lab-suite/>
- **Print Paperback (`6" x 9"`, ISBN `979-8-1447-3586-5`) & Kindle Edition:** Coming to Amazon (links wired upon KDP release).

---

## Overview: Neural Weights as a Directly Addressable Program Representation

Standard post-training pipelines (`SFT`, `LoRA`, `DPO`, `RLHF`) modify transformer checkpoints indirectly by running stochastic gradient descent over batches of token activations. Yet modern linear algebra, compiler theory, and mechanistic interpretability make it possible to **compile, audit, surgically edit, and merge parameter tensors directly in closed form**—often in milliseconds on CPU/GPU with **zero `.backward()` calls**.

Across production PyTorch and Hugging Face checkpoints, two silent numerical hazards break naive weight-space scripts:

### 1. Silent Matrix-Orientation Transposes (`[d_out, d_in]` vs. `[d_in, d_out]`)

When a square projection satisfies $`d_{\text{out}} = d_{\text{in}}`$, applying a rank-one `ROME` update or orthogonal subspace projector under the wrong storage convention silently corrupts orthogonal directions **without raising a runtime shape error**:

| Ecosystem / Layer Class | Weight Tensor Shape | Forward Contraction | Rank-$`r`$ Adapter Contraction |
| :--- | :---: | :---: | :---: |
| **PyTorch `nn.Linear`** (`LLaMA`, `Mistral`, `Qwen`, `Gemma`) | $`[d_{\text{out}}, d_{\text{in}}]`$ | $`y = x W^{\top} + b`$ | $`\Delta W = \frac{\alpha}{r} B A`$ with $`B \in \mathbb{R}^{d_{\text{out}} \times r}`$, $`A \in \mathbb{R}^{r \times d_{\text{in}}}`$ |
| **Hugging Face `Conv1D`** (`GPT-2`, `fan_in_fan_out=True`) | $`[d_{\text{in}}, d_{\text{out}}]`$ | $`y = x W + b`$ | $`\Delta W = \frac{\alpha}{r} A^{\top} B^{\top}`$ |
| **`TransformerLens` (`HookedTransformer`)** | $`[n_{\text{heads}}, d_{\text{model}}, d_{\text{head}}]`$ | $`z_h = x W_V^{(h)}`$ | Head-sliced row-vector convention |

### 2. `bfloat16` Gram Aliasing in Null-Space Knowledge Editing (`AlphaEdit`)

To insert new factual associations $`(K_1, V_1)`$ into an MLP down-projection $`W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}`$ without disturbing preserved knowledge keys $`K_0`$, null-space editing projects the update onto the left null space of $`K_0`$:

```math
P_{\text{null}} = I - U_{\text{act}} U_{\text{act}}^{\top}, \qquad \Delta W = \bigl(V_1 - W K_1\bigr)\bigl(P_{\text{null}} K_1\bigr)^{\top}\Bigl(\bigl(P_{\text{null}} K_1\bigr)\bigl(P_{\text{null}} K_1\bigr)^{\top} + \lambda I\Bigr)^{-1} P_{\text{null}}
```

Computing $`K_0 K_0^{\top}`$ in `bfloat16` ($`\epsilon_{\text{mach}} \approx 7.8 \times 10^{-3}`$) squares singular values and truncates fine active modes around $`\sigma_i / \sigma_0 \approx 3 \times 10^{-3}`$, leaking **`> 5e-2` Frobenius interference** into preserved memories. Performing the SVD directly on $`K_0`$ in `float64` before casting $`W + \Delta W`$ back to the checkpoint dtype holds null-space drift below **`1e-12`** (verified empirically in [`labs/lab01_orientation_and_purity.py`](labs/lab01_orientation_and_purity.py)).

---

## Quickstart: Run Free Sample `Lab 01`

`Lab 01` runs deterministically (`seed=42`) in pure Python 3.10+ and NumPy with an AST static gate banning `torch.optim`, `bitsandbytes`, `trl`, `.backward()`, `.step()`, and `.zero_grad()`:

```bash
pip install numpy
python labs/lab01_orientation_and_purity.py
```

Expected output (`exit 0`):

```text
=== Lab 01: Matrix Orientation, bf16 SVD Trap & AST Purity ===
[PASS] Orientation & rsLoRA scaling verified (std=2.0, rs=8.0)
[PASS] Dual-convention K-composition & abliteration exact (<1e-13)
[PASS] Null-space drift: float64=3.140e-14 vs bfloat16_gram=4.342e+00 (edit_res=5.380e-06)
[PASS] Zero-gradient AST purity verified across 1 lab(s)
```

---

## Complete 8-Chapter & 8-Lab Architecture

This public repository includes **Free Sample `Lab 01`** and the **40-Paper Curated Bibliography** ([`REFERENCES.md`](REFERENCES.md) / [`CITATION.bib`](CITATION.bib)). The complete 8-lab suite (`128` deterministic numerical assertions) ships with the [Complete Handbook + Python Lab Bundle](https://whop.com/weight-space-engineering/weight-space-engineering-handbook-python-lab-suite/):

| Ch / Lab | Module | Core Algorithms & Verified Invariants | Availability |
| :---: | :--- | :--- | :---: |
| **01** | `lab01_orientation_and_purity.py` | `[d_out, d_in]` vs `[d_in, d_out]` LoRA/rsLoRA contracts, `float64` vs `bf16` SVD null-space trap, AST zero-autograd gate | **Included Here (MIT)** |
| **02** | `lab02_constructive_mlp.py` | Exact ReLU Boolean logic gates, 1D/2D `CPWL` splines, 2-neuron bias-free `SwiGLU` bilinear multiplier | [Complete Suite](https://whop.com/weight-space-engineering/weight-space-engineering-handbook-python-lab-suite/) |
| **03** | `lab03_compiling_transformers.py` | `RASP`-style SSA residual-stream allocation, `BOS` attention sinks, $`O(\log L)`$ parabolic hull lookup memory | [Complete Suite](https://whop.com/weight-space-engineering/weight-space-engineering-handbook-python-lab-suite/) |
| **04** | `lab04_static_weight_audit.py` | Zero-forward-pass `QK`/`OV` bilinear head spectra, Shannon effective rank (`erank`), cold Logit Lens | [Complete Suite](https://whop.com/weight-space-engineering/weight-space-engineering-handbook-python-lab-suite/) |
| **05** | `lab05_alphaedit_nullspace.py` | Closed-form rank-1 `ROME`, multi-fact `MEMIT`, centered-$`K_0`$ `float64` `AlphaEdit` null-space solver | [Complete Suite](https://whop.com/weight-space-engineering/weight-space-engineering-handbook-python-lab-suite/) |
| **06** | `lab06_abliteration_and_clamps.py` | Contrastive directional orthogonalization (abliteration), `RMSNorm` norm guardrails, `LASER` spectral rank clamps | [Complete Suite](https://whop.com/weight-space-engineering/weight-space-engineering-handbook-python-lab-suite/) |
| **07** | `lab07_qr_svd_and_rebasin.py` | Task Arithmetic, `DARE-TIES`, `SLERP`, dense-free `QR-SVD` multi-LoRA compression, `Git Re-Basin` Hungarian solver | [Complete Suite](https://whop.com/weight-space-engineering/weight-space-engineering-handbook-python-lab-suite/) |
| **08** | `lab08_dual_gate_pipeline.py` | `< 1%` hyper-sparse delta masking, Dual Efficacy + Locality KL Gate, post-surgery `INT8` / `NF4` quantization folding | [Complete Suite](https://whop.com/weight-space-engineering/weight-space-engineering-handbook-python-lab-suite/) |

---

## Citation

If this handbook or reference suite helps your research or engineering workflow, cite via [`CITATION.bib`](CITATION.bib):

```bibtex
@book{aguentil2026weightspace,
  title     = {Weight-Space Engineering: Compiling, Editing, and Merging Neural Network Weights Without Gradient Descent},
  author    = {Aguentil, H.},
  year      = {2026},
  edition   = {1st},
  isbn      = {979-8-1447-3586-5},
  url       = {https://weight-space-engineering.weight-space-engineering.workers.dev/}
}
```

---

## Disclaimer, Trademarks & Responsible Use

1. **Educational & Research Purpose (`AS IS`):** All mathematical derivations, reference scripts, and companion documentation are provided `"AS IS"` for educational and scientific research purposes without warranty of any kind.
2. **Nominative Trademark Notice:** `PyTorch`, `Hugging Face`, `LLaMA`, `GPT-2`, `Mistral`, `Qwen`, `Gemma`, and other framework or model names referenced for tensor-layout compatibility are trademarks of their respective owners; this publication is independent and not affiliated with or endorsed by them.
3. **Checkpoint License Compliance:** Practitioners applying weight-space auditing, editing, subspace projection, or merging techniques to third-party pretrained checkpoints are solely responsible for complying with the underlying model's license, attribution requirements, and acceptable-use policies.
