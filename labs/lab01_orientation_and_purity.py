#!/usr/bin/env python3
"""Lab 01 — Matrix-Orientation Validator, bf16 SVD Trap & AST Purity Gate.

Companion executable lab for Chapter 1 of Weight-Space Engineering.
Runs deterministically (seed=42) in pure Python + NumPy with zero autograd.
"""
from __future__ import annotations

import ast
from pathlib import Path
import numpy as np

LABS_DIR = Path(__file__).resolve().parent

BANNED_MODULES = frozenset({"torch.optim", "bitsandbytes", "trl"})
BANNED_ATTR_CALLS = frozenset({"backward", "step", "zero_grad"})


def to_bf16_f64(x: np.ndarray) -> np.ndarray:
    """Exact IEEE-754 round-to-nearest-even bfloat16 truncation cast to float64."""
    f32 = np.asarray(x, dtype=np.float32)
    u32 = f32.view(np.uint32)
    round_bias = ((u32 >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    bf16_bits = (u32 + round_bias) & np.uint32(0xFFFF0000)
    return bf16_bits.view(np.float32).astype(np.float64)


def verify_peft_lora_orientation(
    lora_A: np.ndarray,
    lora_B: np.ndarray,
    d_in: int,
    d_out: int,
    fan_in_fan_out: bool = False,
) -> int:
    """Fail-closed orientation validator for column-vector [d_out, d_in] LoRA."""
    if fan_in_fan_out:
        raise ValueError(
            "Rejected fan_in_fan_out=True (Conv1D [d_in, d_out] convention)."
        )
    if lora_A.ndim != 2 or lora_B.ndim != 2:
        raise ValueError("LoRA factors must be 2D matrices.")
    r_a, cols_a = lora_A.shape
    rows_b, r_b = lora_B.shape
    if r_a != r_b:
        raise ValueError(f"Rank mismatch: A has r={r_a}, B has r={r_b}.")
    if cols_a != d_in or rows_b != d_out:
        raise ValueError(
            f"Shape mismatch: expected A=[r,{d_in}], B=[{d_out},r]; "
            f"got A={lora_A.shape}, B={lora_B.shape}."
        )
    if r_a > min(d_in, d_out):
        raise ValueError(f"Invalid rank r={r_a} > min(d_in, d_out).")
    return r_a


def compute_peft_scaling(
    r: int,
    lora_alpha: float,
    use_rslora: bool = False,
    module_name: str = "",
    alpha_pattern: dict[str, float] | None = None,
) -> float:
    """Compute exact PEFT LoRA / rsLoRA scaling factor with alpha_pattern overrides."""
    eff_alpha = float(lora_alpha)
    if alpha_pattern:
        for pattern, override in alpha_pattern.items():
            if pattern in module_name:
                eff_alpha = float(override)
                break
    denom = np.sqrt(float(r)) if use_rslora else float(r)
    return eff_alpha / denom


def nullspace_edit_f64(
    W: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    V1: np.ndarray,
    reg: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Column-vector [d_out, d_in] float64 null-space associative update."""
    assert W.ndim == 2 and W.shape[1] == K0.shape[0]
    W64 = W.astype(np.float64)
    K0_64 = K0.astype(np.float64)
    U, s, _ = np.linalg.svd(K0_64, full_matrices=False)
    u_act = U[:, s > 1e-11 * s[0]]
    eye = np.eye(W64.shape[1], dtype=np.float64)
    p_null = eye - u_act @ u_act.T
    p_null = 0.5 * (p_null + p_null.T)

    k1_p = p_null @ K1.astype(np.float64)
    resid = V1.astype(np.float64) - W64 @ K1.astype(np.float64)
    gram = k1_p @ k1_p.T + reg * eye
    w_tilde = resid @ np.linalg.solve(gram, k1_p).T
    w_new = (W64 + w_tilde @ p_null).astype(W.dtype)
    return w_new, p_null


def find_ast_gradient_violations(source_code: str) -> list[str]:
    """Return any banned autograd/optimizer imports or method calls in Python AST."""
    tree = ast.parse(source_code)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == b or alias.name.startswith(b + ".")
                    for b in BANNED_MODULES
                ):
                    violations.append(f"import:{alias.name}@L{node.lineno}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if any(
                node.module == b or node.module.startswith(b + ".")
                for b in BANNED_MODULES
            ):
                violations.append(f"from:{node.module}@L{node.lineno}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in BANNED_ATTR_CALLS:
                violations.append(f"call:.{node.func.attr}()@L{node.lineno}")
    return violations


def run_lab() -> None:
    rng = np.random.default_rng(42)
    print("=== Lab 01: Matrix Orientation, bf16 SVD Trap & AST Purity ===")

    # 1. Orientation & PEFT scaling checks
    d_in, d_out, r = 64, 96, 16
    A = rng.standard_normal((r, d_in))
    B = rng.standard_normal((d_out, r))
    assert verify_peft_lora_orientation(A, B, d_in, d_out) == r

    for bad_case in (
        (A, B, d_in, d_out, True),
        (A.T, B.T, d_in, d_out, False),
    ):
        caught = False
        try:
            verify_peft_lora_orientation(*bad_case)
        except ValueError:
            caught = True
        assert caught, "Failed to reject invalid orientation!"

    s_std = compute_peft_scaling(r=16, lora_alpha=32.0, use_rslora=False)
    s_rs = compute_peft_scaling(r=16, lora_alpha=32.0, use_rslora=True)
    assert np.isclose(s_std, 2.0) and np.isclose(s_rs, 8.0)
    print(f"[PASS] Orientation & rsLoRA scaling verified (std={s_std}, rs={s_rs})")

    # 2. Dual-convention equivalence (Exercise 1.1)
    d_model, d_k = 48, 16
    W_Q = rng.standard_normal((d_k, d_model))
    W_K = rng.standard_normal((d_k, d_model))
    W_V = rng.standard_normal((d_k, d_model))
    W_O = rng.standard_normal((d_model, d_k))
    W_Q2 = rng.standard_normal((d_k, d_model))
    W_K2 = rng.standard_normal((d_k, d_model))

    W_QK_col = W_Q.T @ W_K
    W_OV_col = W_O @ W_V
    W_Kcomp_col = (W_Q2.T @ W_K2) @ W_OV_col

    W_Q2_row, W_K2_row = W_Q2.T, W_K2.T
    W_QK2_row = W_Q2_row @ W_K2_row.T
    W_OV1_row = W_V.T @ W_O.T
    W_Kcomp_row = W_QK2_row @ W_OV1_row.T
    assert np.max(np.abs(W_Kcomp_col - W_Kcomp_row)) < 1e-13

    r_vec = rng.standard_normal(d_model)
    r_hat = r_vec / np.linalg.norm(r_vec)
    W_out_ab_col = (np.eye(d_model) - np.outer(r_hat, r_hat)) @ W_O
    W_out_ab_row = W_O.T @ (np.eye(d_model) - np.outer(r_hat, r_hat))
    assert np.max(np.abs(W_out_ab_col.T - W_out_ab_row)) < 1e-14
    print("[PASS] Dual-convention K-composition & abliteration exact (<1e-13)")

    # 3. float64 vs bfloat16 SVD null-space precision trap (Exercise 1.2)
    U_raw, _ = np.linalg.qr(rng.standard_normal((64, 40)))
    V_raw, _ = np.linalg.qr(rng.standard_normal((40, 40)))
    # 30 strong modes + 10 fine active modes at s_i / s_0 = 3e-3
    sigmas = np.concatenate([np.geomspace(10.0, 0.5, 30), np.full(10, 0.03)])
    K0 = (U_raw * sigmas) @ V_raw.T
    W0 = rng.standard_normal((32, 64))
    K1 = rng.standard_normal((64, 4))
    V1 = rng.standard_normal((32, 4))

    W_f64, P_f64 = nullspace_edit_f64(W0, K0, K1, V1)
    drift_f64 = np.linalg.norm((W_f64 - W0) @ K0, ord="fro")
    edit_err_f64 = np.linalg.norm(V1 - W_f64 @ K1, ord="fro")

    # Simulate bfloat16 covariance / Gram SVD at relative cutoff tau = 1e-2
    K0_bf = to_bf16_f64(K0)
    gram_bf = to_bf16_f64(K0_bf @ K0_bf.T)
    U_bf, lam_bf, _ = np.linalg.svd(gram_bf, full_matrices=True)
    U_null_bf = U_bf[:, lam_bf <= 1e-2 * lam_bf[0]]
    P_bf = to_bf16_f64(U_null_bf @ U_null_bf.T)
    K1_p_bf = P_bf @ K1
    R_bf = V1 - W0 @ K1
    w_tilde_bf = R_bf @ np.linalg.solve(
        K1_p_bf @ K1_p_bf.T + 1e-6 * np.eye(64), K1_p_bf
    ).T
    W_bf = W0 + w_tilde_bf @ P_bf
    drift_bf = np.linalg.norm((W_bf - W0) @ K0, ord="fro")

    assert drift_f64 < 1e-12, f"Expected float64 drift < 1e-12, got {drift_f64}"
    assert edit_err_f64 < 1e-4, f"Expected accurate edit, got {edit_err_f64}"
    assert drift_bf > 0.05, f"Expected bf16 aliasing drift > 0.05, got {drift_bf}"
    print(
        f"[PASS] Null-space drift: float64={drift_f64:.3e} vs "
        f"bfloat16_gram={drift_bf:.3e} (edit_res={edit_err_f64:.3e})"
    )

    # 4. AST zero-gradient purity linter across labs/
    bad_snippet = "import torch.optim as opt\nloss.backward()\nopt.step()\n"
    detected = find_ast_gradient_violations(bad_snippet)
    assert len(detected) == 3, f"Expected 3 synthetic violations, got {detected}"

    lab_files = sorted(LABS_DIR.glob("lab*.py"))
    assert len(lab_files) >= 1
    for lf in lab_files:
        v = find_ast_gradient_violations(lf.read_text(encoding="utf-8"))
        assert not v, f"AST purity violation in {lf.name}: {v}"
    print(f"[PASS] Zero-gradient AST purity verified across {len(lab_files)} lab(s)")


if __name__ == "__main__":
    run_lab()
