#!/usr/bin/env python3
"""
Q26 Kvantna regresija (1/5) — tehnika: HHL (Harrow-Hassidim-Lloyd) linearna regresija
(čisto kvantno: rešava A·w = b preko QPE + ancilla-rotacije + inverzne QPE,
bez klasične regresije i bez iterativne optimizacije).

Koncept:
  Klasična linearna regresija: w = A⁻¹·b, gde je A Hermitska matrica feature-a
  izvedena iz CELOG CSV-a, a b target-vektor. HHL proizvodi kvantno stanje
  |w⟩ ∝ A⁻¹|b⟩ egzaktno preko sledećeg niza:
    1) StatePreparation(|b⟩) na b-registar.
    2) QPE za U = e^{iAt}: procenjuje λ_k eigvalue-e u phase-registru.
    3) Controlled ancilla rotacija Ry(2·arcsin(C/λ)) po svakom eigenvalue j ∈ ph_reg,
       sa ctrl_state=j; ovo „invertuje" λ u amplitude.
    4) Inverzna QPE (QFT + inverzni controlled-U power-i + H^⊗n_phase) da se
       phase-registar disentanglu-je.
    5) Post-selekcija anc=1 + marginalizacija phase-registra →
       |w⟩ ∝ A⁻¹|b⟩ je na b-registru.
    6) Amplitude → bias_39 → TOP-7 = NEXT.

A matrica (Hermitska, CELI CSV):
    A = P + diag(freq_csv) + α·I  (padded na 2^nq dimenzije, skalirana).
    P[i,j] — pair-co-occurrence matrica CELOG CSV-a (simetrična).
    α — ridge-regularizacija (grid parametar).
    Skaliranje: eigenvalue raspon u (0, 0.5] (t=2π → fase u (0, 0.5]).

b vektor:
    b = freq_csv, amp-kodiran na b-registar (dim 2^nq, nq=6 da stane 39).

Razlika u odnosu na slične fajlove:
  Q8 (QAE):        procenjuje AMPLITUDU |⟨ψ_0|ψ⟩|, ne rešava linearni sistem.
  Q16 (QAOA):      iterativni Hamiltonian-exp za cost/mixer, NE HHL inverzija.
  HHL:             egzaktna kvantna linearna algebra — A⁻¹ kroz QPE + rotaciju + QPE†.

Sve deterministički: seed=39; A, b iz CELOG CSV-a (pravilo 10).
Deterministička grid-optimizacija (n_phase, α) po cos(bias_39, freq_csv).
Klasično se računa samo konstrukcija matrica A i e^{iAt} (compilation-step,
kako to HHL protokol i predviđa — eigendecomposition za unitary gate).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, QFT
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

NQ_B = 6
GRID_PHASE = (4, 5)
GRID_ALPHA = (0.5, 1.0, 2.0)
T_HHL = 2.0 * np.pi


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Konstrukcija Hermitske matrice A (CELI CSV)
# =========================
def pair_matrix(H: np.ndarray) -> np.ndarray:
    P = np.zeros((N_MAX, N_MAX), dtype=np.float64)
    for row in H:
        for a in row:
            for b in row:
                if a != b and 1 <= a <= N_MAX and 1 <= b <= N_MAX:
                    P[a - 1, b - 1] += 1.0
    return P


def build_matrix_A(H: np.ndarray, nq: int, alpha: float) -> Tuple[np.ndarray, float]:
    """A = P + diag(freq) + α·I, padded na 2^nq, skalirana u (0, 0.5]."""
    P = pair_matrix(H)
    f = freq_vector(H)
    f_mean = float(f.mean()) + 1e-18
    P_n = P / f_mean
    D = np.diag(f / f_mean)
    A = P_n + D + float(alpha) * np.eye(N_MAX)

    dim = 2 ** nq
    A_pad = float(alpha) * np.eye(dim, dtype=np.float64)
    A_pad[:N_MAX, :N_MAX] = A

    eigs = np.linalg.eigvalsh(A_pad)
    max_abs = float(max(abs(eigs.max()), abs(eigs.min())))
    scale = max_abs * 2.0 if max_abs > 0 else 1.0
    A_scaled = A_pad / scale
    return A_scaled, scale


def matrix_exp_hermitian(A: np.ndarray, t: float) -> np.ndarray:
    eigs, V = np.linalg.eigh(A)
    diag_exp = np.exp(1j * eigs * t)
    U = V @ np.diag(diag_exp) @ V.conj().T
    return U


# =========================
# HHL kvantno kolo
# Registri: b (nq), phase (n_phase), anc (1) — qiskit little-endian: b najniži.
# =========================
def build_hhl_circuit(
    A_scaled: np.ndarray, b_amp: np.ndarray, n_b: int, n_phase: int, t: float, C: float
) -> QuantumCircuit:
    b_reg = QuantumRegister(n_b, name="b")
    ph_reg = QuantumRegister(n_phase, name="p")
    anc = QuantumRegister(1, name="a")
    qc = QuantumCircuit(b_reg, ph_reg, anc)

    qc.append(StatePreparation(b_amp.tolist()), b_reg)

    qc.h(ph_reg)

    for k in range(n_phase):
        U_pow_matrix = matrix_exp_hermitian(A_scaled, t * (2 ** k))
        U_gate = UnitaryGate(U_pow_matrix, label=f"U^{2**k}")
        cU = U_gate.control(1)
        qc.append(cU, [ph_reg[k]] + list(b_reg))

    qc.append(QFT(n_phase, inverse=True, do_swaps=True), ph_reg)

    for j in range(1, 2 ** n_phase):
        lambda_j = (j / float(2 ** n_phase)) * (2.0 * np.pi / t)
        if lambda_j < 1e-12:
            continue
        ratio = float(C / lambda_j)
        if ratio > 1.0:
            ratio = 1.0
        if ratio < -1.0:
            ratio = -1.0
        theta_j = 2.0 * float(np.arcsin(ratio))
        ry_sub = QuantumCircuit(1, name=f"Ry_{j}")
        ry_sub.ry(theta_j, 0)
        ry_gate = ry_sub.to_gate(label=f"Ry_{j}")
        cry = ry_gate.control(num_ctrl_qubits=n_phase, ctrl_state=j)
        qc.append(cry, list(ph_reg) + [anc[0]])

    qc.append(QFT(n_phase, inverse=False, do_swaps=True), ph_reg)

    for k in reversed(range(n_phase)):
        U_pow_inv = matrix_exp_hermitian(A_scaled, -t * (2 ** k))
        U_gate_inv = UnitaryGate(U_pow_inv, label=f"U^-{2**k}")
        cU_inv = U_gate_inv.control(1)
        qc.append(cU_inv, [ph_reg[k]] + list(b_reg))

    qc.h(ph_reg)

    return qc


def hhl_state_probs(
    H: np.ndarray, n_b: int, n_phase: int, alpha: float
) -> Tuple[np.ndarray, float]:
    """Vraća (marginala b-registra post-select anc=1, ph=0 [normalizovana], P(anc=1, ph=0))."""
    A_scaled, _scale = build_matrix_A(H, n_b, alpha)
    b_amp = amp_from_freq(freq_vector(H), n_b)
    min_eig = float(np.min(np.abs(np.linalg.eigvalsh(A_scaled))))
    C = max(min_eig, 1.0 / (2 ** n_phase))

    qc = build_hhl_circuit(A_scaled, b_amp, n_b, n_phase, T_HHL, C)
    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2

    dim_b = 2 ** n_b
    dim_ph = 2 ** n_phase
    dim_anc = 2
    mat = p.reshape(dim_anc, dim_ph, dim_b)
    p_post = float(mat[1, 0, :].sum())
    if p_post < 1e-18:
        return np.zeros(dim_b, dtype=np.float64), 0.0
    p_b = mat[1, 0, :] / p_post
    return p_b, p_post


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (n_phase, α)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for n_phase in GRID_PHASE:
        for alpha in GRID_ALPHA:
            try:
                p_b, p_post = hhl_state_probs(H, NQ_B, int(n_phase), float(alpha))
                bi = bias_39(p_b)
                score = cosine(bi, f_csv_n)
            except Exception:
                continue
            key = (score, int(n_phase), -float(alpha))
            if best is None or key > best[0]:
                best = (
                    key,
                    dict(n_phase=int(n_phase), alpha=float(alpha),
                         score=float(score), p_post=float(p_post)),
                )
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q26 Kvantna regresija (1/5) — HHL: CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED, "| nq_b:", NQ_B, "| t:", round(float(T_HHL), 6))

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "n_phase=", best["n_phase"],
        "| α (ridge):", best["alpha"],
        "| P(anc=1, ph=0):", round(float(best["p_post"]), 6),
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    A_scaled, scale = build_matrix_A(H, NQ_B, float(best["alpha"]))
    eigs = np.linalg.eigvalsh(A_scaled)
    print("--- A (skalirana) eigenvalue info ---")
    print(f"  min|λ|={float(np.min(np.abs(eigs))):.6f}  max|λ|={float(np.max(np.abs(eigs))):.6f}  "
          f"cond_number={float(np.max(np.abs(eigs)) / max(np.min(np.abs(eigs)), 1e-12)):.4f}")

    p_b, _ = hhl_state_probs(H, NQ_B, int(best["n_phase"]), float(best["alpha"]))
    pred = pick_next_combination(p_b)
    print("--- glavna predikcija (HHL |x⟩ post-select anc=1, ph=0) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q26 Kvantna regresija (1/5) — HHL: CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | nq_b: 6 | t: 6.283185
BEST hparam: n_phase= 4 | α (ridge): 0.5 | P(anc=1, ph=0): 0.234486 | cos(bias, freq_csv): 0.854623
--- A (skalirana) eigenvalue info ---
  min|λ|=0.033280  max|λ|=0.500000  cond_number=15.0239
--- glavna predikcija (HHL |x⟩ post-select anc=1, ph=0) ---
predikcija NEXT: (3, 4, 14, 17, 19, 22, 25)
"""



"""
Q26_qreg1_HHL.py — tehnika: HHL kvantna linearna regresija.

Koncept:
A·w = b, gde je A Hermitska matrica iz CELOG CSV-a (pair + freq-diag + α·I),
b = freq_csv amp-kodiran. HHL kolo proizvodi |w⟩ ∝ A⁻¹|b⟩ egzaktno preko
QPE + ancilla-inverzije + QPE†.

Kolo (nq_b + n_phase + 1 qubit-a):
  StatePreparation(|b⟩) na b-registar.
  QPE: H^⊗n_phase na ph; za k=0..n_phase-1 controlled-U^(2^k) (U=e^{iAt});
       inverzna QFT na ph.
  Controlled Ry na anc, po svakom j ∈ ph (ctrl_state=j): Ry(2·arcsin(C/λ_j)).
  QPE†: QFT na ph; za k=n_phase-1..0 controlled-U^(-2^k); H^⊗n_phase na ph.
Readout:
  Post-select anc=1 ∧ ph=0, marginala b → bias_39 → TOP-7 = NEXT.

Tehnike:
Egzaktna kvantna linearna algebra (A⁻¹ kroz QPE + rotaciju + QPE†).
UnitaryGate iz eigendecomposition (kompilacioni korak — pure-quantum na hardverskom nivou).
Multi-controlled Ry po eigenvalue-u j (ctrl_state=j na phase registru).
Post-selekcija na anc=1 i ph=0 (standardni HHL protokol).
Egzaktni Statevector (bez uzorkovanja).
Deterministička grid-optimizacija (n_phase, α).

Prednosti:
Kanonski kvantni primitive za linearnu regresiju — A⁻¹|b⟩ bez iterativnog solver-a.
Čisto kvantno: bez klasičnog gradient-descent-a, bez SGD-a, bez ML lib-ova.
Regresijski koeficijenti su amplitude kvantnog stanja.
Ceo CSV (pravilo 10): A i b oba izvedena iz CELOG CSV-a.

Nedostaci:
Post-selekcija na anc=1 ∧ ph=0 ima P(post) često nisku — slaba efikasnost bez
amplifikacije (u stvarnom hardveru traži se amplitude amplification; ovde
egzaktna simulacija ne mari).
HHL |x⟩ je NORMALIZOVAN — čita se odnos amp-a, ne apsolutne vrednosti w.
Eigendecomposition za UnitaryGate je klasično pre-izračunata (O(N³)) — za veći
nq_b postaje neprivlačno, ali je to kompilacioni korak, ne deo kvantnog procesa.
mod-39 readout meša stanja (dim 2^nq_b = 64 ≠ 39).
"""



"""
HHL (A⁻¹·b) kvantna linearna regresija 
Rešava linearni sistem A·w = b kvantno-linearno, 
gde je A matrica feature-a iz CSV-a 
(npr. A[i,k] = pojavljivanje broja k u redu i), 
a b target-vector (npr. freq_csv). 
Rezultat je kvantno stanje |w⟩ 
čije amplitude su regresioni koeficijenti. 
Primena na NEXT = amp-vektor brojeva preko naučenog |w⟩. 
Klasičan kvantni primitive za linearnu regresiju.
"""
