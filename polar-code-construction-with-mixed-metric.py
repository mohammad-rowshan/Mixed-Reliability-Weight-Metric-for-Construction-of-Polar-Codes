# -----------------------------------------------------------------------------
# Copyright (c) 2026 Mohammad Rowshan
#
# This file is part of the repository:
#   Mixed Reliability–Weight Metric for Construction of Polar Codes
#   https://github.com/mohammad-rowshan/Mixed-Reliability-Weight-Metric-for-Construction-of-Polar-Codes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

import math
from typing import Dict, List, Any

def generate_mixed_order_polar(N: int,
                               K: int,
                               alpha: float,
                               EbN0_dB: float,
                               restrict_degree_to_rel: bool = True) -> Dict[str, Any]:
    """
    Design two information sets for a polar-like code of length N and dimension K:

      1) I_rel   – constructed purely from synthetic-channel reliabilities (GA for AWGN),
      2) I_mixed – constructed from a mixed cost that combines:
           - P_e(i) from GA, and
           - the monomial contribution to the truncated union bound term
             A_{w_min} * Z(W)^{w_min}, via the closed-form formula
             contrib_{w_min}(f) = 2^{r + |lambda_f|} for each max-degree monomial f.

    The channel is real BPSK over AWGN, using Gaussian Approximation (GA) for polar DE.

    Parameters
    ----------
    N : int
        Block length (must be a power of 2).
    K : int
        Dimension (number of information bits).
    EbN0_dB : float
        E_b/N_0 in dB for the design.
    alpha: int
        the utning parameter 
    restrict_degree_to_rel : bool
        If True (recommended), the mixed design only considers monomials of degree
        <= r_rel (the max degree in the reliability-based code) to keep the same
        w_min class.

    Returns
    -------
    dict
        A dictionary with:
          - "N", "K", "EbN0_dB"
          - "r_rel", "wmin_rel"
          - "Z_base", "gamma"
          - "I_rel", "I_mixed"
          - "indices_in_rel_not_mixed"
          - "indices_in_mixed_not_rel"
          - "Pe"  (list of P_e(i))
          - "D"   (list of distance-term contributions per index)
          - "J"   (list of mixed costs per index)
          - "metrics_rel", "metrics_mixed" (for both sets I)
            each metrics dict has:
              * "w_min"
              * "A_w_min"
              * "sum_Pe"
              * "UB_trunc"      = A_{w_min} * Z(W)^{w_min}
              * "sumPe_plus_UB" = sum_Pe + UB_trunc
    """
    # ------------------ basic checks ------------------
    m = int(round(math.log2(N)))
    if 2**m != N:
        raise ValueError("N must be a power of 2.")
    R = K / float(N)

    # ------------------ helper: GA for AWGN ------------------
    def Q(x: float) -> float:
        return 0.5 * math.erfc(x / math.sqrt(2.0))

    def phi(x: float) -> float:
        # Chung-type approximation used in GA for LDPC / polar
        if x <= 0:
            return 1.0
        return math.exp(-0.4527 * x**0.86 + 0.0218)

    def phi_inv(y: float) -> float:
        # Numerical inverse of phi on y in (0,1]
        if y >= 1.0:
            return 0.0
        if y <= 0.0:
            return 100.0  # effectively infinite reliability
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if phi(mid) > y:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def GA_means_awgn(EbN0_dB: float, N: int, K: int):
        """Return (means, Pe) for all N synthetic channels W_N^{(i)}."""
        m_ = int(round(math.log2(N)))
        R_ = K / float(N)
        SNRb = 10**(EbN0_dB / 10.0)
        # AWGN: sigma^2 = 1 / (2 R E_b/N0)
        sigma2 = 1.0 / (2 * R_ * SNRb)
        m0 = 2.0 / sigma2               # = 4 R E_b/N0

        means = [m0]
        for _ in range(m_):
            new = []
            for mu in means:
                mu_minus = phi_inv(1.0 - (1.0 - phi(mu))**2)  # W^-
                mu_plus  = 2.0 * mu                           # W^+
                new.append(mu_minus)
                new.append(mu_plus)
            means = new
        Pe = [Q(math.sqrt(mu / 2.0)) for mu in means]
        return means, Pe

    # ------ reliability of synthetic channels via GA ------
    means, Pe = GA_means_awgn(EbN0_dB, N, K)
    all_idx = list(range(N))

    # ------------------ monomial utilities ------------------
    # Mapping between polar row index i and monomial f:
    # f = product of x_j for all j where bit_j(i) == 0.
    def monomial_support_from_index(i: int):
        return [j for j in range(m) if ((i >> j) & 1) == 0]

    def monomial_degree(i: int) -> int:
        return len(monomial_support_from_index(i))

    def lambda_total_from_index(i: int) -> int:
        """
        |lambda_f| = sum(S) - s(s-1)/2, where S is the sorted support of f
        and s = |S| = deg(f).
        This is the partition weight used in the orbit-size formula:
           |orbit(f)| = 2^{deg(f) + |lambda_f|}.
        """
        S = monomial_support_from_index(i)
        s = len(S)
        return sum(S) - s * (s - 1) // 2

    def Awmin_and_wmin(I: List[int]):
        """Compute w_min and A_{w_min}(I) from the closed-form formula."""
        if not I:
            return None, 0
        r = max(monomial_degree(i) for i in I)
        Aw = 0
        for i in I:
            if monomial_degree(i) == r:
                lam = lambda_total_from_index(i)
                Aw += 2**(r + lam)
        wmin = 2**(m - r)
        return wmin, Aw

    # ------------------ reliability-only information set ------------------
    sorted_by_Pe = sorted(all_idx, key=lambda i: Pe[i])
    I_rel = sorted(sorted_by_Pe[:K])

    # Maximum degree in reliability-based code
    r_rel = max(monomial_degree(i) for i in I_rel)
    wmin_rel = 2**(m - r_rel)

    # ------------------ distance term from truncated union bound ------------------
    # Underlying channel Bhattacharyya for BPSK-AWGN:
    #    Z(W) = exp(-R * Eb/N0)
    SNRb = 10**(EbN0_dB / 10.0)
    Z_base = math.exp(-R * SNRb)

    # Factor Z(W)^{w_min} appearing in UB_min = A_{w_min} Z(W)^{w_min}
    gamma = Z_base**wmin_rel

    # Per-index distance contribution D[i] = contrib_{w_min}(f) * Z(W)^{w_min}
    # Only for degree r_rel monomials.
    D = [0.0] * N
    for i in range(N):
        deg = monomial_degree(i)
        if deg == r_rel:
            lam = lambda_total_from_index(i)
            contrib = 2**(r_rel + lam)  # contribution to A_{w_min}
            D[i] = gamma * contrib
        else:
            D[i] = 0.0

    # ------------------ mixed cost and mixed information set ------------------
    if restrict_degree_to_rel:
        # We stay in the same "family" of min-distance codes: no degree > r_rel
        candidate_idx = [i for i in all_idx if monomial_degree(i) <= r_rel]
    else:
        candidate_idx = all_idx

    # Per-index mixed cost: J(i) = P_e(i) + D(i)
    J = [float('inf')] * N
    for i in candidate_idx:
        J[i] = Pe[i] + alpha * D[i]

    sorted_by_J = sorted(candidate_idx, key=lambda i: J[i])
    I_mixed = sorted(sorted_by_J[:K])

    # ------------------ metrics for I_rel and I_mixed ------------------
    def code_metrics(I: List[int]) -> Dict[str, float]:
        wmin, Awmin = Awmin_and_wmin(I)
        sumPe = sum(Pe[i] for i in I)
        if wmin is None:
            return {
                "w_min": None,
                "A_w_min": 0,
                "sum_Pe": sumPe,
                "UB_trunc": None,
                "sumPe_plus_UB": None,
            }
        Z = math.exp(-R * SNRb)  # = Z_base again
        UB_trunc = (Z**wmin) * Awmin
        return {
            "w_min": wmin,
            "A_w_min": Awmin,
            "sum_Pe": sumPe,
            "UB_trunc": UB_trunc,
            "sumPe_plus_UB": sumPe + UB_trunc,
        }

    metrics_rel = code_metrics(I_rel)
    metrics_mixed = code_metrics(I_mixed)

    I_rel_set = set(I_rel)
    I_mixed_set = set(I_mixed)

    return {
        "N": N,
        "K": K,
        "EbN0_dB": EbN0_dB,
        "r_rel": r_rel,
        "wmin_rel": wmin_rel,
        "Z_base": Z_base,
        "gamma": gamma,                  # Z(W)^{w_min}
        "I_rel": I_rel,
        "I_mixed": I_mixed,
        "indices_in_rel_not_mixed": sorted(I_rel_set - I_mixed_set),
        "indices_in_mixed_not_rel": sorted(I_mixed_set - I_rel_set),
        "Pe": Pe,                        # reliability per synthetic channel
        "D": D,                          # UB-distance term per index
        "J": J,                          # mixed cost per index
        "metrics_rel": metrics_rel,      # {w_min, A_w_min, sum_Pe, UB_trunc, sumPe_plus_UB}
        "metrics_mixed": metrics_mixed,
    }


# ------------------ example usage / sanity check ------------------
if __name__ == "__main__":
    dSNR, alpha = 4.0, 100
    #for N, K in [(64, 32), (128, 64), (256, 128), (512, 256), (1024, 512), (2**12, 2**11), (2**15, 2**14), (2**18, 2**17), (2**20, 2**19)]:
    for N, K in [(2**8, int(2**8*2/4))]:
        res = generate_mixed_order_polar(N, K, alpha=alpha, EbN0_dB=dSNR, restrict_degree_to_rel=True)
        print(f"=== N={N}, K={K}, alpha={alpha}, dSNR={dSNR} ===")
        print("r_rel =", res["r_rel"], "wmin_rel =", res["wmin_rel"])
        print("Indices in I_mixed:", res["I_mixed"])
        print("Indices in I_rel but not in I_mixed:", res["indices_in_rel_not_mixed"])
        print("Indices in I_mixed but not in I_rel:", res["indices_in_mixed_not_rel"])
        print("metrics_rel   =", res["metrics_rel"])
        print("metrics_mixed =", res["metrics_mixed"])
        print()
