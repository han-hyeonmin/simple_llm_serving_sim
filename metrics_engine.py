"""
metrics_engine.py

Core metrics engine + CSV utilities.

- Prefill collective time is included in prefill layer time. run.py now schedules those
  collectives on the shared D2D link so they do not overlap with KV transfers.
- Decode can start after prefill compute finishes (not after kv_done), and decode collectives
  contend with KV transfers on a shared D2D resource in run.py.

This module is intended to be imported by run.py.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple
from itertools import islice

from config import (
    N,
    TP,
    H,
    I,
    n_heads,
    n_kv,
    TFLOPs_per_die,
    channel_bw_GBps,
    memory_bw_TBps,
    channel_latency_prefill_us,
    channel_latency_decode_us,
    bytes_per_element,
)

# ============================================================
# CSV utilities
# ============================================================


def load_timestamp_zero(csv_path: str, n: int = 2) -> List[Tuple[int, int]]:
    """
    Load (ContextTokens, GeneratedTokens) for rows with TIMESTAMP == 0.
    """
    rows: List[Tuple[int, int]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in islice(reader, n):
            if int(r["TIMESTAMP"]) == 0:
                rows.append((int(r["ContextTokens"]), int(r["GeneratedTokens"])))
    return rows


# ============================================================
# Batch-dependent config
# ============================================================


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Batch-dependent configuration (prefill length and decode length)."""

    L_prefill: int
    L_decode: int


# ============================================================
# Metrics engine
# ============================================================


@dataclass(slots=True)
class MetricsEngine:
    """
    Metrics engine with invariant precomputation.

    Notes:
    - All internal times are in seconds.
    - assemble() preserves your original conversion behavior, but run.py mainly consumes *_time_s.
    """

    # Invariants
    P: int = TP
    N: int = N
    H: int = H
    I: int = I
    n_heads: int = n_heads
    n_kv: int = n_kv
    bytes_el: int = bytes_per_element

    # Hardware invariants
    peak: float = TFLOPs_per_die * 1e12  # FLOPs/s
    Mem_BW: float = memory_bw_TBps * 1e12  # bytes/s
    D2D_BW: float = channel_bw_GBps * 1e9  # bytes/s
    alpha_p: float = channel_latency_prefill_us / 1e6
    alpha_d: float = channel_latency_decode_us / 1e6

    # Model-derived invariants
    head_dim: int = 0
    H_kv: int = 0

    # Common scalars
    invP: float = 0.0
    H_over_P: float = 0.0
    I_over_P: float = 0.0
    H_plus_2Hkv_over_P: float = 0.0
    H_plus_Hkv_over_P: float = 0.0
    Hkv_over_P: float = 0.0

    def __post_init__(self) -> None:
        if self.H % self.n_heads != 0:
            raise ValueError("H must be divisible by n_heads.")
        self.head_dim = self.H // self.n_heads
        self.H_kv = self.head_dim * self.n_kv

        self.invP = 1.0 / self.P
        self.H_over_P = self.H * self.invP
        self.I_over_P = self.I * self.invP
        self.H_plus_2Hkv_over_P = (self.H + 2 * self.H_kv) * self.invP
        self.H_plus_Hkv_over_P = (self.H + self.H_kv) * self.invP
        self.Hkv_over_P = self.H_kv * self.invP

    # --------------------------------------------------------
    # assemble()
    # --------------------------------------------------------

    def assemble(
        self,
        F: float,
        M: float,
        time_multiplier: float,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out["F"] = F
        out["T"] = F / 1e6  # GFLOPs

        compute_time_s = F / self.peak
        out["compute_time_s"] = compute_time_s
        out["compute_time_converted"] = compute_time_s * time_multiplier

        if M is not None:
            out["M"] = M
            out["memory_size_converted"] = M / 1e6  # MB

            memory_time_s = M / self.Mem_BW
            out["memory_time_s"] = memory_time_s
            out["memory_time_converted"] = memory_time_s * time_multiplier

        return out

    # --------------------------------------------------------
    # KV cache transfer primitives
    # --------------------------------------------------------

    def kv_cache_bytes_per_layer_per_die(self, B: int, L_prefill: int) -> float:
        """
        KV cache size per layer per die [bytes]:
          B * L_prefill * (H_kv / P) * 2 * bytes_el
        """
        return B * L_prefill * self.Hkv_over_P * 2 * self.bytes_el

    def kv_transfer_time_s(self, bytes_to_transfer: float, *, hops: int = 2) -> float:
        """
        2-hop transfer model:
          hops * bytes / D2D_BW
        """
        return hops * bytes_to_transfer / self.D2D_BW

    # --------------------------------------------------------
    # Prefill per-layer per-step split times
    # --------------------------------------------------------

    def prefill_layer_times_s(self, *, B: int, L_prefill: int) -> Dict[str, float]:
        """
        Return split times for one prefill layer:
        - attn_non_d2d_s: qkv_prefill + attn_qk_prefill + attn_pv_prefill + attn_out_prefill (compute+memory)
        - attn_collective_s: attn_allreduce_decode (D2D)
        - ffn_non_d2d_s: ffn_in_prefill + ffn_out_prefill (compute+memory)
        - ffn_collective_s: ffn_allreduce_decode (D2D)
        """
        attn_non = 0.0

        f_sum = 0.0
        m_sum = 0.0

        # Ready prefill
        F = 0
        M = B * L_prefill * self.H * self.bytes_el
        o = self.assemble(F, M, 1e3)
        attn_non += o["memory_time_s"]

        f_sum = 0.0
        m_sum = 0.0

        # QKV prefill
        F = 2 * B * L_prefill * self.H * self.H_plus_2Hkv_over_P
        M = B * L_prefill + self.H_plus_Hkv_over_P * self.bytes_el
        o = self.assemble(F, M, 1e3)
        f_sum += o["compute_time_s"]
        m_sum += o["memory_time_s"]

        # QK prefill
        F = 2 * B * (L_prefill**2) * self.H_over_P
        M = B * L_prefill * self.Hkv_over_P * self.bytes_el
        o = self.assemble(F, M, 1e3)
        f_sum += o["compute_time_s"]
        m_sum += o["memory_time_s"]

        # PV prefill
        F = 2 * B * (L_prefill**2) * self.H_over_P
        M = (self.H**2) * self.invP * self.bytes_el
        o = self.assemble(F, M, 1e3)
        f_sum += o["compute_time_s"]
        m_sum += o["memory_time_s"]

        # Attn out prefill
        F = 2 * B * L_prefill * (self.H**2) * self.invP
        M = 2 * self.H * self.I_over_P * self.bytes_el
        o = self.assemble(F, M, 1e3)
        f_sum += o["compute_time_s"]
        m_sum += o["memory_time_s"]

        attn_non += max(f_sum, m_sum)

        # Attn write prefill
        F = 0
        M = B * L_prefill * self.H * self.bytes_el
        o = self.assemble(F, M, 1e3)
        attn_non += o["memory_time_s"]

        # Attn collective (D2D)
        msg_bytes = B * L_prefill * self.H * self.bytes_el
        attn_coll = (
            2 * (self.N - 1) * (self.alpha_p + msg_bytes / (self.N * self.D2D_BW))
        )

        ffn_non = 0.0

        # Attn read prefill
        F = 0
        M = B * L_prefill * self.H * self.bytes_el
        o = self.assemble(F, M, 1e3)
        attn_non += o["memory_time_s"]

        f_sum = 0.0
        m_sum = 0.0

        # FFN in prefill
        F = 4 * B * L_prefill * self.H * self.I_over_P
        M = self.H * self.I_over_P * self.bytes_el
        o = self.assemble(F, M, 1e3)
        f_sum += o["compute_time_s"]
        m_sum += o["memory_time_s"]

        # FFN out prefill
        F = 2 * B * L_prefill * self.H * self.I_over_P
        M = self.H * self.H_plus_2Hkv_over_P * self.bytes_el
        o = self.assemble(F, M, 1e3)
        f_sum += o["compute_time_s"]
        m_sum += o["memory_time_s"]

        ffn_non += max(f_sum, m_sum)

        # FFN write prefill
        F = 0
        M = B * L_prefill * self.H * self.bytes_el
        o = self.assemble(F, M, 1e3)
        ffn_non += o["memory_time_s"]

        # Attn collective (D2D)
        msg_bytes = B * L_prefill * self.H * self.bytes_el
        ffn_coll = (
            2 * (self.N - 1) * (self.alpha_p + msg_bytes / (self.N * self.D2D_BW))
        )

        return {
            "attn_non_d2d_s": attn_non,
            "attn_collective_s": attn_coll,
            "ffn_non_d2d_s": ffn_non,
            "ffn_collective_s": ffn_coll,
        }

    # --------------------------------------------------------
    # Decode per-layer per-step split times
    # --------------------------------------------------------

    def decode_layer_times_s(
        self,
        *,
        L_cur_list: List[int],
    ) -> Dict[str, float]:
        """
        Return split times for one decode layer at (B_current, L_cur):

        - attn_non_d2d_s: qkv_decode + attn_qk_decode + attn_pv_decode + attn_out_decode (compute+memory)
        - attn_collective_s: attn_allreduce_decode (D2D)
        - ffn_non_d2d_s: ffn_in_decode + ffn_out_decode (compute+memory)
        - ffn_collective_s: ffn_allreduce_decode (D2D)
        """
        attn_non = 0.0

        f_sum = 0.0
        m_sum = 0.0

        B_current = 1.0

        for i in range(len(L_cur_list)):
            L_cur = L_cur_list[i]
            # Ready decode
            F = 0
            M = B_current * 1 * self.H * self.bytes_el
            o = self.assemble(F, M, 1e6)
            attn_non += o["memory_time_s"]

            # QKV decode
            F = 2 * B_current * 1 * self.H * self.H_plus_2Hkv_over_P
            M = B_current * (1 * self.H + L_cur * self.H_kv) * self.invP * self.bytes_el
            o = self.assemble(F, M, 1e6)
            f_sum += o["compute_time_s"]
            m_sum += o["memory_time_s"]

            # QK decode
            F = 2 * B_current * L_cur * self.H_over_P
            M = B_current * L_cur * self.Hkv_over_P * self.bytes_el
            o = self.assemble(F, M, 1e6)
            f_sum += o["compute_time_s"]
            m_sum += o["memory_time_s"]

            # PV decode (comp)
            F = 2 * B_current * L_cur * self.H_over_P
            M = 0
            o = self.assemble(F, M, 1e6)
            f_sum += o["compute_time_s"]

            # Attn out decode (comp)
            F = 2 * B_current * 1 * (self.H**2) * self.invP
            M = 0
            o = self.assemble(F, M, 1e6)
            f_sum += o["compute_time_s"]

        # PV decode (W0 read only once)
        F = 0
        M = (self.H**2) * self.invP * self.bytes_el
        o = self.assemble(F, M, 1e6)
        m_sum += o["memory_time_s"]

        # Attn out decode (W1 read only once)
        F = 0
        M = 2 * self.H * self.I_over_P * self.bytes_el
        o = self.assemble(F, M, 1e6)
        m_sum += o["memory_time_s"]

        attn_non += max(f_sum, m_sum)

        # merge into one batch
        B_current = len(L_cur_list)

        # Attn write decode
        F = 0
        M = B_current * 1 * self.H * self.bytes_el
        o = self.assemble(F, M, 1e6)
        attn_non += o["memory_time_s"]

        # Attn collective (D2D)
        msg_bytes = B_current * self.H * self.bytes_el
        attn_coll = (
            2 * (self.N - 1) * (self.alpha_d + msg_bytes / (self.N * self.D2D_BW))
        )

        ffn_non = 0.0

        # Attn read decode
        F = 0
        M = B_current * 1 * self.H * self.bytes_el
        o = self.assemble(F, M, 1e6)
        ffn_non += o["memory_time_s"]

        f_sum = 0.0
        m_sum = 0.0

        # FFN in decode
        F = 4 * B_current * self.H * self.I_over_P
        M = self.H * self.I_over_P * self.bytes_el
        o = self.assemble(F, M, 1e6)
        f_sum += o["compute_time_s"]
        m_sum += o["memory_time_s"]

        # FFN out decode
        F = 2 * B_current * self.H * self.I_over_P
        M = self.H * self.H_plus_2Hkv_over_P * self.bytes_el
        o = self.assemble(F, M, 1e6)
        f_sum += o["compute_time_s"]
        m_sum += o["memory_time_s"]

        ffn_non += max(f_sum, m_sum)

        # FFN write decode
        F = 0
        M = B_current * 1 * self.H * self.bytes_el
        o = self.assemble(F, M, 1e6)
        ffn_non += o["memory_time_s"]

        # FFN collective (D2D)
        msg_bytes = B_current * self.H * self.bytes_el
        ffn_coll = (
            2 * (self.N - 1) * (self.alpha_d + msg_bytes / (self.N * self.D2D_BW))
        )

        return {
            "attn_non_d2d_s": attn_non,
            "attn_collective_s": attn_coll,
            "ffn_non_d2d_s": ffn_non,
            "ffn_collective_s": ffn_coll,
        }
