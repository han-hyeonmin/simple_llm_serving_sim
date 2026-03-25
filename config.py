"""
Global shared configuration for model, topology, and hardware.
All values are treated as constants.
"""

# ============================================================
# Topology
# ============================================================

N: int = 4  # Number of nodes
TP: int = N  # Tensor-parallel degree

# ============================================================
# Model (LLaMA3 8B)
# ============================================================

H: int = 4096
I: int = 14336
n_heads: int = 32
n_kv: int = 8
n_layers: int = 32

# ============================================================
# Model (LLaMA3 70B)
# ============================================================

# H: int = 8192
# I: int = 28672
# n_heads: int = 64
# n_kv: int = 8
# n_layers: int = 80

# ============================================================
# Hardware
# ============================================================

# Dojo-style compute die
TFLOPs_per_core: float = 1.02
core_per_die: int = 16**2
TFLOPs_per_die: float = TFLOPs_per_core * core_per_die

channel_bw_GBps: float = 32.0
memory_bw_TBps: float = 3.35
channel_latency_prefill_us: float = 0.4
channel_latency_decode_us: float = 0.1


# ============================================================
# Datatype
# ============================================================

bytes_per_element: int = 2  # FP16 / BF16
