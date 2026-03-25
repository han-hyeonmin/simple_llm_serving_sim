# =============================================================
# Continuous (dynamic) decode batch scheduling
# - Prefill batches finish on two independent pools.
# - Decode runs token-by-token with a fixed capacity per step.
# - Newly finished prefill batches are decomposed into requests and admitted
#   at token boundaries as long as capacity allows.
# =============================================================
"""
Disaggregated serving simulator (continuous decode).

Key behaviors captured here:
- Prefill collective time is part of the prefill critical path, but its D2D
  contention is intentionally ignored.
- Decode may begin once prefill compute finishes, even if KV transfers are
  still running. Decode collectives share the D2D link with KV transfers, so
  either side can stall the other.

System model:
- Prefill pools: 2 independent timelines
- Decode pool : 1 timeline, re-batching every token up to `decode_batch_size`
- D2D link   : shared by KV cache transfers (2-hop) and decode collectives
"""

from __future__ import annotations

import argparse
import heapq
import itertools
from json import dumps
from dataclasses import dataclass
from typing import List, Tuple, Optional
from math import ceil

from config import n_layers
from metrics_engine import (
    MetricsEngine,
    BatchConfig,
    load_timestamp_zero,
)


# ============================================================
# D2D scheduling primitives
# ============================================================


@dataclass(frozen=True, slots=True)
class KVJob:
    """A KV transfer job released at ready_t, with fixed duration."""

    ready_t: float
    duration: float
    batch_tag: str
    layer_idx: int

    def __repr__(self) -> str:
        return (
            "KVJob("
            f"batch='{self.batch_tag}', "
            f"layer={self.layer_idx:3d}, "
            f"ready={(self.ready_t * 1e6):.3g}\tμs, "
            f"duration={(self.duration * 1e6):.3g}\tμs"
            ")"
        )


@dataclass
class D2DState:
    """
    Single-server D2D resource.

    available_t: end time of the last scheduled D2D task
    last_task_kind: "kv" or "decode" or None
    """

    available_t: float = 0.0
    last_task_kind: Optional[str] = None


class KVQueue:
    """Min-heap of KV jobs ordered by ready time."""

    def __init__(self) -> None:
        self._h: List[Tuple[float, int, KVJob]] = []
        self._seq = 0

    def push(self, job: KVJob) -> None:
        self._seq += 1
        heapq.heappush(self._h, (job.ready_t, self._seq, job))

    def peek(self) -> Optional[KVJob]:
        return self._h[0][2] if self._h else None

    def pop(self) -> KVJob:
        return heapq.heappop(self._h)[2]

    def __len__(self) -> int:
        return len(self._h)

    def __repr__(self) -> str:
        if not self._h:
            return "KVQueue([])"

        h_copy = list(self._h)
        heapq.heapify(h_copy)
        h_sorted = [heapq.heappop(h_copy) for _ in range(len(h_copy))]

        items = [
            f"("
            # f"seq={seq:3d}, "
            f"ready_t={(t*1e6):.3g}\tμs, job={job})"
            for (t, seq, job) in h_sorted
        ]
        return "KVQueue([\n  " + ",\n  ".join(items) + "\n])"


def run_kv_until(d2d: D2DState, kvq: KVQueue, t_limit: float) -> float:
    """
    Run KV transfers on D2D up to time t_limit.

    Returns:
      stall_kv_due_to_decode_collective_s accumulated during this interval.
    """
    stall = 0.0

    while len(kvq) > 0:
        job = kvq.peek()
        assert job is not None

        if job.ready_t > t_limit:
            break

        start = max(d2d.available_t, job.ready_t)
        if start >= t_limit:
            break

        if job.ready_t < d2d.available_t and d2d.last_task_kind == "decode":
            stall += d2d.available_t - job.ready_t

        d2d.available_t = start + job.duration
        d2d.last_task_kind = "kv"
        kvq.pop()

    return stall


def reserve_decode_collective(
    d2d: D2DState,
    kvq: KVQueue,
    *,
    t_req: float,
    duration: float,
) -> Tuple[float, float]:
    """
    Reserve D2D for a decode collective.

    Returns:
      (end_time, stall_decode_collective_due_to_kv_s)
    """
    _ = run_kv_until(d2d, kvq, t_req)

    start = max(t_req, d2d.available_t)
    stall = 0.0
    if t_req < d2d.available_t and d2d.last_task_kind == "kv":
        stall = d2d.available_t - t_req

    d2d.available_t = start + duration
    d2d.last_task_kind = "decode"
    return d2d.available_t, stall


# ============================================================
# Batch runtime state
# ============================================================


@dataclass
class BatchRuntime:
    tag: str
    cfg: BatchConfig
    req_rows: List[Tuple[int, int]]
    B_init: int
    kv: float

    prefill_compute_done_t: float = 0.0
    decode_start_t: float = 0.0
    decode_done_t: float = 0.0


@dataclass
class RequestRuntime:
    parent: BatchRuntime
    idx_in_parent: int

    decode_start_t: float = 0.0
    decode_done_t: float = 0.0

    stall_decode: float = 0.0
    stall_kv: float = 0.0


def init_remaining_tokens(req_rows: List[Tuple[int, int]]) -> List[int]:
    return [out for _, out in req_rows]


def step_batch_and_update_B(remaining: List[int]) -> int:
    """Consume one token for each in-flight request and return active count."""
    for i in range(len(remaining)):
        if remaining[i] > 0:
            remaining[i] -= 1
    return sum(1 for x in remaining if x > 0)


def prefill_batch_to_decode_batch(
    batch_ready: List[BatchRuntime],
    remaining: List[int],
    L_cur_list: List[int],
    batch: List[RequestRuntime],
) -> Tuple[float, List[int], List[RequestRuntime]]:
    """
    Flatten all ready prefill batches into the decode queues.

    Returns the latest prefill completion time and the updated L_cur/batch lists.
    """
    prefill_end = 0.0

    for b in batch_ready:
        if b.prefill_compute_done_t > prefill_end:
            prefill_end = b.prefill_compute_done_t

        for i in range(len(b.req_rows)):
            remaining.append(b.req_rows[i][1] - 1)
            L_cur_list.append(b.cfg.L_prefill + 1)
            batch.append(RequestRuntime(b, i))

    return prefill_end, L_cur_list, batch


def build_prefill_queues_by_finish_time(
    engine: MetricsEngine,
    *,
    cfg_rows_pairs: List[Tuple[BatchConfig, List[Tuple[int, int]]]],
    batch_size: int,
    layers: int,
) -> Tuple[List[BatchRuntime], List[BatchRuntime]]:
    """Assign prefill batches to two pools in arrival order.

    - All requests arrive at t=0 (per assumption).
    - Batches are consumed in CSV order (no reordering).
    - Each batch is assigned to the pool that becomes available first;
      ties prefer pool 1 (higher priority).
    """

    candidates = []

    for idx, (cfg, rows) in enumerate(cfg_rows_pairs):
        B_init = min(batch_size, len(rows))
        kv_time = engine.kv_transfer_time_s(
            engine.kv_cache_bytes_per_layer_per_die(B=B_init, L_prefill=cfg.L_prefill),
            hops=2,
        )
        times = engine.prefill_layer_times_s(B=B_init, L_prefill=cfg.L_prefill)
        layer_time = sum(times.values())
        prefill_duration = layer_time * layers

        candidates.append(
            {
                "cfg": cfg,
                "rows": rows,
                "B_init": B_init,
                "kv": kv_time,
                "prefill_duration": prefill_duration,
                "idx": idx,  # stable tie-breaker
            }
        )

    # Min-heap of (available_time, pool_idx); pool_idx ensures pool1 priority on ties
    pool_heap: List[Tuple[float, int]] = [(0.0, 0), (0.0, 1)]
    pool_items = {0: [], 1: []}

    for cand in candidates:
        available_t, pool_idx = heapq.heappop(pool_heap)
        pool_items[pool_idx].append(cand)
        heapq.heappush(pool_heap, (available_t + cand["prefill_duration"], pool_idx))

    batch1_queue = [
        BatchRuntime("batch1", c["cfg"], c["rows"], c["B_init"], c["kv"])
        for c in pool_items[0]
    ]
    batch2_queue = [
        BatchRuntime("batch2", c["cfg"], c["rows"], c["B_init"], c["kv"])
        for c in pool_items[1]
    ]

    return batch1_queue, batch2_queue


# ============================================================
# KV job construction
# ============================================================


def build_kv_jobs_for_batch(
    engine: MetricsEngine,
    *,
    batch_tag: str,
    B: int,
    L_prefill: int,
    layers: int,
    kv_hops: int = 2,
    t_start: float = 0.0,
) -> Tuple[float, List[KVJob]]:
    """Create one KV transfer job per layer for a completed prefill batch."""
    times = engine.prefill_layer_times_s(B=B, L_prefill=L_prefill)
    layer_time = sum(times.values())

    kv_bytes = engine.kv_cache_bytes_per_layer_per_die(B=B, L_prefill=L_prefill)
    kv_dur = engine.kv_transfer_time_s(kv_bytes, hops=kv_hops)

    t = t_start
    jobs: List[KVJob] = []
    for li in range(layers):
        t += layer_time
        jobs.append(KVJob(t, kv_dur, batch_tag, li))

    return t, jobs


# ============================================================
# Decode simulation
# ============================================================


def simulate_decode_for_batch_1_tok(
    engine: MetricsEngine,
    d2d: D2DState,
    kvq: KVQueue,
    t: float,
    remaining_selected: List[int],
    L_cur_selected: List[int],
    *,
    batch: List[RequestRuntime],
    layers: int,
) -> Tuple[float, float, float, List[str], List[str], int, List[str]]:
    """
    Run one decode token step for the current batch selection.

    Returns end time, stall accumulators, logs, next active batch size, and
    (optionally) the first-layer breakdown for verbose output.
    """
    stall_kv_due_decode = run_kv_until(d2d, kvq, t)

    # initialize
    stall_decode_due_kv = 0.0
    log = []
    stall_log = []
    decode_layer_time_list = []
    stop_triggered = False
    L_generated_list = []
    batch_tag_list = []

    # pre-calculate decode_time
    times = engine.decode_layer_times_s(L_cur_list=L_cur_selected)
    B_cur = len(L_cur_selected)

    for i in range(len(L_cur_selected)):
        L_generated = L_cur_selected[i] - batch[i].parent.cfg.L_prefill
        L_generated_list.append(L_generated)
        batch_tag_list.append(batch[i].parent.tag)

    # 1 token generation with layers
    for layer in range(layers):
        layer_log = []
        if (any(L == 1 for L in L_generated_list)) & (layer == 0):
            decode_layer_time = f" - The first decode layer time"
            for k, v in times.items():
                decode_layer_time += f"\n     - {k:22s}: {(v*1e6):9.3g} μs"
            decode_layer_time += "\n" + "-" * 65
            for _ in range(L_generated_list.count(1)):
                decode_layer_time_list.append(decode_layer_time)

        if 10 <= (layer + 1) % 100 <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get((layer + 1) % 10, "th")

        # attn_non_d2d_s
        t_next = t + times["attn_non_d2d_s"]

        if t_next == t:
            log.append(
                f"[STOP]"
                "\n"
                f"current time: {t:.3g}s, decode stage increment({times["attn_non_d2d_s"]:.3g}s) lost!"
                "\n"
                f"Batch L_generated: {L_generated_list} \n"
                f"{layer+1}{suffix} layer, B={B_cur}"
            )
            stop_triggered = True
            break

        stall = run_kv_until(d2d, kvq, t_next)
        stall_kv_due_decode += stall
        if stall > 0:
            layer_log.append(f"       - kv_stall     : {(stall*1e6):8.3g} μs")

        # Attn AllReduce
        t = t_next

        t, stall = reserve_decode_collective(
            d2d, kvq, t_req=t, duration=times["attn_collective_s"]
        )
        stall_decode_due_kv += stall
        if stall > 0:
            layer_log.append(f"       - decode_stall : {(stall*1e6):8.3g} μs")

        # ffn_non_d2d_s
        t_next = t + times["ffn_non_d2d_s"]
        stall = run_kv_until(d2d, kvq, t_next)
        stall_kv_due_decode += stall
        if stall > 0:
            layer_log.append(f"       - kv_stall     : {(stall*1e6):8.3g} μs")

        # FFN AllReduce
        t = t_next

        t, stall = reserve_decode_collective(
            d2d, kvq, t_req=t, duration=times["ffn_collective_s"]
        )
        stall_decode_due_kv += stall
        if stall > 0:
            layer_log.append(f"       - decode_stall : {(stall*1e6):8.3g} μs")

        if layer == 0:
            log.append("")
            log.append(
                f" - layer              : {layer+1}{suffix}\n"
                f" - Batch size         : {B_cur}\n"
                f" - Batch L_generated  : {L_generated_list}\n"
                f" - Prefill batch tags : {batch_tag_list}\n"
            )
            log.append("-" * 60)

        if len(layer_log) > 0:
            if len(stall_log) == 0:
                stall_log.append("")
                stall_log.append(
                    f" - Batch L_generated  : {L_generated_list}"
                    # f"\n - Batch size         : {B_cur}"
                    # f"\n - Prefill batch tags : {batch_tag_list}"
                )
            stall_log.append(f"\n    - [{layer+1}{suffix} layer] stall log")
            stall_log += layer_log

        if stop_triggered:
            break

    if len(stall_log) > 0:
        stall_log.append("-" * 60)

    B_nxt = step_batch_and_update_B(remaining_selected)

    return (
        t,
        stall_decode_due_kv,
        stall_kv_due_decode,
        log,
        stall_log,
        B_nxt,
        decode_layer_time_list,
    )


# ============================================================
# Overall simulation
# ============================================================


def simulate_disaggregated(
    engine: MetricsEngine,
    *,
    batch1_queue: List[BatchRuntime],
    batch2_queue: List[BatchRuntime],
    decode_batch_size: int,
    layers: int,
) -> dict:
    """Full timeline simulation: prefill → KV transfers → token-level decode."""
    d2d = D2DState()
    kvq = KVQueue()

    # prefill machine pool 1 prefill done queue
    b1_done = 0.0
    for batch1 in batch1_queue:
        b1_time, b1_jobs = build_kv_jobs_for_batch(
            engine,
            batch_tag=batch1.tag,
            B=batch1.B_init,
            L_prefill=batch1.cfg.L_prefill,
            layers=layers,
            t_start=b1_done,
        )
        b1_done = b1_time
        batch1.prefill_compute_done_t = b1_done
        for j in b1_jobs:
            kvq.push(j)

    # prefill machine pool 2 prefill done queue
    b2_done = 0.0
    for batch2 in batch2_queue:
        b2_time, b2_jobs = build_kv_jobs_for_batch(
            engine,
            batch_tag=batch2.tag,
            B=batch2.B_init,
            L_prefill=batch2.cfg.L_prefill,
            layers=layers,
            t_start=b2_done,
        )
        b2_done = b2_time
        batch2.prefill_compute_done_t = b2_done
        for j in b2_jobs:
            kvq.push(j)

    # initialize
    decode_pool_available = 0.0
    stall_decode_total = 0.0
    stall_kv_total = 0.0

    prefill_log = []
    decode_log = {"decode_batch_log": [], "decode_stall_log": []}
    decode_layer_time = []
    kv = []

    remaining = []
    L_cur_list = []
    decode_batch = []
    sorted_batch_queue = []

    # merge prefill batch queue and heapify
    batch_queue = batch1_queue + batch2_queue

    counter = itertools.count()
    batch_heap = [(b.prefill_compute_done_t, next(counter), b) for b in batch_queue]
    heapq.heapify(batch_heap)

    # repeat until all the prefill queue -> decode
    while True:
        # prefill_done queue
        batch_ready = []
        while batch_heap and batch_heap[0][0] <= decode_pool_available:
            _, _, batch = heapq.heappop(batch_heap)
            batch_ready.append(batch)
            sorted_batch_queue.append(batch)

        if not batch_ready:
            if all(r == 0 for r in remaining):
                if not batch_heap:
                    break
                else:
                    _, _, batch = heapq.heappop(batch_heap)
                    batch_ready.append(batch)
                    sorted_batch_queue.append(batch)

        # decompose prefill batch and make decode batch by requests
        prefill_compute_done_t, L_cur_list, decode_batch = (
            prefill_batch_to_decode_batch(
                batch_ready, remaining, L_cur_list, decode_batch
            )
        )

        stop_triggered = False

        # decode start time
        s = max(decode_pool_available, prefill_compute_done_t)

        # make decode batch
        valid_L_cur_idx = []
        remaining_selected = []
        L_cur_selected = []
        decode_batch_selected = []

        for i, v in enumerate(remaining):
            if len(valid_L_cur_idx) < decode_batch_size:
                if v > 0:
                    valid_L_cur_idx.append(i)
                    remaining_selected.append(v)
                    L_cur_selected.append(L_cur_list[i])
                    decode_batch_selected.append(decode_batch[i])
            else:
                break

        # simulate_decode_for_batch_1_tok
        (
            e,
            sd,
            sk,
            decode_batch_log,
            decode_stall_log,
            B_nxt,
            first_decode_layer_time,
        ) = simulate_decode_for_batch_1_tok(
            engine,
            d2d,
            kvq,
            s,
            remaining_selected,
            L_cur_selected,
            batch=decode_batch_selected,
            layers=layers,
        )

        # decode stage increment lost -> break
        if s == e:
            stop_triggered = True
            break

        # decode_layer_time
        if len(first_decode_layer_time) > 0:
            decode_layer_time += first_decode_layer_time
        # decode_log
        decode_log["decode_batch_log"] += decode_batch_log
        if decode_stall_log != []:
            decode_log["decode_stall_log"] += decode_stall_log

        # value updates
        for i, v in enumerate(remaining_selected):
            idx = valid_L_cur_idx[i]

            # update remaining
            remaining[idx] = v
            L_cur_list[idx] += 1

            # update decode start/done time
            if decode_batch[idx].decode_start_t == 0:
                decode_batch[idx].decode_start_t = s
                if decode_batch[idx].parent.decode_start_t == 0:
                    decode_batch[idx].parent.decode_start_t = s
            elif v == 0:
                decode_batch[idx].decode_done_t = e
                decode_batch[idx].parent.decode_done_t = e

            decode_batch[idx].stall_decode += sd
            decode_batch[idx].stall_kv += sk

        decode_pool_available = e
        stall_decode_total += sd
        stall_kv_total += sk

        if stop_triggered:
            break

    # prefill log and kv
    for b in sorted_batch_queue:
        prefill_log.append(
            (
                f"[{b.tag}]",
                b.cfg.L_prefill,
                b.cfg.L_decode,
                b.prefill_compute_done_t,
                b.decode_start_t,
                b.decode_done_t,
                (b.decode_done_t - b.decode_start_t) / layers,
            )
        )
        kv.append(b.kv)
    E2E = max(b.decode_done_t for b in sorted_batch_queue)
    stall_total = stall_decode_total + stall_kv_total

    by_request_log = []
    req_stall_ratio_list = []

    by_request_log.append(
        "\n"
        " L_prefill | L_decode | Decode start  | Decode end    | Decode time(ms) | stall_total(ms) | stall_ratio(%)\n"
        "-----------+----------+---------------+---------------+-----------------+-----------------+---------------"
    )

    for r in decode_batch:
        lp, ld = r.parent.req_rows[r.idx_in_parent]
        dt = r.decode_done_t - r.decode_start_t
        request_stall_total = r.stall_decode + r.stall_kv
        stall_ratio = request_stall_total / dt
        by_request_log.append(
            f" {lp:9d} | {ld:8d} | {r.decode_start_t:13.6g} | {r.decode_done_t:13.6g} | "
            f"{(dt*1e3):15.3g} | {(request_stall_total*1e3):15.3g} | {(stall_ratio*100):13.3g}"
        )

        req_stall_ratio_list.append(stall_ratio)

    if not req_stall_ratio_list:
        raise ValueError("empty list")

    xs = sorted(req_stall_ratio_list)
    idx = ceil(0.99 * len(xs)) - 1  # 0-based index

    stall_p99 = xs[idx]

    return {
        "E2E": E2E,
        "stall_decode_collective_due_to_kv_s": stall_decode_total,
        "stall_kv_due_to_decode_collective_s": stall_kv_total,
        "stall_total_s": stall_total,
        "stall_ratio": stall_total / E2E if E2E > 0 else 0.0,
        "stall_ratio_p99": stall_p99,
        "prefill log": prefill_log,
        "decode log": decode_log,
        "by request log": by_request_log,
        "decode layer time": decode_layer_time,
        "batch1_queue": batch1_queue,
        "batch2_queue": batch2_queue,
        "kv cache transfer time": kv,
    }


# ============================================================
# CLI / main
# ============================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Disaggregated LLM serving simulator (updated)."
    )
    p.add_argument("--input", type=str, default="inputs/requests.csv")
    p.add_argument("--req_num", type=int, default=None)
    p.add_argument("--batch", type=int, default=1, help="prefill batch size")
    p.add_argument("--db", type=int, default=4, help="decode batch scaler: db*batch")
    p.add_argument(
        "--aggregate",
        type=str,
        default="max",
        choices=["max", "first", "sum"],
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def run_experiment(
    *,
    csv_path: str,
    request_num: int,
    batch: int,
    decode_batch_size: int,
    aggregate: str = "max",
) -> dict:
    """Helper used by CLI and tests; wires CSV parsing into the simulator."""
    engine = MetricsEngine()

    rows_t0 = load_timestamp_zero(csv_path, request_num)

    def agg(batch_rows: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not batch_rows:
            raise ValueError("Empty batch.")
        if aggregate == "first":
            return batch_rows[0]
        if aggregate == "max":
            return max(x for x, _ in batch_rows), max(y for _, y in batch_rows)
        if aggregate == "sum":
            return sum(x for x, _ in batch_rows), sum(y for _, y in batch_rows)
        raise ValueError(f"Unknown aggregate_policy: {aggregate}")

    cfg_rows_pairs: List[Tuple[BatchConfig, List[Tuple[int, int]]]] = []
    for i in range(0, len(rows_t0), batch):
        chunk = rows_t0[i : i + batch]
        Lp, Ld = agg(chunk)
        cfg_rows_pairs.append((BatchConfig(L_prefill=Lp, L_decode=Ld), chunk))

    batch1_queue, batch2_queue = build_prefill_queues_by_finish_time(
        engine,
        cfg_rows_pairs=cfg_rows_pairs,
        batch_size=batch,
        layers=n_layers,
    )

    result = simulate_disaggregated(
        engine,
        batch1_queue=batch1_queue,
        batch2_queue=batch2_queue,
        decode_batch_size=decode_batch_size,
        layers=n_layers,
    )
    return result


def main() -> None:
    args = parse_args()

    if args.req_num == None:
        request_num = args.batch * 2
    else:
        request_num = args.req_num

    decode_batch_size = args.db * args.batch

    out = run_experiment(
        csv_path=args.input,
        request_num=request_num,
        batch=args.batch,
        decode_batch_size=decode_batch_size,
        aggregate=args.aggregate,
    )

    print()
    print("Simulation Setting")
    print("=" * 40)
    print(f"input         : {args.input}")
    print(f"prefill batch : {args.batch}")
    print(f"decode batch  : {decode_batch_size}")
    # print(f"aggregate     : {args.aggregate}")
    print(f"request_num   : {request_num}")
    print("=" * 40)
    print()

    if args.verbose:
        # print("Prefill Batch Log")
        # print("=" * 65)
        # for i, (
        #     batch_tag,
        #     L_prefill,
        #     L_decode,
        #     prefill_end,
        #     decode_start,
        #     decode_end,
        #     avg_decode_time_per_layer,
        # ) in enumerate(out["prefill log"]):
        #     print(
        #         "\n"
        #         f"{batch_tag}"
        #         f"\n - L_prefill                 : {L_prefill:9d}"
        #         # f"\n - L_decode                  : {L_decode:9d}"
        #         f"\n - prefill_end               : {prefill_end:9.3g} s"
        #         # f"\n - decode_start              : {decode_start:9.3g} s"
        #         # f"\n - decode_end                : {decode_end:9.3e} s"
        #         # f"\n - decode_time               : {(decode_end-decode_start):9.3g} s"
        #         # f"\n - avg_decode_time_per_layer : {avg_decode_time_per_layer:9.3g} s"
        #     )
        #     if (decode_start - prefill_end) > 0:
        #         print(
        #             f" - queuing delay             : {(decode_start-prefill_end):9.3g} s"
        #         )
        #     print(
        #         f" - kv cache transfer         : {(out["kv cache transfer time"][i] * 1e6):9.3g} μs"
        #     )
        #     print(out["decode layer time"][i])
        # print("=" * 65)
        # print()

        # print("Decode Batch Log by Layer")
        # print("=" * 60)
        # for log in out["decode log"]["decode_batch_log"]:
        #     print(log)
        # print("=" * 60)
        # print()

        # print("Decode Stall Log")
        # print("=" * 60)
        # for log in out["decode log"]["decode_stall_log"]:
        #     print(log)
        # print("=" * 60)
        # print()

        print("Requests")
        print("=" * 106)
        for log in out["by request log"]:
            print(log)
        print("=" * 106)
        print()

    print("Summary")
    print("=" * 55)
    print(f"E2E Latency                         : {out["E2E"]:8.3g} s")
    if out["stall_decode_collective_due_to_kv_s"] > 0:
        print(
            f"stall_decode_collective_due_to_kv_s : "
            f"{(out["stall_decode_collective_due_to_kv_s"] * 1e3):8.3g} ms"
        )
    if out["stall_kv_due_to_decode_collective_s"] > 0:
        print(
            f"stall_kv_due_to_decode_collective_s : "
            f"{(out["stall_kv_due_to_decode_collective_s"] * 1e3):8.3g} ms"
        )
    if out["stall_total_s"] > 0:
        print(
            f"stall_total_s                       : "
            f"{(out["stall_total_s"] * 1e3):8.3g} ms"
        )
    print(f"stall_ratio                         : {out["stall_ratio"]*100:8.3g} %")
    print(
        f"stall_ratio_p99                     : {(out["stall_ratio_p99"] * 100):8.3g} %"
    )
    print("=" * 55)


if __name__ == "__main__":
    main()
