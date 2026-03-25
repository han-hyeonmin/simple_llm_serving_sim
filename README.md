# disagg-llm-serving-sim (Continuous Batching)

A time-based simulator for **disaggregated LLM serving** that models contention
between **KV cache transfers** and **decode collectives** on a shared
device-to-device (D2D) interconnect. Decode is **continuous / dynamic**: the
decode batch is rebuilt every token up to a fixed capacity.

---

## What this variant models

- **Prefill pools:** 2 independent timelines. Collective time is included in the
  prefill critical path, but D2D contention for those collectives is ignored by
  design.
- **Decode pool:** 1 timeline that re-batches each token with capacity
  `decode_batch_size = 4 × batch`.
- **D2D link:** shared by KV cache transfers (2-hop after each prefill layer)
  and decode collectives (attention + FFN all-reduce). Either side can stall
  the other.
- **Start rule:** decode can begin as soon as prefill compute ends; KV transfers
  may still be in flight.

The key metric is the **stall ratio**  
`stall_total_s / E2E` (fraction of total time lost to D2D contention).

---

## Repository Layout

```text
.
├── config.py          # Model/topology/hardware constants
├── metrics_engine.py  # FLOP, memory, and communication cost models
├── run.py             # Continuous decode scheduler (single experiment)
├── sweep.py           # Batch/request sweeps with optional CSV + plotting
└── inputs/
    └── requests.csv   # Example request trace
```

---

## Inputs

- `inputs/requests.csv` must contain `TIMESTAMP, ContextTokens, GeneratedTokens`.
- Only rows with `TIMESTAMP == 0` are used.  
- Requests are split evenly into two prefill pools of size `batch` each.

Defaults inside `run.py`:

- `request_num` defaults to `2 * batch` if omitted.
- `decode_batch_size` is always `4 * batch`.
- `aggregate` (how per-batch lengths are derived) defaults to `max`.

---

## Run a single experiment

```bash
python run.py --batch 4                # uses inputs/requests.csv
python run.py --batch 4 --req_num 16   # override request count
python run.py --batch 4 --aggregate sum
python run.py --batch 4 --verbose      # show stalls and per-request timing
```

### Sweep helpers

```bash
python sweep.py                        # stall ratios + bar chart
python sweep.py -o out.csv --no-plot   # CSV only
python sweep.py -r --batch 4 --req_num 16 --no-plot
```

---

## Output metrics

- `E2E` — total completion time (s)  
- `stall_decode_collective_due_to_kv_s` — decode collectives stalled by KV  
- `stall_kv_due_to_decode_collective_s` — KV stalled by decode collectives  
- `stall_total_s` — combined stall time  
- `stall_ratio` — `stall_total_s / E2E`

These are printed by `run.py`; `--verbose` also dumps per-layer stall logs and
per-request timelines.

---

## Notes

- Time-based (not cycle-accurate); absolute latency depends on the model
  assumptions in `config.py` and `metrics_engine.py`.
- Use this variant to study token-level scheduling effects and how continuous
  admission changes stall patterns versus a static policy.
