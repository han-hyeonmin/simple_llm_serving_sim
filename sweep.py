"""
Batch sweep runner.

Runs run.py logic for multiple batch sizes and aggregates results.
Also plots stall_ratio as a bar chart.
"""

import argparse
import csv
from typing import List, Optional

import matplotlib.pyplot as plt

from run import run_experiment


def sweep_batches(
    *,
    csv_path: str,
    batch_list: List[int],
    aggregate: str = "max",
    output_csv: Optional[str] = None,
    plot: bool = True,
) -> None:
    """
    Run sweep over batch_list.

    If output_csv is provided, results are written to that CSV file.
    If plot is True, a bar chart of stall_ratio is displayed.
    """
    writer = None
    csv_file = None

    if output_csv is not None:
        csv_file = open(output_csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(
            ["batch", "E2E", "stall_total", "stall_ratio", "stall_ratio_p99"]
        )

    print()
    print(f"decode_batch = 4 * prefill_batch")
    print(f"req_num      = 2 * prefill_batch")
    print()
    print(
        "batch         , E2E latency     , stall_total    , stall_ratio    , stall_ratio_p99"
    )

    # For plotting
    batches: List[int] = []
    stall_ratios: List[float] = []
    stall_ratios_p99: List[float] = []

    for B in batch_list:
        result = run_experiment(
            csv_path=csv_path,
            request_num=2 * B,
            batch=B,
            decode_batch_size=4 * B,
            aggregate=aggregate,
        )

        E2E = result["E2E"]
        stall_total = result["stall_total_s"]
        stall_ratio = result["stall_ratio"] * 100
        stall_ratio_p99 = result["stall_ratio_p99"] * 100

        print(
            f"{B:15d}, {E2E:15.3e}, {stall_total:15.3e}, {stall_ratio:15.3g}, {stall_ratio_p99:15.3g}"
        )

        if writer is not None:
            writer.writerow(
                [
                    B,
                    f"{E2E:.6e}",
                    f"{stall_total:.6e}",
                    f"{stall_ratio:.6e}",
                    f"{stall_ratio_p99:.6e}",
                ]
            )

        batches.append(B)
        stall_ratios.append(stall_ratio)
        stall_ratios_p99.append(stall_ratio_p99)

    if csv_file is not None:
        csv_file.close()

    # Plot stall ratio
    if plot:
        y_pos = list(range(len(batches)))

        bar_h = 0.38
        offset = bar_h / 2

        plt.figure()

        plt.barh(
            [y - offset for y in y_pos],
            stall_ratios,
            height=bar_h,
            label="stall_ratio",
        )

        plt.barh(
            [y + offset for y in y_pos],
            stall_ratios_p99,
            height=bar_h,
            label="stall_ratio_p99",
        )

        plt.yticks(y_pos, [str(b) for b in batches])
        plt.gca().invert_yaxis()
        plt.xlabel("Stall ratio (%)")
        plt.ylabel("Batch size")
        plt.title(f"Batch size vs stall ratio")
        plt.legend()
        plt.tight_layout()
        plt.show()


def sweep_requests(
    *,
    csv_path: str,
    batch: int,
    request_list: List[int],
    aggregate: str = "max",
    output_csv: Optional[str] = None,
    plot: bool = True,
) -> None:
    """
    Run sweep over request_list.

    If output_csv is provided, results are written to that CSV file.
    If plot is True, a bar chart of stall_ratio is displayed.
    """
    writer = None
    csv_file = None
    decode_batch = 8 * batch

    if output_csv is not None:
        csv_file = open(output_csv + f"_pb{batch}_db{decode_batch}", "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(
            ["request", "E2E", "stall_total", "stall_ratio", "sweep_ratio_p99"]
        )

    print()
    print(f"prefill_batch : {batch}")
    print(f"decode_batch  : {decode_batch}")
    print()
    print(
        "request        , E2E latency    , stall_total    ,  stall_ratio   , stall_ratio_p99"
    )

    # For plotting
    requests: List[int] = []
    stall_ratios: List[float] = []
    stall_ratios_p99: List[float] = []

    for R in request_list:
        result = run_experiment(
            csv_path=csv_path,
            request_num=R,
            batch=batch,
            decode_batch_size=decode_batch,
            aggregate=aggregate,
        )

        E2E = result["E2E"]
        stall_total = result["stall_total_s"]
        stall_ratio = result["stall_ratio"] * 100
        stall_ratio_p99 = result["stall_ratio_p99"] * 100

        print(
            f"{R:15d}, {E2E:15.3e}, {stall_total:15.3e}, {stall_ratio:15.3g}, {stall_ratio_p99:15.3g}"
        )

        if writer is not None:
            writer.writerow(
                [
                    R,
                    f"{E2E:.6e}",
                    f"{stall_total:.6e}",
                    f"{stall_ratio:.6e}",
                    f"{stall_ratio_p99:.6e}",
                ]
            )

        requests.append(R)
        stall_ratios.append(stall_ratio)
        stall_ratios_p99.append(stall_ratio_p99)

    if csv_file is not None:
        csv_file.close()

    # Plot stall ratio
    if plot:
        y_pos = list(range(len(requests)))

        bar_h = 0.4
        offset = bar_h / 2

        plt.figure()

        plt.barh(
            [y - offset for y in y_pos], stall_ratios, height=bar_h, label="stall_ratio"
        )

        plt.barh(
            [y + offset for y in y_pos],
            stall_ratios_p99,
            height=bar_h,
            label="stall_ratio_p99",
        )

        plt.yticks(y_pos, [str(r) for r in requests])
        plt.gca().invert_yaxis()

        plt.xlabel("Stall ratio (%)")
        plt.ylabel("Request num")
        plt.title(f"Request num vs stall ratio (Batch size: {batch})")

        plt.legend()
        plt.tight_layout()
        plt.show()


def sweep_decode_batch(
    *,
    csv_path: str,
    batch: int,
    db_scalar_list: List[int],
    request_num: int,
    aggregate: str = "max",
    output_csv: Optional[str] = None,
    plot: bool = True,
) -> None:
    """
    If output_csv is provided, results are written to that CSV file.
    If plot is True, a bar chart of stall_ratio is displayed.
    """
    writer = None
    csv_file = None

    if output_csv is not None:
        csv_file = open(output_csv + f"_b{batch}_r{request_num}", "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(
            ["decode_batch", "E2E", "stall_total", "stall_ratio", "stall_ratio_p99"]
        )

    print()
    print(f"prefill_batch : {batch}")
    print(f"req_num       : {request_num}")
    print()
    print(
        "decode_batch   , E2E latency    , stall_total    , stall_ratio    , stall_ratio_p99"
    )

    # For plotting
    batches: List[int] = []
    stall_ratios: List[float] = []
    stall_ratios_p99: List[float] = []

    for x in db_scalar_list:
        decode_batch_size = x * batch
        result = run_experiment(
            csv_path=csv_path,
            request_num=request_num,
            batch=batch,
            decode_batch_size=decode_batch_size,
            aggregate=aggregate,
        )

        E2E = result["E2E"]
        stall_total = result["stall_total_s"]
        stall_ratio = result["stall_ratio"] * 100
        stall_ratio_p99 = result["stall_ratio_p99"] * 100

        print(
            f"{decode_batch_size:15d}, {E2E:15.3e}, {stall_total:15.3e}, {stall_ratio:15.3g}, {stall_ratio_p99:15.3g}"
        )

        if writer is not None:
            writer.writerow(
                [
                    decode_batch_size,
                    f"{E2E:.6e}",
                    f"{stall_total:.6e}",
                    f"{stall_ratio:.6e}",
                    f"{stall_ratio_p99:.6e}",
                ]
            )

        batches.append(decode_batch_size)
        stall_ratios.append(stall_ratio)
        stall_ratios_p99.append(stall_ratio_p99)

    if csv_file is not None:
        csv_file.close()

    # Plot stall ratio
    if plot:
        # Categorical y positions
        y_pos = list(range(len(batches)))

        bar_h = 0.38
        offset = bar_h / 2

        plt.figure()

        plt.barh(
            [y - offset for y in y_pos],
            stall_ratios,
            height=bar_h,
            label="stall_ratio",
        )

        plt.barh(
            [y + offset for y in y_pos],
            stall_ratios_p99,
            height=bar_h,
            label="stall_ratio_p99",
        )

        plt.yticks(y_pos, [str(b) for b in batches])
        plt.gca().invert_yaxis()
        plt.xlabel("Stall ratio (%)")
        plt.ylabel("Decode batch size")
        plt.title(f"Decode batch size vs stall ratio (Prefill batch size: {batch})")
        plt.tight_layout()
        plt.show()


def exhaustive_search(
    *,
    csv_path: str,
    aggregate: str,
) -> None:
    max_stall = {
        "batch": 0,
        "req_num": 0,
        "stall_ratio": 0.0,
        "stall_ratio_p99": 0.0,
        "decode_batch": 0,
    }

    for b in range(1, 30):
        for r in range((b * 2), (b * 2) * 3 + 1, (b * 2)):
            for db in [1, 4, 8, 16]:
                decode_batch = db * b
                result = run_experiment(
                    csv_path=csv_path,
                    request_num=r,
                    batch=b,
                    decode_batch_size=decode_batch,
                    aggregate=aggregate,
                )
                # if result["stall_ratio"] > max_stall["stall_ratio"]:
                if result["stall_ratio_p99"] > max_stall["stall_ratio_p99"]:
                    max_stall["batch"] = b
                    max_stall["req_num"] = r
                    max_stall["stall_ratio"] = result["stall_ratio"]
                    max_stall["stall_ratio_p99"] = result["stall_ratio_p99"]
                    max_stall["decode_batch"] = decode_batch
                else:
                    print(
                        f"Skipped!\tp_batch: {b:3d}, req_num: {r:3d}, d_batch: {decode_batch:3d}"
                    )

    print()
    print("Summary")
    print("=" * 50)
    print(f"prefill batch   : {max_stall["batch"]}")
    print(f"decode_batch    : {max_stall["decode_batch"]}")
    # print(f"aggregate       : {args.aggregate}")
    print(f"request_num     : {max_stall["req_num"]}")
    print(f"stall_ratio     : {max_stall["stall_ratio"]*100:.3g}%")
    print(f"stall_ratio_p99 : {max_stall["stall_ratio_p99"]*100:.3g}%")
    print("=" * 50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch sweep runner.")
    p.add_argument(
        "--input",
        type=str,
        default="inputs/requests.csv",
        help="Path to requests CSV.",
    )
    p.add_argument("--req_num", type=int, default=None)
    p.add_argument(
        "--aggregate",
        type=str,
        default="max",
        choices=["max", "first", "sum"],
        help="Aggregation policy for (L_prefill, L_decode).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write results to CSV file (e.g., output.csv).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting stall ratio.",
    )
    p.add_argument("--batch", type=int, default=1)
    group = p.add_mutually_exclusive_group()
    group.add_argument("-b", action="store_true", help="Batch_size sweep mode")
    group.add_argument("-db", action="store_true", help="Decode_batch_size sweep mode")
    group.add_argument("-r", action="store_true", help="Request_num sweep mode")
    group.add_argument("-es", action="store_true", help="Max Exhaustive Search")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.b:
        batch_list = [1, 4, 16, 64]
        # batch_list = list(range(1, 65))

        sweep_batches(
            csv_path=args.input,
            batch_list=batch_list,
            aggregate=args.aggregate,
            output_csv=args.output,
            plot=not args.no_plot,
        )
    elif args.r:
        request_list = list(
            range((args.batch * 2) * 2, (args.batch * 2) * 16 + 1, (args.batch * 2) * 2)
        )
        # request_list.append(10000)

        sweep_requests(
            csv_path=args.input,
            request_list=request_list,
            batch=args.batch,
            aggregate=args.aggregate,
            output_csv=args.output,
            plot=not args.no_plot,
        )
    elif args.es:
        exhaustive_search(csv_path=args.input, aggregate=args.aggregate)

    elif args.db:
        if not args.req_num:
            # req_num = 50
            req_num = args.batch * 50
        else:
            req_num = args.req_num

        db_scalar_list = [1, 4, 8, 16]

        sweep_decode_batch(
            csv_path=args.input,
            batch=args.batch,
            db_scalar_list=db_scalar_list,
            request_num=req_num,
            aggregate=args.aggregate,
            output_csv=args.output,
            plot=not args.no_plot,
        )
    else:
        print("Use '-b' or '-r' or '-es' tag!")


if __name__ == "__main__":
    main()
