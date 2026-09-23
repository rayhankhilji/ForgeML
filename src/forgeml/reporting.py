from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

from forgeml.measurement import summarize, write_report


def _bytes(spec: dict) -> int:
    sizes = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,
        "bool": 1,
    }
    return math.prod(spec["shape"]) * sizes[spec["dtype"]]


def validate_memory_plan(explanation: dict) -> None:
    graph = explanation["graph"]
    plan = explanation["memory_plan"]
    internal = {
        node["name"]: node for node in graph["nodes"] if node["name"] not in graph["outputs"]
    }
    naive = sum(_bytes(node["spec"]) for node in internal.values())
    planned = sum(_bytes(spec) for spec in plan["slot_specs"].values())
    if naive != plan["naive_bytes"] or planned != plan["planned_bytes"]:
        raise ValueError("memory totals do not agree with the recorded IR and slots")
    if set(plan["allocations"]) != set(internal):
        raise ValueError("memory plan must cover exactly the intermediate values")
    slots: dict[int, list[tuple[int, int]]] = {}
    for name, allocation in plan["allocations"].items():
        size = _bytes(internal[name]["spec"])
        capacity = _bytes(plan["slot_specs"][str(allocation["slot"])])
        if allocation["size_bytes"] != size or size > capacity:
            raise ValueError("invalid slot capacity")
        interval = (allocation["first"], allocation["last"])
        if interval[0] > interval[1]:
            raise ValueError("invalid lifetime")
        previous = slots.setdefault(allocation["slot"], [])
        if any(max(first, interval[0]) <= min(last, interval[1]) for first, last in previous):
            raise ValueError("overlapping live values share a slot")
        previous.append(interval)


def compiler_summary(report: dict) -> list[dict]:
    if report.get("suite") != "forgeml.compiler" or report.get("measured") is not True:
        raise ValueError("a measured compiler report is required")
    rows = []
    for workload in report["workloads"]:
        for result in workload["correctness"].values():
            if result["passed"] is not True:
                raise ValueError("cannot render a failed correctness gate as performance data")
        timings = {}
        for name, recorded in workload["timings"].items():
            derived = summarize(recorded["samples_ms"])
            for field in ("median_ms", "p95_ms", "count", "min_ms", "max_ms"):
                if not math.isclose(derived[field], recorded[field], rel_tol=1e-12):
                    raise ValueError(f"stored {field} disagrees with raw samples")
            if derived["count"] != report["settings"]["repeats"]:
                raise ValueError("sample count disagrees with benchmark settings")
            timings[name] = derived
        for variant in ("optimized", "unoptimized"):
            validate_memory_plan(workload[variant])
        eager = timings["eager"]["median_ms"]
        optimized = timings["optimized"]["median_ms"]
        speedup = eager / optimized
        if not math.isclose(
            speedup, workload["timings"]["optimized"]["speedup_vs_eager"], rel_tol=1e-12
        ):
            raise ValueError("speedup disagrees with latency samples")
        rows.append(
            {
                "name": workload["name"],
                "eager_ms": eager,
                "unoptimized_ms": timings["unoptimized"]["median_ms"],
                "optimized_ms": optimized,
                "optimized_p95_ms": timings["optimized"]["p95_ms"],
                "speedup": speedup,
                "original_nodes": workload["optimized"]["original_nodes"],
                "optimized_nodes": workload["optimized"]["optimized_nodes"],
                "unoptimized_naive_kib": workload["unoptimized"]["memory_plan"]["naive_bytes"]
                / 1024,
                "unoptimized_planned_kib": workload["unoptimized"]["memory_plan"]["planned_bytes"]
                / 1024,
                "optimized_planned_kib": workload["optimized"]["memory_plan"]["planned_bytes"]
                / 1024,
            }
        )
    return rows


def bar_chart(
    rows: list[dict], series: list[tuple[str, str, str]], *, title: str, unit: str, note: str
) -> str:
    width = 1060
    row_height = len(series) * 19 + 30
    height = 150 + len(rows) * row_height
    left = 230
    plot_width = 690
    maximum = max(row[field] for row in rows for field, _, _ in series) or 1.0
    maximum *= 1.12
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(note)}</desc>',
        '<rect width="100%" height="100%" rx="16" fill="#0b1220"/>',
        '<g font-family="ui-monospace,SFMono-Regular,Consolas,monospace">',
        f'<text x="28" y="38" fill="#f1f5f9" font-size="21" font-weight="700">{html.escape(title)}</text>',
        f'<text x="28" y="63" fill="#94a3b8" font-size="12">{html.escape(note)}</text>',
    ]
    for index, (_, label, color) in enumerate(series):
        x = 28 + index * 300
        parts.append(f'<rect x="{x}" y="82" width="11" height="11" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 19}" y="92" fill="#cbd5e1" font-size="12">{html.escape(label)}</text>'
        )
    for tick in range(5):
        value = maximum * tick / 4
        x = left + plot_width * tick / 4
        parts.append(f'<path d="M{x:.1f} 115 V{height - 36}" stroke="#263244" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{height - 15}" fill="#94a3b8" font-size="11">{value:.2f} {html.escape(unit)}</text>'
        )
    for index, row in enumerate(rows):
        top = 122 + index * row_height
        label = html.escape(row["name"])
        parts.append(f'<text x="28" y="{top + 15}" fill="#e2e8f0" font-size="12">{label}</text>')
        for offset, (field, _, color) in enumerate(series):
            value = row[field]
            y = top + offset * 19
            length = value / maximum * plot_width
            parts.append(
                f'<rect x="{left}" y="{y}" width="{length:.2f}" height="12" rx="2" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{left + length + 7:.2f}" y="{y + 11}" fill="#cbd5e1" font-size="11">{value:.3f}</text>'
            )
    parts.append("</g></svg>\n")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render figures from frozen compiler measurements")
    parser.add_argument("report")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    raw = Path(args.report).read_bytes()
    report = json.loads(raw)
    rows = compiler_summary(report)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    env = report["environment"]
    settings = report["settings"]
    device = env["device"].split(":")[0]
    if device not in ("cpu", "cuda"):
        raise ValueError("report device must be cpu or cuda")
    hardware = env.get("gpu_name", f"{env['machine']} {env['platform'].split('-')[0]}")
    (destination / f"compiler-{device}.json").write_bytes(raw)
    write_report(
        {
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "environment": report["environment"],
            "rows": rows,
        },
        destination / "compiler-summary.json",
    )
    (destination / "compiler-latency.svg").write_text(
        bar_chart(
            rows,
            [
                ("eager_ms", "PyTorch eager", "#94a3b8"),
                ("unoptimized_ms", "ForgeML unoptimized", "#fbbf24"),
                ("optimized_ms", "ForgeML optimized", "#67e8f9"),
            ],
            title=f"{device.upper()} latency / raw measurements",
            unit="ms",
            note=(
                f"{hardware} · PyTorch {env['torch']} · {settings['dtype']} · "
                f"{env['torch_threads']} thread(s) · median of {settings['repeats']} calls"
            ),
        ),
        encoding="utf-8",
    )
    (destination / "compiler-memory.svg").write_text(
        bar_chart(
            rows,
            [
                ("unoptimized_naive_kib", "Unoptimized / distinct buffers", "#94a3b8"),
                ("unoptimized_planned_kib", "Unoptimized / reused slots", "#fbbf24"),
                ("optimized_planned_kib", "Optimized / reused slots", "#67e8f9"),
            ],
            title="Compiler-planned intermediate storage",
            unit="KiB",
            note="Derived from captured IR plans · excludes inputs, weights, outputs and backend scratch · NOT peak RSS/VRAM",
        ),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
