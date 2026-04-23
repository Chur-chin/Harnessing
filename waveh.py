#!/usr/bin/env python3
"""
waveh.py  —  Wave Harnessing CLI
Dispatch wave-physics computations to the best available chip.

Usage:
    python waveh.py devices
    python waveh.py run lyapunov --w0 1.0 --g1 0.8
    python waveh.py run param_sweep --n 12 --T 120
    python waveh.py run stdp_batch --n_samples 5000
    python waveh.py run synapse_batch --n_samples 2000
    python waveh.py run hbn_epsilon --n_points 500
    python waveh.py benchmark
    python waveh.py run lyapunov --force gpu --w0 1.5
"""

import argparse
import json
import sys


def parse_payload(extra):
    payload = {}
    i = 0
    while i < len(extra):
        if extra[i].startswith("--"):
            key = extra[i][2:]
            if i+1 < len(extra) and not extra[i+1].startswith("--"):
                raw = extra[i+1]
                try:    val = int(raw)
                except ValueError:
                    try:    val = float(raw)
                    except ValueError: val = raw
                payload[key] = val; i += 2
            else:
                payload[key] = True; i += 1
        else:
            i += 1
    return payload


def print_result(result):
    dev  = result.get("device_used", "?")
    fn   = result.get("function",    "?")
    ms   = result.get("elapsed_ms",  0)
    icon = {"CPU": "🔵 CPU", "GPU": "🟢 GPU", "NPU": "🟣 NPU"}.get(dev, dev)
    print(f"\n{'='*60}")
    print(f"  Function  : {fn}")
    print(f"  Device    : {icon}")
    print(f"  Elapsed   : {ms} ms")
    print(f"  Scheduled : {result.get('schedule_info','')}")
    print(f"{'='*60}")
    print(json.dumps(result.get("data", {}), indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        prog="waveh",
        description="Wave Harnessing CLI — auto CPU/GPU/NPU dispatch",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("devices",   help="Show detected hardware")
    sub.add_parser("benchmark", help="Benchmark all available chips")

    run_p = sub.add_parser("run", help="Run a wave function")
    run_p.add_argument("function", help="Function name")
    run_p.add_argument("--force", default=None,
                       help="Force backend: cpu | gpu | npu")
    run_p.add_argument("--verbose", action="store_true", default=True)

    args, extra = parser.parse_known_args()

    if args.command == "devices":
        from harnessing import WaveScheduler
        sch = WaveScheduler(verbose=False)
        print(sch.device_info())

    elif args.command == "benchmark":
        from harnessing import WaveScheduler
        sch = WaveScheduler(verbose=False)
        print("\n⏱  Benchmarking all chips (lyapunov, T=100)...\n")
        bm = sch.benchmark(T_ode=100.0)
        for chip, res in bm.items():
            icon = {"CPU":"🔵","GPU":"🟢","NPU":"🟣"}.get(chip, "⚪")
            if res["status"] == "ok":
                print(f"  {icon} {chip:4s}  {res['elapsed_ms']:>8.1f} ms")
            else:
                print(f"  ✗  {chip:4s}  {res['status']}")

    elif args.command == "run":
        payload = parse_payload(extra)
        from harnessing import WaveScheduler
        sch = WaveScheduler(verbose=args.verbose, force=args.force)
        result = sch.run(args.function, payload)
        print_result(result)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
