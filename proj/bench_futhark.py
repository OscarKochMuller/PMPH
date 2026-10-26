#!/usr/bin/env python3
# Runs the Futhark baselines on the datasets written by test7.cu (data_n<n>_r<r>.in)
# and prints the same kind of table as the CUDA program.
#
# Setup:   futhark pkg add github.com/diku-dk/sorts && futhark pkg sync
#          futhark cuda sort.fut          (creates the executable ./sort)
# Usage:   python3 bench_futhark.py [n1 n2 ...]      (default: the same sizes as test7.cu)

import math, os, subprocess, sys, tempfile

SIZES = [1000, 10000, 100000, 1000000, 10000000, 100000000]
RUNS = 10
T_CRIT_95 = 2.262  # two-sided 95% Student-t critical value, df = RUNS-1 = 9
assert RUNS == 10, "T_CRIT_95 is for df = 9; update it if RUNS changes"
EXE = "./sort"

def time_ms(entry, path):
    # -r 2: the first run is an untimed warm-up, the second one is the measurement
    with tempfile.NamedTemporaryFile(mode="r") as tf:
        with open(path, "rb") as f:
            subprocess.run([EXE, "-e", entry, "-n", "-r", "2", "-t", tf.name],
                           stdin=f, stdout=subprocess.DEVNULL, check=True)
        lines = [l for l in tf.read().split() if l]
    return float(lines[-1]) / 1000.0  # microseconds -> ms

def is_ok(entry, path):
    with open(path, "rb") as f:
        out = subprocess.run([EXE, "-e", entry], stdin=f, stdout=subprocess.PIPE, check=True).stdout
    return out.strip() == b"true"

def mean_ci(xs):
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    h = T_CRIT_95 * s / math.sqrt(len(xs))
    return m, h

sizes = [int(a) for a in sys.argv[1:]] or SIZES

for n in sizes:
    paths = ["data_n%d_r%d.in" % (n, r) for r in range(RUNS)]
    for p in paths:
        if not os.path.exists(p):
            sys.exit("missing %s (run the CUDA program first)" % p)

    rs_times = []
    rs_ok = True
    for p in paths:
        rs_times.append(time_ms("rs", p))
        rs_ok = rs_ok and is_ok("rs_ok", p)

    print("\n=== n = %d  (radix=%s) ===" % (n, "OK" if rs_ok else "FAIL"))
    print("%6s %12s" % ("run", "radix [ms]"))
    for r in range(RUNS):
        print("%6d %12.3f" % (r, rs_times[r]))
    t_rs, h_rs = mean_ci(rs_times)
    print("%6s %12.3f" % ("avg", t_rs))
    print("%6s [%.3f, %.3f]" % ("95%CI", t_rs - h_rs, t_rs + h_rs))
