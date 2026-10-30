import math, os, subprocess, sys, tempfile

SIZES = [1000, 10000, 100000, 1000000, 10000000, 100000000]
RUNS = 10
T_CRIT_95 = 2.262  
EXE = "./sort"

def time_ms(entry, path):
    with tempfile.NamedTemporaryFile(mode="r") as tf:
        with open(path, "rb") as f:
            subprocess.run([EXE, "-e", entry, "-n", "-r", "2", "-t", tf.name],
                           stdin=f, stdout=subprocess.DEVNULL, check=True)
        lines = [l for l in tf.read().split() if l]
    return float(lines[-1]) / 1000.0  

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
