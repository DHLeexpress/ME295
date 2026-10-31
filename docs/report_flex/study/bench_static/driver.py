# Run the benchmark sweep: one subprocess per model (run_model.py), at most NPROC at a time, skipping models
# whose output JSON already exists with status ok. Usage: python driver.py [nproc] [subset]
import sys, os, json, subprocess, itertools, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import cfg_id
OUT = os.path.join(HERE, "out")
PY = "/home/dohyun/miniforge3/bin/python"


def configs():
    cs = []
    for v in ("full_corr", "full_raw"):
        for nt in (1, 2, 4, 6, 8, 12, 16):
            for nz in (2, 3, 4, 5, 6, 7):
                for nu in (0.45, 0.49):
                    cs.append(dict(variant=v, nt=nt, nz=nz, nu=nu))
    for nt in (1, 2, 4, 6, 8, 12, 16):
        for cz in (1, 2, 3, 4, 6):
            for nu in (0.45, 0.49):
                cs.append(dict(variant="trilinear", nt=nt, cz=cz, nu=nu))
    for nt in (1, 2, 4, 6, 8):
        for cz in (1, 2, 3):
            for nu in (0.45, 0.49):
                cs.append(dict(variant="quadratic", nt=nt, cz=cz, nu=nu))
    # nu = 0.499 (near-incompressible) on a subset
    for nt in (2, 4, 8):
        for nz in (3, 5, 7):
            cs.append(dict(variant="full_corr", nt=nt, nz=nz, nu=0.499))
        for cz in (1, 2, 3):
            cs.append(dict(variant="trilinear", nt=nt, cz=cz, nu=0.499))
            cs.append(dict(variant="quadratic", nt=nt, cz=cz, nu=0.499))
    return cs


def cost(c):
    # rough size for scheduling: biggest first
    o = 2 if c["variant"] == "quadratic" else 1
    nz = c.get("nz", (c.get("cz", 1) * o + 1))
    return (5 * c["nt"] * o + 1) * (7 * c["nt"] * o + 1) * nz * (3 if o == 2 else 1)


def done(c):
    f = os.path.join(OUT, cfg_id(c) + ".json")
    if not os.path.exists(f):
        return False
    try:
        return json.load(open(f)).get("status") == "ok"
    except Exception:
        return False


def go(c):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    r = subprocess.run([PY, os.path.join(HERE, "run_model.py"), json.dumps(c), OUT], capture_output=True, text=True,
                       env=env, cwd=HERE)
    line = [l for l in r.stdout.splitlines() if l.startswith(cfg_id(c))]
    msg = line[-1] if line else ("FAILED rc=%d %s" % (r.returncode, r.stderr[-300:]))
    print(f"[{time.strftime('%H:%M:%S')}] {msg} ({time.time()-t0:.0f}s)", flush=True)
    return c, r.returncode


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    cs = [c for c in configs() if not done(c)]
    if len(sys.argv) > 2:
        sub = sys.argv[2]
        cs = [c for c in cs if eval(sub, {}, dict(c=c))]
    cs.sort(key=cost, reverse=True)
    print(len(cs), "models to run with", nproc, "processes", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(nproc) as ex:
        list(ex.map(go, cs))
