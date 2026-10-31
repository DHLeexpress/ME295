import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l); o = r["o"]
    if "error" in r: print(o["tag"], "ERROR", r["error"][:150]); continue
    hw = r.get("held_window", {}); mx = r.get("max_over_run", {})
    g = lambda k: hw.get(k, [float('nan')] * 2)
    print(f"{o['tag']:26s} bad {r['bad']} fat {str(r['fatal'])[:20]} ms {r.get('ms_per_step_all',0):5.1f} it {g('niter')[0]:5.0f} | held nL {g('nL')[0]:7.1f}±{g('nL')[1]:5.1f} nR {g('nR')[0]:7.1f}±{g('nR')[1]:5.1f} rigid {g('rigid_n')[0]:7.1f} "
          f"comp {1e3*g('compL')[0]:6.3f}/{1e3*g('compR')[0]:6.3f}mm inv {g('invL')[0]:.0f}/{g('invR')[0]:.0f} | max nL {mx.get('nL',0):9.0f} rigid {mx.get('rigid_n',0):7.0f} comp {1e3*mx.get('compL',0):.3f} | lift {r.get('box_lift_mm',0):5.1f}")
