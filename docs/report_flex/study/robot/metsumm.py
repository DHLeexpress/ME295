"""Compact table of out/metric_*.json."""
import json, glob, sys
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "out/metric_*.json")):
    r = json.load(open(f))
    print(f"{r['tag']:20s} nv {r['nv']:5d} isl_nv {r['island_nv']:5d} | flexblock relerr mono {r['relerr_mono_vs_h2Kflex']['flex-flex']:.1e} isl {r['relerr_isl_vs_h2Kflex']['flex-flex']:.1e}"
          f" | all relerr mono {r['relerr_mono_vs_full']['all']:.1e} | |S_isl flex| {r['absnorm_S_isl']['flex-flex'] if 'absnorm_S_isl' in r else r['norm_isl_flexflex']:.3g} |h2K| {r['norm_h2Kfd_flexflex']:.3g} |M| {r['norm_M_flexflex']:.3g}"
          f" | eigmin A_ref {r['eig_A_ref']['min']:.3g} M+Smono {r['eig_M+S_mono']['min']:.3g} M+Sisl {r['eig_M+S_isl']['min']:.3g} nneg {r['eig_A_ref']['n_neg']},{r['eig_M+S_mono']['n_neg']}"
          f" | maxgeneig {r['max_eig_(M+S_isl)^-1(M+S_mono)']:.4g} | flex-other abs mono {r.get('absnorm_S_mono',{}).get('flex-other',float('nan')):.2g} ref {r.get('absnorm_ref_full',{}).get('flex-other',float('nan')):.2g}")
