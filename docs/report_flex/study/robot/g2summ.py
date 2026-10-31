import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l); o = r["o"]
    if "error" in r: print(o["tag"], "ERROR", r["error"][:200]); continue
    fin = r.get("final", {})
    print(f"{o['tag']:28s} {'pymj' if 'pymj' in r['mujoco_file'] else 'wheel'} steps {r['steps']} bad {r['bad']} fatal {str(r['fatal'])[:50]} "
          f"ms/step {r.get('ms_per_step_all',0):.1f} (contact {r.get('ms_per_step_contact') or 0:.1f}) niter {r.get('mean_niter',0):.0f}/{r.get('max_niter',0)} "
          f"| final nL {fin.get('nL',0):.1f} nR {fin.get('nR',0):.1f} tL {fin.get('tL',0):.1f} tR {fin.get('tR',0):.1f} cL {fin.get('cL',0):.0f} cR {fin.get('cR',0):.0f} other {fin.get('nO',0):.0f} "
          f"| end {r.get('end_normal_N')} inv {r.get('inverted_tets_end')} lift {r.get('box_lift_mm',0):.1f}mm yaw {r['phase_t'].get('box_yaw_deg')}")
    sk = r.get("skin_left")
    if sk: print("    skin_left:", str(sk)[:400])
