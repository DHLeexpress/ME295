"""Inventory of MuJoCo 3.17.0 flex example models and tests: parameter regime vs ours.
Static scan (regex) of model/flex/**/*.xml and test/**/*.cc; for example models also compile them and read the
compiled flex parameters (dim, dof/interp, nvert, young/poisson from flex_stiffness presence, radius, solref, solimp,
integrator, timestep, element edge length). Output: out/t2_models_tests.{txt,json}"""
import glob, json, os, re, sys, warnings
import numpy as np, mujoco
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import dump
SRC = "<scratch>/src/mujoco"
INT = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast", 4: "discrete"}
rows = []
for f in sorted(glob.glob(SRC + "/model/flex/**/*.xml", recursive=True)):
    txt = open(f).read()
    if "flex" not in txt or os.path.basename(f) == "scene.xml":
        continue
    attrs = dict(young=re.findall(r'young="([^"]+)"', txt), poisson=re.findall(r'poisson="([^"]+)"', txt),
                 dof=re.findall(r'dof="([^"]+)"', txt), solref=re.findall(r'solref="([^"]+)"', txt),
                 solimp=re.findall(r'solimp="([^"]+)"', txt), elastic2d=re.findall(r'elastic2d="([^"]+)"', txt))
    r = dict(file=os.path.relpath(f, SRC), **{k: sorted(set(v)) for k, v in attrs.items()})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = mujoco.MjModel.from_xml_path(f)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        fl = []
        for i in range(m.nflex):
            ea, en = m.flex_edgeadr[i], m.flex_edgenum[i]
            L = m.flexedge_length0[ea:ea + en] if en else np.array([0.0])
            fl.append(dict(dim=int(m.flex_dim[i]), interp=int(m.flex_interp[i]), nvert=int(m.flex_vertnum[i]),
                           radius=float(m.flex_radius[i]), edge_mm=round(float(np.median(L)) * 1e3, 3),
                           stiff=bool(m.flex_stiffnessadr[i] >= 0), solref=[float(x) for x in m.flex_solref[i]],
                           solimp=[round(float(x), 4) for x in m.flex_solimp[i][:3]]))
        r.update(integrator=INT.get(int(m.opt.integrator), int(m.opt.integrator)), timestep=float(m.opt.timestep),
                 solver=int(m.opt.solver), flexes=fl)
    except Exception as e:
        r.update(compile_error=str(e)[:160])
    rows.append(r)
tests = []
for f in sorted(glob.glob(SRC + "/test/**/*.cc", recursive=True)):
    txt = open(f).read()
    for mt in re.finditer(r'TEST_F?\((\w+),\s*(\w+)\)', txt):
        name = mt.group(2)
        body = txt[mt.end():mt.end() + 6000]
        nxt = re.search(r'\nTEST_F?\(', body)
        body = body[:nxt.start()] if nxt else body
        if not re.search(r'flex', body, re.I):
            continue
        tests.append(dict(file=os.path.relpath(f, SRC), test=f"{mt.group(1)}.{name}",
                          young=sorted(set(re.findall(r'young=\\?"([^"\\]+)', body))),
                          poisson=sorted(set(re.findall(r'poisson=\\?"([^"\\]+)', body))),
                          integrator=sorted(set(re.findall(r'integrator=\\?"([^"\\]+)', body))),
                          dof=sorted(set(re.findall(r'dof=\\?"([^"\\]+)', body)))))
p = dump("t2_models_tests.json", dict(models=rows, tests=tests))
with open(p.replace(".json", ".txt"), "w") as fo:
    for r in rows:
        fo.write(json.dumps(r) + "\n")
    fo.write(f"\n{len(tests)} flex-related tests\n")
    for t in tests:
        if t["young"] or t["poisson"]:
            fo.write(json.dumps(t) + "\n")
print(p)
