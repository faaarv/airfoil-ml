
import os, json, numpy as np
from glob import glob

DATA_DIR = "json"
OUT_NPZ  = "dataset.npz"

def as_f32(x): return np.asarray(x, dtype=np.float32)

def main():
    rows = []

    json_files = sorted(glob(os.path.join(DATA_DIR, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in '{DATA_DIR}'")

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            J = json.load(f)

        name = J.get("name")

        yps = as_f32(J["yps"])  
        yss = as_f32(J["yss"])  

        for p in J.get("polars", []):
            try:
                Re     = float(p["Re"])
                Ncrit  = float(p["Ncrit"])
                alpha  = float(p["alpha"])
                Cl     = float(p["Cl"])
                Cd     = float(p["Cd"])
                Cm     = float(p["Cm"])
                Cdp    = float(p.get("Cdp", np.nan))
                cp_ps  = as_f32(p["Cp_ps"])
                cp_ss  = as_f32(p["Cp_ss"])


                rows.append(dict(
                    name=name,
                    Re=Re, Ncrit=Ncrit, alpha=alpha,
                    Cl=Cl, Cd=Cd, Cm=Cm, Cdp=Cdp,
                    y_ps=yps, y_ss=yss,
                    Cp_ps=cp_ps, Cp_ss=cp_ss
                ))
                
            except KeyError as e:
                print(f"[WARN] {name}: missing key {e} in a polar, skipping that case.")
                continue

    if not rows:
        raise RuntimeError("No valid samples collected. Check your JSON structure and lengths.")

    N = len(rows)
    name   = np.asarray([r["name"]  for r in rows], dtype=object)
    Re     = np.asarray([r["Re"]    for r in rows], dtype=np.float64) 
    Ncrit  = np.asarray([r["Ncrit"] for r in rows], dtype=np.float32)
    alpha  = np.asarray([r["alpha"] for r in rows], dtype=np.float32)
    Cl     = np.asarray([r["Cl"]    for r in rows], dtype=np.float32)
    Cd     = np.asarray([r["Cd"]    for r in rows], dtype=np.float32)
    Cm     = np.asarray([r["Cm"]    for r in rows], dtype=np.float32)
    Cdp    = np.asarray([r["Cdp"]   for r in rows], dtype=np.float32)

    y_ps  = np.stack([r["y_ps"]  for r in rows], axis=0).astype(np.float32)
    y_ss  = np.stack([r["y_ss"]  for r in rows], axis=0).astype(np.float32)
    Cp_ps = np.stack([r["Cp_ps"] for r in rows], axis=0).astype(np.float32)
    Cp_ss = np.stack([r["Cp_ss"] for r in rows], axis=0).astype(np.float32)

 
    unique_names, inv = np.unique(name, return_inverse=True)
    airfoil_id = inv.astype(np.int32)
    id2name = {int(i): str(n) for i, n in enumerate(unique_names.tolist())}
    name2id = {v: k for k, v in id2name.items()}
    
    # Save mapping
    with open("airfoil_id_map.json", "w", encoding="utf-8") as f:
        json.dump({"id2name": id2name, "name2id": name2id}, f, ensure_ascii=False, indent=2)

    # Save in one compressed NPZ
    np.savez_compressed(
        OUT_NPZ,
        airfoil_id=airfoil_id,
        y_ps=y_ps, y_ss=y_ss,
        Re=Re, Ncrit=Ncrit, alpha=alpha,
        Cl=Cl, Cd=Cd, Cm=Cm, Cdp=Cdp,
        Cp_ps=Cp_ps, Cp_ss=Cp_ss,
    )

    print(f"Samples (N): {N}")
    print(f"Airfoils   : {len(unique_names)}")

if __name__ == "__main__":
    main()
