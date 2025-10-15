import numpy as np , joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# ----- helpers -----
_EPS = 1e-6
def log1p_safe(x):
    return np.log1p(np.maximum(x, _EPS))
def expm1_safe(x):
    return np.expm1(x)
def as2d(x):
    x = np.asarray(x)
    return x[:, None] if x.ndim == 1 else x


def main():
    D = np.load('dataset.npz', allow_pickle=False)
    airfoil_id = D["airfoil_id"]
    y_ps  = D["y_ps"]
    y_ss  = D["y_ss"]
    Re    = D["Re"]
    Ncrit = D["Ncrit"]
    alpha = D["alpha"]


    Cl    = D["Cl"];  Cd = D["Cd"]
    Cm = D["Cm"];  Cdp = D["Cdp"]
    Cp_ps = D["Cp_ps"]; Cp_ss = D["Cp_ss"]
    N = airfoil_id.shape[0]

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(np.zeros(len(airfoil_id)), groups=airfoil_id ))
    gss2 = GroupShuffleSplit(test_size=0.20, random_state=42)
    subtrain_idx, val_rel = next(gss2.split(np.zeros(len(train_idx)), groups=airfoil_id [train_idx]))
    val_idx   = train_idx[val_rel]
    train_idx = train_idx[subtrain_idx]

    print(f"Split sizes -> train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")
    print(f"Unique airfoils -> train:{len(np.unique(airfoil_id[train_idx]))}  "
      f"val:{len(np.unique(airfoil_id[val_idx]))}  test:{len(np.unique(airfoil_id[test_idx]))}")

    split_airfoils = {
        "train_airfoils": np.unique(airfoil_id[train_idx]).astype(np.int32),
        "val_airfoils":   np.unique(airfoil_id[val_idx]).astype(np.int32),
        "test_airfoils":  np.unique(airfoil_id[test_idx]).astype(np.int32),
    }

    meta = {
        "split_kind": "GroupShuffleSplit 72/18/10 by airfoil_id",
        "random_state": 42,
        "sizes": {
            "train_rows": int(train_idx.size),
            "val_rows":   int(val_idx.size),
            "test_rows":  int(test_idx.size),
            "train_airfoils": int(split_airfoils["train_airfoils"].size),
            "val_airfoils":   int(split_airfoils["val_airfoils"].size),
            "test_airfoils":  int(split_airfoils["test_airfoils"].size),
        }
    }

    joblib.dump({"split_airfoils": split_airfoils, "meta": meta}, 'SPLITS_META_Data.pkl', compress=3)
    print("Saved:  'SPLITS_Data.pkl'")

    # ===== SCALERS =====
    scalers = {}
    logRe_tr = np.log10(Re[train_idx])
    sc_logRe = StandardScaler().fit(as2d(logRe_tr))
    scalers["sc_logRe"] = sc_logRe
    scalers["sc_alpha"]   = StandardScaler().fit(as2d(alpha[train_idx]))
    scalers["ncrit_div"]  = np.float32(9.0)
    scalers["sc_Cl"]      = StandardScaler().fit(as2d(Cl[train_idx]))
    scalers["sc_Cm"]      = StandardScaler().fit(as2d(Cm[train_idx]))
    scalers["sc_Cd_log"]  = StandardScaler().fit(as2d(log1p_safe(Cd[train_idx])))
    scalers["sc_Cdp_log"] = StandardScaler().fit(as2d(log1p_safe(Cdp[train_idx])))
    scalers["sc_Cp_ps"]   = StandardScaler().fit(Cp_ps[train_idx])
    scalers["sc_Cp_ss"]   = StandardScaler().fit(Cp_ss[train_idx])

    joblib.dump({"scalers": scalers}, "scalers.pkl", compress=3)
    print("Saved scalers.pkl")
    return

if __name__ == "__main__":
    main()
