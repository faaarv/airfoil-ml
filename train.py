
import numpy as np
import joblib
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

D = np.load("dataset.npz", allow_pickle=False)
S = joblib.load("scalers.pkl")["scalers"]
M = joblib.load("SPLITS_Data.pkl")  
af = M["split_airfoils"]

def idx_from_airfoil_sets(aid: np.ndarray, wanted_ids: np.ndarray) -> np.ndarray:
    mask = np.isin(aid, wanted_ids)
    return np.nonzero(mask)[0].astype(np.int64)

_EPS = 1e-6
airfoil_id = D["airfoil_id"].astype(np.int64)

y_ps  = D["y_ps"].astype(np.float32)
y_ss  = D["y_ss"].astype(np.float32)
Re    = D["Re"].astype(np.float32)
Ncrit = D["Ncrit"].astype(np.float32)
alpha = D["alpha"].astype(np.float32)

Cl    = D["Cl"].astype(np.float32)
Cd    = D["Cd"].astype(np.float32)
Cm    = D["Cm"].astype(np.float32)
Cdp   = D["Cdp"].astype(np.float32)
Cp_ps = D["Cp_ps"].astype(np.float32)
Cp_ss = D["Cp_ss"].astype(np.float32)

X = np.hstack([y_ps, y_ss,
                   Re[:,None],
                   alpha[:,None],
                   Ncrit[:,None]
                  ]).astype(np.float32)
Y = np.hstack([
    np.stack([Cl, Cd, Cm, Cdp], axis=1),
    Cp_ps, Cp_ss
]).astype(np.float32)


train_idx = idx_from_airfoil_sets(airfoil_id, af["train_airfoils"])
val_idx   = idx_from_airfoil_sets(airfoil_id, af["val_airfoils"])
test_idx  = idx_from_airfoil_sets(airfoil_id, af["test_airfoils"])

print("Split sizes:",
      len(train_idx), len(val_idx), len(test_idx))
print("Unique airfoils:",
      len(np.unique(airfoil_id[train_idx])),
      len(np.unique(airfoil_id[val_idx])),
      len(np.unique(airfoil_id[test_idx])))

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]

np.savez_compressed(
    "test_split_raw_v3.npz",
    X_test=X_test.astype(np.float32),
    Y_test=Y_test.astype(np.float32),
    test_idx=test_idx.astype(np.int64),
    airfoil_id=airfoil_id[test_idx].astype(np.int32),
)

print("Train:", X_train.shape, Y_train.shape)
print("Val  :", X_val.shape,   Y_val.shape)
print("Test :", X_test.shape,  Y_test.shape)

#scaling
#defining indexes
X_YPS   = slice(0, 100)
X_YSS   = slice(100, 200)
X_RE = 200
X_ALPHA = 201
X_NCRIT = 202
Y_CL     = 0
Y_CD     = 1
Y_CM     = 2
Y_CDP    = 3
Y_CP_PS  = slice(4, 104)
Y_CP_SS  = slice(104, 204)

def transform_X(X_):
    re_col = X_[:, X_RE:X_RE+1]
    re_log = np.log10(re_col)
    X_[:, X_RE] = S["sc_logRe"].transform(re_log).ravel()

    X_[:, X_ALPHA] = S["sc_alpha"].transform(X_[:, X_ALPHA:X_ALPHA+1]).ravel()
    X_[:, X_NCRIT] = X_[:, X_NCRIT] / np.float32(S["ncrit_div"])

    return X_

def transform_Y(Y_):

    Y_[:, Y_CL] = S["sc_Cl"].transform(Y_[:, Y_CL:Y_CL+1]).ravel()

    Y_[:, Y_CM] = S["sc_Cm"].transform(Y_[:, Y_CM:Y_CM+1]).ravel()

    cd_log = np.log1p(np.maximum(Y_[:, Y_CD:Y_CD+1], _EPS))
    Y_[:, Y_CD] = S["sc_Cd_log"].transform(cd_log).ravel()
    cdp_log = np.log1p(np.maximum(Y_[:, Y_CDP:Y_CDP+1], _EPS))
    Y_[:, Y_CDP] = S["sc_Cdp_log"].transform(cdp_log).ravel()
    Y_[:, Y_CP_PS] = S["sc_Cp_ps"].transform(Y_[:, Y_CP_PS])
    Y_[:, Y_CP_SS] = S["sc_Cp_ss"].transform(Y_[:, Y_CP_SS])

    return Y_

X_train_s, Y_train_s = transform_X(X_train), transform_Y(Y_train)
X_val_s,   Y_val_s   = transform_X(X_val),   transform_Y(Y_val)
X_test_s,  Y_test_s  = transform_X(X_test),  transform_Y(Y_test)

np.savez_compressed(
    "test_split_scaled_v3.npz",
    X_test_s=X_test_s.astype(np.float32),
    Y_test_s=Y_test_s.astype(np.float32),
    test_idx=test_idx.astype(np.int64),
    airfoil_id=airfoil_id[test_idx].astype(np.int32),
)

def to_tensor(x): 
    return torch.from_numpy(x)

ds_tr = TensorDataset(to_tensor(X_train_s), to_tensor(Y_train_s))
ds_va = TensorDataset(to_tensor(X_val_s),   to_tensor(Y_val_s))
ds_te = TensorDataset(to_tensor(X_test_s),  to_tensor(Y_test_s))


modelConfig = {
        "lr": 3e-4,
        "hidden": [512, 256, 256],
        "batch": 512,
        "epochs": 100,
        "patience": 10,
}

dl_tr = DataLoader(ds_tr, batch_size=modelConfig["batch"], shuffle=True,  pin_memory=True, num_workers=0)
dl_va = DataLoader(ds_va, batch_size=1024, shuffle=False, pin_memory=True, num_workers=0)
dl_te = DataLoader(ds_te, batch_size=1024, shuffle=False, pin_memory=True, num_workers=0)

class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: list[int], act=nn.ReLU, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
            return self.net(x)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


W_SCALARS = 1.0
W_CP      = 2.0
d_in  = 203
d_out = 204

model = MLP(d_in, d_out, hidden=modelConfig["hidden"], act=nn.ReLU, dropout=0.0).to(device)
print("trainable params:", count_params(model))
print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)

mse = nn.MSELoss()
best_val = float("inf")
bad = 0
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

train_hist, val_hist = [], []

def eval_epoch(dl):
    model.eval()
    losses = []
    for Xb, Yb in dl:
        Xb = Xb.to(device, non_blocking=True).float()
        Yb = Yb.to(device, non_blocking=True).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            Yhat = model(Xb)
            loss_scal = mse(Yhat[:, :4], Yb[:, :4])
            loss_cp   = mse(Yhat[:, 4:], Yb[:, 4:])
            loss = W_SCALARS*loss_scal + W_CP*loss_cp
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan


def train_epoch(dl):
    model.train()
    losses = []
    for Xb, Yb in dl:
        Xb = Xb.to(device, non_blocking=True).float()
        Yb = Yb.to(device, non_blocking=True).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            Yhat = model(Xb)
            loss_scal = mse(Yhat[:, :4],  Yb[:, :4])
            loss_cp   = mse(Yhat[:, 4:],  Yb[:, 4:])
            loss = W_SCALARS*loss_scal + W_CP*loss_cp
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan



for epoch in range(1, modelConfig["epochs"]+1):
    tr_loss = train_epoch(dl_tr)
    va_loss = eval_epoch(dl_va)
    print(f"epoch {epoch:03d} | train {tr_loss:.5f} | val {va_loss:.5f}")
    train_hist.append(tr_loss)
    val_hist.append(va_loss)

    if va_loss + 1e-6 < best_val:
        best_val = va_loss
        bad = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        bad += 1
        if bad >= modelConfig["patience"]:
            print("Early stopping.")
            break

model.load_state_dict(torch.load("best_model.pt", map_location=device))
test_loss = eval_epoch(dl_te)
print("test MSE:", test_loss)

torch.save({
    "model": model.state_dict(),
    "opt": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
}, "best_ckpt.pt")

plt.figure()
plt.plot(train_hist, label="train")
plt.plot(val_hist,   label="val")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("Saved loss plot to loss_curve.png")
