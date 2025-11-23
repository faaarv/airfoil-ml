# imports
import torch
import torch.nn as nn
import joblib
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR 
import matplotlib.pyplot as plt

Device = "cuda" if torch.cuda.is_available() else "cpu"

#loads : ds , scaler , splitting_ids , 
xfoil_ds_path = 'dataset.npz'
D = np.load(xfoil_ds_path,allow_pickle=False)
S = joblib.load("scaler.pkl")['scalers']
SD = joblib.load("split_data.pkl")
af = SD["split_airfoils"]
# 
airfoil_id = D["airfoil_id"].astype(np.int64)
y_ps  = D["y_ps"].astype(np.float32)
y_ss  = D["y_ss"].astype(np.float32)
Re    = D["Re"].astype(np.float32)
Ncrit = D["Ncrit"].astype(np.float32)
alpha = D["alpha"].astype(np.float32)
Cp_ps = D["Cp_ps"].astype(np.float32)
Cp_ss = D["Cp_ss"].astype(np.float32)
# 
X = np.hstack([y_ps, y_ss,
                Re[:,None],
                alpha[:,None],
                Ncrit[:,None]
                ]).astype(np.float32)

Y = np.hstack([Cp_ps, Cp_ss]).astype(np.float32)


train_airfoil_ids = af["train_airfoils"]
val_airfoil_ids = af["val_airfoils"]
test_airfoil_ids = af["test_airfoils"]

train_idx = np.where(np.isin(airfoil_id, train_airfoil_ids))[0]
test_idx = np.where(np.isin(airfoil_id,test_airfoil_ids))
val_idx =np.where(np.isin(airfoil_id,val_airfoil_ids))

########################  sanity check ####################################################
print(f'airfoil id shape: {airfoil_id.shape}')
print(f'lenght train:{len(train_idx)}')
print(f'lenght test:{len(test_idx)}')
print(f'lenght val:{len(val_idx)}')
print("Unique airfoils:",
      len(np.unique(airfoil_id[train_idx])),
      len(np.unique(airfoil_id[val_idx])),
      len(np.unique(airfoil_id[test_idx])))
total_samples = len(train_idx) + len(val_idx) + len(test_idx)
print(f"\nTotal samples from splits: {total_samples}")
print(f"Total samples in dataset: {len(airfoil_id)}")

assert total_samples == len(airfoil_id) , "mismatch in samples length"
##########################################################################################

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]
print("Train:", X_train.shape, Y_train.shape)
print("Val  :", X_val.shape,   Y_val.shape)
print("Test :", X_test.shape,  Y_test.shape)

####################### Scaling ###########################################################
X_YPS   = slice(0, 100)
X_YSS   = slice(100, 200)
X_RE = 200
X_ALPHA = 201
X_NCRIT = 202
Y_CP_PS  = slice(0, 100)
Y_CP_SS  = slice(100, 200)

def transform_X(X_):
    re_col = X_[:, X_RE:X_RE+1]
    re_log = np.log10(re_col)
    X_[:, X_RE] = S["sc_logRe"].transform(re_log).ravel()

    X_[:, X_ALPHA] = S["sc_alpha"].transform(X_[:, X_ALPHA:X_ALPHA+1]).ravel()
    X_[:, X_NCRIT] = X_[:, X_NCRIT] / np.float32(S["ncrit_div"])

    return X_

def transform_Y(Y_):
    Y_[:, Y_CP_PS] = S["sc_Cp_ps"].transform(Y_[:, Y_CP_PS])
    Y_[:, Y_CP_SS] = S["sc_Cp_ss"].transform(Y_[:, Y_CP_SS])
    return Y_

X_train_s, Y_train_s = transform_X(X_train), transform_Y(Y_train)
X_val_s,   Y_val_s   = transform_X(X_val),   transform_Y(Y_val)
X_test_s,  Y_test_s  = transform_X(X_test),  transform_Y(Y_test)

def to_tensor(x):
    return torch.from_numpy(x)



d_in  = 203
d_out = 200
modelConfig = {
        "lr": 1e-4,
        "hidden": [1024, 512, 256 ],
        "batch": 512,
        "epochs": 100,
        "patience": 10,
}

ds_tr = TensorDataset(to_tensor(X_train_s), to_tensor(Y_train_s))
ds_va = TensorDataset(to_tensor(X_val_s),   to_tensor(Y_val_s))
ds_te = TensorDataset(to_tensor(X_test_s),  to_tensor(Y_test_s))

dl_tr = DataLoader(ds_tr, batch_size=modelConfig["batch"], shuffle=True,  pin_memory=True, num_workers=0)
dl_va = DataLoader(ds_va, batch_size=1024, shuffle=False, pin_memory=True, num_workers=0)
dl_te = DataLoader(ds_te, batch_size=1024, shuffle=False, pin_memory=True, num_workers=0)


############## Define MLP #################################################
class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: list[int], 
                 act=nn.GELU, dropout: float = 0.0):
        super().__init__()
        self.act = act  
        
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), act()] 
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        self.encoder = nn.Sequential(*layers)
        d_features = hidden[-1]
        d_out_head = d_out // 2
        
        self.head_ps = nn.Sequential(
            nn.Linear(d_features, d_features),
            act(),
            nn.Linear(d_features, d_out_head)
        )
        self.head_ss = nn.Sequential(
            nn.Linear(d_features, d_features),
            act(),
            nn.Linear(d_features, d_out_head)
        )
        
        self._init_weights()

    def _init_weights(self):
        nonlinearity = 'relu' if self.act == nn.ReLU else 'linear'
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if nonlinearity == 'relu':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        features = self.encoder(x)
        out_ps = self.head_ps(features)
        out_ss = self.head_ss(features)
        return torch.cat([out_ps, out_ss], dim=1)
    
############################################################################

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = MLP(d_in, d_out, hidden=modelConfig["hidden"], act=nn.ReLU, dropout=0.1).to(Device)
print("trainable params:", count_params(model))
print(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=modelConfig["lr"], weight_decay=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=modelConfig["epochs"])

mse = nn.MSELoss()
best_val = float("inf")
bad = 0
scaler = torch.amp.GradScaler(enabled=(Device == "cuda"))
train_hist, val_hist = [], []

def eval_epoch(dl):
    model.eval()
    losses = []
    for Xb, Yb in dl:
        Xb = Xb.to(Device, non_blocking=True).float()
        Yb = Yb.to(Device, non_blocking=True).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(Device=="cuda")):
            Yhat = model(Xb)
            loss  = mse(Yhat[:,:], Yb[:,:])
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan

def train_epoch(dl):
    model.train()
    losses = []
    for Xb, Yb in dl:
        Xb = Xb.to(Device, non_blocking=True).float()
        Yb = Yb.to(Device, non_blocking=True).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(Device=="cuda")):
            Yhat = model(Xb)
            loss  = mse(Yhat[:, :],  Yb[:, :])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan

########################### Training ##########################
for epoch in range(1, modelConfig["epochs"]+1):
    tr_loss = train_epoch(dl_tr)
    va_loss = eval_epoch(dl_va)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"epoch {epoch:03d} | train {tr_loss:.5f} | val {va_loss:.5f} | LR {current_lr:.2e}")
    train_hist.append(tr_loss)
    val_hist.append(va_loss)

    if va_loss + 1e-6 < best_val:
        best_val = va_loss
        bad = 0

        checkpoint = {
            'epoch': epoch,
            'best_val_loss': best_val,
            'model_config': modelConfig,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scalers_object': S, 
            'train_history': train_hist,
            'val_history': val_hist,
        }
        torch.save(checkpoint, "best_ckpt.pt")
        print(f"  ✓ New best model saved to best_ckpt.pt (Epoch {epoch}, Val Loss: {best_val:.5f})")

    else:
        bad += 1
        if bad >= modelConfig["patience"]:
            print(f"Early stopping at epoch {epoch}.")
            break
print("\nLoading best model from 'best_ckpt.pt' for final evaluation...")

# Load the comprehensive checkpoint
try:
    checkpoint = torch.load("best_ckpt.pt", map_location=Device, weights_only=False)

    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint.get('epoch', 'N/A')}")

    # Run evaluation on test set
    test_loss = eval_epoch(dl_te)
    print("---------------------------------")
    print(f"Test MSE: {test_loss:.5f}")
    print("---------------------------------")

except FileNotFoundError:
    print("ERROR: 'best_ckpt.pt' not found. Could not run final evaluation.")
    test_loss = -1.0


# --- (Plotting remains the same) ---
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
