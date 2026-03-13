#!/usr/bin/env python3
"""
train_qat_and_export.py – FX-QAT → ap_fixed headers

Pipeline stages:
- weighted training-data loading
- FX-based quantization-aware training
- activation and cell-state statistics collection
- fake-quant and INT8 evaluation
- fixed-point format suggestion
- HLS-compatible header export
"""

import os
import time
import json
import math
import random

import h5py
import numpy as np
import scipy.io as sio
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (TensorDataset, DataLoader,
                              WeightedRandomSampler)
from torch.ao.quantization import (
    QConfigMapping, fake_quantize, observer, quantize_fx, QConfig
)
from torch.ao.quantization.observer import NoopObserver

# ───────────────────────── CONFIG ─────────────────────────────────────────
# Note:
# The original .mat dataset and weight files are not included in this repository.
# Update these paths locally before running the script.

class Cfg:
    WEIGHTS_MAT = "path/to/rawNetWeights.mat"
    CLEAN_DATA  = "path/to/qat_data.mat"
    TRAIN_KEY   = "cleanTrainData"
    TEST_KEY    = "cleanTestData"

    INPUT_SIZE  = 1
    HIDDEN_SIZE = 200
    OUTPUT_SIZE = 4
    SEG_LEN     = 5000

    QAT_EPOCHS  = 30
    BS_TRAIN    = 200
    BS_TEST     = 256
    LR          = 1e-4
    LR_DROP_EP  = 15
    LR_GAMMA    = 0.3

    F_BITS      = 10                     # ap_fixed<16,6> style (INT = 6, FRAC = 10)
    OUT_DIR     = "hls_headers_int8"
    CKPT_INT8   = "qat_int8_finetuned.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────── DATA LOADERS ───────────────────────────────────────
def load_clean(path: str, key: str):
    with h5py.File(path, "r") as f:
        raw = f[key][()]
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T                                            # unify shape
    M, _ = raw.shape
    X = np.zeros((M, Cfg.SEG_LEN), np.float32)
    Y = np.zeros((M, Cfg.SEG_LEN), np.int64)
    with h5py.File(path, "r") as f:
        for i in range(M):
            seg, lbl = raw[i, 0], raw[i, 1]
            X[i] = f[seg][()].squeeze()
            Y[i] = f[lbl][()].squeeze()
    return X, Y

def make_train_loader(X, Y):
    frac_nna = [(np.bincount(y, minlength=Cfg.OUTPUT_SIZE)[0] / y.size)
                for y in Y]
    weights = 1.0 + 2.0 * (1.0 - np.array(frac_nna))           # >1 ⇒ oversample
    return DataLoader(TensorDataset(torch.from_numpy(X),
                                    torch.from_numpy(Y)),
                      batch_size=Cfg.BS_TRAIN,
                      sampler=WeightedRandomSampler(weights,
                                                    len(weights),
                                                    replacement=True))

def make_test_loader(X, Y):
    return DataLoader(TensorDataset(torch.from_numpy(X),
                                    torch.from_numpy(Y)),
                      batch_size=Cfg.BS_TEST)

# ─────────────────────────── MODEL ────────────────────────────────────────
class LSTMClassifierFX(nn.Module):
    """
    A tiny 1-layer LSTMCell unrolled SEG_LEN steps + 1 FC.
    Adds an **Identity** “input_stub” so that the raw ECG always
    passes through a Fake-Quant observer/quantizer node.
    """
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.seq_len = cfg.SEG_LEN
        H = cfg.HIDDEN_SIZE
        self.input_stub = nn.Identity()               # <- observer goes here
        self.cell = nn.LSTMCell(cfg.INPUT_SIZE, H)
        self.fc   = nn.Linear(H, cfg.OUTPUT_SIZE)

    def forward(self, x):
        x = self.input_stub(x)                        # Fake-Quant inserted here
        B, T, _ = x.shape
        H = self.fc.in_features
        h = x.new_zeros(B, H)
        c = h.clone()
        outs = []
        for t in range(self.seq_len):
            h, c = self.cell(x[:, t, :], (h, c))
            outs.append(h)
        return self.fc(torch.stack(outs, 1).view(B*T, H)).view(B, T, -1)

def load_matlab_weights(m: nn.Module, mat_path: str):
    d = sio.loadmat(mat_path, struct_as_record=False,
                    squeeze_me=True)["learnables"]
    if isinstance(d, np.ndarray):
        d = d.item()
    with torch.no_grad():
        m.cell.weight_ih.copy_(torch.tensor(d.Layer2_InputWeights).view(-1, 1))
        m.cell.weight_hh.copy_(torch.tensor(d.Layer2_RecurrentWeights))
        m.cell.bias_ih .copy_(torch.tensor(d.Layer2_Bias))
        m.cell.bias_hh.zero_()
        m.fc.weight    .copy_(torch.tensor(d.Layer3_Weights))
        m.fc.bias      .copy_(torch.tensor(d.Layer3_Bias))

# ────────────── QCONFIGS ──────────────────────────────────────────────────
# fake-quant for activations (per-tensor sym.)
act_fq = fake_quantize.FakeQuantize.with_args(
    observer=observer.MovingAverageMinMaxObserver,
    quant_min=-128, quant_max=127,
    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

# fake-quant for weights (per-channel sym.)
wch_fq = fake_quantize.FakeQuantize.with_args(
    observer=observer.PerChannelMinMaxObserver,
    quant_min=-127, quant_max=127,
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric)

per_ch_cfg = QConfig(activation=act_fq, weight=wch_fq)

# QConfig applied to the input stub
inp_cfg = QConfig(
    activation=act_fq,
    weight=NoopObserver.with_args(dtype=torch.float)   # callable dummy
)

# QAT configuration mapping
QCFG_MAP = (QConfigMapping()
            .set_global(per_ch_cfg)
            .set_object_type(nn.Linear,   per_ch_cfg)
            .set_object_type(nn.LSTMCell, per_ch_cfg)
            .set_module_name("input_stub", inp_cfg))

# ─────────────── STATS  (incl. Welford) ───────────────────────────────────
stats = {
    "activations": {},
    "gmax"       : 0.0,
    "cell_min"   :  1e9,
    "cell_max"   : -1e9,
    # Welford accumulators
    "cell_mu"    : 0.0,
    "cell_m2"    : 0.0,
    "cell_cnt"   : 0,
    # reservoir sample for percentiles
    "cell_vals"  : []
}

def act_hook(name):
    def _fn(_, __, out):
        lo, hi = out.min().item(), out.max().item()
        arr = stats["activations"].setdefault(name, [lo, hi])
        arr[0] = min(arr[0], lo);  arr[1] = max(arr[1], hi)
    return _fn

def gate_observer_hook(mod):
    ap = getattr(mod, "activation_post_process", None)
    if ap is not None and hasattr(ap, "scale"):
        s = ap.scale.detach().cpu()
        if s.numel() == 1:
            stats["gmax"] = max(stats["gmax"], 127.0 * s.item())

def cell_state_hook(_, __, out):
    if isinstance(out, tuple) and len(out) == 2:       # (h_next, c_next)
        c = out[1].detach()
        stats["cell_min"] = min(stats["cell_min"], c.min().item())
        stats["cell_max"] = max(stats["cell_max"], c.max().item())
        # Welford
        n  = c.numel()
        delta = c.mean().item() - stats["cell_mu"]
        stats["cell_cnt"] += n
        stats["cell_mu"]  += delta * n / stats["cell_cnt"]
        stats["cell_m2"]  += ((c - stats["cell_mu"])**2).sum().item()
        # reservoir sample ≤50 000
        if len(stats["cell_vals"]) < 50_000:
            stats["cell_vals"].extend(
                c.flatten()[:(50_000-len(stats["cell_vals"]))].cpu().tolist())
        else:
            for val in c.flatten()[:1024]:
                if random.random() < 1024 / stats["cell_cnt"]:
                    idx = random.randrange(50_000)
                    stats["cell_vals"][idx] = val.item()

# ──────────── helpers ────────────────────────────────────────────────────

def compute_frac_shift(stats, model):
    if stats["gmax"] == 0:
        print("   ⓘ  gmax=0 → weight-based fallback")
        with torch.no_grad():
            W_ih = model.cell.weight_ih.detach().abs().cpu()
            W_hh = model.cell.weight_hh.detach().abs().cpu()
            B    = model.cell.bias_ih.detach().abs().cpu()
            stats["gmax"] = float((W_ih.squeeze() + W_hh.sum(1) + B).max())
    return int(math.ceil(math.log2(max(stats["gmax"], 1e-6) / 32.0)))

def choose_state_format(W_all, W_fc, frac_bits, max_total=18):
    max_w = float(np.abs(np.concatenate([W_all.flatten(),
                                         W_fc.flatten()])).max())
    int_bits   = int(math.ceil(math.log2(max(max_w, 1e-9)))) + 1
    total_bits = int_bits + frac_bits
    if total_bits > max_total:
        print(f"⚠️  Need {total_bits} bits, cap {max_total}.")
        total_bits = max_total
        int_bits   = total_bits - frac_bits
    headroom = (2**(int_bits-1)-1) / max_w
    used_pct = max_w / (2**(int_bits-1)-1) * 100
    print(f"✦  Suggested state_t : ap_fixed<{total_bits},{int_bits}>")
    print(f"   – weight head-room ×{headroom:.2f}")
    print(f"   – MSB utilisation  {used_pct:.1f}%")
    with open(os.path.join(Cfg.OUT_DIR, "datatype_suggestion.txt"), "w") as f:
        f.write(f"state_t ap_fixed<{total_bits},{int_bits}>\n")
    return total_bits, int_bits

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    conf = np.zeros((Cfg.OUTPUT_SIZE, Cfg.OUTPUT_SIZE), int)
    tot  = np.zeros(Cfg.OUTPUT_SIZE, int)
    ok   = np.zeros(Cfg.OUTPUT_SIZE, int)
    for xb, yb in loader:
        xb = xb.unsqueeze(-1).to(device)
        yb = yb.to(device)
        B,T,_ = xb.shape
        pred  = model(xb).view(B*T, -1).argmax(1)
        targ  = yb.view(-1)
        p,t = pred.cpu().numpy(), targ.cpu().numpy()
        for ti, pi in zip(t, p):
            tot[ti] += 1
            if pi == ti: ok[ti] += 1
            conf[ti, pi] += 1
    per_cls = ok / np.maximum(tot, 1)
    overall = ok.sum() / tot.sum()
    return overall, per_cls, conf

# ───────────────────────── helper to format rows ─────────────────────────
def _fmt_row(row) -> str:
    """Return comma-separated decimal literals WITHOUT leading ‘+’ signs."""
    return ", ".join(f"{v:.6f}" for v in row)

# ─────────── header writers (now output floats, not ints) ────────────────
def dump2d(path, name, arr: np.ndarray):
    guard = name.upper()
    with open(path, "w") as f:
        r, c = arr.shape
        f.write(f"#ifndef {guard}\n#define {guard}\n")
        f.write('#include "lstm_forward.h"\n')
        f.write(f"static const state_t {name}[{r}][{c}] = {{\n")
        for row in arr:
            f.write(f"  {{ {_fmt_row(row)} }},\n")
        f.write("};\n#endif\n")

def dump1d(path, name, arr: np.ndarray):
    guard = name.upper()
    with open(path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n")
        f.write('#include "lstm_forward.h"\n')
        f.write(f"static const state_t {name}[{arr.size}] = "
                f"{{ {_fmt_row(arr)} }};\n#endif\n")

# ───────────────────────────── MAIN ───────────────────────────────────────
def main():
    """Run QAT, evaluate floating-point and INT8 models, collect activation
    statistics, and export HLS-compatible parameter headers."""
    cfg = Cfg(); os.makedirs(cfg.OUT_DIR, exist_ok=True)
    t_all = time.time()

    # Data
    print("▶  Loading data …")
    t0=time.time()
    Xtr,Ytr = load_clean(cfg.CLEAN_DATA, cfg.TRAIN_KEY)
    Xte,Yte = load_clean(cfg.CLEAN_DATA, cfg.TEST_KEY)
    print(f"   train {len(Xtr)}, test {len(Xte)} segments  ({time.time()-t0:.1f}s)")
    tr_loader = make_train_loader(Xtr, Ytr)
    te_loader = make_test_loader(Xte, Yte)

    # Model build
    print("▶  Building QAT model …")
    t0=time.time()
    model = LSTMClassifierFX(cfg).to(DEVICE)
    load_matlab_weights(model, cfg.WEIGHTS_MAT)

    model_fp32 = copy.deepcopy(model).to(DEVICE)
    model_fp32.eval()

    model = quantize_fx.prepare_qat_fx(
        model, QCFG_MAP,
        example_inputs=(torch.randn(1,cfg.SEG_LEN,1).to(DEVICE),))

    # Was the input Fake-Quant inserted?
    first_inp_fq_scale = None
    for name, mod in model.named_modules():
        if name.endswith("input_stub.activation_post_process"):
            first_inp_fq_scale = mod.scale

    if first_inp_fq_scale is None:
        print("⚠️  Could not find a per-tensor input FakeQuant – continuing anyway.")
    else:
        print(f"   ECG scale factor (s_in) ≈ {first_inp_fq_scale.item():.8f}")

    # hooks
    for n,m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.register_forward_hook(act_hook(n))
        if isinstance(m, nn.LSTMCell):
            m.register_forward_hook(lambda m,i,o: gate_observer_hook(m))
            m.register_forward_hook(cell_state_hook)

    print(f"   model ready  ({time.time()-t0:.1f}s)")

    # Calibration
    print("▶  Quick calibration (20 batches) …")
    t0=time.time()
    model.train(); it=iter(tr_loader)
    with torch.no_grad():
        for _ in range(20):
            xb,_ = next(it); model(xb.unsqueeze(-1).to(DEVICE))
    print(f"   done ({time.time()-t0:.1f}s)")

    # QAT loop
    opt   = optim.Adam(model.parameters(), lr=cfg.LR)
    sched = optim.lr_scheduler.StepLR(opt, cfg.LR_DROP_EP, gamma=cfg.LR_GAMMA)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, cfg.QAT_EPOCHS+1):
        t0=time.time(); model.train(); running=0.0
        for xb,yb in tr_loader:
            xb = xb.unsqueeze(-1).to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            B,T,_ = xb.shape
            loss = loss_fn(model(xb).view(B*T,-1), yb.view(-1))
            loss.backward(); opt.step()
            running += loss.item()
        sched.step()
        print(f"EP{ep:02d}  loss={running / len(tr_loader):.4f} "
              f"({time.time() - t0:.1f}s)")


    fp32_sd = model_fp32.state_dict()
    qat_sd = model.state_dict()
    for k in fp32_sd.keys():
        fp32_sd[k] = qat_sd[k]
    model_fp32.load_state_dict(fp32_sd)

    # Evaluate the QAT-trained floating-point model
    print("\n▶  Evaluation (pure float-32 QAT-trained model) …")
    t0 = time.time()
    acc_f, cls_f, conf_f = evaluate(model_fp32, te_loader, DEVICE)
    print(f"   overall {acc_f * 100:5.2f}%  ({time.time() - t0:.1f}s)")
    for k, p in enumerate(cls_f):
        print(f"     class {k}: {p * 100:5.2f}%")
    print("   confusion matrix:\n", conf_f)

    # Fake-quant eval
    print("\n▶  Evaluation (fake-quant GPU) …")
    t0=time.time(); acc,cls,conf = evaluate(model, te_loader, DEVICE)
    print(f"   overall {acc*100:5.2f}%  ({time.time()-t0:.1f}s)")
    for k,p in enumerate(cls): print(f"     class {k}: {p*100:5.2f}%")
    print("   confusion matrix:\n", conf)

    # Convert → int8
    print("\n▶  Converting to INT-8 …")
    t0=time.time()
    m_int8 = quantize_fx.convert_fx(model.cpu())
    torch.save(m_int8.state_dict(), os.path.join(cfg.OUT_DIR, cfg.CKPT_INT8))
    print(f"   checkpoint saved  ({time.time()-t0:.1f}s)")

    # INT-8 eval
    print("▶  Evaluation (INT-8 CPU) …")
    t0=time.time(); acc,cls,conf = evaluate(m_int8, te_loader, "cpu")
    print(f"   overall {acc*100:5.2f}%  ({time.time()-t0:.1f}s)")
    for k,p in enumerate(cls): print(f"     class {k}: {p*100:5.2f}%")
    print("   confusion matrix:\n", conf)

    # Stats
    print("\n▶  Computing & writing stats …")
    t0=time.time()
    stats["FRAC_SHIFT"] = compute_frac_shift(stats, model)
    # mean / std / percentiles
    cnt = max(stats["cell_cnt"], 1)
    cell_mu = stats["cell_mu"]
    cell_sigma = math.sqrt(stats["cell_m2"] / max(cnt-1,1))
    p05, p995 = (-1.0, 1.0)
    if stats["cell_vals"]:
        p05, p995 = np.percentile(stats["cell_vals"], [0.5, 99.5])
    stats.update({"cell_mu": cell_mu,
                  "cell_sigma": cell_sigma,
                  "cell_p0.5": float(p05),
                  "cell_p99.5": float(p995)})
    with open(os.path.join(cfg.OUT_DIR, "qat_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(cfg.OUT_DIR, "act_ranges.json"), "w") as f:
        json.dump(stats["activations"], f, indent=2)
    print(f"   stats written  ({time.time() - t0:.1f}s)")
    print(f"     μ={cell_mu:.3f}, σ={cell_sigma:.3f}, 5σ≈{5 * cell_sigma:.3f}")
    print(f"     p0.5={p05:.3f}, p99.5={p995:.3f}")

    # ────────────────── Header export  (FLOAT LITERALS) ──────────────────
    print("▶  Writing float-literal headers …")
    t0 = time.time()

    w_all = np.concatenate(
        [m_int8.cell.weight_ih.detach().cpu().numpy(),
         m_int8.cell.weight_hh.detach().cpu().numpy()],
        axis=1,
    )
    b_all = m_int8.cell.bias_ih.detach().cpu().numpy()
    w_fc = m_int8.fc.weight.detach().cpu().numpy()
    b_fc = m_int8.fc.bias.detach().cpu().numpy()

    # Print and store the suggested fixed-point format for state_t
    choose_state_format(w_all, w_fc, cfg.F_BITS, max_total=18)

    dump2d(os.path.join(cfg.OUT_DIR, "W_all_fixed.h"), "W_all_fixed", w_all)
    dump1d(os.path.join(cfg.OUT_DIR, "B_all_fixed.h"), "B_all_fixed", b_all)
    dump2d(os.path.join(cfg.OUT_DIR, "W_fc_fixed.h"), "W_fc_fixed", w_fc)
    dump1d(os.path.join(cfg.OUT_DIR, "b_fc_fixed.h"), "b_fc_fixed", b_fc)

    print(f"   headers done  ({time.time() - t0:.1f}s)")
    print(f"\n✅  Finished in {(time.time() - t_all) / 60:.1f} min → {cfg.OUT_DIR}")

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
