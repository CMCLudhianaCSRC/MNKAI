import argparse, os, json, math, random, warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import timm
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support, roc_curve, precision_recall_curve,
    balanced_accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

# ---------- Small helpers ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def softmax_np(x): return np.exp(x) / np.clip(np.exp(x).sum(axis=1, keepdims=True), 1e-9, None)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def read_image_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")

def detect_manifest_columns(df):
    cols = {c.lower(): c for c in df.columns}
    path_keys = ("filepath", "path", "image", "image_path", "file_path", "filename")
    label_keys = ("label", "label_name", "target", "diagnosis", "class", "category")
    path_col = next((cols[k] for k in path_keys if k in cols), None)
    label_col = next((cols[k] for k in label_keys if k in cols), None)
    return path_col, label_col

# ---------- Dataset ----------
class SimpleImageCSVDataset(Dataset):
    def __init__(self, df, classes, img_size=352):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.img_size = img_size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = read_image_rgb(row["filepath"])
        img = ImageOps.exif_transpose(img)
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        x = np.array(img).astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2,0,1))  # CHW
        y = self.class_to_idx[str(row["label"]).strip().lower()]
        return {"image": torch.from_numpy(x), "target": y, "path": row["filepath"]}

# ---------- Corruptions (light/fast) ----------
def apply_corruption(img_np, kind, level):
    img = Image.fromarray(np.clip(img_np,0,255).astype(np.uint8))
    if kind=="tint":
        # +/- color cast
        tint = (np.array([20, -5, -5])*level).astype(int)
        arr = img_np + tint
        return np.clip(arr, 0, 255)
    if kind=="blur":
        return np.array(img.filter(ImageFilter.GaussianBlur(radius=0.5*level)))
    if kind=="jpeg":
        # recompress
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=int(max(10, 100-15*level)))
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))
    return img_np

# ---------- Model loading ----------
def build_model(model_name, num_classes, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model.eval()
    return model

def load_ckpt_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    # If keys are 'module.' prefixed, strip them
    new_state = {}
    for k,v in state.items():
        nk = k.replace("module.","")
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)} from {ckpt_path}")
    model.to(device)
    return ckpt

# ---------- Temperature scaling ----------
class TempScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = torch.nn.Parameter(torch.zeros(1))  # T=1

    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T

def fit_temperature(logits, labels, max_iter=200, lr=0.01):
    """Platt/tempscale on logits (torch tensors on CPU) to minimize NLL."""
    logits = logits.detach()
    y = labels.long().detach()
    scaler = TempScaler()
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    nll = torch.nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = nll(scaler(logits), y)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        T = torch.exp(scaler.log_temp).item()
    return scaler, T

def ece_score(probs, labels, n_bins=15):
    """Expected Calibration Error (ECE)."""
    probs = np.clip(probs, 1e-12, 1-1e-12)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    labels = labels.astype(int)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece, brier = 0.0, np.mean(np.sum((probs - (np.eye(probs.shape[1])[labels]))**2, axis=1))
    accs, confs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            accs.append(0); confs.append((lo+hi)/2); counts.append(0)
            continue
        acc = (preds[m] == labels[m]).mean()
        avg_conf = conf[m].mean()
        ece += (m.sum()/len(conf)) * abs(acc - avg_conf)
        accs.append(acc); confs.append(avg_conf); counts.append(int(m.sum()))
    return float(ece), float(brier), (np.array(confs), np.array(accs), np.array(counts))

# ---------- Evaluation core ----------
@torch.no_grad()
def infer_logits(model, loader, device, tta=False):
    logits_list, y_list, paths = [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = torch.as_tensor(batch["target"]).to(device)
        out = model(x)
        if tta:
            out_h = model(torch.flip(x, dims=[3]))
            out_v = model(torch.flip(x, dims=[2]))
            out_hv = model(torch.flip(x, dims=[2,3]))
            out = (out + out_h + out_v + out_hv) / 4.0
        logits_list.append(out.cpu())
        y_list.append(y.cpu())
        paths += batch["path"]
    return torch.cat(logits_list, 0), torch.cat(y_list, 0), paths

def op_point_sens_at_spec(y_true, y_scores, target_spec=0.95):
    """Binary one-vs-rest: find threshold that gives >= target specificity."""
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    spec = 1.0 - fpr
    # pick the highest sensitivity where spec >= target
    mask = spec >= target_spec
    if not mask.any():
        idx = np.argmax(spec)  # closest we can get
    else:
        idx = np.argmax(tpr[mask]) + np.where(mask)[0][0]
    return float(thr[idx]), float(tpr[idx]), float(spec[idx])

def table_thresholds(y_true, y_scores, thresholds):
    rows=[]
    for th in thresholds:
        pred = (y_scores >= th).astype(int)
        TP = int(((pred==1)&(y_true==1)).sum())
        TN = int(((pred==0)&(y_true==0)).sum())
        FP = int(((pred==1)&(y_true==0)).sum())
        FN = int(((pred==0)&(y_true==1)).sum())
        sens = TP/max(1,TP+FN); spec = TN/max(1,TN+FP)
        ppv  = TP/max(1,TP+FP);  npv  = TN/max(1,TN+FN)
        rows.append({"threshold":th,"sensitivity":sens,"specificity":spec,"PPV":ppv,"NPV":npv,
                     "TP":TP,"TN":TN,"FP":FP,"FN":FN})
    return pd.DataFrame(rows)

def coverage_risk_curve(y_true, conf, preds):
    """Abstain lowest-confidence x% and compute error on rest."""
    order = np.argsort(conf)  # low->high
    cov, err = [], []
    for drop_frac in np.linspace(0, 0.6, 13):  # 0..60% abstention
        k = int(len(conf)*drop_frac)
        keep = np.ones(len(conf), bool)
        keep[order[:k]] = False
        if keep.sum()==0:
            cov.append(0.0); err.append(np.nan); continue
        acc = (preds[keep]==y_true[keep]).mean()
        cov.append(keep.mean()); err.append(1-acc)
    return np.array(cov), np.array(err)

# ---------- Plots ----------
def plot_confmat(cm_norm, classes, out):
    fig, ax = plt.subplots(figsize=(8,7))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, ylabel='True', xlabel='Pred')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i,j]*100:.1f}%", ha="center", va="center",
                    color="white" if cm_norm[i,j]>0.5 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def plot_roc_pr(classes, ys, ps, out_roc, out_pr):
    fig1, ax1 = plt.subplots(figsize=(7,6))
    fig2, ax2 = plt.subplots(figsize=(7,6))
    macro_aucs, macro_aupr = [], []
    for c in range(len(classes)):
        y = (ys==c).astype(int)
        s = ps[:,c]
        if len(np.unique(y))<2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        auc = roc_auc_score(y, s)
        aupr = average_precision_score(y, s)
        macro_aucs.append(auc); macro_aupr.append(aupr)
        ax1.plot(fpr, tpr, label=f"{classes[c]} (AUC {auc:.3f})")
        ax2.plot(rec, prec, label=f"{classes[c]} (AP {aupr:.3f})")
    ax1.plot([0,1],[0,1],'k--',alpha=0.4); ax1.set(xlabel="FPR", ylabel="TPR", title="ROC (OvR)"); ax1.legend(fontsize=8)
    ax2.set(xlabel="Recall", ylabel="Precision", title="PR (OvR)"); ax2.legend(fontsize=8)
    fig1.tight_layout(); fig1.savefig(out_roc, dpi=200); plt.close(fig1)
    fig2.tight_layout(); fig2.savefig(out_pr, dpi=200); plt.close(fig2)
    return (np.mean(macro_aucs) if macro_aucs else float("nan"),
            (np.mean(macro_aupr) if macro_aupr else float("nan")))

def plot_reliability(confs, accs, out_png, title):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot([0,1],[0,1],'k--',alpha=0.5)
    ax.plot(confs, accs, marker='o')
    ax.set(xlabel="Confidence", ylabel="Accuracy", title=title)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_coverage_curve(cov, err, out):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(cov*100, 100*err, marker='o')
    ax.set(xlabel="Coverage (% kept)", ylabel="Error on kept (%)", title="Coverage vs Error (abstain lowest-confidence)")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

def plot_corruption(results_dict, out):
    # results_dict[kind] = list of (level, acc_clean, acc_tta)
    fig, ax = plt.subplots(figsize=(7,5))
    for kind, rows in results_dict.items():
        lv = [r[0] for r in rows]
        a0 = [r[1] for r in rows]
        a1 = [r[2] for r in rows]
        ax.plot(lv, a0, marker='o', label=f"{kind} (no TTA)")
        ax.plot(lv, a1, marker='s', label=f"{kind} (+TTA)")
    ax.set(xlabel="Severity", ylabel="Accuracy", title="Corruption sweep")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True, help="CSV with columns: filepath,label[,source,device,lighting,fitzpatrick,site,age,sex]")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--classes", nargs="+", required=True, help="class names in the model's order")
    ap.add_argument("--primary-class", default="mel", help="for operating point (sens@95%spec)")
    ap.add_argument("--model-name", default="tf_efficientnet_b3_ns")
    ap.add_argument("--img-size", type=int, default=352)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = ensure_dir(Path(args.outdir))

    # 1) Load data
    df = pd.read_csv(args.manifest)
    # normalize columns
    path_col, label_col = detect_manifest_columns(df)
    if path_col is None:
        raise ValueError("Manifest must have a column with image paths (e.g. filepath, image_path, filename).")
    if label_col is None:
        raise ValueError("Manifest must have a column with class labels (e.g. label, label_name, diagnosis).")
    df = df.rename(columns={path_col: "filepath", label_col: "label"})
    # keep only rows where file exists and label in classes (normalized)
    classes = [str(c).strip().lower() for c in args.classes]
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(classes)].copy()
    df = df[df["filepath"].map(lambda p: Path(p).is_file())].reset_index(drop=True)
    if len(df)==0: raise ValueError("No valid rows in manifest after filtering by files/classes.")

    # 2) Dataloader
    ds = SimpleImageCSVDataset(df, classes, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # 3) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name, num_classes=len(classes), pretrained=False)
    ckpt = load_ckpt_into_model(model, args.ckpt, device)

    # 4) Infer logits (base)
    logits, y_true_t, paths = infer_logits(model, dl, device, tta=args.tta)
    logits_np = logits.detach().cpu().numpy()
    y_true = y_true_t.numpy()
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    preds = probs.argmax(1)

    # 5) Overall metrics
    macro_f1 = f1_score(y_true, preds, average="macro")
    bal_acc  = balanced_accuracy_score(y_true, preds)

    # One-vs-rest macro ROC-AUC & PR-AUC
    m_auc, m_ap = plot_roc_pr(
        classes, y_true, probs,
        out_roc=outdir/"roc_curves_one_vs_rest.png",
        out_pr=outdir/"pr_curves_one_vs_rest.png"
    )

    # 6) Per-class metrics + confusion matrix
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, preds, labels=list(range(len(classes))), zero_division=0)
    per_class = pd.DataFrame({
        "class": classes, "precision":prec, "recall":rec, "f1":f1, "support":sup
    })
    per_class.to_csv(outdir/"per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, preds, labels=list(range(len(classes))))
    cm_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    plot_confmat(cm_norm, classes, outdir/"confusion_matrix_normalized.png")

    # High-risk (FN/NPV) for primary class
    primary = classes.index(args.primary_class) if args.primary_class in classes else 0
    y_bin = (y_true==primary).astype(int)
    s_bin = probs[:,primary]
    # sensitivity@95% specificity
    thr_95, sens_95, spec_95 = op_point_sens_at_spec(y_bin, s_bin, target_spec=0.95)
    # threshold table (0.1..0.9)
    ths = np.linspace(0.1, 0.9, 9)
    thr_table = table_thresholds(y_bin, s_bin, ths)
    thr_table.to_csv(outdir/"threshold_table_primary.csv", index=False)

    # FN count & NPV at default 0.5 (report)
    default_tb = table_thresholds(y_bin, s_bin, [0.5]).iloc[0].to_dict()

    # 7) Calibration: ECE/Brier + reliability (before)
    ece0, brier0, (c0, a0, _) = ece_score(probs, y_true, n_bins=15)
    plot_reliability(c0, a0, outdir/"reliability_before.png", "Reliability (before temp scale)")

    # Fit temperature on half split (calib/eval) to avoid bias
    idx = np.arange(len(y_true))
    rng = np.random.default_rng(2025)
    rng.shuffle(idx)
    mid = len(idx)//2
    cal_idx, ev_idx = idx[:mid], idx[mid:]
    logits_tensor = torch.from_numpy(logits_np)
    scaler, T = fit_temperature(logits_tensor[cal_idx], torch.from_numpy(y_true[cal_idx]))
    logits_ts = scaler(logits_tensor).detach().numpy()
    probs_ts = softmax_np(logits_ts)
    ece1, brier1, (c1, a1, _) = ece_score(probs_ts[ev_idx], y_true[ev_idx], n_bins=15)
    plot_reliability(c1, a1, outdir/"reliability_after_tempscale.png", f"Reliability (after temp scale, T={T:.2f})")
    with open(outdir/"calibration.json","w") as f:
        json.dump({"ece_before":ece0, "brier_before":brier0, "ece_after_evalhalf":ece1, "brier_after_evalhalf":brier1, "temperature":T}, f, indent=2)

    # 8) Coverage vs risk (abstention)
    conf = probs.max(1)
    cov, err = coverage_risk_curve(y_true, conf, preds)
    plot_coverage_curve(cov, err, outdir/"coverage_risk_curve.png")

    # 9) Robustness / subgroup breakdowns (if columns exist)
    subgroup_cols = [c for c in ["source","device","lighting","fitzpatrick","site","sex"] if c in df.columns]
    rob_rows = []
    if subgroup_cols:
        for col in subgroup_cols:
            for val, sub in df.groupby(col):
                m = sub.index.values
                acc = (preds[m]==y_true[m]).mean()
                rob_rows.append({"group":col, "value":val, "n":len(m), "acc":float(acc)})
        pd.DataFrame(rob_rows).to_csv(outdir/"robustness_breakdown.csv", index=False)

    # 10) Corruption tests (+/- TTA benefit)
    # sample up to 150 images for quick sweep
    samp = np.random.choice(len(df), size=min(150, len(df)), replace=False)
    x_small = []
    for i in samp:
        im = read_image_rgb(df.iloc[i]["filepath"]).resize((args.img_size,args.img_size), Image.BICUBIC)
        x_small.append(np.array(im))
    x_small = np.stack(x_small,0)

    def batch_logits(arr, tta=False):
        arrf = arr.astype(np.float32)/255.0
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        arrf = (arrf-mean)/std
        arrf = np.transpose(arrf, (0,3,1,2))
        with torch.no_grad():
            x = torch.from_numpy(arrf).to(device)
            out = model(x)
            if tta:
                out = (out + model(torch.flip(x,[3])) + model(torch.flip(x,[2])) + model(torch.flip(x,[2,3]))) / 4.0
        return out.softmax(-1).cpu().numpy()

    corr_results = {}
    for kind in ["tint","blur","jpeg"]:
        rows=[]
        for level in [1,2,3,4]:
            arr_c = np.stack([apply_corruption(im, kind, level) for im in x_small],0)
            p0 = batch_logits(arr_c, tta=False)
            p1 = batch_logits(arr_c, tta=True)
            # use available labels for this subset
            y_s = y_true[samp]
            a0 = (p0.argmax(1)==y_s).mean()
            a1 = (p1.argmax(1)==y_s).mean()
            rows.append((level, float(a0), float(a1)))
        corr_results[kind] = rows
    plot_corruption(corr_results, outdir/"corruption_sweep.png")

    # 11) Macro summaries
    overall = {
        "macro_f1": float(macro_f1),
        "macro_roc_auc_ovr": float(m_auc) if not math.isnan(m_auc) else None,
        "macro_pr_auc_ovr": float(m_ap) if not math.isnan(m_ap) else None,
        "balanced_accuracy": float(bal_acc),
        "primary_class": args.primary_class,
        "operating_point_sens_at_95spec": {"threshold":thr_95, "sensitivity":sens_95, "specificity":spec_95},
        "default_threshold_primary": default_tb,
        "tta_used": bool(args.tta),
        "n_eval": int(len(df))
    }
    with open(outdir/"metrics_overall.json","w") as f:
        json.dump(overall, f, indent=2)

    print("\n=== DONE ===")
    print(f"Reports in: {outdir.as_posix()}")
    print(json.dumps(overall, indent=2))
    print("Per-class metrics -> per_class_metrics.csv")
    print("Calibration -> reliability_before.png, reliability_after_tempscale.png, calibration.json")
    print("Thresholds -> threshold_table_primary.csv")
    print("Curves -> roc_curves_one_vs_rest.png, pr_curves_one_vs_rest.png, coverage_risk_curve.png")
    print("Robustness -> robustness_breakdown.csv (if subgroup cols existed)")
    print("Corruptions -> corruption_sweep.png")

if __name__ == "__main__":
    main()
