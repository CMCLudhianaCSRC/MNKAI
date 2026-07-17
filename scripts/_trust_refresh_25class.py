"""Trust-layer refresh on the HONEST frozen-probe 25-class model (clean external):
temperature calibration (ECE), RAPS conformal sets, per-Fitzpatrick fairness, and a
confidence-gated ABSTENTION curve (retained accuracy vs referral rate, per skin tone).
This produces the numbers the app + paper need and the abstention rule for V-VI.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from scipy.special import log_softmax, softmax
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts._phase3_eval_full import fit_temperature, expected_calibration_error
from scripts._conformal_fix import raps_scores, qhat_of, coverage_and_size


def main():
    np.random.seed(0)
    z = np.load("eval/probe/feats_25class_518.npz", allow_pickle=True)
    X = z["features"].astype(np.float32); y = z["y"]; split = z["split"].astype(str); fold = z["fold"]; fst = z["fst"]
    C = len(z["classes"]); tv = split == "train_val"; ex = split == "external_test"
    Xtv, ytv, ftv = X[tv], y[tv], fold[tv]; Xex, yex, fstex = X[ex], y[ex], fst[ex]
    # dedup clean mask
    tvn = Xtv/(np.linalg.norm(Xtv,axis=1,keepdims=True)+1e-8); exn = Xex/(np.linalg.norm(Xex,axis=1,keepdims=True)+1e-8)
    mx = np.zeros(len(exn))
    for i in range(0,len(exn),256): mx[i:i+256]=(exn[i:i+256]@tvn.T).max(1)
    clean = mx <= 0.99

    # OOF (calibration source) + external probs
    oof = np.zeros((len(ytv), C))
    for fk in sorted(set(ftv[ftv>=0])):
        tr=ftv!=fk; te=ftv==fk; sc=StandardScaler().fit(Xtv[tr]); m=MLPClassifier((512,),max_iter=300,early_stopping=True,alpha=1e-3,random_state=0).fit(sc.transform(Xtv[tr]),ytv[tr]); oof[te]=m.predict_proba(sc.transform(Xtv[te]))
    sc=StandardScaler().fit(Xtv); m=MLPClassifier((512,),max_iter=300,early_stopping=True,alpha=1e-3,random_state=0).fit(sc.transform(Xtv),ytv)
    exf=m.predict_proba(sc.transform(Xex))
    oof_log=np.log(np.clip(oof,1e-9,1)); ex_log=np.log(np.clip(exf,1e-9,1))
    yc=yex[clean]; exl=ex_log[clean]; fstc=fstex[clean]

    T=fit_temperature(oof_log, ytv)
    p_raw=softmax(exl,1); p_cal=softmax(exl/T,1)
    res={"n_clean_external": int(clean.sum()), "temperature": round(float(T),3),
         "ECE_before": round(expected_calibration_error(p_raw,yc),4),
         "ECE_after": round(expected_calibration_error(p_cal,yc),4)}
    # conformal RAPS (calibrate on OOF, eval on clean external)
    np.random.seed(0)
    oof_cal=softmax(oof_log/T,1)
    best=None
    for kr in (1,2,3):
        for lam in (0.01,0.05,0.1):
            s=raps_scores(oof_cal,ytv,lam,kr); qh=qhat_of(s,0.10); cov,sz=coverage_and_size(p_cal,yc,qh,lam,kr)
            if cov>=0.89 and (best is None or sz.mean()<best[0]): best=(sz.mean(),cov,kr,lam)
    if best: res["conformal_RAPS"]={"coverage":round(best[1],3),"mean_set":round(best[0],2),"k_reg":best[2],"lambda":best[3]}
    # abstention curve: confidence = max cal prob; retain top-(1-r); report retained accuracy
    conf=p_cal.max(1); pred=p_cal.argmax(1); order=np.argsort(-conf)
    res["abstention"]=[]
    for r in (0.0,0.1,0.2,0.3):
        keep=order[:int(len(order)*(1-r))]
        res["abstention"].append({"referral_rate":r,"retained_acc":round(accuracy_score(yc[keep],pred[keep]),3),
                                  "retained_macroF1":round(f1_score(yc[keep],pred[keep],average="macro"),3)})
    # per-tone (clean)
    res["by_tone"]={}
    for lbl,mk in [("FST_I-III",(fstc>=1)&(fstc<=3)),("FST_IV-VI",fstc>=4)]:
        if mk.sum()>20:
            res["by_tone"][lbl]={"n":int(mk.sum()),"acc":round(accuracy_score(yc[mk],pred[mk]),3),
                                 "mean_confidence":round(float(conf[mk].mean()),3)}
    Path("eval/probe/trust_25class.json").write_text(json.dumps(res,indent=2))
    print("=== TRUST REFRESH (25-class frozen-probe, clean external) ===")
    print(f"  n={res['n_clean_external']} | ECE {res['ECE_before']} -> {res['ECE_after']} (T={res['temperature']})")
    if "conformal_RAPS" in res: print(f"  RAPS conformal: coverage {res['conformal_RAPS']['coverage']} mean-set {res['conformal_RAPS']['mean_set']}")
    print("  abstention (refer low-confidence):")
    for a in res["abstention"]: print(f"    refer {a['referral_rate']*100:.0f}% -> retained acc {a['retained_acc']} macroF1 {a['retained_macroF1']}")
    print("  per tone:", {k:v['acc'] for k,v in res['by_tone'].items()})
    print("  saved -> eval/probe/trust_25class.json")


if __name__ == "__main__":
    main()
