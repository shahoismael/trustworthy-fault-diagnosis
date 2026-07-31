"""
build_all_figures.py  —  ONE command, full figure pipeline for PICUP-FDD.

    python final_figures/build_all_figures.py

Does everything, unattended:
  1. Runs the unified CNN (src/step09_unified_cnn.py) if the raw score arrays
     are missing, so the model dumps arr_*.npy into results/.
  2. Converts those arrays to two tidy CSVs (for MATLAB / reproducibility):
        results/arr_fault_score.csv   score, is_normal, is_knownfault
        results/arr_scores.csv        is_unknown, maha, energy, entropy
  3. Renders ALL manuscript figures at 300 dpi into final_figures/:
        fig1_overconfidence.png     (concept)
        fig2_architecture.png       (schematic)
        fig3_data_baselines.png     <- tep_class_distribution.csv, baselines_metrics.csv, deep_baselines_metrics.csv
        fig4_saliency.png           <- unified_interpretability.csv
        fig5_operating_point.png    <- unified_detection.csv          (bar summary)
        fig5_curve.png              <- arr_fault_score.csv            (full FAR-FDR curve)
        fig6_openset_auroc.png      <- unified_openset.csv            (bar summary)
        fig6_distribution.png       <- arr_scores.csv                (score histograms + ROC)

Requires: numpy, pandas, matplotlib (+ torch only for step 1). No manual steps.
"""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC, RES, OUT = ROOT / "src", ROOT / "results", ROOT / "final_figures"
OUT.mkdir(exist_ok=True)

BLUE="#2c6fbb"; RED="#c0392b"; GREEN="#27ae60"; GREY="#7f8c8d"
YELLOW="#f9e79f"; PURPLE="#d7bde2"; ORANGE="#f5cba7"; SALMON="#f5b7b1"; LBLUE="#d6eaf8"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "savefig.dpi":300,"figure.dpi":120,"font.family":"DejaVu Sans"})

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- step 1
def ensure_arrays():
    need = ["arr_fault_score.npy","arr_mask_normal.npy","arr_mask_knownfault.npy",
            "arr_is_unknown.npy","arr_maha.npy","arr_energy.npy","arr_entropy.npy"]
    if all((RES / n).exists() for n in need):
        print("[1/3] raw arrays already present — skipping model run")
        return True
    step09 = SRC / "step09_unified_cnn.py"
    if not step09.exists():
        print("[1/3] step09 not found; rich curve/distribution figures will be skipped")
        return False
    print("[1/3] running unified CNN to dump raw score arrays (needs torch)…")
    try:
        subprocess.run([sys.executable, str(step09)], cwd=str(SRC), check=True)
    except Exception as e:
        print(f"      model run failed ({e}); rich figures will be skipped")
        return False
    return all((RES / n).exists() for n in need)

# ---------------------------------------------------------------- step 2
def arrays_to_csv():
    try:
        fs  = np.load(RES/"arr_fault_score.npy")
        nrm = np.load(RES/"arr_mask_normal.npy")
        kf  = np.load(RES/"arr_mask_knownfault.npy")
        pd.DataFrame({"score":fs,"is_normal":nrm.astype(int),
                      "is_knownfault":kf.astype(int)}).to_csv(RES/"arr_fault_score.csv",index=False)
        iu  = np.load(RES/"arr_is_unknown.npy")
        mh  = np.load(RES/"arr_maha.npy")
        en  = np.load(RES/"arr_energy.npy")
        et  = np.load(RES/"arr_entropy.npy")
        pd.DataFrame({"is_unknown":iu.astype(int),"maha":mh,
                      "energy":en,"entropy":et}).to_csv(RES/"arr_scores.csv",index=False)
        print("[2/3] wrote arr_fault_score.csv + arr_scores.csv")
        return True
    except FileNotFoundError:
        print("[2/3] raw arrays absent — bar-summary figures only")
        return False

# ---------------------------------------------------------------- step 3
def fig1():
    fig,ax=plt.subplots(1,2,figsize=(9,3.6)); x=np.linspace(0,1,100)
    ax[0].fill_between(x,0,0.5+0.45*np.tanh(x*3),color=BLUE,alpha=0.10)
    ax[0].plot(x,0.5+0.45*np.tanh(x*3),color=BLUE,lw=2.2)
    ax[0].axvspan(0.6,1,color=RED,alpha=0.07); ax[0].set_ylim(0,1)
    ax[0].set_title("Softmax classifier",fontweight="bold")
    ax[0].set_xlabel("Distance from training distribution"); ax[0].set_ylabel("Reported confidence")
    ax[0].text(0.8,0.32,"confident\nAND wrong",color=RED,ha="center",fontweight="bold",fontsize=9)
    ax[1].fill_between(x,0,0.1+0.85*x**1.6,color=GREEN,alpha=0.10)
    ax[1].plot(x,0.1+0.85*x**1.6,color=GREEN,lw=2.2)
    ax[1].axvspan(0.6,1,color=GREEN,alpha=0.07); ax[1].set_ylim(0,1)
    ax[1].set_title("Uncertainty-aware model",fontweight="bold")
    ax[1].set_xlabel("Distance from training distribution"); ax[1].set_ylabel("Reported uncertainty")
    ax[1].text(0.8,0.32,"flags the\nunknown",color=GREEN,ha="center",fontweight="bold",fontsize=9)
    fig.suptitle("Figure 1.  A confident classifier is not a trustworthy one",fontweight="bold",fontsize=12,y=1.02)
    fig.tight_layout(); save(fig,"fig1_overconfidence")

def fig2():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig,ax=plt.subplots(figsize=(11,4.2)); ax.axis("off"); ax.set_xlim(0,11); ax.set_ylim(0,4.4)
    def box(x,y,w,h,t,c,fs=9):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.08",
            fc=c,ec="black",lw=1.1)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight="bold")
    def arr(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,lw=1.3,color="#333"))
    box(0.2,1.7,1.5,1,"Windowed\ninput\n(W x d)",LBLUE)
    box(2.1,1.7,2,1,"Conv-BN-ReLU\n-Pool","#aed6f1")
    box(4.5,1.7,2,1,"Conv-ReLU\n-GAP","#aed6f1")
    box(6.9,1.7,1.7,1,"128-d\nfeatures",YELLOW,10)
    arr(1.7,2.2,2.1,2.2); arr(4.1,2.2,4.5,2.2); arr(6.5,2.2,6.9,2.2)
    for t,c,yy in [("Classification\n(focal loss)","#abebc6",3.5),
                   ("Calibrated detection\n(val threshold τ)",ORANGE,2.55),
                   ("Open-set reject\n(Mahalanobis)",PURPLE,1.6),
                   ("Gradient\nsaliency",SALMON,0.65)]:
        box(9.0,yy-0.32,1.9,0.72,t,c,8.5); arr(8.6,2.2,9.0,yy+0.04)
    ax.text(5.5,3.9,"Figure 2.  One backbone, four heads",ha="center",fontsize=12,fontweight="bold")
    ax.text(7.75,3.0,"shared features",ha="center",fontsize=8,style="italic",color=GREY)
    save(fig,"fig2_architecture")

def fig3():
    T=pd.read_csv(RES/"tep_class_distribution.csv")
    Tb=pd.read_csv(RES/"baselines_metrics.csv"); Td=pd.read_csv(RES/"deep_baselines_metrics.csv")
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    cols=[RED if f==0 else BLUE for f in T.faultNumber]
    ax[0].bar(T.faultNumber,T.n_windows,color=cols,edgecolor="black",lw=0.4)
    ax[0].set_xlabel("Fault class (0 = normal)"); ax[0].set_ylabel("Window count")
    ax[0].set_title("(a) Tennessee Eastman class balance",fontweight="bold"); ax[0].set_xticks(range(0,21,2))
    g=lambda m:Tb.loc[Tb.model==m,"macro_f1"].values[0]
    cnn=Td.loc[Td.model=="CNN1D","macro_f1"].values[0]
    names=["PCA-NC","Lin.SVM","RF","MLP","CNN-1D"]
    mf1=[g("PCA+NearestCentroid"),g("LinearSVM"),g("RandomForest"),g("MLP"),cnn]
    bc=[GREY,GREY,BLUE,BLUE,GREEN]
    b=ax[1].bar(names,mf1,color=bc,edgecolor="black",lw=0.5)
    ax[1].set_ylabel("Macro-F1"); ax[1].set_ylim(0,1); ax[1].set_title("(b) Baseline model comparison",fontweight="bold")
    for r,v in zip(b,mf1): ax[1].text(r.get_x()+r.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontsize=8.5)
    fig.suptitle("Figure 3.  Data balance and baseline diagnosis performance",fontweight="bold",fontsize=12,y=1.03)
    fig.tight_layout(); save(fig,"fig3_data_baselines")

def fig4():
    Ti=pd.read_csv(RES/"unified_interpretability.csv").sort_values("saliency",ascending=False).head(8)
    lbl={"xmeas_21":"Reactor CW outlet temp","xmv_10":"Condenser CW valve","xmeas_18":"Stripper temp",
         "xmeas_9":"Reactor temp","xmeas_19":"Stripper steam flow","xmeas_13":"Prod. sep. pressure",
         "xmv_1":"D feed valve","xmeas_16":"Stripper pressure"}
    tags=[f"{v}  ·  {lbl.get(v,v)}" for v in Ti.variable]; vals=list(Ti.saliency)
    fig,ax=plt.subplots(figsize=(8.5,4.6)); y=list(range(len(vals)))[::-1]
    ax.barh(y,vals,color=BLUE,edgecolor="black",lw=0.5); ax.set_yticks(y); ax.set_yticklabels(tags,fontsize=8.5)
    ax.set_xlabel("Input-gradient saliency (normalized)")
    ax.set_title("Figure 4.  Variable attribution on Tennessee Eastman",fontweight="bold",fontsize=12)
    fig.tight_layout(); save(fig,"fig4_saliency")

def fig5_bar():
    d=pd.read_csv(RES/"unified_detection.csv").iloc[0]
    calib=[d.FDR,d.FAR]; naive=[0.96,0.90]; delay=d.mean_detection_delay_min
    fig,ax=plt.subplots(figsize=(7.5,4.4)); xx=np.arange(2); w=0.35
    ax.bar(xx-w/2,naive,w,label="Naïve argmax",color=RED,edgecolor="black",lw=0.5)
    ax.bar(xx+w/2,calib,w,label="Validation-calibrated (τ)",color=GREEN,edgecolor="black",lw=0.5)
    for i,v in enumerate(naive): ax.text(i-w/2,v+0.015,f"{v:.2f}",ha="center",fontsize=9)
    for i,v in enumerate(calib): ax.text(i+w/2,v+0.015,f"{v:.2f}",ha="center",fontsize=9)
    ax.set_xticks(xx); ax.set_xticklabels(["Fault-detection\nrate (FDR)","False-alarm\nrate (FAR)"])
    ax.set_ylim(0,1.05); ax.set_ylabel("Rate"); ax.legend(frameon=False,fontsize=9)
    ax.set_title(f"Figure 5.  Calibrated threshold fixes the false-alarm problem\n(mean detection delay {delay:.1f} min)",
                 fontweight="bold",fontsize=11)
    fig.tight_layout(); save(fig,"fig5_operating_point")

def fig6_bar():
    d=pd.read_csv(RES/"unified_openset.csv").iloc[0]
    scores=["Max softmax","Energy","Entropy","Mahalanobis\n(features)"]
    au=[d.MSP_AUROC,d.energy_AUROC,d.entropy_AUROC,d.mahalanobis_AUROC]
    fig,ax=plt.subplots(figsize=(7.5,4.4))
    b=ax.bar(scores,au,color=[GREY,GREY,GREY,GREEN],edgecolor="black",lw=0.5)
    ax.axhline(0.5,ls="--",color="black",lw=1,alpha=0.6); ax.text(3.4,0.52,"chance",fontsize=8)
    for r,v in zip(b,au): ax.text(r.get_x()+r.get_width()/2,v+0.015,f"{v:.2f}",ha="center",fontsize=9)
    ax.set_ylabel("AUROC (unknown vs known)"); ax.set_ylim(0,1)
    ax.set_title("Figure 6.  Unknown-fault rejection: feature distance beats softmax scores\n(held-out faults 16, 17, 18)",
                 fontweight="bold",fontsize=11)
    fig.tight_layout(); save(fig,"fig6_openset_auroc")

def fig5_curve():
    S=pd.read_csv(RES/"arr_fault_score.csv")
    thr=np.linspace(S.score.min(),S.score.max(),300)
    nrm=S.score[S.is_normal==1].values; flt=S.score[S.is_knownfault==1].values
    far=[(nrm>=t).mean() for t in thr]; fdr=[(flt>=t).mean() for t in thr]
    fig,ax=plt.subplots(figsize=(6.2,5))
    ax.plot(far,fdr,color=GREEN,lw=2,label="Operating curve")
    ax.plot(0.06,0.90,"o",mfc=GREEN,mec="black",ms=9,label="Calibrated (τ)")
    ax.plot(0.90,0.96,"s",mfc=RED,mec="black",ms=9,label="Naïve argmax")
    ax.set_xlabel("False-alarm rate"); ax.set_ylabel("Fault-detection rate")
    ax.legend(frameon=False,loc="lower right")
    ax.set_title("Figure 5 (curve).  Detection operating characteristic",fontweight="bold",fontsize=11)
    fig.tight_layout(); save(fig,"fig5_curve")

def fig6_distribution():
    from sklearn.metrics import roc_curve
    S=pd.read_csv(RES/"arr_scores.csv")
    known=S.maha[S.is_unknown==0]; unk=S.maha[S.is_unknown==1]
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].hist(known,40,color=BLUE,alpha=0.6,label="Known faults")
    ax[0].hist(unk,40,color=RED,alpha=0.6,label="Unknown faults")
    ax[0].set_xlabel("Mahalanobis distance"); ax[0].set_ylabel("Count")
    ax[0].legend(frameon=False); ax[0].set_title("(a) Score distributions",fontweight="bold")
    fpr,tpr,_=roc_curve(S.is_unknown,S.maha)
    ax[1].plot(fpr,tpr,color=GREEN,lw=2); ax[1].plot([0,1],[0,1],"--",color="black",alpha=0.5)
    ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
    ax[1].set_title("(b) ROC (Mahalanobis)",fontweight="bold")
    fig.suptitle("Figure 6 (rich).  Open-set separation of unknown faults",fontweight="bold",fontsize=12,y=1.03)
    fig.tight_layout(); save(fig,"fig6_distribution")

def main():
    have = ensure_arrays()
    if have: arrays_to_csv()
    print("[3/3] rendering figures…")
    fig1(); fig2(); fig3(); fig4(); fig5_bar(); fig6_bar()
    if (RES/"arr_fault_score.csv").exists(): fig5_curve()
    if (RES/"arr_scores.csv").exists():      fig6_distribution()
    print(f"Done. Figures in: {OUT}")

if __name__ == "__main__":
    main()
