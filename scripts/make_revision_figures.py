from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':9.5,'axes.labelsize':9.5,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
C=['#24557d','#be652d','#39755a']
files=[R/'modal/ladder_linear-760m/ablation_results_linear760m.json',R/'modal/probe_linear-1.3b.json',R/'modal/probe_mlp-760m.json']
names=['Linear-760M','Linear-1.3B','MLP-760M']
datasets=[json.loads(p.read_text()) for p in files]
fig,ax=plt.subplots(1,3,figsize=(6.8,2.65),layout='constrained')
for d,name,col in zip(datasets,names,C):
 rows=d['delta_term_per_layer']; ls=sorted(rows,key=int); y=[rows[k]['delta_fraction_mean'] for k in ls]; x=[rows[k]['gamma_cv2'] for k in ls]
 ax[0].plot(list(map(int,ls)),y,'o-',ms=2.5,lw=1,color=col,label=name)
 ax[1].scatter(x,y,s=12,color=col,alpha=.8)
ax[0].set(xlabel='Layer',ylabel=r'Mean $\|g_{\rm pred}\|_F/\|g\|_F$',ylim=(0,.82),title='(a) Gradient ratio')
ax[0].legend(frameon=False)
ax[1].set(xlabel=r'Head-averaged $\mathrm{CV}(\gamma^2)$',ylabel='Mean gradient norm ratio',ylim=(0,.82),title='(b) Gain variation')
for d,name,col in zip(datasets[:2],names[:2],C[:2]):
 e=d['eta_k2_per_layer']; ls=sorted(e,key=int); ax[2].plot(list(map(int,ls)),[e[k]['median'] for k in ls],'o-',ms=2.5,lw=1,color=col,label=name)
ax[2].set(xlabel='Layer',ylabel=r'Median $\eta\|k\|^2$',yscale='log',title='(c) Effective step')
ax[2].legend(frameon=False)
from matplotlib.ticker import NullFormatter
ax[2].yaxis.set_minor_formatter(NullFormatter())
fig.savefig(OUT/'probe_fig.pdf'); fig.savefig(OUT/'probe_fig.png', dpi=180); plt.close(fig)

lad=[datasets[0],json.loads((R/'modal/ladder_linear-1.3b/ablation_results_linear13b.json').read_text())]
fig,ax=plt.subplots(1,2,figsize=(6.8,2.65),layout='constrained',gridspec_kw={'width_ratios':[1.1,1]})
arms=['descent','hebbian','hebbian_nm','ascent','frozen']; labels=['Baseline','Prediction\nremoved','Norm\nmatched','Ascent','Frozen']; x=np.arange(5)
for d,name,col,off in zip(lad,names[:2],C[:2],[-.19,.19]):
 y=[d['per_arm'][a]['ppl'] for a in arms]; ax[0].bar(x+off,y,width=.36,color=col,label=name)
 for xx,yy in zip(x+off,y): ax[0].text(xx,yy*1.15,f'{yy:.1f}' if yy<100 else f'{yy:.0f}',ha='center',fontsize=6.4)
ax[0].set(yscale='log',ylim=(8,18000),ylabel='Perplexity',title='(a) Inference-time ablations')
ax[0].set_xticks(x,labels,fontsize=7); ax[0].legend(frameon=False,loc='upper left',fontsize=6.8,ncol=2)
for arm,col,label,style in [('hebbian',C[0],'Prediction removed','-'),('hebbian_nm',C[1],'Norm matched','--')]:
 b=lad[0]['deltas_vs_descent'][arm]['by_position']; ax[1].plot([r['pos_mid'] for r in b],[r['delta_nll'] for r in b],style,color=col,marker='o',ms=3,lw=1.2,label=label)
ax[1].axhline(0,color='#888888',lw=.6); ax[1].set(xscale='log',xlabel='Context position',ylabel=r'$\Delta$NLL from baseline (nats)',title='(b) Context effect on 760M'); ax[1].legend(frameon=False,loc='upper left')
fig.savefig(OUT/'ladder_fig.pdf'); fig.savefig(OUT/'ladder_fig.png', dpi=180); plt.close(fig)

A=json.loads((R/'overwrite_sweep_results.json').read_text())
fig,ax=plt.subplots(1,3,figsize=(6.8,2.65),layout='constrained')
for rule,col,label in [('additive',C[0],'Additive'),('mse',C[1],'Delta rule')]:
 rows=A['eta1_vs_noise'][rule]; ax[0].errorbar([r['noise'] for r in rows],[r['mean'] for r in rows],yerr=[1.96*r['se'] for r in rows],fmt='o-',color=col,ms=3,lw=1.1,capsize=2,label=label)
 rows=A['eta_sweep_noise005'][rule]; ax[1].errorbar([r['eta'] for r in rows],[r['mean'] for r in rows],yerr=[1.96*r['se'] for r in rows],fmt='o-',color=col,ms=3,lw=1.1,capsize=2)
ax[0].set(xlabel=r'Key noise $\sigma$',ylabel='Cosine margin for new value',title='(a) Unit step'); ax[0].legend(frameon=False)
ax[1].axvspan(.001387437,.041920524,color='#555555',alpha=.10,lw=0); ax[1].text(.008,.94,'760M median\nstep range',ha='center',va='top',fontsize=6.7,color='#555555'); ax[1].set(xscale='log',xlabel=r'Effective step $\eta\|k\|^2$',title='(b) Noise 0.05')
r=np.arange(1,2001)
for eta,col in zip([.04,.01,.001],C):
 q=(1-eta)**r; m=(1-q)/np.sqrt(1+q*q); ax[2].plot(r,m,color=col,lw=1.3,label=rf'$\eta={eta:g}$')
ax[2].axhline(0,color='#777777',lw=1,ls='--',label='Additive'); ax[2].set(xscale='log',xlabel=r'Writes per value $r$',title='(c) Identical-key calculation'); ax[2].legend(frameon=False,fontsize=6.7,loc='upper left')
for a in ax: a.set_ylim(-.05,1.07)
fig.savefig(OUT/'overwrite_fig.pdf'); fig.savefig(OUT/'overwrite_fig.png', dpi=180); plt.close(fig)
print('Wrote three vector PDF figures')
