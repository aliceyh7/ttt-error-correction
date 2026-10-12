from pathlib import Path
import json, sys, hashlib
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/modal'
OUT=ROOT/'results/revision'
OUT.mkdir(exist_ok=True)
paths={'Linear-760M':R/'ladder_linear-760m/ablation_results_linear760m.json','Linear-1.3B':R/'probe_linear-1.3b.json','MLP-760M':R/'probe_mlp-760m.json'}
def corr(x,y):
 a=rankdata(x,axis=-1); b=rankdata(y,axis=-1); a-=a.mean(axis=-1,keepdims=True); b-=b.mean(axis=-1,keepdims=True)
 return (a*b).sum(axis=-1)/np.sqrt((a*a).sum(axis=-1)*(b*b).sum(axis=-1))
def boot(x,y):
 idx=np.random.default_rng(0).integers(0,len(x),(10000,len(x))); vals=corr(x[idx],y[idx]); return np.quantile(vals,[.025,.975]).tolist()
summary={}
for name,p in paths.items():
 d=json.loads(p.read_text()); rows=d['delta_term_per_layer']; rows=[rows[k] for k in sorted(rows,key=int)]
 x=np.array([v['gamma_cv2'] for v in rows]); y=np.array([v['delta_fraction_mean'] for v in rows]); mask=np.arange(len(x))!=x.argmax()
 e=d['eta_k2_per_layer']; med=[v['median'] for v in e.values()]
 summary[name]={'source':str(p.relative_to(ROOT)),'mode':d['hebbian_mode'],'n_chunks_per_layer':rows[0]['n_minibatches'],'probe_mean':float(y.mean()),'probe_range':[float(y.min()),float(y.max())],'cv_median':float(np.median(x)),'cv_mean':float(x.mean()),'rho':float(corr(x,y)),'ci':boot(x,y),'rho_without_max':float(corr(x[mask],y[mask])),'ci_without_max':boot(x[mask],y[mask]),'eta_median_range':[min(med),max(med)]}
for key,file in [('Linear-760M',R/'ladder_linear-760m/ablation_results_linear760m.json'),('Linear-1.3B',R/'ladder_linear-1.3b/ablation_results_linear13b.json')]:
 d=json.loads(file.read_text()); base=d['per_arm']['descent']['ppl']; s=summary[key]
 s['ppl']={k:v['ppl'] for k,v in d['per_arm'].items()}; s['n_tokens']=int(d['per_arm']['descent']['n_scored_tokens']); s['ppl_ratios']={k:v['ppl']/base for k,v in d['per_arm'].items()}
 s['delta_nll']={k:v['overall']['delta_nll'] for k,v in d['deltas_vs_descent'].items()}; s['normmatch_mean']=d['normmatch_ratio']['mean']; s['normmatch_median']=d['normmatch_ratio']['median']
 for k,v in d['per_arm'].items(): assert abs(np.log(v['ppl'])-v['mean_nll'])<1e-12
(OUT/'verified_metrics.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
# Compare the closed-form two-write simulation with the stored sweep, using the same RNG order.
A=json.loads((ROOT/'results/overwrite_sweep_results.json').read_text())
def samples(n,noise,seed):
 rng=np.random.default_rng(seed); q=[]
 def unit(x): return x/np.linalg.norm(x)
 for _ in range(n):
  k=unit(rng.standard_normal(128)); k1=unit(k+noise*rng.standard_normal(128)); k2=unit(k+noise*rng.standard_normal(128)); v1=unit(rng.standard_normal(128)); v2=unit(rng.standard_normal(128)); q.append([k,k1,k2,v1,v2])
 return np.array(q)
def margins(q,eta,rule,r=1):
 k,k1,k2,v1,v2=q.transpose(1,0,2); dot=lambda x,y:(x*y).sum(-1)
 if rule=='mse':
  a=-np.expm1(r*np.log1p(-eta)) if eta<1 else 1.
  y=a*dot(k1,k)[:,None]*v1+a*dot(k2,k)[:,None]*(v2-a*dot(k1,k2)[:,None]*v1)
 else: y=eta*r*(dot(k1,k)[:,None]*v1+dot(k2,k)[:,None]*v2)
 return dot(y,v2-v1)/np.linalg.norm(y,axis=-1)
worst=0.
for group,param,seed in [('eta1_vs_noise','noise',1),('eta_sweep_noise005','eta',2)]:
 for rule,rows in A[group].items():
  for v in rows:
   noise=v['noise'] if param=='noise' else .05; eta=v['eta'] if param=='eta' else 1.
   m=margins(samples(1000,noise,seed),eta,rule); worst=max(worst,abs(m.mean()-v['mean']))
assert worst<1e-12
print('Two-write sweep reproduced from independent closed form. Maximum mean error',worst)
(OUT/'overwrite_verification.json').write_text(json.dumps({'max_mean_error':worst,'cells':28,'n_trials_per_cell':1000},indent=2))
# Reference-kernel checks against autodiff and exact local norm matching.
sys.path.insert(0,str(ROOT/'validation_kit'))
import torch,ttt,ttt_arms
import torch.nn.functional as F
torch.manual_seed(41); torch.set_num_threads(2)
x=torch.randn(2,3,8,16,dtype=torch.float64,requires_grad=True); target=torch.randn_like(x); gamma=torch.rand(1,3,1,16,dtype=torch.float64)+.3; beta=torch.randn_like(gamma)*.1
loss=.5*((ttt.ln_fwd(x,gamma,beta)-target)**2).sum(); auto,=torch.autograd.grad(loss,x); full=ttt.ln_fused_l2_bwd(x,target,gamma,beta)
err=float((auto-full).abs().max()); assert err<1e-12
D,std,zhat=ttt_arms._ln_stats(x,gamma,beta,1e-6)
z=ttt_arms._project(D,std,zhat,gamma*gamma*zhat); off=ttt_arms._project(D,std,zhat,gamma*beta)
drop=ttt_arms.make_hebbian_bwd('drop_pred')(x,target,gamma,beta); dropw=ttt_arms.make_hebbian_bwd('drop_w')(x,target,gamma,beta)
assert torch.allclose(full-drop,z+off,atol=1e-12,rtol=1e-12)
assert torch.allclose(full-dropw,z,atol=1e-12,rtol=1e-12)
nm=ttt_arms.make_normmatched_hebbian_bwd(ttt.ln_fused_l2_bwd)(x,target,gamma,beta)
assert torch.allclose(nm.norm(dim=-1),full.norm(dim=-1),atol=1e-12,rtol=1e-12)
# Uniform-gain epsilon residual.
g=torch.ones_like(gamma); be=torch.zeros_like(beta)
D,s,h=ttt_arms._ln_stats(x,g,be,1e-6); actual=ttt_arms._project(D,s,h,h); expected=h*1e-6/s**3
assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-8)
# Same CV, different orientation and projected norm at epsilon zero.
h=torch.tensor([1.,-1.,1.,-1.],dtype=torch.float64); proj=lambda v:v-v.mean()-h*(h*v).mean()
v1=proj(torch.tensor([1.,2.,1.,2.])*h); v2=proj(torch.tensor([1.,1.,2.,2.])*h)
assert v1.norm()==0 and v2.norm()>0
# Nonlinear dot-product gauge on a multi-step inner loop.
def dynamics(ascent,sign):
 torch.manual_seed(10); w=sign*torch.randn(5,7,dtype=torch.float64); theta=torch.randn(7,4,dtype=torch.float64); p=sign*torch.randn(3,5,dtype=torch.float64); outputs=[]
 keys=torch.randn(11,4,dtype=torch.float64); values=torch.randn(11,5,dtype=torch.float64)
 for k,v in zip(keys,values):
  w=w.detach().requires_grad_(); theta=theta.detach().requires_grad_(); loss=-(v*(w@torch.tanh(theta@k))).sum(); dw,dt=torch.autograd.grad(loss,(w,theta)); direction=1 if ascent else -1; w=w+direction*.03*dw; theta=theta+direction*.03*dt; outputs.append(p@w@torch.tanh(theta@k))
 return torch.stack(outputs).detach()
gauge_error=float((dynamics(False,1)-dynamics(True,-1)).abs().max()); assert gauge_error<1e-12
# Frozen arm and baseline identity on a tiny released model.
cfg=ttt.TTTConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,mini_batch_size=4,ttt_layer_type='linear',share_qk=True,use_gate=True)
model=ttt.TTTForCausalLM(cfg).eval(); ids=torch.arange(16).reshape(1,-1)%32
with torch.no_grad():
 original=model(input_ids=ids).logits
 with ttt_arms.arm('descent',ttt): desc=model(input_ids=ids).logits
 assert torch.equal(original,desc)
 with ttt_arms.arm('frozen',ttt):
  with ttt_arms.StateRecorder(ttt) as rec: model(input_ids=ids)
  drift=rec.max_drift()
  assert all(v['max_pairwise_spread']==0 for v in drift.values())
checks={'autodiff_max_abs_error':err,'drop_pred_matches_prediction_plus_offset':True,'drop_w_matches_prediction_only':True,'token_norm_matching':True,'epsilon_residual':True,'CV_orientation_counterexample':True,'nonlinear_gauge_max_abs_error':gauge_error,'baseline_bitwise_equal':True,'frozen_state_constant':True}
(OUT/'kernel_checks.json').write_text(json.dumps(checks,indent=2)); print(checks)
