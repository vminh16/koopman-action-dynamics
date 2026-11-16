import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment import OUT as out
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,system,arms in [
    (axes[0,0],'pendulum',['baseline','epochmean','noenc','noenc_epochmean']),
    (axes[0,1],'cartpole',['baseline','negative_head','baseline_start0','negative_head_start0'])]:
    for arm in arms:
        r=json.loads((out/f'{system}-{arm}-s42/result.json').read_text())
        ax.plot([h['epoch'] for h in r['history']],[h['validation_first15_mse'] for h in r['history']],marker='.',label=arm)
    ax.set(yscale='log',xlabel='Epoch (seed 42)',ylabel='Validation H=15 MSE',title=system.capitalize()+' — controlled interventions')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
for ax,system,reference in [(axes[1,0],'pendulum',.0318),(axes[1,1],'cartpole',.0024)]:
    d=json.loads((out/f'{system}_final_summary.json').read_text());labels=[];means=[];std=[]
    for arm,metrics in d['summary'].items():
        a=metrics['open_loop_H15'];labels.append(arm);means.append(a['mean']);std.append(a['sample_std'])
    if system=='cartpole':
        a=json.loads((out/'cartpole_start0_diagnostic_summary.json').read_text());labels.append('start0\nDIAGNOSTIC');means.append(a['negative_head_start0_mean']);std.append(a['negative_head_start0_sample_std'])
    x=np.arange(len(labels));ax.bar(x,means,yerr=std,capsize=5,color=['#4c78a8','#f58518','#54a24b'][:len(labels)],alpha=.85)
    ax.axhline(reference,color='black',linestyle='--',label='Paper numeric reference')
    ax.set(xticks=x,xticklabels=labels,yscale='log',ylabel='Test H=15 MSE (mean ± sample SD)',title=system.capitalize()+' — 3 seeds, 1000 test trajectories')
    ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8)
fig.suptitle('P1 causal diagnosis — numerical scores do not certify exact paper-protocol reproduction',fontsize=12)
fig.savefig(out/'causal_results.png',dpi=180)
print(out/'causal_results.png')
