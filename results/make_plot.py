"""Rebuild results/learning_curves.png from results/runs.json (BirdSet PER re-run, Oct 2026)."""
from pathlib import Path
import json,numpy as np,collections,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=json.load(open(Path(__file__).parent/'runs.json')); g=collections.defaultdict(list)
for r in rows: g[r['cond']].append(r)
C=['#2a78d6','#eb6834','#1baf7a','#6250d6']; INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'; BG='#fcfcfb'
def band(cond):
    rs=g[cond]; lo=max(r['curve'][0][0] for r in rs); hi=min(r['curve'][-1][0] for r in rs)
    x=np.linspace(lo,hi,60); Y=np.array([np.interp(x,*zip(*r['curve'])) for r in rs])*100
    return x,Y.mean(0),Y.std(0,ddof=1)
def panel(ax,items,title):
    ax.set_facecolor(BG)
    for (cond,label),c in zip(items,C):
        x,m,s=band(cond); ax.fill_between(x,m-s,m+s,color=c,alpha=.12,lw=0); ax.plot(x,m,color=c,lw=2,label=f'{label} — {m[-1]:.1f}% at {x[-1]:.0f} labels')
    ax.set_title(title,loc='left',fontsize=11,color=INK,pad=10)
    ax.set_xlabel('Labeled clips used for training',color=INK2,fontsize=9); ax.set_ylabel('Test accuracy (%)',color=INK2,fontsize=9)
    ax.grid(axis='y',color=GRID,lw=.8); ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2,labelsize=8); ax.set_ylim(0,95); ax.legend(frameon=False,fontsize=8,loc='lower right')
fig,axs=plt.subplots(1,2,figsize=(12,4.8),facecolor=BG)
panel(axs[0],[('A_margin_fixed17','Margin'),('A_entropy_fixed17','Entropy'),('A_random_fixed17','Random'),('A_coreset_fixed17','Coreset')],'Query strategy (budget 17/round, 20 rounds)')
panel(axs[1],[('A_entropy_fixed17','Fixed 17/round'),('B_entropy_ucb_orig','UCB bandit (original)'),('B_entropy_ucb_fixed_c141','UCB bandit (corrected reward)'),('B_entropy_fixed5','Fixed 5/round')],'Budget policy, entropy sampling')
fig.text(0.01,0.01,'BirdSet PER, 25 species, BirdNET embeddings + random forest. Lines: mean of 3 seeds; bands: ±1 SD.',fontsize=8,color=INK2)
plt.tight_layout(rect=(0,0.04,1,1)); plt.savefig(Path(__file__).parent/'learning_curves.png',dpi=160,facecolor=BG)
