"""Export attacks using the current publication's symbols and panel numbers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-data',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
DATA=args.source_data.resolve();WORK=args.output.resolve();DEST=WORK
if DATA==WORK or DATA in WORK.parents:raise SystemExit('Output must be outside the input source-data directory.')
FIG=DEST/'figures';SRC=DEST/'source_data';FIG.mkdir(parents=True,exist_ok=True);SRC.mkdir(exist_ok=True)

plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','mathtext.fontset':'dejavusans'})
STATES=[f'M{i}' for i in range(8)]
FULL=['mean_response','exact_template_replay','ridge_clone','kernel_ridge_clone','random_fourier_ridge_clone','small_mlp_clone']
PART=['mean_leaked_response','nearest_leaked_challenge','ridge_clone','kernel_ridge_clone','random_fourier_ridge_clone','small_mlp_clone']
NAMES=dict(zip(FULL,['Mean-response baseline','Exact replay','Ridge','Kernel Ridge','RFF Ridge','Small MLP']))
NAMES.update({'mean_leaked_response':'Mean-response baseline','nearest_leaked_challenge':'Nearest-challenge baseline'})
COLORS=dict(zip(FULL,['#8D9CAB','#63B35D','#149DA0','#F35D60','#DB7BB1','#B95336']))
COLORS.update({'mean_leaked_response':'#8D9CAB','nearest_leaked_challenge':'#57A4DB'})
GREEN='#59A64D'; BLUE='#58A4D8'; RED='#ED5658'; NORM=TwoSlopeNorm(vmin=-.12,vcenter=0,vmax=1)

def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=400,bbox_inches='tight',pad_inches=.09)
    from matplotlib.text import Text
    (WORK/f'{name}_displayed_text.json').write_text(json.dumps([x.get_text() for x in fig.findobj(Text) if x.get_text()],ensure_ascii=False,indent=2));plt.close(fig)

def letter(ax,s):ax.text(-.16,1.07,f'({s})',transform=ax.transAxes,fontsize=11,va='bottom')
def matrix(df,col):return df.groupby(['source_state','target_state'])[col].median().unstack().reindex(index=STATES,columns=STATES)
def heat(ax,m,title):
    im=ax.imshow(m.to_numpy(),cmap='RdBu',norm=NORM,aspect='auto')
    ax.set(xticks=range(8),xticklabels=STATES,yticks=range(8),yticklabels=STATES,xlabel='Target mechanical state',ylabel='Source mechanical state',title=title)
    for i in range(8):
        for j in range(8):
            v=m.iloc[i,j];ax.text(j,i,('0.00' if abs(v)<.005 else f'{v:.2f}'),ha='center',va='center',fontsize=6.6,color='white' if v>.6 or v<-.07 else '#222')
    ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True);ax.grid(which='minor',color='white',lw=.7);ax.tick_params(which='both',length=0)
    return im

def export(df,cols,name):
    t=df[list(cols)].rename(columns=cols)
    if 'method' in t:t['method']=t.method.map(NAMES)
    t.to_csv(SRC/name,index=False)

reverse_names={v:k for k,v in NAMES.items() if k in FULL}
def read_public(name,columns,partial=False):
    table=pd.read_csv(DATA/name).rename(columns=columns)
    if 'attack_method' in table:
        names={v:k for k,v in NAMES.items() if k in (PART if partial else FULL)}
        table['attack_method']=table.attack_method.map(names)
        if table.attack_method.isna().any():raise ValueError('Unrecognized publication method name')
    return table
keys={'device':'device_id','source_mechanical_state':'source_state','target_mechanical_state':'target_state','method':'attack_method','same_state':'is_diagonal'}
c=read_public('Fig8a_complete_disclosure_unit_scores.csv',{**keys,'attack_score_median':'median_S_A','reconstruction_or_replay_success':'source_state_valid','genuine_score_median':'median_genuine'})
d=read_public('Fig8b_and_S3_complete_disclosure_transfer.csv',{**keys,'attack_score_median':'median_S_A','state_specificity_gap':'delta_clone_same_to_cross'})
assert len(c)==480 and not c.duplicated(['device_id','source_state','attack_method']).any()
assert len(d)==3200 and not d.duplicated(['device_id','source_state','target_state','attack_method']).any()
assert set(c.attack_method)==set(FULL),set(c.attack_method)
export(c,{'device_id':'device','source_state':'source_mechanical_state','attack_method':'method','median_S_A':'attack_score_median','source_state_valid':'reconstruction_or_replay_success','median_genuine':'genuine_score_median'},'Fig8a_complete_disclosure_unit_scores.csv')
export(d,{'device_id':'device','source_state':'source_mechanical_state','target_state':'target_mechanical_state','attack_method':'method','is_diagonal':'same_state','median_S_A':'attack_score_median','delta_clone_same_to_cross':'state_specificity_gap'},'Fig8b_and_S3_complete_disclosure_transfer.csv')
band=c[c.attack_method=='exact_template_replay'].median_genuine.quantile([.05,.95]).to_numpy()
pd.DataFrame([{'genuine_lower_tail_quantile':band[0],'genuine_upper_tail_quantile':band[1]}]).to_csv(SRC/'Fig8a_genuine_reference.csv',index=False)
summary=[]; ranges={}; mats={}
for method in FULL[1:]:
    sub=d[d.attack_method==method];mats[method]=matrix(sub,'median_S_A')
    mats[method].to_csv(SRC/f'FigS3_{NAMES[method].lower().replace(" ","_")}_matrix.csv')
    summary.append({'method':method,'same':sub[sub.is_diagonal].median_S_A.median(),'cross':sub[~sub.is_diagonal].median_S_A.median(),'gap':sub[~sub.is_diagonal].delta_clone_same_to_cross.median()})
    diag=np.diag(mats[method]);off=mats[method].to_numpy()[~np.eye(8,dtype=bool)]
    ranges[NAMES[method]]={'state_diagonal_min':float(diag.min()),'state_diagonal_max':float(diag.max()),'state_off_diagonal_min':float(off.min()),'state_off_diagonal_max':float(off.max())}
summary=pd.DataFrame(summary);export(summary,{'method':'method','same':'same_state_attack_score_median','cross':'cross_state_attack_score_median','gap':'state_specificity_gap'},'Fig8b_same_and_cross_state_summary.csv')
fig,axs=plt.subplots(2,3,figsize=(9,6.1),layout='constrained')
for ax,method,l in zip(axs.flat,FULL[1:],'abcde'):im=heat(ax,mats[method],NAMES[method]);letter(ax,l)
axs[1,2].axis('off');fig.colorbar(im,ax=list(axs.flat),fraction=.022,pad=.025,label=r'Attack score $S_A$')
save(fig,'FigS3')
metrics={'complete_disclosure_success_counts':{NAMES[k]:int(v) for k,v in c.groupby('attack_method').source_state_valid.sum().items()},'complete_disclosure_state_matrix_ranges':ranges,'state_specificity_gap':{NAMES[r.method]:r.gap for r in summary.itertuples()}}
(WORK/'attack_publication_values.json').write_text(json.dumps(metrics,indent=2))

pk={**keys,'partition':'repetition','partition_seed':'split_seed','disclosed_crp_count':'leak_size','hidden_challenge_count':'n_hidden','hidden_attack_score_median':'median_S_A_partial','hidden_genuine_score_median':'median_S_G_hidden','hidden_challenge_retrieval_accuracy':'Top1_hidden','top1_lift':'normalized_retrieval_lift','hidden_challenge_mean_residual_ncc':'median_residual_ncc','retained_pca_dimension':'effective_pca_dim'}
p=read_public('Fig8c_and_S4_partial_disclosure_unit_results.csv',pk,partial=True)
p['common_dominance_flag']=np.where(p.common_dominated_evaluation,'HIGH_NCC_DOMINATED_BY_COMMON_RESPONSE','OK')
q=read_public('Fig8d_and_S4a_partial_disclosure_transfer.csv',pk,partial=True)
model=read_public('Fig8d_disclosure_summary.csv',{'method':'attack_method','same_state_attack_score_median':'diagonal_median_S_A','cross_state_attack_score_median':'off_diagonal_median_S_A'},partial=False)
model=model[model.disclosure=='Partial: 96/128'].copy();model['leak_size']=96
assert len(p)==9600 and len(q)==76800
assert not q.duplicated(['device_id','source_state','target_state','repetition','leak_size','attack_method']).any()
pc={'device_id':'device','source_state':'source_mechanical_state','repetition':'partition','split_seed':'partition_seed','leak_size':'disclosed_crp_count','attack_method':'method','n_hidden':'hidden_challenge_count','median_S_A_partial':'hidden_attack_score_median','median_S_G_hidden':'hidden_genuine_score_median','Top1_hidden':'hidden_challenge_retrieval_accuracy','normalized_retrieval_lift':'top1_lift','median_residual_ncc':'hidden_challenge_mean_residual_ncc','effective_pca_dim':'retained_pca_dimension'}
p['common_dominated_evaluation']=p.common_dominance_flag.eq('HIGH_NCC_DOMINATED_BY_COMMON_RESPONSE')
pc['common_dominated_evaluation']='common_dominated_evaluation'
export(p,pc,'Fig8c_and_S4_partial_disclosure_unit_results.csv')
export(q,{**{k:v for k,v in pc.items() if k not in ['median_residual_ncc','effective_pca_dim','common_dominated_evaluation']},'target_state':'target_mechanical_state','is_diagonal':'same_state'},'Fig8d_and_S4a_partial_disclosure_transfer.csv')
curves=p.groupby(['attack_method','leak_size'])[['median_S_A_partial','normalized_retrieval_lift']].median().reset_index()
hidden=p.groupby('leak_size').median_S_G_hidden.median()
export(curves,{'attack_method':'method','leak_size':'disclosed_crp_count','median_S_A_partial':'hidden_attack_score_median','normalized_retrieval_lift':'top1_lift_median'},'Fig8c_partial_disclosure_summary.csv')
hidden.rename_axis('disclosed_crp_count').rename('hidden_genuine_score_median').to_csv(SRC/'Fig8c_hidden_genuine_reference.csv')
sub=q[(q.attack_method=='ridge_clone')&(q.leak_size==96)];partialmatrix=matrix(sub,'median_S_A_partial');ref=float(sub.median_S_G_hidden.median());partialmatrix.to_csv(SRC/'FigS4a_ridge_96_disclosed_matrix.csv')
audit=[]
for (method,n),g in p[p.attack_method.isin(FULL[2:])].groupby(['attack_method','leak_size']):
    dominated=g.common_dominance_flag.eq('HIGH_NCC_DOMINATED_BY_COMMON_RESPONSE')
    audit.append({'method':method,'disclosed_crp_count':n,'retained_pca_dimension':g.effective_pca_dim.median(),'median_unit_mean_residual_ncc':g.median_residual_ncc.median(),'common_dominated_evaluation_count':int(dominated.sum()),'evaluation_count':len(g),'common_dominated_evaluation_fraction':dominated.mean()})
audit=pd.DataFrame(audit);export(audit,{x:x for x in audit.columns},'FigS4bcd_model_audit.csv')
figure_rows=[]
for row in summary.itertuples():figure_rows.append({'disclosure':'Complete','method':row.method,'same_state':row.same,'cross_state':row.cross})
for method in FULL[2:5]:
    r=model[(model.attack_method==method)&(model.leak_size==96)].iloc[0]
    figure_rows.append({'disclosure':'Partial: 96/128','method':method,'same_state':r.diagonal_median_S_A,'cross_state':r.off_diagonal_median_S_A})
export(pd.DataFrame(figure_rows),{'disclosure':'disclosure','method':'method','same_state':'same_state_attack_score_median','cross_state':'cross_state_attack_score_median'},'Fig8d_disclosure_summary.csv')

fig=plt.figure(figsize=(9.1,7.4));gs=fig.add_gridspec(2,2,left=.095,right=.975,bottom=.08,top=.94,hspace=.72,wspace=.57)
ax=fig.add_subplot(gs[0,0]);ax.axhspan(*band,color=GREEN,alpha=.15)
for i,method in enumerate(FULL):
    vals=c[c.attack_method==method].median_S_A.to_numpy();ax.scatter(np.full(len(vals),i),vals,s=8,alpha=.45,color=COLORS[method],edgecolors='none')
    ax.plot([i-.18,i+.18],[np.median(vals)]*2,color=COLORS[method],lw=1.8)
    n=int(c[c.attack_method==method].source_state_valid.sum());ax.text(i,1.085,f'{n}/80',ha='center',color=GREEN if n else '#555',fontsize=7.5)
ax.text(5.75,.965,r'Genuine $q_G$: $Q_{5\%}$–$Q_{95\%}$',ha='right',color=GREEN,fontsize=7.5)
ax.set(xticks=range(6),xticklabels=['Mean-\nresponse\nbaseline','Exact\nreplay','Ridge','Kernel\nRidge','RFF\nRidge','Small\nMLP'],ylabel=r'Attack score $S_A$',xlabel='Attack method',ylim=(-.05,1.12),xlim=(-.5,5.9));letter(ax,'a')
sg=gs[0,1].subgridspec(2,1,height_ratios=[1.45,1],hspace=.62);ax=fig.add_subplot(sg[0]);heat(ax,mats['exact_template_replay'],'Exact replay');ax.tick_params(labelsize=6.5);ax.set_xlabel('Target mechanical state',fontsize=8);ax.set_ylabel('Source mechanical state',fontsize=8);letter(ax,'b')
ax=fig.add_subplot(sg[1])
for i,r in enumerate(summary.itertuples()):
    y=4-i;ax.plot([r.cross,r.same],[y,y],color='#D0DCE6',lw=1);ax.plot(r.same,y,'o',ms=4,color=BLUE);ax.plot(r.cross,y,'o',ms=4,mfc='white',mec=RED);ax.text(1.08,y,f'{r.gap:.2f}',ha='center',va='center',fontsize=7)
ax.text(1.08,4.85,r'$\Delta S_A$',ha='center',fontsize=8);ax.set(yticks=range(5),yticklabels=[NAMES[x] for x in reversed(FULL[1:])],xlabel=r'Median attack score $S_A$',xlim=(-.05,1.18),ylim=(-.5,5.15));ax.text(.02,1,'○ Cross-state',color=RED,transform=ax.transAxes,fontsize=7);ax.text(.5,1,'● Same-state',color=BLUE,transform=ax.transAxes,fontsize=7)
sg=gs[1,0].subgridspec(2,1,height_ratios=[1.2,1],hspace=.52);ax=fig.add_subplot(sg[0])
for method in PART:
    t=curves[curves.attack_method==method];ax.plot(t.leak_size,t.median_S_A_partial,'o-',ms=3,lw=1,color=COLORS[method],label=NAMES[method])
ax.axhline(0,color='#CDD6DF',ls=':',lw=.8);ax.set(xticks=[16,32,64,96],xticklabels=[],ylabel='Hidden-challenge\n'+r'attack score $S_A$',ylim=(-.135,.025));letter(ax,'c')
ax.legend(loc='lower left',bbox_to_anchor=(-.06,1.02),ncol=2,frameon=False,fontsize=6.4,columnspacing=.8,handlelength=1.3)
ax.text(.98,.96,rf'Genuine $q_G$ median: {hidden.median():.2f}',transform=ax.transAxes,ha='right',va='top',color=GREEN,fontsize=7.5)
ax=fig.add_subplot(sg[1]);lift=curves.pivot(index='attack_method',columns='leak_size',values='normalized_retrieval_lift').reindex(PART)
ax.imshow(lift,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('lift',['#FBE4E4','#E1F0DF']),vmin=0,vmax=1,aspect='auto')
for i in range(6):
    for j in range(4):ax.text(j,i,f'{lift.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7)
ax.set(xticks=range(4),xticklabels=[16,32,64,96],yticks=range(6),yticklabels=[NAMES[m] for m in PART],xlabel=r'Number of disclosed CRPs $N_L$ (of 128)');ax.tick_params(axis='y',labelsize=6.4);ax.set_title('Top-1 lift: 1 = chance level',fontsize=7.5,pad=5)
ax=fig.add_subplot(gs[1,1]);letter(ax,'d')
for i,row in enumerate(figure_rows):
    y=9-i if i<5 else 7-i
    ax.plot([row['cross_state'],row['same_state']],[y,y],color='#D0DCE6',lw=1);ax.plot(row['same_state'],y,'o',color=GREEN,ms=4);ax.plot(row['cross_state'],y,'o',mfc='white',mec=RED,ms=4)
ax.set(yticks=[9,8,7,6,5,2,1,0],yticklabels=[NAMES[r['method']] for r in figure_rows],xlabel=r'Median attack score $S_A$',xlim=(-.14,1.02),ylim=(-.8,11))
ax.text(.99,.99,r'Genuine $q_G$: $Q_{5\%}$–$Q_{95\%}$',transform=ax.transAxes,ha='right',va='top',color=GREEN,fontsize=7)
ax.text(.04,10.1,'Complete database disclosure',color=GREEN,fontsize=8);ax.text(.04,3.2,'Partial CRP disclosure (96/128)',color=RED,fontsize=8)
ax.axvspan(*band,ymin=.38,ymax=1,color=GREEN,alpha=.12);ax.legend(handles=[Line2D([],[],marker='o',color=GREEN,lw=0,label='Same-state'),Line2D([],[],marker='o',color=RED,mfc='white',lw=0,label='Cross-state')],loc='lower right',fontsize=7,frameon=False)
save(fig,'Fig8')

fig,axs=plt.subplots(2,2,figsize=(7.6,6.5),layout='constrained',gridspec_kw={'height_ratios':[1.2,1]})
im=heat(axs[0,0],partialmatrix,'Ridge: 96/128 disclosed');fig.colorbar(im,ax=axs[0,0],fraction=.045,pad=.02,label=r'Attack score $S_A$');axs[0,0].text(.5,-.24,f'10 devices × 5 partitions; 32 hidden challenges\nGenuine hidden-response reference: {ref:.3f}',ha='center',va='top',transform=axs[0,0].transAxes,fontsize=7)
specs=[('retained_pca_dimension',r'Retained PCA dimension $d_{\mathrm{PCA}}$',(0,70)),('median_unit_mean_residual_ncc','Median residual NCC',(-.04,.04)),('common_dominated_evaluation_fraction','Fraction of common-dominated\nevaluations',(-.04,1.04))]
for ax,(col,ylabel,ylim) in zip([axs[0,1],axs[1,0],axs[1,1]],specs):
    for method,marker in zip(FULL[2:],['^','D','v','P']):
        t=audit[audit.method==method];ax.plot(t.disclosed_crp_count,t[col],marker+'-',color=COLORS[method],lw=1,ms=4,label=NAMES[method])
    ax.set(xticks=[16,32,64,96],xlabel=r'Disclosed CRPs $N_L$ (of 128)',ylabel=ylabel,ylim=ylim)
axs[0,1].legend(loc='lower right',fontsize=7,frameon=False)
for ax,l in zip(axs.flat,'abcd'):letter(ax,l)
save(fig,'FigS4')
metrics.update({'partial_max_median_top1_lift':float(curves.normalized_retrieval_lift.max()),'partial_hidden_genuine_by_disclosure':hidden.to_dict(),'FigS4a_genuine_reference':ref,'FigS4a_diagonal_min':float(np.diag(partialmatrix).min()),'FigS4a_diagonal_max':float(np.diag(partialmatrix).max()),'FigS4c_min':float(audit.median_unit_mean_residual_ncc.min()),'FigS4c_max':float(audit.median_unit_mean_residual_ncc.max())})
(WORK/'attack_publication_values.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
