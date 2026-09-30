"""Publication terminology is explicit here; internal analysis aliases stop at this boundary."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-data',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
DATA=args.source_data.resolve();WORK=args.output.resolve();DEST=WORK
if DATA==WORK or DATA in WORK.parents:raise SystemExit('Output must be outside the input source-data directory.')
FIG=DEST/'figures';SRC=DEST/'source_data';FIG.mkdir(parents=True,exist_ok=True);SRC.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','mathtext.fontset':'dejavusans'})
navy='#39598B';green='#59A64D';red='#ED5658';blue='#58A4D8';orange='#EDA71D'
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=400,bbox_inches='tight',pad_inches=.08)
 # Record all displayed strings so notation can be checked mechanically.
 from matplotlib.text import Text
 (WORK/f'{name}_displayed_text.json').write_text(json.dumps([x.get_text() for x in fig.findobj(Text) if x.get_text()],ensure_ascii=False,indent=2));plt.close(fig)
def letter(ax,s):ax.text(-.16,1.06,f'({s})',transform=ax.transAxes,ha='left',va='bottom',fontsize=11)
def heatmap(ax,m,title,cmap='Blues',vmin=None,vmax=None,states=True):
 im=ax.imshow(m.to_numpy(),cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
 ax.set_xticks(range(len(m.columns)),m.columns);ax.set_yticks(range(len(m)),m.index)
 for i in range(len(m)):
  for j in range(len(m.columns)):
   v=m.iloc[i,j];norm=im.norm(v);ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6.9,color='white' if norm>.66 else '#222222')
 ax.set_xticks(np.arange(-.5,len(m.columns),1),minor=True);ax.set_yticks(np.arange(-.5,len(m),1),minor=True);ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='both',length=0)
 ax.set_title(title,pad=7);return im
# Fig.4(e): report the true denominator and the undefined resamples.
f=pd.read_csv(DATA/'Fig4e_bootstrap_selection_frequency.csv');f.to_csv(SRC/'Fig4e_bootstrap_selection_frequency.csv',index=False)
pd.read_csv(DATA/'Fig4e_bootstrap_replicates.csv').rename(columns={f'L{x}':f'minimum_separation_margin_{x}_cm' for x in [7,9,11,13,15]}).to_csv(SRC/'Fig4e_bootstrap_replicates.csv',index=False)
fig,ax=plt.subplots(figsize=(3.1,2.9));ax.axhspan(8.6,9.4,color='#EDF4FA')
for row in f.itertuples():
 col=navy if row.fiber_length_cm==9 else '#8D9AA8';ax.hlines(row.fiber_length_cm,0,row.selection_frequency,color=col,lw=1.2);ax.plot(row.selection_frequency,row.fiber_length_cm,'o',ms=6 if row.fiber_length_cm==9 else 4,mfc=col if row.fiber_length_cm==9 else 'white',mec=col)
 ax.text(.055 if row.fiber_length_cm!=9 else row.selection_frequency-.035,row.fiber_length_cm+.24,f'{row.selection_count:,}/{row.complete_replicates:,}',ha='left' if row.fiber_length_cm!=9 else 'right',fontsize=7.5,color=col)
ax.set(yticks=[7,9,11,13,15],ylim=(6.5,15.7),xlim=(-.01,1.04),xlabel='Bootstrap selection frequency',ylabel=r'Length $L$ (cm)');ax.set_xticks([0,.25,.5,.75,1]);letter(ax,'e');fig.text(.5,-.06,'5,000 draws; 4,967 complete; 33 undefined',ha='center',fontsize=7.5);save(fig,'Fig4e')
# Same-state and cross-state summaries: only affected values replace historical rows.
a=pd.read_csv(DATA/'Fig7a_and_S2ab_device_state_quality.csv').rename(columns={'device':'device_id','mechanical_state':'state_id','robust_challenge_margin':'rg_challenge','challenge_retrieval_accuracy':'top1','equal_error_rate':'eer_challenge'})
a['state_conclusion']='DATABASE_AUTHENTICATION_'+a.quality_class.str.upper()
points=pd.read_csv(DATA/'Fig7b_similarity_summary_points.csv')
a['median_S_G']=points.loc[points.comparison=='Same device, state and challenge','green_credential_score'].to_numpy()
a['median_S_C']=points.loc[points.comparison=='Different challenge','green_credential_score'].to_numpy()
b=pd.read_csv(DATA/'Fig7c_device_state_comparisons.csv').rename(columns={'device':'device_id','source_mechanical_state':'source_state','target_mechanical_state':'target_state','cross_state_similarity_median':'median_S_X','cross_state_revocation_margin':'rg_cross_state_credential'})
sd=pd.read_csv(DATA/'FigS2cd_inter_device_discrimination.csv').rename(columns={'mechanical_state':'state_id','inter_device_comparison_count':'n_S_D','inter_device_similarity_median':'median_S_D','inter_device_similarity_upper_tail_quantile':'q95_S_D','device_robust_gap':'rg_device','device_auc':'auc_device','device_eer':'eer_device'})
assert len(a)==80 and not a.duplicated(['device_id','state_id']).any();assert len(b)==640 and not b.duplicated(['device_id','source_state','target_state']).any()
pa=a[['device_id','state_id','rg_challenge','top1','eer_challenge','state_conclusion']].rename(columns={'device_id':'device','state_id':'mechanical_state','rg_challenge':'robust_challenge_margin','top1':'challenge_retrieval_accuracy','eer_challenge':'equal_error_rate','state_conclusion':'quality_class'});pa.quality_class=pa.quality_class.str.replace('DATABASE_AUTHENTICATION_','').str.title();pa.to_csv(SRC/'Fig7a_and_S2ab_device_state_quality.csv',index=False)
states=[f'M{i}' for i in range(8)];devices=[f'F{i:02}' for i in range(1,11)]
rg=a.pivot(index='device_id',columns='state_id',values='rg_challenge').reindex(index=devices,columns=states);top=a.pivot(index='device_id',columns='state_id',values='top1').reindex(index=devices,columns=states);sd=sd.set_index('state_id').reindex(states).reset_index()
sd.rename(columns={'state_id':'mechanical_state','n_S_D':'inter_device_comparison_count','median_S_D':'inter_device_similarity_median','q95_S_D':'inter_device_similarity_upper_tail_quantile','rg_device':'device_robust_gap','auc_device':'device_auc','eer_device':'device_eer'}).to_csv(SRC/'FigS2cd_inter_device_discrimination.csv',index=False)
fig,axs=plt.subplots(2,2,figsize=(7.6,6.1),layout='constrained',gridspec_kw={'height_ratios':[1.5,1]})
im=heatmap(axs[0,0],rg,r'Challenge robust gap $RG_C$',vmin=0,vmax=.6);fig.colorbar(im,ax=axs[0,0],fraction=.045,pad=.02,label=r'$RG_C$');axs[0,0].set(xlabel='Mechanical state',ylabel='Device')
im=heatmap(axs[0,1],top,'Top-1 retrieval: 128 challenges',vmin=.85,vmax=1);fig.colorbar(im,ax=axs[0,1],fraction=.045,pad=.02,label='Top-1 accuracy');axs[0,1].set(xlabel='Mechanical state',ylabel='Device')
x=np.arange(8);axs[1,0].errorbar(x,sd.median_S_D,yerr=[np.zeros(8),sd.q95_S_D-sd.median_S_D],fmt='o',color=red,ms=4,capsize=3,lw=.8);axs[1,0].axhline(0,color='#CBD3DB',lw=.7);axs[1,0].set(xticks=x,xticklabels=states,xlabel='Mechanical state',ylabel=r'Inter-device similarity $S_{\mathrm{inter},d}$');axs[1,0].text(.02,.96,'Median with upper whisker to '+r'$Q_{95\%}$'+'\n11,520 comparisons per state',transform=axs[1,0].transAxes,va='top',fontsize=7)
axs[1,1].plot(x,sd.rg_device,'o-',color=blue,ms=4,lw=1);axs[1,1].set(xticks=x,xticklabels=states,xlabel='Mechanical state',ylabel=r'Device robust gap $RG_{\mathrm{device}}$',ylim=(0,.75));axs[1,1].text(.03,.1,f'Device AUC: {sd.auc_device.min():.4f}–{sd.auc_device.max():.4f}',transform=axs[1,1].transAxes,fontsize=7.5)
for ax,l in zip(axs.flat,'abcd'):letter(ax,l)
save(fig,'FigS2')
# Fig.7(b) retains the publication's aggregation over unit/pair-level medians.
off=b[~b.is_diagonal];groups=[a.median_S_G.to_numpy(),a.median_S_C.to_numpy(),sd.median_S_D.to_numpy(),off.median_S_X.to_numpy()]
labels=['Same device, state and challenge','Different challenge','Different device','Different mechanical state'];symbols=[r'$S_{\mathrm{intra}}$',r'$S_{\mathrm{inter},c}$',r'$S_{\mathrm{inter},d}$',r'$S_{\mathrm{inter},s}$'];colors=[green,orange,blue,red]
pd.concat([pd.DataFrame({'comparison':label,'green_credential_score':values}) for label,values in zip(labels,groups)]).to_csv(SRC/'Fig7b_similarity_summary_points.csv',index=False)
mat=b.groupby(['source_state','target_state']).median_S_X.median().unstack().reindex(index=states,columns=states);mat.to_csv(SRC/'Fig7c_cross_state_matrix.csv')
b[['device_id','source_state','target_state','is_diagonal','median_S_X','rg_cross_state_credential']].rename(columns={'device_id':'device','source_state':'source_mechanical_state','target_state':'target_mechanical_state','median_S_X':'cross_state_similarity_median','rg_cross_state_credential':'cross_state_revocation_margin'}).to_csv(SRC/'Fig7c_device_state_comparisons.csv',index=False)
fig=plt.figure(figsize=(8.6,6.5));gs=fig.add_gridspec(2,3,height_ratios=[1.1,1],left=.08,right=.98,bottom=.09,top=.95,wspace=.55,hspace=.65)
ax=fig.add_subplot(gs[0,0]);ax.axvline(.05,color='#888',ls=':',lw=.8);ax.axhline(.9,color='#888',ls=':',lw=.8)
for quality,col,marker in [('Valid',red,'o'),('Partial',green,'^'),('Failed','#333','x')]:
 t=pa[pa.quality_class==quality]
 if len(t):ax.scatter(t.robust_challenge_margin,t.challenge_retrieval_accuracy,s=18,c=col,marker=marker,edgecolors='white' if marker!='x' else None,linewidths=.3,label=f'{quality} ({len(t)})')
ax.set(xlabel=r'Challenge robust gap $RG_C(i,s)$',ylabel='Challenge-retrieval\nTop-1 accuracy',ylim=(.86,1.015),xlim=(-.02,.62));ax.text(.055,.99,r'$RG_C=0.05$',rotation=90,va='top',fontsize=7);ax.text(.59,.902,'Top-1 = 0.90',ha='right',va='bottom',fontsize=7);ax.legend(loc='lower right',fontsize=7,frameon=False);letter(ax,'a')
ax=fig.add_subplot(gs[0,1]);
for index,(v,col,sym) in enumerate(zip(groups,colors,symbols)):
 y=3-index;grid=np.linspace(min(v)-.04,max(v)+.04,250);den=gaussian_kde(v)(grid);den=den/den.max()*.3;ax.fill_between(grid,y,y+den,color=col,alpha=.25);ax.plot(grid,y+den,color=col,lw=.8);q=.05 if index==0 else .95;value=np.quantile(v,q);ax.text(.98,y+.13,rf'$Q_{{{int(q*100)}\%}}={value:.3f}$',ha='right',fontsize=7.4);ax.text(-.17,y+.08,sym,ha='right',fontsize=8)
ax.set(xlim=(-.17,1.03),ylim=(-.25,3.6),yticks=[],xlabel=r'Green credential score $q_G$');ax.axvline(0,color='#BDC7D1',lw=.6,ls=':');letter(ax,'b')
sub=gs[0,2].subgridspec(2,1,height_ratios=[1.65,1],hspace=.4);ax=fig.add_subplot(sub[0]);heatmap(ax,mat,'',vmin=-.04,vmax=.96);ax.set(xlabel='Target mechanical state',ylabel='Source mechanical state');ax.tick_params(labelsize=6.5);ax.text(0,1.06,'(c)',transform=ax.transAxes,fontsize=11,va='bottom')
ax=fig.add_subplot(sub[1]);per=b.assign(kind=np.where(b.is_diagonal,'same','cross')).groupby(['device_id','kind']).median_S_X.median().unstack()
for i,row in enumerate(per.itertuples()):ax.plot([row.cross,row.same],[i,i],color='#CCD6E0',lw=.8);ax.plot(row.cross,i,'o',mfc='white',mec=red,ms=3);ax.plot(row.same,i,'o',color=blue,ms=3)
ax.set(yticks=[],xlabel=r'Median green credential score $q_G$',xlim=(-.05,1.05),ylim=(-2,12));ax.text(.02,.91,'Cross-state',color=red,transform=ax.transAxes,fontsize=7);ax.text(.98,.91,'Same-state',color=blue,transform=ax.transAxes,fontsize=7,ha='right');ax.text(.02,.02,rf'min $RG_{{\mathrm{{state}}}}={off.rg_cross_state_credential.min():.3f}$',transform=ax.transAxes,fontsize=7)
# Red-channel results are unchanged by replacement of two green recordings.
r=pd.read_csv(DATA/'Fig7d_red_identity_scores.csv').rename(columns={'red_identity_score':'S_R'});r['r0']=r.comparison.eq('Same device, same state');r['r1']=r.comparison.eq('Same device, cross state');r['same']=~r.comparison.eq('Different device');dev=pd.read_csv(DATA/'Fig7d_red_identity_discrimination.csv').rename(columns={'red_identity_auc':'AUC','red_identity_eer':'EER'});redmeta={'global_identity':{'EER':float(pd.read_csv(DATA/'Fig7d_global_identity_summary.csv').global_red_identity_eer.iloc[0])}};ax=fig.add_subplot(gs[1,:2]);redgroups=[r[r.r0].S_R.to_numpy(),r[r.r1].S_R.to_numpy(),r[~r.same].S_R.to_numpy()];redlabels=['Same device, same state','Same device, cross state','Different device']
for y,v,label,col in zip([2,1,0],redgroups,redlabels,[red,red,green]):
 grid=np.linspace(min(v),max(v),400);den=gaussian_kde(v)(grid);den=den/den.max()*.65;ax.fill_between(grid,y,y+den,color=col,alpha=.3);ax.plot(grid,y+den,color=col,lw=.8);ax.text(.015,(y+.25)/3,label+f': {np.median(v):.2f}',color=col,transform=ax.transAxes,fontsize=8)
ax.set(xlabel=r'Red identity score $q_R$',yticks=[],ylim=(-.1,2.9));letter(ax,'d')
ax=fig.add_subplot(gs[1,2]);order=dev.sort_values('EER');ax.plot(order.EER,np.arange(len(order)),'o',color=green,ms=4);ax.set(yticks=np.arange(len(order)),yticklabels=order.device,xlabel='Identity EER',xlim=(-.004,.112));global_eer=redmeta['global_identity']['EER'];ax.axvline(global_eer,color=red,ls=':',lw=1);ax.text(global_eer,9.3,f'Global\n{global_eer:.3f}',ha='center',fontsize=7.5)
pd.concat([pd.DataFrame({'comparison':label,'red_identity_score':values}) for label,values in zip(redlabels,redgroups)]).to_csv(SRC/'Fig7d_red_identity_scores.csv',index=False)
dev.rename(columns={'device':'device','AUC':'red_identity_auc','EER':'red_identity_eer'}).to_csv(SRC/'Fig7d_red_identity_discrimination.csv',index=False)
pd.DataFrame([{'global_red_identity_eer':global_eer}]).to_csv(SRC/'Fig7d_global_identity_summary.csv',index=False)
save(fig,'Fig7')
summary={'Fig7a_valid':int((pa.quality_class=='Valid').sum()),'Fig7a_partial':int((pa.quality_class=='Partial').sum()),'Fig7a_failed':int((pa.quality_class=='Failed').sum()),'robust_challenge_margin_median':float(a.rg_challenge.median()),'Fig7b_quantiles':[float(np.quantile(v,.05 if i==0 else .95)) for i,v in enumerate(groups)],'Fig7c_same_state_median':float(b[b.is_diagonal].median_S_X.median()),'Fig7c_cross_state_median':float(off.median_S_X.median()),'minimum_cross_state_revocation_margin':float(off.rg_cross_state_credential.min()),'devices_positive_revocation_margin':int((off.groupby('device_id').rg_cross_state_credential.min()>0).sum()),'F02_M1_Top1':float(a[(a.device_id=='F02')&(a.state_id=='M1')].top1.iloc[0])}
(WORK/'publication_recomputed_values.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
