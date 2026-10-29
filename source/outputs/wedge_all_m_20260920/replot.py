from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).parent;d=pd.read_csv(O/'areas.csv');v=pd.read_csv(O/'contour_vertices.csv');s=d[(d.label_mass==.683)&d.m.notna()]
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in'})
colors=['#222222','#56B4E9','#0072B2','#009E73','#E69F00','#CC79A7'];fig,(ax,bx)=plt.subplots(1,2,figsize=(9,4.1),layout='constrained')
for (_,r),col in zip(s.iterrows(),colors):
 ax.plot(r.m,r.display_over_radial,'o',ms=6,color=col)
 ax.annotate(f'{r.display_over_radial:.3f}',(r.m,r.display_over_radial),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
 z=v[(v.branch==r.branch)&(v.mass==.683)]
 for k,seg in z.groupby('segment'):
  bx.plot(seg.omega_m,seg.w,color=col,lw=1.1,ls='--' if r.m==0 else '-',label=('Radial only' if r.m==0 else f'$m={r.m:.3f}$') if k==0 else None)
ax.axhline(1,color='0.65',ls=':',lw=.8);ax.set(xlabel=r'Wedge slope $m$',ylabel=r'$\mathcal{A}_{\rm display}(m)/\mathcal{A}_{\rm display}(0)$',title='(a) Displayed inner-contour area',xlim=(-.13,1.86));ax.margins(y=.23)
bx.plot(.3111,-1,'*',color='black',ms=8);bx.set(xlabel=r'$\Omega_m$',ylabel=r'$w$',title='(b) Inner contours, common 1285-point grid');bx.legend(fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3)
for a in [ax,bx]:a.tick_params(top=True,right=True)
for ext in ['png','pdf']:fig.savefig(O/f'fig__intrinsic_subtracted_contours__all_m_display_area.{ext}',dpi=230)
