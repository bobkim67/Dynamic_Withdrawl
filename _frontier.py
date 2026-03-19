import sys, io, pickle, numpy as np, plotly.graph_objects as go
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('product_paths.pkl','rb') as f:
    data = pickle.load(f)

W0=100.0

def sim(path, wr, mode, bl=0.05, bh=0.05, thr=1.0):
    tr=wr/12;nav=W0;cum=0;prev_wd=W0*tr;wds=[]
    for t,r in enumerate(path):
        if mode=='vol':
            if t>=12:
                s=np.std(path[t-12:t],ddof=1)
                sr=abs(path[t-1])/s if s>0 else 0
                band=bh if sr>=thr else bl
            else: band=bl
        else: band=bl
        wd=prev_wd
        if nav>0:
            u=tr*(1+band);l=tr*(1-band);ratio=wd/nav
            if ratio>u:wd=u*nav
            elif ratio<l:wd=l*nav
        wd=min(wd,max(nav,0));nav=max(nav-wd,0)*(1+r);prev_wd=wd;cum+=wd;wds.append(wd)
    ws=np.array(wds);wp=ws[ws>0]
    cuts=[(ws[i]-ws[i-1])/ws[i-1] for i in range(1,len(ws)) if ws[i-1]>0]
    return {'nav':nav,'cum':cum,'tot':nav+cum,'std':np.std(wp) if len(wp)>0 else 0,'worst':min(cuts) if cuts else 0}

def med(results,key):
    return np.median([r[key] for r in results])

bands=[0.001,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.40,0.50,0.75,1.0,2.0,5.0]
lset={0.01,0.02,0.03,0.05,0.10,0.15,0.20,0.30,0.50,1.0,5.0}

vol_combos=[]
for bl in [0.01,0.02,0.03,0.05]:
    for bh_add in [0.03,0.05,0.08,0.10,0.13,0.15]:
        bh=bl+bh_add
        if bh>0.20: continue
        vol_combos.append((bl,bh))

flist=['Golden Growth','MS GROWTH','MS STABLE']
funds_wr={}
for fund in flist:
    mr=data['fund_paths'][fund]['monthly_returns']
    funds_wr[fund]=((1+mr.mean())**12-1)+0.01

chart_defs=[
    ('nav_cum','기말잔액','누적인출','nav','cum',False,False),
    ('tot_std','총가치','인출Std','tot','std',False,True),
    ('tot_worst','총가치','Worst삭감(%)','tot','worst_abs',False,True),
]

html='<!DOCTYPE html><html><head><meta charset="utf-8"><title>Frontier Alpha</title>'
html+='<style>@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");'
html+='body{font-family:"Pretendard",sans-serif;margin:0;background:#f5f5f5}'
html+='.tabs{display:flex;background:#1a237e;flex-wrap:wrap}.tab{padding:11px 16px;color:rgba(255,255,255,0.6);cursor:pointer;font-size:12px;font-weight:600;border-bottom:3px solid transparent}'
html+='.tab.active{color:white;border-bottom-color:#ff9800;background:rgba(255,255,255,0.08)}'
html+='.content{display:none;padding:12px}.content.active{display:block}</style></head><body><div class="tabs">'

tab_i=0
for fund in flist:
    for _,ylabel,_2,_3,_4,_5,_6 in chart_defs:
        act=' active' if tab_i==0 else ''
        ct_id_tab=chart_defs[tab_i % 3][0]
        short={'nav_cum':'NAV/인출','tot_std':'Tot/Std','tot_worst':'Tot/Worst'}[ct_id_tab]
        html+=f'<div class="tab{act}" onclick="switchTab({tab_i})">{fund} [{short}]</div>'
        tab_i+=1
html+='</div>'

tab_idx=0
for fi,fund in enumerate(flist):
    wr=funds_wr[fund];tr=wr/12;mwd=W0*tr
    paths=data['fund_paths'][fund]['bootstrap_paths']

    # 비율밴드 데이터
    bd_data=[]
    for band in bands:
        res=[sim(p,wr,'fixed',band) for p in paths]
        bd_data.append({k:med(res,k) for k in ['nav','cum','tot','std','worst']})
        bd_data[-1]['worst_abs']=abs(bd_data[-1]['worst'])*100
        bd_data[-1]['band']=band

    # Fixed Amount
    fa_res=[sim(p,wr,'fixed',99.0) for p in paths]
    fa={k:med(fa_res,k) for k in ['nav','cum','tot','std','worst']}
    fa['worst_abs']=abs(fa['worst'])*100

    # Fixed Rate
    fr_list=[]
    for p in paths:
        n=W0;c=0;wds=[]
        for r in p: wd=n*tr;n=max(n-wd,0)*(1+r);c+=wd;wds.append(wd)
        ws=np.array(wds);wp=ws[ws>0];cuts=[(ws[i]-ws[i-1])/ws[i-1] for i in range(1,len(ws)) if ws[i-1]>0]
        fr_list.append({'nav':n,'cum':c,'tot':n+c,'std':np.std(wp) if len(wp)>0 else 0,'worst':min(cuts) if cuts else 0})
    fr={k:np.median([r[k] for r in fr_list]) for k in ['nav','cum','tot','std','worst']}
    fr['worst_abs']=abs(fr['worst'])*100

    # Guard 5%
    g5_res=[sim(p,wr,'fixed',0.05) for p in paths]
    g5={k:med(g5_res,k) for k in ['nav','cum','tot','std','worst']}
    g5['worst_abs']=abs(g5['worst'])*100

    # Vol 조합들
    vol_data=[]
    for bl,bh in vol_combos:
        res=[sim(p,wr,'vol',bl,bh) for p in paths]
        d={k:med(res,k) for k in ['nav','cum','tot','std','worst']}
        d['worst_abs']=abs(d['worst'])*100
        d['label']=f'{bl*100:.0f}/{bh*100:.0f}%'
        d['bl']=bl;d['bh']=bh
        vol_data.append(d)

    for ct_id,ylabel,xlabel,ykey,xkey,xrev,xrev2 in chart_defs:
        act=' active' if tab_idx==0 else ''
        fig=go.Figure()

        # 비율밴드 곡선
        bx=[d[xkey] for d in bd_data]
        by=[d[ykey] for d in bd_data]
        bl_labels=[f'{d["band"]*100:.0f}%' if d["band"] in lset else '' for d in bd_data]
        fig.add_trace(go.Scatter(x=bx,y=by,mode='lines+markers+text',name='비율밴드',line=dict(color='#1565C0',width=3),marker=dict(size=5),text=bl_labels,textposition='top center',textfont=dict(size=8,color='#1565C0')))

        # FA-FR 직선
        fig.add_trace(go.Scatter(x=[fa[xkey],fr[xkey]],y=[fa[ykey],fr[ykey]],mode='lines',name='FA-FR직선',line=dict(color='#BDBDBD',width=2,dash='dash')))

        # Fixed Amount / Rate
        fig.add_trace(go.Scatter(x=[fa[xkey]],y=[fa[ykey]],mode='markers+text',name='Fixed Amt',marker=dict(size=14,color='#9E9E9E',symbol='square'),text=['FA'],textposition='top left',textfont=dict(size=10)))
        fig.add_trace(go.Scatter(x=[fr[xkey]],y=[fr[ykey]],mode='markers+text',name='Fixed Rate',marker=dict(size=14,color='#4CAF50',symbol='square'),text=['FR'],textposition='bottom right',textfont=dict(size=10,color='#2E7D32')))

        # Guard 5%
        fig.add_trace(go.Scatter(x=[g5[xkey]],y=[g5[ykey]],mode='markers+text',name='Guard 5%',marker=dict(size=16,color='red',symbol='star'),text=['G5%'],textposition='bottom left',textfont=dict(size=10,color='red')))

        # Vol 조합들
        # 알파 판별: guard 5% 대비 두 축 모두 개선
        for d in vol_data:
            if ct_id=='nav_cum':
                is_alpha=d['nav']>g5['nav']+0.05 and d['cum']>g5['cum']+0.05
            elif ct_id=='tot_std':
                ft=np.interp(d['std'],sorted([dd['std'] for dd in bd_data]),[bd_data[i]['tot'] for i in np.argsort([dd['std'] for dd in bd_data])])
                is_alpha=d['tot']>ft+0.05
            else:
                ft=np.interp(d['worst_abs'],sorted([dd['worst_abs'] for dd in bd_data]),[bd_data[i]['tot'] for i in np.argsort([dd['worst_abs'] for dd in bd_data])])
                is_alpha=d['tot']>ft+0.05

            color='#FF6F00' if is_alpha else '#90A4AE'
            size=12 if is_alpha else 6
            fig.add_trace(go.Scatter(x=[d[xkey]],y=[d[ykey]],mode='markers+text',
                name=d['label'] if is_alpha else None,showlegend=bool(is_alpha),
                marker=dict(size=size,color=color,symbol='diamond',line=dict(width=1,color='black') if is_alpha else dict(width=0)),
                text=[d['label']] if is_alpha else [''],textposition='top right',textfont=dict(size=8,color='#FF6F00')))

        rev=xrev2
        lbl={'nav_cum':'기말잔액 vs 누적인출','tot_std':'총가치 vs 인출Std','tot_worst':'총가치 vs Worst삭감'}[ct_id]
        fig.update_layout(title=f'{fund} [{lbl}] WR={wr*100:.1f}%<br><sub>주황=알파(frontier 바깥), 회색=frontier 안쪽</sub>',
            xaxis_title=xlabel+(' <- 좋음' if rev else ' -> 좋음'),yaxis_title=ylabel+' -> 좋음',
            font=dict(family='Pretendard',size=12),height=620,width=950,plot_bgcolor='white',
            xaxis=dict(showgrid=True,gridcolor='#f0f0f0',autorange='reversed' if rev else None),
            yaxis=dict(showgrid=True,gridcolor='#f0f0f0'),
            legend=dict(x=0.01,y=0.01,yanchor='bottom',bgcolor='rgba(255,255,255,0.8)',font=dict(size=9)))

        chart_html=fig.to_html(include_plotlyjs='cdn' if tab_idx==0 else False,full_html=False)
        html+=f'<div class="content{act}" id="tab{tab_idx}">{chart_html}</div>'
        tab_idx+=1

html+='<script>function switchTab(idx){document.querySelectorAll(".tab").forEach(function(t,i){t.classList.toggle("active",i===idx)});document.querySelectorAll(".content").forEach(function(c,i){c.classList.toggle("active",i===idx)});}</script></body></html>'

with open('efficient_frontier_vol_ratio.html','w',encoding='utf-8') as f:
    f.write(html)
print('efficient_frontier_vol_ratio.html saved (9 tabs, 3 funds x 3 charts)')
