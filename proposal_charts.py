"""
제안서 핵심장표 생성기 v4.0
============================
판매사 본부 설득용 제안서 (12장 + 부록)
Port_9.0% / 12% 인출(1.2억 월100만) / Band +-5% / Rolling / 총가치 KPI

실행: python proposal_charts.py
"""

import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import sys, io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================================
# 상수
# ============================================================================

W0 = 1.2e8          # 초기 투자금 1.2억
MONTHLY_WD = 100e4  # 월 100만원
INIT_WR = 0.12      # 연 12%
BAND = 0.05         # +-5%
MAIN_PORT = 'Port_9.0%'
TODAY = date.today().strftime('%Y.%m.%d')
TOTAL_PAGES = 11  # 표지~마무리 (플로우차트 삭제로 12→11)

DISCLAIMER = (
    "본 분석은 과거 데이터 기반 시뮬레이션이며, 미래 수익률을 보장하지 않습니다. "
    "투자 성과는 시장 상황에 따라 달라질 수 있으며, 원금 손실이 발생할 수 있습니다. "
    "모든 수치는 명목 기준(인플레이션 미조정)이며, 시뮬레이션 기간은 10년입니다."
)

FONT_IMPORT = '@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");'
FONT_FAMILY = '"Pretendard", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif'
PLOTLY_FONT = dict(family=FONT_FAMILY, size=15)

COMMON_CSS = f"""
<style>
{FONT_IMPORT}
* {{ box-sizing: border-box; }}
body {{ font-family: {FONT_FAMILY}; margin:0; padding:0; background:#eef0f4; color:#222; font-size:15px; }}
.slide {{ width:1280px; min-height:720px; margin:20px auto; background:white;
    box-shadow:0 4px 24px rgba(0,0,0,0.10); border-radius:8px; overflow:hidden; position:relative; }}
.slide-hd {{ background:linear-gradient(135deg,#0d1b4a 0%,#283593 100%);
    color:white; padding:30px 52px 22px; }}
.slide-hd h1 {{ margin:0 0 6px; font-size:30px; font-weight:800; }}
.slide-hd .sub {{ font-size:15px; color:rgba(255,255,255,0.65); margin:0; }}
.slide-bd {{ padding:28px 52px 16px; }}
.slide-ft {{ position:absolute; bottom:0; left:0; right:0; padding:10px 52px;
    font-size:11px; color:#aaa; border-top:1px solid #eee; background:#fafafa;
    display:flex; justify-content:space-between; }}
.pn {{ font-weight:700; color:#1a237e; }}
.callout {{ background:#e8eaf6; border-left:4px solid #3949ab;
    padding:14px 20px; border-radius:0 6px 6px 0; margin:14px 0; font-size:14px; color:#1a237e; line-height:1.7; }}
.warn {{ background:#fff3e0; border-left:4px solid #f57c00;
    padding:14px 20px; border-radius:0 6px 6px 0; margin:14px 0; font-size:14px; color:#e65100; line-height:1.7; }}
.disc {{ font-size:11px; color:#aaa; line-height:1.5; margin-top:12px; padding-top:10px; border-top:1px solid #eee; }}
table.dt {{ border-collapse:collapse; width:100%; font-size:14px; }}
table.dt th {{ background:#37474F; color:white; padding:11px 16px; font-weight:700; text-align:center; }}
table.dt td {{ padding:11px 16px; text-align:center; border-bottom:1px solid #eee; }}
table.dt .rl {{ text-align:left; font-weight:600; color:#555; background:#f8f9fa; }}
</style>
"""

def slide(title, sub, body, pg):
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>{COMMON_CSS}</head><body>
<div class="slide"><div class="slide-hd"><h1>{title}</h1><p class="sub">{sub}</p></div>
<div class="slide-bd">{body}</div>
<div class="slide-ft"><span>{DISCLAIMER[:80]}...</span><span class="pn">{pg} / {TOTAL_PAGES}</span></div>
</div></body></html>"""

def plotly_slide(title, sub, fig, pg, extra=''):
    chart = fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False})
    body = f'{chart}{extra}<div class="disc">{DISCLAIMER}</div>'
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>{COMMON_CSS}</head><body>
<div class="slide" style="min-height:auto"><div class="slide-hd"><h1>{title}</h1><p class="sub">{sub}</p></div>
<div class="slide-bd" style="padding:16px 40px 12px">{body}</div>
<div class="slide-ft" style="position:relative"><span>{DISCLAIMER[:80]}...</span><span class="pn">{pg} / {TOTAL_PAGES}</span></div>
</div></body></html>"""

def fmt(v, unit=''):
    """1.2억 -> '1.2억', 100만 -> '100만'"""
    if abs(v) >= 1e8:
        return f'{v/1e8:.1f}억{unit}'
    elif abs(v) >= 1e4:
        return f'{v/1e4:,.0f}만{unit}'
    return f'{v:,.0f}{unit}'


# ============================================================================
# 데이터 로딩
# ============================================================================

def load_data():
    """Rolling 경로 + 시뮬레이션 결과 반환"""
    from withdrawal_backtest import DataPreprocessor, PORTFOLIOS
    from engine import daily_to_monthly_returns, generate_paths_rolling, simulate_withdrawal_on_path

    with open('benchmark_data.pkl', 'rb') as f:
        bm = pickle.load(f)
    prep = DataPreprocessor(bm)
    mr = daily_to_monthly_returns(prep.returns[MAIN_PORT])
    mr = mr.values if hasattr(mr, 'values') else mr

    # 월초 날짜 매핑 (경로 인덱스 -> 시작 년월)
    # month_starts는 bool Series — True인 날짜만 추출
    ms_series = prep.month_starts
    month_dates = ms_series.index[ms_series].tolist()  # list of Timestamps
    paths = generate_paths_rolling(mr, T_months=120)

    return paths, month_dates, simulate_withdrawal_on_path


# ============================================================================
# Page 0: 표지
# ============================================================================

def page0_cover():
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>표지</title>{COMMON_CSS}
<style>
.cover {{ width:1280px; height:720px; margin:20px auto;
    background:linear-gradient(135deg,#0d1b4a 0%,#1a237e 40%,#283593 100%);
    border-radius:8px; box-shadow:0 4px 24px rgba(0,0,0,0.15);
    display:flex; flex-direction:column; justify-content:center; align-items:center;
    color:white; text-align:center; position:relative; }}
.cover h1 {{ font-size:46px; font-weight:900; margin:0 0 14px; }}
.cover h2 {{ font-size:22px; font-weight:400; color:rgba(255,255,255,0.65); margin:0 0 48px; }}
.cover-div {{ width:80px; height:3px; background:#ff9800; margin:0 auto 36px; }}
.badges {{ display:flex; gap:36px; }}
.badge {{ background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12);
    border-radius:10px; padding:18px 28px; text-align:center; }}
.badge .n {{ font-size:28px; font-weight:800; color:#ff9800; }}
.badge .l {{ font-size:13px; color:rgba(255,255,255,0.55); margin-top:6px; }}
.cover-ft {{ font-size:14px; color:rgba(255,255,255,0.4); position:absolute; bottom:36px; }}
</style></head><body>
<div class="cover">
  <div style="font-size:14px;letter-spacing:5px;color:rgba(255,255,255,0.4);margin-bottom:28px">RETIREMENT WITHDRAWAL STRATEGY</div>
  <h1>Guardrail 인출 전략 제안서</h1>
  <h2>시장 상황에 연동한 스마트 인출로 연금 자산을 지킵니다</h2>
  <div class="cover-div"></div>
  <div class="badges">
    <div class="badge"><div class="n">{fmt(W0)}</div><div class="l">초기 투자금</div></div>
    <div class="badge"><div class="n">월 {fmt(MONTHLY_WD)}</div><div class="l">목표 인출액</div></div>
    <div class="badge"><div class="n">+-5%</div><div class="l">Guardrail Band</div></div>
  </div>
  <div class="cover-ft">솔루션전략부 &middot; {TODAY} &middot; 과거 데이터 Rolling 시뮬레이션</div>
</div></body></html>"""
    _write('proposal_00_cover.html', html)
    print('  [0] 표지')


# ============================================================================
# Page 1: 상품 한눈에 보기 (Mock W/NAV 밴드 차트)
# ============================================================================

def page1_overview():
    """viewer.py Tab1 밴드 차트 + 3행 차트(NAV/인출/누적인출) 충실 재현 (mock, 240개월)"""
    # === viewer.py와 동일한 mock 시뮬레이션 ===
    _np_seed = 7
    _n = 240
    init_wr = 0.05  # viewer.py 기본 init_wr (사이드바 기본값)
    beta = 0.5
    _target_ratio = init_wr / 12
    _band_pct = 0.05
    _lo_band = _target_ratio * (1 - _band_pct)
    _hi_band = _target_ratio * (1 + _band_pct)
    _nav0 = 100.0
    _w0 = _target_ratio * _nav0

    rng = np.random.RandomState(_np_seed)
    _rets = np.concatenate([
        rng.normal(0.005, 0.020, 60),   # 안정기
        rng.normal(-0.018, 0.038, 60),  # 하락기 (GFC급)
        rng.normal(0.018, 0.020, 60),   # 회복기 (V자 반등)
        rng.normal(0.013, 0.016, 60),   # 후기 랠리
    ])

    # 시장 NAV (인출 없음)
    _mkt = np.zeros(_n+1); _mkt[0] = _nav0
    for t in range(_n): _mkt[t+1] = _mkt[t] * (1 + _rets[t])

    # Fixed 인출
    _nav_f = np.zeros(_n+1); _nav_f[0] = _nav0
    _wf = np.zeros(_n+1); _wf[0] = _w0
    for t in range(_n):
        g = _nav_f[t] * (1 + _rets[t])
        if g > 0:
            _wf[t+1] = _w0; _nav_f[t+1] = max(g - _w0, 0)
        else:
            _wf[t+1] = 0; _nav_f[t+1] = 0

    # Guardrail 인출
    _nav_g = np.zeros(_n+1); _nav_g[0] = _nav0
    _wg = np.zeros(_n+1); _wg[0] = _w0
    for t in range(_n):
        g = _nav_g[t] * (1 + _rets[t])
        if g <= 0:
            _nav_g[t+1] = 0; _wg[t+1] = 0; continue
        ratio = _wg[t] / g
        if ratio > _hi_band:
            _wg[t+1] = _hi_band * g
        elif ratio < _lo_band:
            _wg[t+1] = _lo_band * g
        else:
            _wg[t+1] = _wg[t]
        _nav_g[t+1] = max(g - _wg[t+1], 0)

    _time = np.arange(_n+1)
    _cum_f = np.cumsum(_wf)
    _cum_g = np.cumsum(_wg)
    _target_line = _nav0 * beta
    _main_ruin_idx = np.where((_nav_f == 0) & (_time > 0))[0]
    _main_ruin_t = int(_main_ruin_idx[0]) if len(_main_ruin_idx) > 0 else -1

    # === 밴드 메커니즘 차트 (viewer.py 동일: seed=99, +-15%) ===
    _bm_band = 0.15
    _bm_target = _target_ratio
    _bm_hi = _bm_target * (1 + _bm_band)
    _bm_lo = _bm_target * (1 - _bm_band)
    _bm_w0 = _bm_target * _nav0

    _bm_rng = np.random.RandomState(99)
    _bm_rets = np.concatenate([
        _bm_rng.normal(0.005, 0.015, 60),
        _bm_rng.normal(-0.005, 0.020, 60),
        _bm_rng.normal(0.010, 0.015, 60),
        _bm_rng.normal(0.006, 0.012, 60),
    ])

    _bm_nav_f = np.zeros(_n+1); _bm_nav_f[0] = _nav0
    _bm_wf = np.zeros(_n+1); _bm_wf[0] = _bm_w0
    for t in range(_n):
        g = _bm_nav_f[t] * (1 + _bm_rets[t])
        if g > 0:
            _bm_wf[t+1] = _bm_w0; _bm_nav_f[t+1] = max(g - _bm_w0, 0)
        else:
            _bm_wf[t+1] = 0; _bm_nav_f[t+1] = 0

    _bm_nav_g = np.zeros(_n+1); _bm_nav_g[0] = _nav0
    _bm_wg = np.zeros(_n+1); _bm_wg[0] = _bm_w0
    for t in range(_n):
        g = _bm_nav_g[t] * (1 + _bm_rets[t])
        if g <= 0:
            _bm_nav_g[t+1] = 0; _bm_wg[t+1] = 0; continue
        ratio = _bm_wg[t] / g
        if ratio > _bm_hi:
            _bm_wg[t+1] = _bm_hi * g
        elif ratio < _bm_lo:
            _bm_wg[t+1] = _bm_lo * g
        else:
            _bm_wg[t+1] = _bm_wg[t]
        _bm_nav_g[t+1] = max(g - _bm_wg[t+1], 0)

    # 정규화
    _bm_grown_f = np.zeros(_n+1); _bm_grown_f[0] = _nav0
    _bm_grown_g = np.zeros(_n+1); _bm_grown_g[0] = _nav0
    _bm_grown_f[1:] = _bm_nav_f[1:] + _bm_wf[1:]
    _bm_grown_g[1:] = _bm_nav_g[1:] + _bm_wg[1:]
    _bm_norm_f = np.where(_bm_grown_f > 0, _bm_wf / (_bm_grown_f * _bm_target), 0)
    _bm_norm_g = np.where(_bm_grown_g > 0, _bm_wg / (_bm_grown_g * _bm_target), 0)
    if _main_ruin_t > 0:
        _bm_norm_f[_main_ruin_t:] = 0

    # === 4행 차트: 밴드 + NAV + 인출 + 누적 ===
    fig = make_subplots(rows=4, cols=1, row_heights=[0.25, 0.35, 0.15, 0.25],
        vertical_spacing=0.04, shared_xaxes=True,
        subplot_titles=['W/NAV 비율 (밴드 메커니즘)', 'NAV 잔액', '월간 인출액', '누적 인출금'])

    phases = [(0,60,'안정기','rgba(226,232,240,0.15)','#94a3b8'),
              (60,120,'하락기','rgba(254,202,202,0.30)','#f87171'),
              (120,180,'회복기','rgba(187,247,208,0.30)','#4ade80'),
              (180,_n,'후기 랠리','rgba(191,219,254,0.15)','#93c5fd')]
    for x0, x1, lbl, fc, lc in phases:
        for row in [1,2,3,4]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=fc, line_width=0, row=row, col=1)

    # Row 1: 밴드 차트
    _bm_band_hi = 1 + _bm_band
    _bm_band_lo = 1 - _bm_band
    fig.add_trace(go.Scatter(
        x=np.concatenate([_time, _time[::-1]]),
        y=np.concatenate([np.full(_n+1, _bm_band_hi), np.full(_n+1, _bm_band_lo)[::-1]]),
        fill='toself', fillcolor='rgba(34,197,94,0.25)', line=dict(width=0),
        name=f'허용 밴드 (+-{_bm_band*100:.0f}%)', showlegend=True), row=1, col=1)
    fig.add_hline(y=1.0, line_dash='dash', line_color='#94a3b8', line_width=1, row=1, col=1)
    fig.add_trace(go.Scatter(x=_time, y=_bm_norm_g, mode='lines', name='Guardrail',
        line=dict(color='#16a34a', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=_time, y=_bm_norm_f, mode='lines', name='고정 인출',
        line=dict(color='#dc2626', width=2, dash='dash')), row=1, col=1)

    # 밴드 이탈 시점
    _bm_exit_idx = np.where(_bm_norm_f > _bm_band_hi)[0]
    if len(_bm_exit_idx) > 0:
        et = int(_bm_exit_idx[0])
        fig.add_annotation(x=et, y=_bm_band_hi,
            text=f'<b>고정: 밴드 이탈</b>', showarrow=True, arrowhead=2, arrowcolor='#dc2626',
            ax=40, ay=20, font=dict(size=11, color='#dc2626'),
            bgcolor='white', bordercolor='#fecaca', borderwidth=1, row=1, col=1)

    fig.update_yaxes(range=[0.78, max(_bm_norm_f.max(), _bm_band_hi)+0.08],
        title_text='W/NAV 비율', tickformat='.2f', showgrid=False, row=1, col=1)

    # Row 2: NAV
    fig.add_trace(go.Scatter(x=_time, y=_mkt, mode='lines', name='시장 (인출 없음)',
        line=dict(color='#d1d5db', width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=_time, y=_nav_f, mode='lines', name='Fixed NAV',
        line=dict(color='#dc2626', width=2.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=_time, y=_nav_g, mode='lines', name='Guardrail NAV',
        line=dict(color='#16a34a', width=2.5), showlegend=False), row=2, col=1)
    fig.add_hline(y=_target_line, line_dash='dash', line_color='#475569', line_width=1, row=2, col=1)

    if _main_ruin_t > 0:
        fig.add_annotation(x=_main_ruin_t, y=0,
            text=f'<b>Fixed 파산 ({_main_ruin_t}개월)</b>',
            showarrow=True, arrowcolor='#dc2626', ax=-40, ay=-30,
            font=dict(size=12, color='#dc2626'), row=2, col=1)
    fig.add_annotation(x=_n, y=_nav_g[-1],
        text=f'<b>Guardrail: {_nav_g[-1]:.1f}</b>',
        showarrow=True, arrowcolor='#16a34a', ax=-50, ay=-20,
        font=dict(size=12, color='#16a34a'), row=2, col=1)

    fig.update_yaxes(title_text='NAV', showgrid=True, gridcolor='#f0f0f0', row=2, col=1)

    # Row 3: 월간 인출
    fig.add_trace(go.Scatter(x=_time, y=_wf, mode='lines',
        line=dict(color='#dc2626', width=1.2, dash='dash'),
        fill='tozeroy', fillcolor='rgba(220,38,38,0.08)', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=_time, y=_wg, mode='lines',
        line=dict(color='#16a34a', width=1.8),
        fill='tozeroy', fillcolor='rgba(22,163,74,0.12)', showlegend=False), row=3, col=1)
    fig.update_yaxes(title_text='인출', showgrid=True, gridcolor='#f0f0f0', row=3, col=1)

    # Row 4: 누적 인출
    fig.add_trace(go.Scatter(x=_time, y=_cum_f, mode='lines',
        line=dict(color='#dc2626', width=2, dash='dash'), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=_time, y=_cum_g, mode='lines',
        line=dict(color='#16a34a', width=2.5), showlegend=False), row=4, col=1)
    fig.add_annotation(x=_n, y=_cum_f[-1],
        text=f'Fixed: {_cum_f[-1]:.1f}', font=dict(size=11, color='#dc2626'),
        showarrow=False, xanchor='right', yshift=10, row=4, col=1)
    fig.add_annotation(x=_n, y=_cum_g[-1],
        text=f'Guard: {_cum_g[-1]:.1f}', font=dict(size=11, color='#16a34a'),
        showarrow=False, xanchor='right', yshift=-10, row=4, col=1)
    fig.update_yaxes(title_text='누적 인출', showgrid=True, gridcolor='#f0f0f0', row=4, col=1)

    fig.update_xaxes(title_text='경과 월수', row=4, col=1)

    fig.update_layout(font=PLOTLY_FONT,
        legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='center', x=0.5, font=dict(size=13)),
        height=720, width=1180, margin=dict(t=60, b=30, l=60, r=60),
        plot_bgcolor='white')

    extra = '''<div class="callout"><b>핵심:</b> Guardrail은 인출/NAV 비율을 밴드 안에 유지합니다. Fixed는 하락장에서 비율 폭등 → 자본 급감 → 파산.
    Guardrail은 인출을 조절하여 자본을 보존하고, 회복기에 복리 효과를 극대화합니다.
    <span style="font-size:12px;color:#888">(설명용 가상 시나리오, beta=50%, 초기인출률 5% 기준 — 실제 백테스트는 후속 장표 참조)</span></div>'''

    out = plotly_slide('상품 한눈에 보기', 'Guardrail: 시장에 연동하여 인출을 자동 조절하는 스마트 인출 전략', fig, 1, extra)
    _write('proposal_01_overview.html', out)
    print('  [1] 상품 한눈에 보기')


# ============================================================================
# Page 2: 투자 포인트 1 (상품 관점)
# ============================================================================

def page2_invest_point1():
    body = f"""
<div style="display:flex;gap:24px;margin-bottom:20px">
  <div style="flex:1;background:linear-gradient(135deg,#e3f2fd,#bbdefb);border-radius:12px;padding:28px;text-align:center">
    <div style="font-size:48px;margin-bottom:12px">&#x1F6E1;</div>
    <h3 style="font-size:20px;color:#0d47a1;margin:0 0 12px">하락장 인출 제한</h3>
    <p style="font-size:15px;color:#333;line-height:1.7;margin:0">
      시장 하락 시 인출액을 <b>자동 축소</b>하여<br>
      운용 자금의 <b>시장 exposure를 최대한 유지</b><br>
      <span style="font-size:13px;color:#666">복리 효과 보존 → 회복 시 자산 극대화</span>
    </p>
  </div>
  <div style="flex:1;background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:12px;padding:28px;text-align:center">
    <div style="font-size:48px;margin-bottom:12px">&#x1F4C8;</div>
    <h3 style="font-size:20px;color:#1b5e20;margin:0 0 12px">상승장 인출 확대</h3>
    <p style="font-size:15px;color:#333;line-height:1.7;margin:0">
      시장 상승 시 인출액을 <b>자동 확대</b>하여<br>
      수급자도 <b>좋은 시기에 더 많이 수령</b><br>
      <span style="font-size:13px;color:#666">양방향 조절 → 수급자 편익 반영</span>
    </p>
  </div>
  <div style="flex:1;background:linear-gradient(135deg,#fce4ec,#f8bbd0);border-radius:12px;padding:28px;text-align:center">
    <div style="font-size:48px;margin-bottom:12px">&#x1F6AB;</div>
    <h3 style="font-size:20px;color:#b71c1c;margin:0 0 12px">연금 조기 소진 차단</h3>
    <p style="font-size:15px;color:#333;line-height:1.7;margin:0">
      인출 자동 조절로 <b>연금 조기 파산을 원천 방지</b><br>
      고정 인출 대비 <b>운용 기간 대폭 연장</b><br>
      <span style="font-size:13px;color:#666">백테스트 기준 파산 확률 0%</span>
    </p>
  </div>
</div>

<div style="background:#f5f5f5;border-radius:10px;padding:24px 32px;margin-top:8px">
  <div style="display:flex;align-items:center;gap:20px">
    <div style="flex:0 0 auto;text-align:center">
      <div style="font-size:16px;color:#555">하락장</div>
      <div style="font-size:32px;font-weight:800;color:#EF5350">인출 축소</div>
    </div>
    <div style="font-size:36px;color:#ccc">&#x27A1;</div>
    <div style="flex:0 0 auto;text-align:center">
      <div style="font-size:16px;color:#555">운용자금 보존</div>
      <div style="font-size:32px;font-weight:800;color:#1565C0">Exposure 유지</div>
    </div>
    <div style="font-size:36px;color:#ccc">&#x27A1;</div>
    <div style="flex:0 0 auto;text-align:center">
      <div style="font-size:16px;color:#555">회복 시</div>
      <div style="font-size:32px;font-weight:800;color:#2E7D32">복리 효과 극대화</div>
    </div>
    <div style="font-size:36px;color:#ccc">&#x27A1;</div>
    <div style="flex:0 0 auto;text-align:center">
      <div style="font-size:16px;color:#555">결과</div>
      <div style="font-size:32px;font-weight:800;color:#FF6F00">총가치 극대화</div>
    </div>
  </div>
</div>
<div class="disc">{DISCLAIMER}</div>"""

    out = slide('투자 포인트 1', 'Guardrail 인출 전략의 3가지 핵심 가치', body, 2)
    _write('proposal_02_invest_point1.html', out)
    print('  [2] 투자 포인트 1')


# ============================================================================
# Page 3: 투자 포인트 2 (판매사 관점 — AUM/보수)
# ============================================================================

def page3_invest_point2():
    body = f"""
<div style="display:flex;gap:28px;margin-bottom:20px">
  <div style="flex:1">
    <div style="background:linear-gradient(135deg,#1a237e,#283593);border-radius:12px;padding:32px;color:white;text-align:center;margin-bottom:20px">
      <div style="font-size:16px;color:rgba(255,255,255,0.6);margin-bottom:8px">Fixed 인출 전략</div>
      <div style="font-size:42px;font-weight:900;color:#EF5350">조기 파산 위험</div>
      <div style="font-size:15px;color:rgba(255,255,255,0.7);margin-top:12px">
        고정 인출 → 하락장에서 자본 급감 → 파산<br>
        <b>AUM 소멸 → 판매보수 중단</b>
      </div>
    </div>
    <div style="background:#ffebee;border-radius:10px;padding:20px;text-align:center">
      <div style="font-size:15px;color:#c62828;font-weight:700">백테스트 결과 (12% 인출)</div>
      <div style="font-size:14px;color:#555;margin-top:8px">파산 경로 존재 → 운용 기간 단축</div>
    </div>
  </div>
  <div style="flex:0 0 60px;display:flex;align-items:center;justify-content:center">
    <div style="font-size:48px;color:#ccc">VS</div>
  </div>
  <div style="flex:1">
    <div style="background:linear-gradient(135deg,#1a237e,#283593);border-radius:12px;padding:32px;color:white;text-align:center;margin-bottom:20px">
      <div style="font-size:16px;color:rgba(255,255,255,0.6);margin-bottom:8px">Guardrail 인출 전략</div>
      <div style="font-size:42px;font-weight:900;color:#4CAF50">파산 0%</div>
      <div style="font-size:15px;color:rgba(255,255,255,0.7);margin-top:12px">
        자동 인출 조절 → 자본 보존 → 운용 지속<br>
        <b>AUM 장기 유지 → 판매보수 지속 수취</b>
      </div>
    </div>
    <div style="background:#e8f5e9;border-radius:10px;padding:20px;text-align:center">
      <div style="font-size:15px;color:#2E7D32;font-weight:700">백테스트 결과 (12% 인출)</div>
      <div style="font-size:14px;color:#555;margin-top:8px">전 구간 파산 0% → 운용 기간 10년 완주</div>
    </div>
  </div>
</div>

<div class="callout">
  <b>판매사 관점 핵심:</b> Guardrail 전략은 고객 자산의 조기 소진을 방지하여
  <b>AUM을 장기간 유지</b>합니다. 이는 판매사의 판매보수 수취 기간 연장으로 직결됩니다.
  고객 보호와 판매사 수익이 동일 방향으로 정렬되는 구조입니다.
</div>
<div class="disc">{DISCLAIMER}</div>"""

    out = slide('투자 포인트 2 : 판매사 관점', '고객 보호 = AUM 유지 = 판매보수 장기화', body, 3)
    _write('proposal_03_invest_point2.html', out)
    print('  [3] 투자 포인트 2 (판매사)')


# ============================================================================
# Page 4: 인출 메커니즘 플로우차트
# ============================================================================

def page4_flowchart():
    # 예시: 1.2억, 월100만, band+-5%
    # target_ratio = 12% / 12 = 1% (월)
    # upper = 1% * 1.05 = 1.05%
    # lower = 1% * 0.95 = 0.95%
    nav_normal = 1.0e8   # 1억 (NAV 약간 하락)
    nav_drop = 0.8e8     # 8천만 (NAV 큰 하락)
    nav_rise = 1.5e8     # 1.5억 (NAV 상승)
    prev_wd = MONTHLY_WD  # 100만

    def calc_case(nav, prev):
        ratio = prev / nav
        target = INIT_WR / 12
        upper = target * (1 + BAND)
        lower = target * (1 - BAND)
        if ratio > upper:
            return upper * nav, '상한 조정', 'red'
        elif ratio < lower:
            return lower * nav, '하한 조정', 'blue'
        else:
            return prev, '밴드 내 (유지)', 'green'

    wd1, label1, c1 = calc_case(nav_normal, prev_wd)
    wd2, label2, c2 = calc_case(nav_drop, prev_wd)
    wd3, label3, c3 = calc_case(nav_rise, prev_wd)

    body = f"""
<div style="max-width:1100px;margin:0 auto">

<div style="background:#f8f9fa;border-radius:10px;padding:24px 32px;margin-bottom:20px">
<h3 style="font-size:18px;color:#1a237e;margin:0 0 16px">인출 판단 흐름</h3>
<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:15px">
  <div style="background:#1565C0;color:white;padding:10px 18px;border-radius:8px;font-weight:700">매월 수익률 적용</div>
  <div style="font-size:24px;color:#ccc">&#x27A1;</div>
  <div style="background:#37474F;color:white;padding:10px 18px;border-radius:8px;font-weight:700">전월 인출액으로 인출 시도</div>
  <div style="font-size:24px;color:#ccc">&#x27A1;</div>
  <div style="background:#FF6F00;color:white;padding:10px 18px;border-radius:8px;font-weight:700">인출액/NAV 비율 체크</div>
  <div style="font-size:24px;color:#ccc">&#x27A1;</div>
  <div style="background:#2E7D32;color:white;padding:10px 18px;border-radius:8px;font-weight:700">밴드 내면 유지<br>밴드 밖이면 조정</div>
</div>
</div>

<div style="display:flex;gap:16px">
  <div style="flex:1;border:2px solid #4CAF50;border-radius:10px;padding:20px">
    <h4 style="color:#2E7D32;margin:0 0 12px;font-size:16px">Case 1: 밴드 내 (인출 유지)</h4>
    <table style="width:100%;font-size:14px;border-collapse:collapse">
      <tr><td style="padding:4px 0;color:#777">NAV</td><td style="text-align:right;font-weight:700">{fmt(nav_normal)}원</td></tr>
      <tr><td style="padding:4px 0;color:#777">전월 인출액</td><td style="text-align:right">{fmt(prev_wd)}원</td></tr>
      <tr><td style="padding:4px 0;color:#777">인출/NAV 비율</td><td style="text-align:right">{prev_wd/nav_normal:.2%}</td></tr>
      <tr><td style="padding:4px 0;color:#777">밴드 범위</td><td style="text-align:right">{INIT_WR/12*(1-BAND):.4%} ~ {INIT_WR/12*(1+BAND):.4%}</td></tr>
      <tr style="border-top:2px solid #4CAF50"><td style="padding:8px 0;color:#2E7D32;font-weight:800">결과: 인출액</td>
          <td style="text-align:right;font-weight:800;color:#2E7D32;font-size:18px">{fmt(wd1)}원</td></tr>
    </table>
  </div>

  <div style="flex:1;border:2px solid #EF5350;border-radius:10px;padding:20px">
    <h4 style="color:#c62828;margin:0 0 12px;font-size:16px">Case 2: 상한 초과 (인출 축소)</h4>
    <table style="width:100%;font-size:14px;border-collapse:collapse">
      <tr><td style="padding:4px 0;color:#777">NAV</td><td style="text-align:right;font-weight:700">{fmt(nav_drop)}원 <span style="color:red">&#x25BC;33%</span></td></tr>
      <tr><td style="padding:4px 0;color:#777">전월 인출액</td><td style="text-align:right">{fmt(prev_wd)}원</td></tr>
      <tr><td style="padding:4px 0;color:#777">인출/NAV 비율</td><td style="text-align:right;color:red;font-weight:700">{prev_wd/nav_drop:.2%} (상한 초과)</td></tr>
      <tr><td style="padding:4px 0;color:#777">밴드 상한</td><td style="text-align:right">{INIT_WR/12*(1+BAND):.4%}</td></tr>
      <tr style="border-top:2px solid #EF5350"><td style="padding:8px 0;color:#c62828;font-weight:800">결과: 인출액</td>
          <td style="text-align:right;font-weight:800;color:#c62828;font-size:18px">{fmt(wd2)}원 &#x25BC;</td></tr>
    </table>
  </div>

  <div style="flex:1;border:2px solid #1565C0;border-radius:10px;padding:20px">
    <h4 style="color:#0d47a1;margin:0 0 12px;font-size:16px">Case 3: 하한 미달 (인출 확대)</h4>
    <table style="width:100%;font-size:14px;border-collapse:collapse">
      <tr><td style="padding:4px 0;color:#777">NAV</td><td style="text-align:right;font-weight:700">{fmt(nav_rise)}원 <span style="color:green">&#x25B2;25%</span></td></tr>
      <tr><td style="padding:4px 0;color:#777">전월 인출액</td><td style="text-align:right">{fmt(prev_wd)}원</td></tr>
      <tr><td style="padding:4px 0;color:#777">인출/NAV 비율</td><td style="text-align:right;color:#1565C0;font-weight:700">{prev_wd/nav_rise:.2%} (하한 미달)</td></tr>
      <tr><td style="padding:4px 0;color:#777">밴드 하한</td><td style="text-align:right">{INIT_WR/12*(1-BAND):.4%}</td></tr>
      <tr style="border-top:2px solid #1565C0"><td style="padding:8px 0;color:#0d47a1;font-weight:800">결과: 인출액</td>
          <td style="text-align:right;font-weight:800;color:#0d47a1;font-size:18px">{fmt(wd3)}원 &#x25B2;</td></tr>
    </table>
  </div>
</div>

<div class="warn" style="margin-top:16px">
  <b>유의:</b> 목표 인출률 연 12% (월 {fmt(MONTHLY_WD)}원) 기준, 시장 상황에 따라 실제 인출액이 변동합니다.
  과거 백테스트 평균 실효인출률은 약 11.4% 입니다.
</div>
</div>
<div class="disc">{DISCLAIMER}</div>"""

    out = slide('인출 메커니즘', f'초기 {fmt(W0)}원 / 목표 월 {fmt(MONTHLY_WD)}원 / Guardrail Band +-5%', body, 4)
    _write('proposal_04_flowchart.html', out)
    print('  [4] 플로우차트')


# ============================================================================
# Page 5: 메커니즘 시각화 (Rolling 실제 경로)
# ============================================================================

def page5_mechanism(paths, month_starts, sim_func):
    """Rolling 경로 중 대표 경로로 Fixed vs Guardrail 비교"""
    # 최악 경로 선택 (Fixed에서 가장 낮은 기말잔액)
    worst_idx = 0
    worst_nav = float('inf')
    for i, p in enumerate(paths):
        res = sim_func(p, init_wr=INIT_WR, band=99.0, beta=0.25, W0=100.0)
        if res['terminal_nav'] < worst_nav:
            worst_nav = res['terminal_nav']
            worst_idx = i

    path = paths[worst_idx]
    res_f = sim_func(path, init_wr=INIT_WR, band=99.0, beta=0.25, W0=W0)
    res_g = sim_func(path, init_wr=INIT_WR, band=BAND, beta=0.25, W0=W0)

    start_dt = month_starts[worst_idx] if worst_idx < len(month_starts) else None
    start_label = f'{start_dt.year}년 {start_dt.month}월' if start_dt else f'경로 #{worst_idx}'

    months = np.arange(len(res_f['W_series']))
    years = months / 12

    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
        shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[f'NAV 잔액 추이 (시작: {start_label})', '월간 인출액'])

    CM = 1e7  # 천만원 단위
    fig.add_trace(go.Scatter(x=years, y=res_f['W_series']/CM, name='Fixed',
        line=dict(color='#EF5350', width=2.5, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=years, y=res_g['W_series']/CM, name='Guardrail',
        line=dict(color='#1565C0', width=3)), row=1, col=1)

    wd_months = np.arange(1, len(res_f['withdraw_series'])+1) / 12
    fig.add_trace(go.Scatter(x=wd_months, y=res_f['withdraw_series']/CM, name='Fixed 인출',
        line=dict(color='#EF5350', width=1.5, dash='dash'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=wd_months, y=res_g['withdraw_series']/CM, name='Guardrail 인출',
        line=dict(color='#1565C0', width=2), showlegend=False), row=2, col=1)

    # 총가치 annotation
    tv_f = res_f['terminal_nav'] + res_f['cum_withdraw']
    tv_g = res_g['terminal_nav'] + res_g['cum_withdraw']

    fig.add_annotation(x=0.98, y=0.95, xref='paper', yref='paper',
        text=f'<b>총가치 (잔액+누적인출)</b><br>Fixed: {fmt(tv_f)}원<br>Guardrail: {fmt(tv_g)}원',
        font=dict(size=14), showarrow=False, bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#ddd', borderwidth=1, borderpad=8)

    fig.update_layout(font=PLOTLY_FONT,
        legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5, font=dict(size=14)),
        height=520, width=1180, margin=dict(t=70, b=30, l=70, r=30), plot_bgcolor='white')
    fig.update_xaxes(title_text='경과 연수', ticksuffix='년', row=2, col=1)
    fig.update_yaxes(title_text='천만원', showgrid=True, gridcolor='#f0f0f0')

    extra = f'<div class="callout"><b>실제 과거 데이터 Rolling 시뮬레이션</b> (가장 불리한 시작 시점: {start_label}). Fixed는 하락기 동일 인출로 자본 급감, Guardrail은 인출 조절로 자본 보존.</div>'
    out = plotly_slide('메커니즘 시각화: 실제 데이터', f'{MAIN_PORT} / 인출률 12% / Band +-5% / Rolling', fig, 4, extra)
    _write('proposal_04_mechanism.html', out)
    print('  [4] 메커니즘 시각화')


# ============================================================================
# Page 6: 백테스트 전체 (스파게티 + 총가치)
# ============================================================================

def page6_backtest_all(paths, sim_func):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
        subplot_titles=['NAV 경로 (Fixed vs Guardrail)', '총가치 분포 (잔액 + 누적인출)'],
        horizontal_spacing=0.1)

    tv_fixed, tv_guard = [], []
    n_show = min(60, len(paths))
    indices = np.linspace(0, len(paths)-1, n_show, dtype=int)

    for idx in indices:
        res_f = sim_func(paths[idx], init_wr=INIT_WR, band=99.0, beta=0.25, W0=W0)
        res_g = sim_func(paths[idx], init_wr=INIT_WR, band=BAND, beta=0.25, W0=W0)
        years = np.arange(len(res_f['W_series'])) / 12

        CM = 1e7
        fig.add_trace(go.Scatter(x=years, y=res_f['W_series']/CM, mode='lines',
            line=dict(color='#EF5350', width=0.5), opacity=0.3, showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=years, y=res_g['W_series']/CM, mode='lines',
            line=dict(color='#1565C0', width=0.5), opacity=0.3, showlegend=False, hoverinfo='skip'), row=1, col=1)

    # 전체 총가치 계산
    CM = 1e7
    for p in paths:
        rf = sim_func(p, init_wr=INIT_WR, band=99.0, beta=0.25, W0=W0)
        rg = sim_func(p, init_wr=INIT_WR, band=BAND, beta=0.25, W0=W0)
        tv_fixed.append((rf['terminal_nav'] + rf['cum_withdraw']) / CM)
        tv_guard.append((rg['terminal_nav'] + rg['cum_withdraw']) / CM)

    fig.add_trace(go.Box(y=tv_fixed, name='Fixed', marker_color='#EF5350', boxmean=True), row=1, col=2)
    fig.add_trace(go.Box(y=tv_guard, name='Guardrail', marker_color='#1565C0', boxmean=True), row=1, col=2)

    fig.update_layout(font=PLOTLY_FONT, height=430, width=1180,
        margin=dict(t=50, b=40, l=60, r=20), plot_bgcolor='white', showlegend=False)
    fig.update_xaxes(title_text='경과 연수', ticksuffix='년', row=1, col=1)
    fig.update_yaxes(title_text='천만원', showgrid=True, gridcolor='#f0f0f0')

    med_f = np.median(tv_fixed)
    med_g = np.median(tv_guard)
    extra = f'<div class="callout">전체 {len(paths)}개 Rolling 경로. 총가치 중앙값: Fixed <b>{med_f:,.1f}천만원</b> vs Guardrail <b>{med_g:,.1f}천만원</b> (Guardrail이 {(med_g-med_f)/med_f*100:+.1f}% 우위)</div>'
    out = plotly_slide('백테스트: 전체 경로 분석', f'{MAIN_PORT} / {len(paths)}개 Rolling 경로 / 12% 인출', fig, 6, extra)
    _write('proposal_06_backtest_all.html', out)
    print('  [6] 백테스트 전체')


# ============================================================================
# Page 7-9: 시대별 백테스트 (단일 경로 스토리텔링)
# ============================================================================

def page_backtest_era(paths, month_starts, sim_func, target_year, end_year, page_num, era_label):
    """특정 시작년도에 해당하는 rolling 경로로 단일 스토리텔링 + 시장 수익률 보조축"""
    best_idx = 0
    best_diff = float('inf')
    for i, dt in enumerate(month_starts):
        if i >= len(paths):
            break
        diff = abs(dt.year - target_year) * 12 + abs(dt.month - 1)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    if best_idx >= len(paths):
        best_idx = len(paths) - 1

    path = paths[best_idx]
    start_dt = month_starts[best_idx]
    start_label = f'{start_dt.year}년 {start_dt.month}월'
    CM = 1e7  # 천만원

    res_f = sim_func(path, init_wr=INIT_WR, band=99.0, beta=0.25, W0=W0)
    res_g = sim_func(path, init_wr=INIT_WR, band=BAND, beta=0.25, W0=W0)

    # 시장 NAV (인출 없음) 계산
    mkt_nav = np.zeros(len(path) + 1)
    mkt_nav[0] = W0
    for t in range(len(path)):
        mkt_nav[t+1] = mkt_nav[t] * (1 + path[t])

    actual_years = [start_dt.year + t/12 for t in range(len(res_f['W_series']))]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
        shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[f'NAV 잔액 ({start_dt.year}~{end_year})', '월간 인출액'],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

    # Fixed / Guardrail NAV (주축, 천만원)
    fig.add_trace(go.Scatter(x=actual_years, y=res_f['W_series']/CM, name='Fixed',
        line=dict(color='#EF5350', width=2.5, dash='dash')), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=actual_years, y=res_g['W_series']/CM, name='Guardrail',
        line=dict(color='#1565C0', width=3)), row=1, col=1, secondary_y=False)

    # 시장 기준가 (보조축, 1000 기준)
    mkt_price = mkt_nav / mkt_nav[0] * 1000
    fig.add_trace(go.Scatter(x=actual_years, y=mkt_price, name='기준가 (인출 없음)',
        line=dict(color='#bdbdbd', width=2), opacity=0.7), row=1, col=1, secondary_y=True)

    # 인출액 (백만원 단위)
    BM = 1e6  # 백만원
    wd_years = [start_dt.year + m/12 for m in range(1, len(res_f['withdraw_series'])+1)]
    fig.add_trace(go.Scatter(x=wd_years, y=res_f['withdraw_series']/BM, showlegend=False,
        line=dict(color='#EF5350', width=1.5, dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=wd_years, y=res_g['withdraw_series']/BM, showlegend=False,
        line=dict(color='#1565C0', width=2)), row=2, col=1)

    tv_f = (res_f['terminal_nav'] + res_f['cum_withdraw']) / CM
    tv_g = (res_g['terminal_nav'] + res_g['cum_withdraw']) / CM
    nav_f_end = res_f['terminal_nav'] / CM
    nav_g_end = res_g['terminal_nav'] / CM
    cum_f = res_f['cum_withdraw'] / CM
    cum_g = res_g['cum_withdraw'] / CM

    # 10년 후 결과는 차트 하단 extra에 표시

    fig.update_layout(font=PLOTLY_FONT,
        legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5, font=dict(size=14)),
        height=500, width=1180, margin=dict(t=70, b=30, l=70, r=80), plot_bgcolor='white')
    fig.update_yaxes(title_text='천만원', showgrid=True, gridcolor='#f0f0f0', row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text='기준가 (1,000 기준)', showgrid=False, row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text='백만원', showgrid=True, gridcolor='#f0f0f0', row=2, col=1)

    diff_abs = (tv_g - tv_f) * CM  # 원 단위 차이
    diff_pct = (tv_g - tv_f) / tv_f * 100 if tv_f > 0 else 0

    extra = f'''<div class="callout"><b>{era_label}</b> 기간 실제 데이터 기반. 초기 {fmt(W0)}원 투자, 월 {fmt(MONTHLY_WD)}원 목표 인출. 회색선 = 인출 없이 운용했을 때의 기준가 (1,000 기준, 보조축).</div>
    <div style="border:1px solid #e0e0e0;border-radius:10px;overflow:hidden;margin-top:8px">
      <table style="width:100%;border-collapse:collapse;font-size:15px;text-align:center">
        <tr style="background:#f5f5f5">
          <th style="padding:10px 16px;font-weight:700;text-align:left;width:160px">10년 후 결과</th>
          <th style="padding:10px 16px">기말 잔액</th>
          <th style="padding:10px 16px">누적 인출금</th>
          <th style="padding:10px 16px;font-weight:800">총가치 (잔액+인출)</th>
        </tr>
        <tr style="border-bottom:1px solid #eee">
          <td style="text-align:left;padding:10px 16px;color:#c62828;font-weight:700">Fixed (고정 인출)</td>
          <td style="padding:10px 16px">{nav_f_end:,.1f}천만</td>
          <td style="padding:10px 16px">{cum_f:,.1f}천만</td>
          <td style="padding:10px 16px;font-weight:800;color:#c62828;font-size:18px">{tv_f:,.1f}천만</td>
        </tr>
        <tr style="border-bottom:1px solid #eee">
          <td style="text-align:left;padding:10px 16px;color:#1565C0;font-weight:700">Guardrail</td>
          <td style="padding:10px 16px">{nav_g_end:,.1f}천만</td>
          <td style="padding:10px 16px">{cum_g:,.1f}천만</td>
          <td style="padding:10px 16px;font-weight:800;color:#1565C0;font-size:18px">{tv_g:,.1f}천만</td>
        </tr>
        <tr style="background:#e8f5e9">
          <td style="text-align:left;padding:10px 16px;color:#2E7D32;font-weight:800">Guardrail 효과</td>
          <td style="padding:10px 16px;color:#2E7D32;font-weight:700">{(nav_g_end-nav_f_end):+,.1f}천만</td>
          <td style="padding:10px 16px;color:#2E7D32;font-weight:700">{(cum_g-cum_f):+,.1f}천만</td>
          <td style="padding:10px 16px;font-weight:900;color:#2E7D32;font-size:18px">{(tv_g-tv_f):+,.1f}천만 ({diff_pct:+.1f}%)</td>
        </tr>
      </table>
    </div>'''
    out = plotly_slide(f'백테스트: {start_dt.year}~{end_year}',
        f'{era_label} / {MAIN_PORT} / 12% 인출 / Band +-5%', fig, page_num, extra)
    _write(f'proposal_{page_num:02d}_backtest_{target_year}.html', out)
    print(f'  [{page_num}] 백테스트 {target_year}~{end_year}')


# ============================================================================
# Page 10: 상품 파라미터
# ============================================================================

def page10_params():
    body = f"""
<table class="dt">
<tr><th style="text-align:left;width:250px">항목</th><th>내용</th></tr>
<tr><td class="rl">대상 포트폴리오</td><td>{MAIN_PORT} (목표수익률 9.0% / 변동성 8.36%)</td></tr>
<tr><td class="rl">자산배분</td><td>미국성장주 42.67% / 한국국고채10년 27.02% / 금 26.29% / 한국주식 3.88% / 신흥국달러채권 0.14%</td></tr>
<tr><td class="rl">인출 전략</td><td>Guardrail +-5% Band (W/NAV 비율 기준)</td></tr>
<tr><td class="rl">목표 인출률</td><td>연 12% (초기 {fmt(W0)}원 기준 월 {fmt(MONTHLY_WD)}원)</td></tr>
<tr><td class="rl">시뮬레이션 방법</td><td>과거 데이터 Rolling (2001~2025, 총 {181}개 경로)</td></tr>
<tr><td class="rl">시뮬레이션 기간</td><td>120개월 (10년)*</td></tr>
<tr><td class="rl">실효 인출률</td><td>백테스트 평균 약 11.4% (목표 12% 대비 시장 상황에 따라 변동)</td></tr>
<tr><td class="rl">성과 지표</td><td>총가치 (기말잔액 + 누적인출금 합산)</td></tr>
</table>

<div class="warn" style="margin-top:20px">
  <b>* 시뮬레이션 기간 유의사항:</b> 본 분석은 10년(120개월) 기준입니다.
  실제 퇴직연금 인출 기간(20~30년)에 적용 시 추천 인출률의 하향 조정이 필요할 수 있습니다.
  장기 시뮬레이션 결과는 별도 협의 시 제공 가능합니다.
</div>

<div class="callout">
  <b>포트폴리오 선택 근거:</b> {MAIN_PORT}는 목표수익률 9%, 변동성 8.36%의 공격적 포트폴리오로,
  Guardrail 효과가 가장 극적으로 나타나는 구간입니다.
  보수적/중립적 위험등급(5%/7%) 결과는 부록에서 확인 가능합니다.
</div>
<div class="disc">{DISCLAIMER}</div>"""

    out = slide('상품 파라미터', '시뮬레이션 전제 조건 및 파라미터', body, 5)
    _write('proposal_05_params.html', out)
    print('  [5] 상품 파라미터')


# ============================================================================
# Page 11: 마무리 + 면책
# ============================================================================

def page11_closing(paths, sim_func):
    # 핵심 수치 계산
    tv_fixed, tv_guard = [], []
    n_ruin_f = 0
    for p in paths:
        rf = sim_func(p, init_wr=INIT_WR, band=99.0, beta=0.25, W0=W0)
        rg = sim_func(p, init_wr=INIT_WR, band=BAND, beta=0.25, W0=W0)
        tv_fixed.append(rf['terminal_nav'] + rf['cum_withdraw'])
        tv_guard.append(rg['terminal_nav'] + rg['cum_withdraw'])
        if rf['ruin_flag']:
            n_ruin_f += 1

    med_f = np.median(tv_fixed)
    med_g = np.median(tv_guard)
    pct_better = sum(1 for f, g in zip(tv_fixed, tv_guard) if g > f) / len(paths) * 100

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>마무리</title>{COMMON_CSS}
<style>
.cls {{ width:1280px; min-height:720px; margin:20px auto; background:white; border-radius:8px;
    box-shadow:0 4px 24px rgba(0,0,0,0.10); overflow:hidden; }}
.cls-hd {{ background:linear-gradient(135deg,#0d1b4a 0%,#283593 100%);
    color:white; padding:36px 52px 28px; text-align:center; }}
.cls-hd h1 {{ font-size:34px; margin:0 0 10px; font-weight:900; }}
.cls-bd {{ padding:28px 52px; }}
.metric-row {{ display:flex; gap:20px; margin-bottom:24px; }}
.metric {{ flex:1; border-radius:12px; padding:24px; text-align:center; }}
.metric .val {{ font-size:36px; font-weight:900; }}
.metric .lbl {{ font-size:14px; color:#666; margin-top:6px; }}
.disc-full {{ background:#fff8e1; border:1px solid #ffe082; border-radius:8px;
    padding:16px 24px; font-size:12px; color:#795548; line-height:1.7; margin-top:16px; }}
.cls-ft {{ text-align:center; padding:14px; font-size:12px; color:#aaa; border-top:1px solid #eee; }}
</style></head><body>
<div class="cls">
  <div class="cls-hd"><h1>분석 결과 요약</h1><p style="font-size:16px;color:rgba(255,255,255,0.6)">{MAIN_PORT} / 12% 인출 / Band +-5% / {TODAY}</p></div>
  <div class="cls-bd">
    <div class="metric-row">
      <div class="metric" style="background:#e3f2fd"><div class="val" style="color:#1565C0">{fmt(med_g)}원</div><div class="lbl">Guardrail 총가치 (중앙값)</div></div>
      <div class="metric" style="background:#ffebee"><div class="val" style="color:#EF5350">{fmt(med_f)}원</div><div class="lbl">Fixed 총가치 (중앙값)</div></div>
      <div class="metric" style="background:#e8f5e9"><div class="val" style="color:#2E7D32">{(med_g-med_f)/med_f*100:+.1f}%</div><div class="lbl">Guardrail 우위</div></div>
    </div>
    <div class="metric-row">
      <div class="metric" style="background:#e8f5e9"><div class="val" style="color:#2E7D32">0%</div><div class="lbl">Guardrail 파산률</div></div>
      <div class="metric" style="background:#ffebee"><div class="val" style="color:#c62828">{n_ruin_f/len(paths)*100:.1f}%</div><div class="lbl">Fixed 파산률</div></div>
      <div class="metric" style="background:#e3f2fd"><div class="val" style="color:#1565C0">{pct_better:.0f}%</div><div class="lbl">Guardrail이 우위인 경로 비율</div></div>
    </div>

    <div class="disc-full">
      <b>면책 조항 (Disclaimer)</b><br>
      {DISCLAIMER}
      본 자료는 정보 제공 목적이며, 특정 금융상품의 매수/매도를 권유하지 않습니다.
      투자 의사결정은 투자자 본인의 판단과 책임 하에 이루어져야 합니다.
      과거의 운용실적이 미래의 수익을 보장하지 않습니다.
      시뮬레이션 기간은 10년이며, 실제 퇴직연금 인출 기간(20~30년)에 적용 시 결과가 달라질 수 있습니다.
    </div>
  </div>
  <div class="cls-ft">솔루션전략부 / {TODAY} / 10 / {TOTAL_PAGES}</div>
</div></body></html>"""
    _write('proposal_10_closing.html', html)
    print('  [10] 마무리')


# ============================================================================
# 부록: 고객용 핵심장표
# ============================================================================

def appendix_customer():
    body = f"""
<div style="text-align:center;margin-bottom:24px">
  <div style="font-size:18px;color:#1a237e;font-weight:700;margin-bottom:8px">고객 설명용 핵심 장표 (2장)</div>
  <div style="font-size:14px;color:#888">판매사 RM이 고객에게 설명 시 활용하는 장표입니다.</div>
</div>

<div style="display:flex;gap:20px;margin-bottom:20px">
  <div style="flex:1;border:1px solid #ddd;border-radius:10px;padding:24px">
    <h3 style="color:#1a237e;margin:0 0 12px;font-size:18px">장표 1: 투자 포인트</h3>
    <p style="font-size:14px;color:#555;line-height:1.7">
      &bull; 하락장 인출 제한 → 운용자금 보존<br>
      &bull; 상승장 인출 확대 → 수급자 편익<br>
      &bull; 연금 조기 소진 방지 → 안전장치<br><br>
      <span style="color:#888;font-size:13px">→ proposal_02_invest_point1.html 참조</span>
    </p>
  </div>
  <div style="flex:1;border:1px solid #ddd;border-radius:10px;padding:24px">
    <h3 style="color:#1a237e;margin:0 0 12px;font-size:18px">장표 2: 백테스트 결과</h3>
    <p style="font-size:14px;color:#555;line-height:1.7">
      &bull; 과거 데이터 기반 실제 시뮬레이션 결과<br>
      &bull; Fixed vs Guardrail 총가치 비교<br>
      &bull; 시대별 대표 사례 (금융위기/코로나 등)<br><br>
      <span style="color:#888;font-size:13px">→ proposal_06 또는 07~09 중 택1 참조</span>
    </p>
  </div>
</div>

<div class="warn">
  <b>고객용 장표 유의사항:</b><br>
  &bull; AUM/보수 관련 내용 (투자 포인트 2) 은 고객용에 포함하지 않습니다.<br>
  &bull; 면책 조항은 반드시 포함하여 교부합니다.
</div>
<div class="disc">{DISCLAIMER}</div>"""

    out = slide('부록: 고객용 핵심장표 가이드', '판매사 RM 활용 안내', body, 11)
    _write('proposal_11_appendix.html', out)
    print('  [11] 부록 (고객용)')


# ============================================================================
# Viewer 업데이트
# ============================================================================

def update_viewer():
    pages = [
        ('proposal_00_cover.html', '표지'),
        ('proposal_01_overview.html', '1. 한눈에 보기'),
        ('proposal_02_invest_point1.html', '2. 투자 포인트'),
        ('proposal_03_invest_point2.html', '3. 판매사 관점'),
        ('proposal_04_flowchart.html', '4. 메커니즘'),
        ('proposal_05_params.html', '5. 파라미터'),
        ('proposal_06_backtest_all.html', '6. 전체 백테스트'),
        ('proposal_07_backtest_2005.html', '7. 2005~2015'),
        ('proposal_08_backtest_2010.html', '8. 2010~2020'),
        ('proposal_09_backtest_2015.html', '9. 2015~2025'),
        ('proposal_10_closing.html', '마무리'),
        ('proposal_11_appendix.html', '부록'),
    ]
    btns = '\n'.join(f'    <button class="tab-btn{" active" if i==0 else ""}" onclick="go({i},this)">{label}</button>' for i, (_, label) in enumerate(pages))
    js_arr = ','.join(f"'{fn}'" for fn, _ in pages)

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>Guardrail 인출 전략 제안서</title>
<style>
{FONT_IMPORT}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:{FONT_FAMILY}; background:#e8eaed; }}
.tab-bar {{ position:sticky; top:0; z-index:100; display:flex; align-items:center;
    background:linear-gradient(135deg,#0d1b4a 0%,#1a237e 100%);
    padding:0 12px; box-shadow:0 2px 8px rgba(0,0,0,0.3); overflow-x:auto; }}
.tab-bar .logo {{ color:white; font-weight:800; font-size:14px; padding:14px 14px 14px 8px;
    white-space:nowrap; border-right:1px solid rgba(255,255,255,0.15); margin-right:4px; }}
.tab-btn {{ background:none; border:none; color:rgba(255,255,255,0.5);
    font-family:inherit; font-size:13px; font-weight:600;
    padding:14px 14px; cursor:pointer; border-bottom:3px solid transparent;
    transition:all 0.15s; white-space:nowrap; }}
.tab-btn:hover {{ color:rgba(255,255,255,0.8); background:rgba(255,255,255,0.05); }}
.tab-btn.active {{ color:white; border-bottom-color:#ff9800; background:rgba(255,255,255,0.08); }}
.frame {{ width:100%; height:calc(100vh - 50px); }}
.frame iframe {{ width:100%; height:100%; border:none; background:#f0f2f5; }}
</style></head><body>
<div class="tab-bar">
    <div class="logo">인출 전략 제안서</div>
{btns}
</div>
<div class="frame"><iframe id="f" src="{pages[0][0]}"></iframe></div>
<script>
const P=[{js_arr}];
function go(i,b){{document.getElementById('f').src=P[i];
document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active'));b.classList.add('active');}}
document.addEventListener('keydown',e=>{{const B=[...document.querySelectorAll('.tab-btn')];
let c=B.findIndex(x=>x.classList.contains('active'));
if(e.key==='ArrowRight'&&c<B.length-1)go(c+1,B[c+1]);
if(e.key==='ArrowLeft'&&c>0)go(c-1,B[c-1]);}});
</script></body></html>"""
    _write('proposal_viewer.html', html)
    print('  [+] viewer 업데이트')


# ============================================================================
# Utility
# ============================================================================

def _write(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("제안서 v4.0 생성 시작...")
    paths, month_starts, sim_func = load_data()
    print(f"  Rolling 경로: {len(paths)}개")

    page0_cover()
    page1_overview()
    page2_invest_point1()
    page3_invest_point2()
    page4_flowchart()                                # 4. 플로우차트 복원
    # page5_mechanism 삭제 (시각화)
    page10_params()                                  # -> 5. 파라미터
    page6_backtest_all(paths, sim_func)              # -> 6. 전체 백테스트

    page_backtest_era(paths, month_starts, sim_func, 2005, 2015, 7, '금융위기 관통')
    page_backtest_era(paths, month_starts, sim_func, 2010, 2020, 8, '강세장 + 코로나')
    page_backtest_era(paths, month_starts, sim_func, 2015, 2025, 9, '코로나 + 금리인상')

    page11_closing(paths, sim_func)                  # -> 10. 마무리
    appendix_customer()                              # -> 11. 부록
    update_viewer()

    print(f"\n완료: {TOTAL_PAGES}장 + 부록 + viewer")
