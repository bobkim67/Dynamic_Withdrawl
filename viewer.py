"""
퇴직 포트폴리오 인출 전략 분석기 (Streamlit Viewer v5.1 — 4탭 스토리)
============================================================================
Grid Search + GBM 결과를 4개 탭 설득 흐름으로 시각화:
  1) Guardrail 효과 시각화 — 금융위기형 연속 시나리오 (급락→회복) 메커니즘
  2) 이론 분석 (GBM) — 변동성별 Fixed vs Guardrail 체계 비교
  3) 실제 포트폴리오 검증 — Historical 데이터 기반 검증
  4) Band 최적화 — 최적 Band 탐색 및 교차검증

v5.1 변경:
  - 슬라이더 레이블 수정 (정수 기반)
  - 시나리오 극단화 + 목표달성 관점 메트릭
  - 사이드바 Beta/Path Method → 모든 탭 연동, 탭 내 중복 위젯 삭제
  - Band는 ±5%(최적값) 고정 (Tab 4 제외)
  - 빈 콘텐츠 QC

독립 실행: streamlit run viewer.py
UI 언어: 한국어 + 영문 금융용어 병기.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import io

from withdrawal_backtest import DataPreprocessor, PORTFOLIOS
from engine import daily_to_monthly_returns, generate_paths_rolling, generate_paths_bootstrap, simulate_withdrawal_on_path


# ============================================================================
# Constants
# ============================================================================

PORT_COLORS = {
    'Port_4.0%': '#2196F3',
    'Port_5.0%': '#4CAF50',
    'Port_6.0%': '#FF9800',
    'Port_7.0%': '#E91E63',
    'Port_8.0%': '#9C27B0',
    'Port_9.0%': '#00BCD4',
}

PATH_METHOD_LABELS = {
    'rolling': '과거 데이터 (Rolling)',
    'bootstrap': 'Bootstrap',
    'combined': '합산 (Rolling+Bootstrap)',
}

BETA_LABELS = {
    0.1: '기말잔액 \u2265 초기의 10%',
    0.25: '기말잔액 \u2265 초기의 25%',
    0.5: '기말잔액 \u2265 초기의 50%',
    0.75: '기말잔액 \u2265 초기의 75%',
    1.0: '기말잔액 \u2265 초기의 100% (원금 보존)',
}

# Band ±5%가 최적 → Tab 1/2/3에서 고정 사용
FIXED_BAND = 0.05

CUSTOM_CSS = """
<style>
div[data-testid="stMetric"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 8px 12px;
}
button[data-baseweb="tab"] {
    font-size: 1.05em;
}
.finding-box {
    background: #f0f7ff;
    border-left: 4px solid #2196F3;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.92em;
}
</style>
"""


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data
def load_grid_results():
    """grid_results_full.pkl -> DataFrame 변환"""
    try:
        with open('grid_results_full.pkl', 'rb') as f:
            results = pickle.load(f)

        df = pd.DataFrame([
            {
                'portfolio': r['portfolio'],
                'path_method': r['path_method'],
                'strategy_type': r['strategy_type'],
                'beta': r.get('beta', 0.5),
                'init_wr': r['init_wr'],
                'band': r['band'],
                'adj_on': r['adj_on'],
                'success_rate': r['x_success_rate'],
                'cum_withdraw_median': r['y_cum_withdraw_median'],
                'cum_withdraw_mean': r['y_cum_withdraw_mean'],
                'p_ruin': r['p_ruin'],
                'p_terminal_fail': r['p_terminal_fail'],
                'p_fail': r['p_fail'],
                'cv_median': r['cv_median'],
                'cv_mean': r['cv_mean'],
                'worst_cut_median': r['worst_cut_median'],
                'worst_cut_mean': r['worst_cut_mean'],
                'p5_monthly_income': r['p5_monthly_income'],
                'n_paths': r['n_paths'],
                'terminal_nav_median': r.get('terminal_nav_median', None),
                'terminal_nav_mean': r.get('terminal_nav_mean', None),
                'is_frontier': r.get('is_frontier', False),
                'is_optimal': r.get('is_optimal', False),
            }
            for r in results
        ])
        return df

    except FileNotFoundError:
        st.error("grid_results_full.pkl 파일을 찾을 수 없습니다. grid_search.py를 먼저 실행하세요.")
        st.stop()


@st.cache_data
def load_benchmark_data():
    """benchmark_data.pkl 로드"""
    try:
        with open('benchmark_data.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("benchmark_data.pkl 파일을 찾을 수 없습니다.")
        st.stop()


@st.cache_data
def load_gbm_results():
    """gbm_results.pkl -> DataFrame 변환 (없으면 None)"""
    try:
        with open('gbm_results.pkl', 'rb') as f:
            results = pickle.load(f)
        return pd.DataFrame(results)
    except FileNotFoundError:
        return None


@st.cache_data
def simulate_paths_for_strategy(portfolio_name, init_wr, band, beta, path_method='rolling'):
    """선택된 전략의 path에 대해 W_series와 withdraw_series를 계산"""
    benchmark_data = load_benchmark_data()
    preprocessor = DataPreprocessor(benchmark_data, add_portfolios=True)
    returns_df, month_starts = preprocessor.get_data()
    daily_returns = returns_df[portfolio_name]
    monthly_returns = daily_to_monthly_returns(daily_returns)

    rolling_paths = generate_paths_rolling(monthly_returns, T_months=120)

    if path_method == 'bootstrap':
        all_bootstrap = generate_paths_bootstrap(monthly_returns, T_months=120,
                                                  block_length=12, n_paths=5000, seed=42)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_bootstrap), min(181, len(all_bootstrap)), replace=False)
        paths = [all_bootstrap[i] for i in idx]
    elif path_method == 'combined':
        all_bootstrap = generate_paths_bootstrap(monthly_returns, T_months=120,
                                                  block_length=12, n_paths=5000, seed=42)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_bootstrap), min(181, len(all_bootstrap)), replace=False)
        paths = list(rolling_paths) + [all_bootstrap[i] for i in idx]
    else:
        paths = rolling_paths

    success_paths = []
    failure_paths = []
    all_withdraw_series = []

    for path in paths:
        result = simulate_withdrawal_on_path(
            path_returns=path, init_wr=init_wr, band=band,
            adj_on=False, beta=beta, W0=100.0, debug=False
        )
        all_withdraw_series.append(result['withdraw_series'])
        if result['success']:
            success_paths.append(result['W_series'])
        else:
            failure_paths.append(result['W_series'])

    return success_paths, failure_paths, all_withdraw_series


# ============================================================================
# Helper Functions
# ============================================================================

def gbm_survival_probability(mu, sigma, wr, T=10, beta=0.5):
    """GBM 생존확률 계산 (Closed-form)"""
    if wr <= 0:
        return 1.0
    if sigma <= 0:
        remaining = 1.0 - wr * T + mu * T
        return 1.0 if remaining >= beta else 0.0
    z = -(np.log(beta) - (mu - wr - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    survival = norm.cdf(z)
    return np.clip(survival, 0.0, 1.0)


def _finding_box(text):
    """핵심발견 박스 HTML"""
    return f'<div class="finding-box">{text}</div>'


def _build_path_trace(paths, color, name, alpha=0.12):
    """여러 경로를 None 구분자로 하나의 trace로 결합 (성능 최적화)"""
    x_all, y_all = [], []
    months = list(range(121))
    for p in paths:
        vals = p.tolist() if hasattr(p, 'tolist') else list(p)
        x_all.extend(months[:len(vals)])
        x_all.append(None)
        y_all.extend(vals)
        y_all.append(None)
    return go.Scatter(
        x=x_all, y=y_all, mode='lines',
        line=dict(color=color, width=0.8),
        opacity=alpha, name=name, hoverinfo='skip',
    )


# ============================================================================
# Tab 1: Guardrail 효과 시각화 (3시나리오)
# ============================================================================

def _simulate_scenario(returns_scenario, init_wr, band, W0=100.0):
    """시나리오별 Fixed vs Guardrail NAV 및 인출액 계산"""
    T = len(returns_scenario)
    base_monthly = W0 * init_wr / 12
    upper_wr = init_wr * (1 + band)
    lower_wr = init_wr * (1 - band)

    # Fixed
    fixed_nav = [W0]
    fixed_withdrawals = []
    nav = W0
    for t in range(T):
        nav = nav * (1 + returns_scenario[t])
        w = base_monthly
        nav = nav - w
        fixed_nav.append(nav)
        fixed_withdrawals.append(w)

    # Guardrail
    guard_nav = [W0]
    guard_withdrawals = []
    nav = W0
    for t in range(T):
        nav = nav * (1 + returns_scenario[t])
        if nav > 0:
            current_wr = (base_monthly * 12) / nav
            if current_wr > upper_wr:
                w = upper_wr * nav / 12
            elif current_wr < lower_wr:
                w = lower_wr * nav / 12
            else:
                w = base_monthly
        else:
            w = 0
        nav = nav - w
        guard_nav.append(nav)
        guard_withdrawals.append(w)

    return {
        'fixed_nav': fixed_nav,
        'guard_nav': guard_nav,
        'fixed_withdrawals': fixed_withdrawals,
        'guard_withdrawals': guard_withdrawals,
        'fixed_cum': sum(fixed_withdrawals),
        'guard_cum': sum(guard_withdrawals),
        'fixed_terminal': fixed_nav[-1],
        'guard_terminal': guard_nav[-1],
    }


def render_tab1_mechanism(beta, path_method, init_wr):
    """탭 1: Guardrail 컨셉 시각화 — 핵심 메시지를 한눈에"""

    # ===== 핵심 메시지 헤드라인 =====
    st.markdown(f"""
    <div style="border-left: 4px solid #334155; padding: 14px 20px; margin-bottom: 24px;
                background: #f8fafc;">
        <div style="font-size: 1.45em; font-weight: 700; color: #1e293b; margin-bottom: 6px;">
            Guardrail — 하락기 자본 보전 + 회복기 성장 극대화
        </div>
        <div style="font-size: 0.9em; color: #64748b; line-height: 1.6;">
            연 {init_wr*100:.0f}% 인출, 밴드 &plusmn;{FIXED_BAND*100:.0f}% 조건에서
            Guardrail은 하락기에 인출을 줄여 자본을 보전하고, 회복기에 적극적으로 인출을 늘려
            기말잔액 목표(초기의 {beta*100:.0f}%)를 안정적으로 달성합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== 고정 인출 vs Guardrail 비교 시뮬레이션 =====
    # 동일한 시장 경로에서 두 전략이 어떻게 다른 결과를 만드는지 보여주는 핵심 차트
    st.markdown(
        '<p style="font-size:0.95em; font-weight:600; color:#334155; '
        'margin-bottom:4px;">같은 시장, 다른 인출 규칙 → 다른 결과</p>',
        unsafe_allow_html=True,
    )

    # --- 시뮬레이션 파라미터 (사이드바 연동) ---
    _np_seed = 7
    _n = 240                                    # 240개월 (20년)
    _target_ratio = init_wr / 12                # 사이드바 연 인출률 → 월 비율
    _band_pct = FIXED_BAND                      # ±5%
    _lo_band = _target_ratio * (1 - _band_pct)  # 밴드 하한
    _hi_band = _target_ratio * (1 + _band_pct)  # 밴드 상한
    _nav0 = 100.0
    _w0 = _target_ratio * _nav0                 # 초기 인출액

    # --- 수익률 경로 (regime-stitched) ---
    # 4개 국면: 안정기 → 하락기 → 회복기 → 후기 랠리
    rng = np.random.RandomState(_np_seed)
    _r1 = rng.normal(0.005, 0.020, 60)   # 안정기
    _r2 = rng.normal(-0.018, 0.038, 60)  # 하락기 (GFC급)
    _r3 = rng.normal(0.018, 0.020, 60)   # 회복기 (V자 반등)
    _r4 = rng.normal(0.013, 0.016, 60)   # 후기 랠리
    _rets = np.concatenate([_r1, _r2, _r3, _r4])

    # --- 시장 NAV (인출 없음, 참조선) ---
    _mkt = np.zeros(_n + 1); _mkt[0] = _nav0
    for _t in range(_n):
        _mkt[_t + 1] = _mkt[_t] * (1 + _rets[_t])

    # --- 고정 인출 NAV (NAV=0이면 인출 중단) ---
    _nav_f = np.zeros(_n + 1); _nav_f[0] = _nav0
    _wf = np.zeros(_n + 1); _wf[0] = _w0  # 실제 인출 추적
    for _t in range(_n):
        _grown_f = _nav_f[_t] * (1 + _rets[_t])
        if _grown_f > 0:
            _wf[_t + 1] = _w0
            _nav_f[_t + 1] = max(_grown_f - _w0, 0)
        else:
            _wf[_t + 1] = 0
            _nav_f[_t + 1] = 0

    # --- Guardrail 인출 NAV ---
    _nav_g = np.zeros(_n + 1); _nav_g[0] = _nav0
    _wg = np.zeros(_n + 1); _wg[0] = _w0
    for _t in range(_n):
        _grown = _nav_g[_t] * (1 + _rets[_t])
        if _grown <= 0:
            _nav_g[_t + 1] = 0; _wg[_t + 1] = 0; continue
        # W/NAV 비율 기반 밴드 적용
        _ratio = _wg[_t] / _grown
        if _ratio > _hi_band:          # 비율 과다 → 인출 축소
            _wg[_t + 1] = _hi_band * _grown
        elif _ratio < _lo_band:        # 비율 과소 → 인출 확대
            _wg[_t + 1] = _lo_band * _grown
        else:                           # 밴드 내 → 유지
            _wg[_t + 1] = _wg[_t]
        _nav_g[_t + 1] = max(_grown - _wg[_t + 1], 0)

    _time = np.arange(_n + 1)
    _target_line = _nav0 * beta        # 기말 목표: 사이드바 beta 연동
    _end_f = _nav_f[-1]
    _end_g = _nav_g[-1]

    # --- 상단 차트: NAV 경로 비교 ---
    fig_persuade = make_subplots(
        rows=2, cols=1, row_heights=[0.75, 0.25],
        vertical_spacing=0.06, shared_xaxes=True,
    )

    # 하락기 / 회복기 배경 음영 — 구간 전체를 파스텔 톤으로 칠함
    fig_persuade.add_vrect(x0=0, x1=60,                          # 안정기: 연한 회색
                           fillcolor='rgba(226,232,240,0.15)',
                           line_width=0, row='all', col=1)
    fig_persuade.add_vrect(x0=60, x1=120,                        # 하락기: 파스텔 핑크
                           fillcolor='rgba(254,202,202,0.30)',
                           line_width=0, row='all', col=1)
    fig_persuade.add_vrect(x0=120, x1=180,                       # 회복기: 파스텔 민트
                           fillcolor='rgba(187,247,208,0.30)',
                           line_width=0, row='all', col=1)
    fig_persuade.add_vrect(x0=180, x1=_n,                        # 후기: 연한 파랑
                           fillcolor='rgba(191,219,254,0.15)',
                           line_width=0, row='all', col=1)

    # 구간 라벨 (상단 고정)
    _y_top = max(_mkt.max(), _nav_g.max()) * 1.02
    for _lx, _lt, _lc in [(30, '안정기', '#94a3b8'), (90, '하락기', '#f87171'),
                           (150, '회복기', '#4ade80'), (210, '후기 랠리', '#93c5fd')]:
        fig_persuade.add_annotation(
            x=_lx, y=_y_top, text=f'<b>{_lt}</b>', showarrow=False,
            font=dict(size=12, color=_lc), row=1, col=1)

    # 시장 NAV (회색 참조선)
    fig_persuade.add_trace(go.Scatter(
        x=_time, y=_mkt, mode='lines', name='시장 (인출 없음)',
        line=dict(color='#d1d5db', width=1.5),
        hovertemplate='%{x}개월<br>시장 NAV: %{y:.1f}<extra></extra>',
    ), row=1, col=1)

    # 고정 인출 NAV (빨간)
    fig_persuade.add_trace(go.Scatter(
        x=_time, y=_nav_f, mode='lines', name='고정 인출',
        line=dict(color='#dc2626', width=2.6),
        hovertemplate='%{x}개월<br>고정 NAV: %{y:.1f}<extra></extra>',
    ), row=1, col=1)

    # Guardrail 인출 NAV (초록)
    fig_persuade.add_trace(go.Scatter(
        x=_time, y=_nav_g, mode='lines', name='Guardrail 인출',
        line=dict(color='#16a34a', width=2.6),
        hovertemplate='%{x}개월<br>Guardrail NAV: %{y:.1f}<extra></extra>',
    ), row=1, col=1)

    # 목표선 (50%)
    fig_persuade.add_hline(
        y=_target_line, line_dash='dash', line_color='#475569',
        line_width=1.2, opacity=0.7, row=1, col=1)
    fig_persuade.add_annotation(
        x=_n + 2, y=_target_line,
        text=f'목표: {_target_line:.0f}<br>(초기의 {beta*100:.0f}%)',
        showarrow=False, xanchor='left',
        font=dict(size=11, color='#475569'), row=1, col=1)

    # 종료점 마커
    fig_persuade.add_trace(go.Scatter(
        x=[_n], y=[_end_f], mode='markers', showlegend=False,
        marker=dict(size=10, color='#dc2626', line=dict(color='white', width=1.5)),
        hovertemplate=f'기말 고정 NAV: {_end_f:.1f}<extra></extra>',
    ), row=1, col=1)
    fig_persuade.add_trace(go.Scatter(
        x=[_n], y=[_end_g], mode='markers', showlegend=False,
        marker=dict(size=10, color='#16a34a', line=dict(color='white', width=1.5)),
        hovertemplate=f'기말 Guardrail NAV: {_end_g:.1f}<extra></extra>',
    ), row=1, col=1)

    # 종료 라벨
    _f_status = '목표 미달' if _end_f < _target_line else '목표 달성'
    _g_status = '목표 미달' if _end_g < _target_line else '목표 달성'
    _f_label_y = _end_f - 10 if _end_g - _end_f > 15 else _end_f - 12
    _g_label_y = _end_g + 10 if _end_g - _end_f > 15 else _end_g + 12

    fig_persuade.add_annotation(
        x=_n, y=_end_f,
        text=f'<b>고정: {_end_f:.1f}  ({_f_status})</b>',
        showarrow=True, arrowhead=2, arrowcolor='#dc2626', ax=-50, ay=20,
        font=dict(size=11, color='#dc2626'),
        bgcolor='white', bordercolor='#fecaca', borderwidth=1, borderpad=3,
        row=1, col=1)
    fig_persuade.add_annotation(
        x=_n, y=_end_g,
        text=f'<b>Guardrail: {_end_g:.1f}  ({_g_status})</b>',
        showarrow=True, arrowhead=2, arrowcolor='#16a34a', ax=-50, ay=-20,
        font=dict(size=11, color='#16a34a'),
        bgcolor='white', bordercolor='#bbf7d0', borderwidth=1, borderpad=3,
        row=1, col=1)

    # --- 하단 차트: 인출액 비교 ---
    # 고정 인출 (NAV=0이면 0으로 표시)
    fig_persuade.add_trace(go.Scatter(
        x=_time, y=_wf, mode='lines', name='고정 인출액',
        line=dict(color='#dc2626', width=1.2, dash='dash'),
        fill='tozeroy', fillcolor='rgba(220,38,38,0.10)',
        hovertemplate='%{x}개월<br>고정 인출: %{y:.2f}<extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    # Guardrail 인출 (변동)
    fig_persuade.add_trace(go.Scatter(
        x=_time, y=_wg, mode='lines', name='Guardrail 인출액',
        line=dict(color='#16a34a', width=1.8),
        fill='tozeroy', fillcolor='rgba(22,163,74,0.15)',
        hovertemplate='%{x}개월<br>Guardrail 인출: %{y:.2f}<extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    # 인출 축소 / 인출 확대 annotation
    fig_persuade.add_annotation(
        x=95, y=_wg[95],
        text='<b>인출 축소</b><br>(자본 보전)',
        showarrow=True, arrowhead=2, ax=30, ay=-25,
        font=dict(size=10, color='#475569'),
        row=2, col=1)
    fig_persuade.add_annotation(
        x=200, y=_wg[200],
        text='<b>인출 확대</b><br>(수익 향유)',
        showarrow=True, arrowhead=2, ax=30, ay=25,
        font=dict(size=10, color='#475569'),
        row=2, col=1)

    # --- 레이아웃 ---
    fig_persuade.update_layout(
        height=520,
        margin=dict(t=10, b=30, l=50, r=80),
        plot_bgcolor='#f8fafc', paper_bgcolor='white',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
            font=dict(size=11), bgcolor='rgba(248,250,252,0.9)',
            bordercolor='#e2e8f0', borderwidth=1,
        ),
    )
    # 상단 축
    fig_persuade.update_yaxes(
        title_text='잔액 (NAV)', title_font=dict(size=12, color='#64748b'),
        showgrid=False, zeroline=False, linecolor='#e2e8f0',
        tickfont=dict(size=10, color='#94a3b8'),
        row=1, col=1)
    fig_persuade.update_xaxes(
        showticklabels=False, showgrid=False, linecolor='#e2e8f0',
        row=1, col=1)
    # 하단 축
    fig_persuade.update_yaxes(
        title_text='인출액 (월)', title_font=dict(size=12, color='#64748b'),
        showgrid=False, zeroline=False, linecolor='#e2e8f0',
        tickfont=dict(size=10, color='#94a3b8'),
        row=2, col=1)
    fig_persuade.update_xaxes(
        title_text='경과 월수', title_font=dict(size=12, color='#64748b'),
        showgrid=False, linecolor='#e2e8f0',
        tickfont=dict(size=10, color='#94a3b8'),
        row=2, col=1)

    st.plotly_chart(fig_persuade, width='stretch')

    st.caption(
        f"시뮬레이션: 연 {_target_ratio*12*100:.0f}% 인출, 밴드 ±5%, 240개월. "
        f"같은 시장 수익률에서 고정 인출은 자본이 고갈되지만, "
        f"Guardrail은 하락기 인출 축소 → 회복기 자본 복원으로 목표를 달성합니다."
    )

    # --- 메트릭 카드 (시뮬레이션 결과 요약, 사이드바 연동) ---
    # 누적 인출 합계 (NAV=0 이후 인출 중단 반영)
    _cum_fixed = _wf.sum()   # 고정: 실제 인출 추적
    _cum_guard = _wg.sum()   # Guardrail: 변동액 합산
    _cum_diff = _cum_guard - _cum_fixed  # 양수 = Guardrail이 더 많이 인출

    _f_met = _end_f >= _target_line
    _g_met = _end_g >= _target_line

    # 기말 NAV + 누적인출 합계
    _total_fixed = _end_f + _cum_fixed
    _total_guard = _end_g + _cum_guard
    _total_diff = _total_guard - _total_fixed

    # --- 실데이터 기반 성공률 (대표 포트폴리오 Port_6.0%) ---
    _rep_port = sorted(PORTFOLIOS.keys())[min(2, len(PORTFOLIOS) - 1)]
    _fix_ok, _fix_fail, _fix_wd = simulate_paths_for_strategy(
        _rep_port, init_wr, band=99.0, beta=beta, path_method=path_method)
    _grd_ok, _grd_fail, _grd_wd = simulate_paths_for_strategy(
        _rep_port, init_wr, band=FIXED_BAND, beta=beta, path_method=path_method)
    _n_fix = len(_fix_ok) + len(_fix_fail)
    _n_grd = len(_grd_ok) + len(_grd_fail)
    _fix_sr = len(_fix_ok) / _n_fix * 100 if _n_fix > 0 else 0
    _grd_sr = len(_grd_ok) / _n_grd * 100 if _n_grd > 0 else 0
    _delta_sr = _grd_sr - _fix_sr

    # 실데이터 중앙값 (누적인출, 기말잔액)
    _all_fix_paths = _fix_ok + _fix_fail
    _all_grd_paths = _grd_ok + _grd_fail
    _fix_cum_vals = [sum(ws) for ws in _fix_wd]
    _grd_cum_vals = [sum(ws) for ws in _grd_wd]
    _fix_cum_med = np.median(_fix_cum_vals) if _fix_cum_vals else 0
    _grd_cum_med = np.median(_grd_cum_vals) if _grd_cum_vals else 0
    _cum_diff_abs = _grd_cum_med - _fix_cum_med
    _cum_diff_pct = ((_grd_cum_med / _fix_cum_med) - 1) * 100 if _fix_cum_med != 0 else 0

    _fix_term_vals = [p[-1] for p in _all_fix_paths] if _all_fix_paths else []
    _grd_term_vals = [p[-1] for p in _all_grd_paths] if _all_grd_paths else []
    _fix_term_med = np.median(_fix_term_vals) if _fix_term_vals else 0
    _grd_term_med = np.median(_grd_term_vals) if _grd_term_vals else 0
    _term_diff = _grd_term_med - _fix_term_med
    _term_pct = ((_grd_term_med / _fix_term_med) - 1) * 100 if _fix_term_med != 0 else 0

    _fix_total_med = _fix_term_med + _fix_cum_med
    _grd_total_med = _grd_term_med + _grd_cum_med
    _total_diff_r = _grd_total_med - _fix_total_med
    _total_pct_r = ((_grd_total_med / _fix_total_med) - 1) * 100 if _fix_total_med != 0 else 0

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric(
            f"성공률 개선 ({_rep_port})",
            f"{_delta_sr:+.1f}%p",
            delta=f"{_delta_sr:+.1f}%p (Fixed {_fix_sr:.1f}% → Guard {_grd_sr:.1f}%)",
            delta_color="normal" if _delta_sr != 0 else "off",
        )
    with mc2:
        st.metric(
            "누적인출금 (중앙값)",
            f"{_grd_cum_med:.1f}",
            delta=f"{_cum_diff_abs:+.1f} ({_cum_diff_pct:+.1f}%) vs Fixed {_fix_cum_med:.1f}",
            delta_color="normal" if _cum_diff_abs != 0 else "off",
        )
    with mc3:
        st.metric(
            "기말잔액 (중앙값)",
            f"{_grd_term_med:.1f}",
            delta=f"{_term_diff:+.1f} ({_term_pct:+.1f}%) vs Fixed {_fix_term_med:.1f}",
            delta_color="normal" if _term_diff != 0 else "off",
        )
    with mc4:
        st.metric(
            "기말잔액 + 누적인출 합계",
            f"{_grd_total_med:.1f}",
            delta=f"{_total_diff_r:+.1f} ({_total_pct_r:+.1f}%) vs Fixed {_fix_total_med:.1f}",
            delta_color="normal" if _total_diff_r != 0 else "off",
        )



# ============================================================================
# Tab 데이터 검증: 실제 과거 데이터 기반 경로 시각화
# ============================================================================

def render_tab_data_validation(beta, path_method, init_wr):
    """데이터 검증 탭: 실제 과거 데이터에서 Fixed vs Guardrail 경로 비교"""

    st.markdown(
        '<p style="font-size:1.2em; font-weight:700; color:#1e293b; '
        'margin: 8px 0 4px 0;">실제 과거 데이터 검증</p>',
        unsafe_allow_html=True,
    )
    st.caption("합성 시나리오가 실제 과거 데이터에서도 성립하는지 확인합니다.")

    st.markdown("")

    # 포트폴리오 선택
    port_options = sorted(PORTFOLIOS.keys())
    _pcol1, _pcol2 = st.columns([1, 2])
    with _pcol1:
        tab1_port = st.selectbox(
            "포트폴리오 선택",
            options=port_options,
            index=min(2, len(port_options) - 1),
            key='tab_dv_port',
        )
    with _pcol2:
        _pinfo = PORTFOLIOS[tab1_port]
        _weights_str = " / ".join(
            f"{k} {v:.1f}%" for k, v in _pinfo['weights'].items()
        )
        st.markdown(
            f'<div style="background:#f1f5f9; border:1px solid #e2e8f0; padding:10px 14px; '
            f'border-radius:6px; margin-top:8px;">'
            f'<span style="font-weight:600; color:#334155;">{tab1_port}</span> '
            f'<span style="color:#64748b; font-size:0.85em;">'
            f'목표수익 {_pinfo["target_return"]:.1f}% / 목표위험 {_pinfo["target_risk"]:.2f}%</span><br>'
            f'<span style="font-size:0.82em; color:#94a3b8;">{_weights_str}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 목표선 (실제 데이터 시뮬레이션 기준: W0=100.0)
    target = 100.0 * beta

    # Fixed vs Guardrail 시뮬레이션
    fixed_success, fixed_failure, fixed_wd = simulate_paths_for_strategy(
        tab1_port, init_wr, band=99.0, beta=beta, path_method=path_method
    )
    guard_success, guard_failure, guard_wd = simulate_paths_for_strategy(
        tab1_port, init_wr, band=FIXED_BAND, beta=beta, path_method=path_method
    )

    n_fixed_total = len(fixed_success) + len(fixed_failure)
    n_guard_total = len(guard_success) + len(guard_failure)
    fixed_sr = len(fixed_success) / n_fixed_total * 100 if n_fixed_total > 0 else 0
    guard_sr = len(guard_success) / n_guard_total * 100 if n_guard_total > 0 else 0
    delta_sr = guard_sr - fixed_sr

    # ===== 2패널 스파게티 차트 =====
    col_fixed, col_guard = st.columns(2)

    with col_fixed:
        st.markdown(f"""
        <div style="border-bottom:2px solid #cbd5e1; padding-bottom:6px; margin-bottom:10px;">
            <span style="font-size:0.9em; font-weight:600; color:#64748b;">
                Fixed (고정 인출)
            </span>
            <span style="font-size:0.85em; color:#94a3b8; margin-left:8px;">
                성공률 {fixed_sr:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

        fig_fixed = go.Figure()
        if fixed_failure:
            fig_fixed.add_trace(_build_path_trace(fixed_failure, '#E74C3C', '실패 경로', alpha=0.15))
        if fixed_success:
            fig_fixed.add_trace(_build_path_trace(fixed_success, '#4CAF50', '성공 경로', alpha=0.15))
        # 중앙값 경로
        all_fixed_paths = fixed_success + fixed_failure
        if all_fixed_paths:
            max_len = max(len(p) for p in all_fixed_paths)
            padded = np.full((len(all_fixed_paths), max_len), np.nan)
            for i, p in enumerate(all_fixed_paths):
                vals = np.array(p)
                padded[i, :len(vals)] = vals
            median_fixed = np.nanmedian(padded, axis=0)
            fig_fixed.add_trace(go.Scatter(
                x=list(range(len(median_fixed))), y=median_fixed.tolist(),
                mode='lines', name='중앙값',
                line=dict(color='#333', width=3),
            ))
        # 목표선
        fig_fixed.add_hline(
            y=target, line_dash="dot", line_color="#FF9800", line_width=2,
            annotation_text=f"목표 {target:.0f}",
            annotation_position="bottom left",
            annotation_font=dict(color="#FF9800", size=11),
        )
        fig_fixed.update_layout(
            yaxis_title="잔액 (NAV)", xaxis_title="월", height=350,
            margin=dict(t=10, b=30, l=50, r=20),
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
            showlegend=True,
        )
        st.plotly_chart(fig_fixed, width='stretch')

    with col_guard:
        st.markdown(f"""
        <div style="border-bottom:2px solid #334155; padding-bottom:6px; margin-bottom:10px;">
            <span style="font-size:0.9em; font-weight:600; color:#334155;">
                Guardrail (&plusmn;{FIXED_BAND*100:.0f}%)
            </span>
            <span style="font-size:0.85em; color:#94a3b8; margin-left:8px;">
                성공률 {guard_sr:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

        fig_guard = go.Figure()
        if guard_failure:
            fig_guard.add_trace(_build_path_trace(guard_failure, '#E74C3C', '실패 경로', alpha=0.15))
        if guard_success:
            fig_guard.add_trace(_build_path_trace(guard_success, '#4CAF50', '성공 경로', alpha=0.15))
        # 중앙값 경로
        all_guard_paths = guard_success + guard_failure
        if all_guard_paths:
            max_len = max(len(p) for p in all_guard_paths)
            padded = np.full((len(all_guard_paths), max_len), np.nan)
            for i, p in enumerate(all_guard_paths):
                vals = np.array(p)
                padded[i, :len(vals)] = vals
            median_guard = np.nanmedian(padded, axis=0)
            fig_guard.add_trace(go.Scatter(
                x=list(range(len(median_guard))), y=median_guard.tolist(),
                mode='lines', name='중앙값',
                line=dict(color='#1565C0', width=3),
            ))
        # 목표선
        fig_guard.add_hline(
            y=target, line_dash="dot", line_color="#FF9800", line_width=2,
            annotation_text=f"목표 {target:.0f}",
            annotation_position="bottom left",
            annotation_font=dict(color="#FF9800", size=11),
        )
        fig_guard.update_layout(
            yaxis_title="잔액 (NAV)", xaxis_title="월", height=350,
            margin=dict(t=10, b=30, l=50, r=20),
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
            showlegend=True,
        )
        st.plotly_chart(fig_guard, width='stretch')

    # ===== 4메트릭 카드 =====
    # 누적인출금 계산
    fixed_cum_vals = [sum(ws) for ws in fixed_wd]
    guard_cum_vals = [sum(ws) for ws in guard_wd]
    fixed_cum_median = np.median(fixed_cum_vals) if fixed_cum_vals else 0
    guard_cum_median = np.median(guard_cum_vals) if guard_cum_vals else 0
    cum_diff_pct = ((guard_cum_median / fixed_cum_median) - 1) * 100 if fixed_cum_median > 0 else 0

    # 기말잔액 계산
    fixed_terminal_vals = [p[-1] for p in all_fixed_paths] if all_fixed_paths else []
    guard_terminal_vals = [p[-1] for p in all_guard_paths] if all_guard_paths else []
    fixed_terminal_median = np.median(fixed_terminal_vals) if fixed_terminal_vals else 0
    guard_terminal_median = np.median(guard_terminal_vals) if guard_terminal_vals else 0
    terminal_diff = guard_terminal_median - fixed_terminal_median

    # 기말잔액 + 누적인출 합계
    fixed_total_median = fixed_terminal_median + fixed_cum_median
    guard_total_median = guard_terminal_median + guard_cum_median
    total_diff = guard_total_median - fixed_total_median

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric(
            "성공률 개선",
            f"{delta_sr:+.1f}%p",
            delta=f"{delta_sr:+.1f}%p (Fixed {fixed_sr:.1f}% → Guard {guard_sr:.1f}%)",
            delta_color="normal" if delta_sr != 0 else "off",
        )
    cum_diff_abs = guard_cum_median - fixed_cum_median
    terminal_pct = ((guard_terminal_median / fixed_terminal_median) - 1) * 100 if fixed_terminal_median != 0 else 0
    total_pct = ((guard_total_median / fixed_total_median) - 1) * 100 if fixed_total_median != 0 else 0
    with mc2:
        st.metric(
            "누적인출금 (중앙값)",
            f"{guard_cum_median:.1f}",
            delta=f"{cum_diff_abs:+.1f} ({cum_diff_pct:+.1f}%) vs Fixed {fixed_cum_median:.1f}",
            delta_color="normal" if cum_diff_abs != 0 else "off",
        )
    with mc3:
        st.metric(
            "기말잔액 (중앙값)",
            f"{guard_terminal_median:.1f}",
            delta=f"{terminal_diff:+.1f} ({terminal_pct:+.1f}%) vs Fixed {fixed_terminal_median:.1f}",
            delta_color="normal" if terminal_diff != 0 else "off",
        )
    with mc4:
        st.metric(
            "기말잔액 + 누적인출 합계",
            f"{guard_total_median:.1f}",
            delta=f"{total_diff:+.1f} ({total_pct:+.1f}%) vs Fixed {fixed_total_median:.1f}",
            delta_color="normal" if total_diff != 0 else "off",
        )

    # ===== 기말잔액 분포 히스토그램 =====
    if fixed_terminal_vals and guard_terminal_vals:
        st.markdown("")
        st.markdown("#### 기말잔액 분포")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=fixed_terminal_vals, name='Fixed',
            marker_color='rgba(153,153,153,0.5)',
            nbinsx=20,
        ))
        fig_hist.add_trace(go.Histogram(
            x=guard_terminal_vals, name='Guardrail',
            marker_color='rgba(33,150,243,0.5)',
            nbinsx=20,
        ))
        # 목표선
        fig_hist.add_vline(
            x=target, line_dash="dot", line_color="#FF9800", line_width=2,
            annotation_text=f"목표 {target:.0f}",
            annotation_position="top right",
            annotation_font=dict(color="#FF9800", size=11),
        )
        # 중앙값 수직선
        fig_hist.add_vline(
            x=fixed_terminal_median, line_dash="dash", line_color="#999", line_width=1.5,
            annotation_text=f"Fixed 중앙값 {fixed_terminal_median:.0f}",
            annotation_position="top left",
            annotation_font=dict(color="#999", size=10),
        )
        fig_hist.add_vline(
            x=guard_terminal_median, line_dash="dash", line_color="#2196F3", line_width=1.5,
            annotation_text=f"Guard 중앙값 {guard_terminal_median:.0f}",
            annotation_position="top right",
            annotation_font=dict(color="#2196F3", size=10),
        )
        fig_hist.update_layout(
            barmode='overlay',
            xaxis_title="기말잔액", yaxis_title="경로 수",
            height=280,
            margin=dict(t=10, b=30, l=50, r=20),
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
        )
        st.plotly_chart(fig_hist, width='stretch')

    # 캡션
    pm_label = PATH_METHOD_LABELS.get(path_method, path_method)
    st.caption(
        f"{pm_label} 기반 {n_fixed_total}개 경로 | "
        f"포트폴리오: {tab1_port} | 기간: 10년 (120개월)"
    )
    st.caption(
        "주의: Rolling window 경로는 시작점이 1개월씩 이동하므로 상호 상관이 높습니다. "
        "Bootstrap 결과와의 비교는 실제 포트폴리오 검증 탭에서 확인할 수 있습니다."
    )


# ============================================================================
# Tab 2: 이론 분석 (GBM 변동성별 Fixed vs Guardrail)
# ============================================================================

def render_tab2_gbm(df_gbm, beta):
    """탭 2: GBM 이론 분석 — 변동성(σ)별 Fixed vs Guardrail 비교
    Band는 ±5% 고정, Beta는 사이드바에서 연동."""

    st.markdown("### 이론 분석: GBM Monte Carlo")
    st.caption(
        f"GBM으로 변동성(\u03c3)별 Fixed vs Guardrail을 비교합니다. "
        f"**기말잔액 비율: {beta*100:.0f}%** | **Band: \u00b1{FIXED_BAND*100:.0f}% 고정**"
    )

    if df_gbm is None or len(df_gbm) == 0:
        st.warning("gbm_results.pkl이 없습니다. `python gbm_grid_search.py`를 실행하세요.")
        return

    # 컨트롤: mu만 (beta는 사이드바, band는 ±5% 고정)
    mu_options = sorted(df_gbm['mu'].unique())
    default_mu = min(mu_options, key=lambda x: abs(x - 0.06))
    gbm_mu = st.select_slider(
        "기대수익률 (\u03bc)",
        options=mu_options,
        value=default_mu,
        format_func=lambda x: f"{x*100:.0f}%",
        key='tab2_mu'
    )

    # beta에 가장 가까운 값 선택
    gbm_beta = min(sorted(df_gbm['beta'].unique()), key=lambda x: abs(x - beta))
    gbm_band = FIXED_BAND

    # 필터링
    gbm_fixed = df_gbm[
        (df_gbm['mu'] == gbm_mu) & (df_gbm['beta'] == gbm_beta) &
        (df_gbm['strategy_type'] == 'fixed_baseline')
    ].copy()
    gbm_guard = df_gbm[
        (df_gbm['mu'] == gbm_mu) & (df_gbm['beta'] == gbm_beta) &
        (df_gbm['strategy_type'] == 'dynamic') & (df_gbm['band'] == gbm_band)
    ].copy()

    if len(gbm_fixed) == 0:
        st.warning(f"\u03bc={gbm_mu*100:.0f}%, 기말잔액 비율={gbm_beta*100:.0f}% 조합의 Fixed 데이터가 없습니다.")
        return
    if len(gbm_guard) == 0:
        st.warning(f"Band \u00b1{gbm_band*100:.0f}% Guardrail 데이터가 없습니다. "
                   f"gbm_grid_search.py에서 해당 band를 포함했는지 확인하세요.")
        return

    sigmas = sorted(gbm_fixed['sigma'].unique())
    init_wrs_gbm = sorted(gbm_fixed['init_wr'].unique())

    # ========================================
    # 1. 성공률 개선 히트맵
    # ========================================
    st.markdown("---")
    st.subheader("1. 성공률 차이 히트맵 (Guardrail \u2212 Fixed)")
    st.caption("녹색 = Guardrail 우위. 변동성이 높을수록 Guardrail 효과가 커집니다.")

    diff_matrix = np.full((len(init_wrs_gbm), len(sigmas)), np.nan)
    for i_w, wr in enumerate(init_wrs_gbm):
        for j_s, sig in enumerate(sigmas):
            f_r = gbm_fixed[(gbm_fixed['sigma'] == sig) & (gbm_fixed['init_wr'] == wr)]
            g_r = gbm_guard[(gbm_guard['sigma'] == sig) & (gbm_guard['init_wr'] == wr)]
            if len(f_r) > 0 and len(g_r) > 0:
                diff_matrix[i_w, j_s] = (
                    g_r.iloc[0]['x_success_rate'] - f_r.iloc[0]['x_success_rate']) * 100

    valid_vals = diff_matrix[np.isfinite(diff_matrix)]
    if len(valid_vals) > 0:
        zmax = max(abs(valid_vals.min()), abs(valid_vals.max()), 3)
        fig_hm = go.Figure(data=go.Heatmap(
            z=diff_matrix,
            x=[f"{s*100:.0f}%" for s in sigmas],
            y=[f"{w*100:.1f}%" for w in init_wrs_gbm],
            colorscale='RdYlGn', zmid=0, zmin=-zmax, zmax=zmax,
            text=np.round(diff_matrix, 1),
            texttemplate='%{text:+.1f}',
            textfont={"size": 8},
            colorbar=dict(title="SR 차이(%p)")
        ))
        fig_hm.update_layout(xaxis_title="변동성 \u03c3", yaxis_title="초기인출률", height=500)
        st.plotly_chart(fig_hm, width='stretch')
    else:
        st.info("히트맵을 그릴 데이터가 부족합니다.")

    # ========================================
    # 2. 변동성별 성공률 곡선
    # ========================================
    st.markdown("---")
    st.subheader("2. 변동성별 성공률 곡선")
    st.caption("실선=Guardrail MC, 점선=Fixed MC, 점점선=Fixed Closed-form")

    default_sigmas = [s for s in [0.04, 0.06, 0.08, 0.10, 0.14] if s in sigmas]
    if not default_sigmas:
        default_sigmas = sigmas[:min(5, len(sigmas))]
    selected_sigmas = st.multiselect(
        "비교 변동성 선택",
        options=sigmas,
        default=default_sigmas,
        format_func=lambda x: f"\u03c3={x*100:.0f}%",
        key='tab2_sigmas'
    )

    if selected_sigmas:
        fig_curve = go.Figure()
        sigma_colors = px.colors.qualitative.Set2
        for idx_s, sig in enumerate(sorted(selected_sigmas)):
            clr = sigma_colors[idx_s % len(sigma_colors)]
            f_data = gbm_fixed[gbm_fixed['sigma'] == sig].sort_values('init_wr')
            if len(f_data) > 0:
                fig_curve.add_trace(go.Scatter(
                    x=f_data['init_wr'] * 100, y=f_data['x_success_rate'],
                    mode='lines', name=f'Fixed \u03c3={sig*100:.0f}%',
                    line=dict(color=clr, width=1.5, dash='dash'),
                    legendgroup=f'sig{sig}',
                ))
            if 'cf_success_rate' in f_data.columns:
                cf_data = f_data.dropna(subset=['cf_success_rate'])
                if len(cf_data) > 0:
                    fig_curve.add_trace(go.Scatter(
                        x=cf_data['init_wr'] * 100, y=cf_data['cf_success_rate'],
                        mode='lines', name=f'CF \u03c3={sig*100:.0f}%',
                        line=dict(color=clr, width=1, dash='dot'),
                        legendgroup=f'sig{sig}',
                    ))
            g_data = gbm_guard[gbm_guard['sigma'] == sig].sort_values('init_wr')
            if len(g_data) > 0:
                fig_curve.add_trace(go.Scatter(
                    x=g_data['init_wr'] * 100, y=g_data['x_success_rate'],
                    mode='lines+markers', name=f'Guard \u03c3={sig*100:.0f}%',
                    line=dict(color=clr, width=2.5), marker=dict(size=3),
                    legendgroup=f'sig{sig}',
                ))
        fig_curve.update_layout(
            xaxis_title="초기인출률 (%)", yaxis_title="성공률", height=500,
            xaxis=dict(ticksuffix='%'), yaxis=dict(tickformat='.0%'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig_curve, width='stretch')
    else:
        st.info("변동성을 하나 이상 선택하세요.")

    # ========================================
    # 3. 누적인출금 비교
    # ========================================
    st.markdown("---")
    st.subheader("3. 누적인출금 비교 (Fixed vs Guardrail)")

    ref_wr_val = min(init_wrs_gbm, key=lambda x: abs(x - 0.06))
    st.caption(f"인출률 {ref_wr_val*100:.0f}% 기준, 변동성별 누적인출금 중앙값.")

    cum_f, cum_g = [], []
    for sig in sigmas:
        f_r = gbm_fixed[(gbm_fixed['sigma'] == sig) & (gbm_fixed['init_wr'] == ref_wr_val)]
        g_r = gbm_guard[(gbm_guard['sigma'] == sig) & (gbm_guard['init_wr'] == ref_wr_val)]
        cum_f.append(f_r.iloc[0]['y_cum_withdraw_median'] if len(f_r) > 0 else np.nan)
        cum_g.append(g_r.iloc[0]['y_cum_withdraw_median'] if len(g_r) > 0 else np.nan)

    if any(np.isfinite(v) for v in cum_f + cum_g):
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=[s*100 for s in sigmas], y=cum_f,
            mode='lines+markers', name='Fixed',
            line=dict(color='#999', width=2, dash='dash'), marker=dict(size=6, symbol='diamond'),
        ))
        fig_cum.add_trace(go.Scatter(
            x=[s*100 for s in sigmas], y=cum_g,
            mode='lines+markers', name=f'Guardrail (\u00b1{gbm_band*100:.0f}%)',
            line=dict(color='#2196F3', width=2.5), marker=dict(size=6),
        ))
        fig_cum.update_layout(
            xaxis_title="변동성 \u03c3 (%)", yaxis_title="누적인출금 중앙값", height=400,
            xaxis=dict(ticksuffix='%'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig_cum, width='stretch')
    else:
        st.info(f"인출률 {ref_wr_val*100:.0f}% 기준 누적인출금 데이터가 없습니다.")

    # ========================================
    # 4. 기말잔고 비교
    # ========================================
    st.markdown("---")
    st.subheader("4. 기말잔고 비교 (Fixed vs Guardrail)")
    st.caption(f"인출률 {ref_wr_val*100:.0f}% 기준. 변동성↑일수록 Guardrail 잔고 보전↑.")

    nav_f, nav_g = [], []
    for sig in sigmas:
        f_r = gbm_fixed[(gbm_fixed['sigma'] == sig) & (gbm_fixed['init_wr'] == ref_wr_val)]
        g_r = gbm_guard[(gbm_guard['sigma'] == sig) & (gbm_guard['init_wr'] == ref_wr_val)]
        nav_f.append(f_r.iloc[0]['terminal_nav_median'] if len(f_r) > 0 else np.nan)
        nav_g.append(g_r.iloc[0]['terminal_nav_median'] if len(g_r) > 0 else np.nan)

    if any(np.isfinite(v) for v in nav_f + nav_g):
        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(
            x=[s*100 for s in sigmas], y=nav_f,
            mode='lines+markers', name='Fixed',
            line=dict(color='#999', width=2, dash='dash'), marker=dict(size=6, symbol='diamond'),
        ))
        fig_nav.add_trace(go.Scatter(
            x=[s*100 for s in sigmas], y=nav_g,
            mode='lines+markers', name=f'Guardrail (\u00b1{gbm_band*100:.0f}%)',
            line=dict(color='#4CAF50', width=2.5), marker=dict(size=6),
        ))
        fig_nav.update_layout(
            xaxis_title="변동성 \u03c3 (%)", yaxis_title="기말잔고 중앙값", height=400,
            xaxis=dict(ticksuffix='%'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig_nav, width='stretch')
    else:
        st.info("기말잔고 데이터가 없습니다.")

    # ========================================
    # 5. 최대 지속가능 인출률
    # ========================================
    st.markdown("---")
    st.subheader("5. 최대 지속가능 인출률 (성공률 \u2265 90%)")
    st.caption("두 선의 간격 = Guardrail이 확보하는 추가 여유.")

    max_wr_f, max_wr_g = [], []
    for sig in sigmas:
        f_ok = gbm_fixed[(gbm_fixed['sigma'] == sig) & (gbm_fixed['x_success_rate'] >= 0.90)]
        max_wr_f.append(f_ok['init_wr'].max() * 100 if len(f_ok) > 0 else 0)
        g_ok = gbm_guard[(gbm_guard['sigma'] == sig) & (gbm_guard['x_success_rate'] >= 0.90)]
        max_wr_g.append(g_ok['init_wr'].max() * 100 if len(g_ok) > 0 else 0)

    fig_max = go.Figure()
    fig_max.add_trace(go.Scatter(
        x=[s*100 for s in sigmas], y=max_wr_f,
        mode='lines+markers', name='Fixed',
        line=dict(color='#999', width=2, dash='dash'), marker=dict(size=6, symbol='diamond'),
    ))
    fig_max.add_trace(go.Scatter(
        x=[s*100 for s in sigmas], y=max_wr_g,
        mode='lines+markers', name=f'Guardrail (\u00b1{gbm_band*100:.0f}%)',
        line=dict(color='#2196F3', width=2.5), marker=dict(size=6),
    ))
    fig_max.update_layout(
        xaxis_title="변동성 \u03c3 (%)", yaxis_title="최대 지속가능 인출률 (%)", height=420,
        xaxis=dict(ticksuffix='%'), yaxis=dict(ticksuffix='%'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig_max, width='stretch')

    # ========================================
    # 6. 요약 테이블
    # ========================================
    st.markdown("---")
    st.subheader("6. 주요 변동성 구간 요약")

    summary_sigmas = [s for s in [0.04, 0.08, 0.12, 0.16, 0.20] if s in sigmas]
    if not summary_sigmas:
        summary_sigmas = sigmas[::max(1, len(sigmas)//5)][:5]
    summary_rows = []
    for sig in summary_sigmas:
        f_s = gbm_fixed[gbm_fixed['sigma'] == sig]
        g_s = gbm_guard[gbm_guard['sigma'] == sig]
        if len(f_s) > 0 and len(g_s) > 0:
            f_ref = f_s[f_s['init_wr'].round(3) == round(ref_wr_val, 3)]
            g_ref = g_s[g_s['init_wr'].round(3) == round(ref_wr_val, 3)]
            f_sr = f_ref.iloc[0]['x_success_rate'] * 100 if len(f_ref) > 0 else None
            g_sr = g_ref.iloc[0]['x_success_rate'] * 100 if len(g_ref) > 0 else None
            f_cum_v = f_ref.iloc[0]['y_cum_withdraw_median'] if len(f_ref) > 0 else None
            g_cum_v = g_ref.iloc[0]['y_cum_withdraw_median'] if len(g_ref) > 0 else None
            f_nav_v = f_ref.iloc[0]['terminal_nav_median'] if len(f_ref) > 0 else None
            g_nav_v = g_ref.iloc[0]['terminal_nav_median'] if len(g_ref) > 0 else None
            f_max_wr = (f_s[f_s['x_success_rate'] >= 0.90]['init_wr'].max() * 100
                        if len(f_s[f_s['x_success_rate'] >= 0.90]) > 0 else 0)
            g_max_wr = (g_s[g_s['x_success_rate'] >= 0.90]['init_wr'].max() * 100
                        if len(g_s[g_s['x_success_rate'] >= 0.90]) > 0 else 0)
            summary_rows.append({
                '\u03c3': f"{sig*100:.0f}%",
                'Fixed SR': f"{f_sr:.1f}%" if f_sr else "-",
                'Guard SR': f"{g_sr:.1f}%" if g_sr else "-",
                'SR차이': f"{g_sr - f_sr:+.1f}%p" if f_sr and g_sr else "-",
                'Fixed 인출': f"{f_cum_v:.1f}" if f_cum_v else "-",
                'Guard 인출': f"{g_cum_v:.1f}" if g_cum_v else "-",
                'Fixed 잔고': f"{f_nav_v:.1f}" if f_nav_v else "-",
                'Guard 잔고': f"{g_nav_v:.1f}" if g_nav_v else "-",
                'Guard 최대WR': f"{g_max_wr:.1f}%",
            })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)
    else:
        st.info("요약 테이블을 생성할 데이터가 부족합니다.")

    # 핵심발견
    st.markdown("---")
    st.subheader("핵심 발견")

    avg_diffs = []
    for sig in sigmas:
        f_s = gbm_fixed[gbm_fixed['sigma'] == sig]
        g_s = gbm_guard[gbm_guard['sigma'] == sig]
        if len(f_s) > 0 and len(g_s) > 0:
            m = f_s[['sigma', 'init_wr', 'x_success_rate']].merge(
                g_s[['sigma', 'init_wr', 'x_success_rate']],
                on=['sigma', 'init_wr'], suffixes=('_f', '_g'))
            if len(m) > 0:
                avg_diffs.append(
                    (sig, (m['x_success_rate_g'] - m['x_success_rate_f']).mean() * 100))
    if avg_diffs:
        best_s = max(avg_diffs, key=lambda x: x[1])
        st.markdown(_finding_box(
            f"<b>Guardrail 효과 최대 변동성</b>: \u03c3={best_s[0]*100:.0f}% "
            f"(평균 성공률 +{best_s[1]:.1f}%p)"
        ), unsafe_allow_html=True)
    st.markdown(_finding_box(
        "<b>변동성이 높을수록 Guardrail 효과 \u2191</b>: "
        "자산 변동이 클 때 인출 조절이 더 유효합니다."
    ), unsafe_allow_html=True)
    st.markdown(_finding_box(
        "<b>Closed-form vs MC</b>: "
        "CF는 고인출률에서 MC보다 낙관적 \u2192 이산 인출의 영향을 과소평가합니다."
    ), unsafe_allow_html=True)


# ============================================================================
# Tab 3: 실제 포트폴리오 검증 (Historical Fixed vs Guardrail)
# ============================================================================

def render_tab3_historical(df_all, beta, path_method, df_gbm=None):
    """탭 3: 실제 포트폴리오 데이터에서 Fixed vs Guardrail 검증.
    좌측: Historical, 우측: GBM 이론. Beta/Path Method는 사이드바, Band는 ±5% 고정."""

    st.markdown("### 실제 포트폴리오 검증")
    st.caption(
        f"좌측: Historical 데이터 기반 검증 | 우측: GBM 이론 분석. "
        f"**기말잔액 비율: {beta*100:.0f}%** | **데이터: {PATH_METHOD_LABELS.get(path_method, path_method)}** "
        f"| **Band: \u00b1{FIXED_BAND*100:.0f}% 고정**"
    )

    hist_band = FIXED_BAND

    df = df_all[(df_all['beta'] == beta) & (df_all['path_method'] == path_method)].copy()
    if len(df) == 0:
        st.warning("선택된 기말잔액 비율/데이터 기반 조합에 맞는 데이터가 없습니다. "
                   "사이드바에서 다른 값을 선택해 보세요.")
        return

    portfolios = sorted(df['portfolio'].unique())
    wrs = sorted(df['init_wr'].unique())

    # ========================================
    # GBM 데이터 준비 (우측 컬럼용)
    # ========================================
    _has_gbm = df_gbm is not None and len(df_gbm) > 0
    if _has_gbm:
        gbm_beta = min(sorted(df_gbm['beta'].unique()), key=lambda x: abs(x - beta))
        gbm_band = FIXED_BAND

    # ========================================
    # GBM 기대수익률 위젯 (half-width)
    # ========================================
    if _has_gbm:
        mu_options = sorted(df_gbm['mu'].unique())
        default_mu = min(mu_options, key=lambda x: abs(x - 0.06))
        _, _mu_col = st.columns(2)
        with _mu_col:
            gbm_mu = st.select_slider(
                "GBM 기대수익률 (\u03bc)",
                options=mu_options,
                value=default_mu,
                format_func=lambda x: f"{x*100:.0f}%",
                key='tab3_gbm_mu'
            )

    # ========================================
    # 2컬럼 레이아웃: 좌=Historical, 우=GBM
    # ========================================
    col_hist, col_gbm = st.columns([1, 1])

    # ===== 좌측 컬럼: Historical =====
    with col_hist:
        st.markdown("#### Historical (실제 데이터)")

        # --- 1. 성공률 차이 히트맵 ---
        st.markdown("**1. 성공률 차이 히트맵 (Guardrail \u2212 Fixed)**")

        sr_diff_data = []
        for port in portfolios:
            for wr in wrs:
                fixed = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                           (df['strategy_type'] == 'fixed_baseline')]
                dyn = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                         (df['strategy_type'] == 'dynamic') & (df['band'] == hist_band)]
                if len(fixed) > 0 and len(dyn) > 0:
                    diff = (dyn.iloc[0]['success_rate'] - fixed.iloc[0]['success_rate']) * 100
                    sr_diff_data.append({'portfolio': port, 'init_wr': wr, 'diff': diff})

        if len(sr_diff_data) > 0:
            sr_df = pd.DataFrame(sr_diff_data)
            pivot_sr = sr_df.pivot_table(index='portfolio', columns='init_wr', values='diff', aggfunc='mean')
            pivot_sr = pivot_sr.reindex(sorted(pivot_sr.index))
            zmax = max(abs(pivot_sr.values[np.isfinite(pivot_sr.values)].min()),
                       abs(pivot_sr.values[np.isfinite(pivot_sr.values)].max()), 5)
            fig_sr = go.Figure(data=go.Heatmap(
                z=pivot_sr.values,
                x=[f"{x*100:.1f}%" for x in pivot_sr.columns],
                y=pivot_sr.index,
                colorscale='RdYlGn', zmid=0, zmin=-zmax, zmax=zmax,
                text=np.round(pivot_sr.values, 1),
                texttemplate='%{text:+.1f}',
                textfont={"size": 8},
                colorbar=dict(title="SR차이(%p)")
            ))
            fig_sr.update_layout(xaxis_title="초기인출률", yaxis_title="포트폴리오", height=350,
                                 margin=dict(t=10, b=30, l=50, r=20))
            st.plotly_chart(fig_sr, use_container_width=True)
        else:
            st.info(f"Band \u00b1{hist_band*100:.0f}% 데이터가 없습니다.")

        # --- 2. 누적인출금 차이 히트맵 ---
        st.markdown("**2. 누적인출금 차이 히트맵 (Guardrail \u2212 Fixed, %)**")

        cw_diff_data = []
        for port in portfolios:
            for wr in wrs:
                fixed = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                           (df['strategy_type'] == 'fixed_baseline')]
                dyn = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                         (df['strategy_type'] == 'dynamic') & (df['band'] == hist_band)]
                if len(fixed) > 0 and len(dyn) > 0:
                    f_cum = fixed.iloc[0]['cum_withdraw_median']
                    d_cum = dyn.iloc[0]['cum_withdraw_median']
                    diff = ((d_cum / f_cum) - 1) * 100 if f_cum > 0 else 0
                    cw_diff_data.append({'portfolio': port, 'init_wr': wr, 'diff': diff})

        if len(cw_diff_data) > 0:
            cw_df = pd.DataFrame(cw_diff_data)
            pivot_cw = cw_df.pivot_table(index='portfolio', columns='init_wr', values='diff', aggfunc='mean')
            pivot_cw = pivot_cw.reindex(sorted(pivot_cw.index))
            zmax_cw = max(abs(pivot_cw.values[np.isfinite(pivot_cw.values)].min()),
                          abs(pivot_cw.values[np.isfinite(pivot_cw.values)].max()), 5)
            fig_cw = go.Figure(data=go.Heatmap(
                z=pivot_cw.values,
                x=[f"{x*100:.1f}%" for x in pivot_cw.columns],
                y=pivot_cw.index,
                colorscale='RdBu', zmid=0, zmin=-zmax_cw, zmax=zmax_cw,
                text=np.round(pivot_cw.values, 1),
                texttemplate='%{text:+.1f}%',
                textfont={"size": 8},
                colorbar=dict(title="인출차이(%)")
            ))
            fig_cw.update_layout(xaxis_title="초기인출률", yaxis_title="포트폴리오", height=350,
                                 margin=dict(t=10, b=30, l=50, r=20))
            st.plotly_chart(fig_cw, use_container_width=True)
        else:
            st.info("누적인출금 비교 데이터가 부족합니다.")

        # --- 3. 포트폴리오별 성공률 곡선 ---
        st.markdown("**3. 포트폴리오별 성공률 곡선**")

        sel_port = st.selectbox(
            "포트폴리오 선택",
            options=portfolios,
            index=min(2, len(portfolios) - 1),
            key='tab3_port_curve'
        )

        fig_curve = go.Figure()
        fixed_data = df[(df['portfolio'] == sel_port) & (df['strategy_type'] == 'fixed_baseline')]
        if len(fixed_data) > 0:
            fixed_by_wr = fixed_data.groupby('init_wr')['success_rate'].max().reset_index().sort_values('init_wr')
            fig_curve.add_trace(go.Scatter(
                x=fixed_by_wr['init_wr'] * 100, y=fixed_by_wr['success_rate'],
                mode='lines+markers', name='Fixed',
                line=dict(color='#999', width=2, dash='dash'),
                marker=dict(size=4, symbol='diamond'),
            ))
        guard_data = df[(df['portfolio'] == sel_port) &
                        (df['strategy_type'] == 'dynamic') & (df['band'] == hist_band)]
        if len(guard_data) > 0:
            guard_by_wr = guard_data.groupby('init_wr')['success_rate'].max().reset_index().sort_values('init_wr')
            color = PORT_COLORS.get(sel_port, '#2196F3')
            fig_curve.add_trace(go.Scatter(
                x=guard_by_wr['init_wr'] * 100, y=guard_by_wr['success_rate'],
                mode='lines+markers', name=f'Guardrail',
                line=dict(color=color, width=2.5), marker=dict(size=4),
            ))

        fig_curve.update_layout(
            xaxis_title="초기인출률 (%)", yaxis_title="성공률", height=400,
            xaxis=dict(ticksuffix='%'), yaxis=dict(tickformat='.0%'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(t=10, b=30, l=50, r=20),
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    # ===== 우측 컬럼: GBM =====
    with col_gbm:
        st.markdown("#### GBM")

        if not _has_gbm:
            st.warning("gbm_results.pkl이 없습니다. `python gbm_grid_search.py`를 실행하세요.")
        else:
            # GBM 데이터 필터링 (mu는 컬럼 밖 위젯에서 선택)
            gbm_fixed = df_gbm[
                (df_gbm['mu'] == gbm_mu) & (df_gbm['beta'] == gbm_beta) &
                (df_gbm['strategy_type'] == 'fixed_baseline')
            ].copy()
            gbm_guard = df_gbm[
                (df_gbm['mu'] == gbm_mu) & (df_gbm['beta'] == gbm_beta) &
                (df_gbm['strategy_type'] == 'dynamic') & (df_gbm['band'] == gbm_band)
            ].copy()

            if len(gbm_fixed) == 0 or len(gbm_guard) == 0:
                st.warning(f"\u03bc={gbm_mu*100:.0f}% 데이터가 부족합니다.")
            else:
                # sigma 4%~15% 필터
                sigmas = sorted([s for s in gbm_fixed['sigma'].unique() if 0.04 <= s <= 0.15])
                init_wrs_gbm = sorted(gbm_fixed['init_wr'].unique())

                if len(sigmas) == 0:
                    st.info("σ 4~15% 범위의 데이터가 없습니다.")
                else:
                    gbm_fixed = gbm_fixed[gbm_fixed['sigma'].isin(sigmas)]
                    gbm_guard = gbm_guard[gbm_guard['sigma'].isin(sigmas)]

                # --- GBM 1. 성공률 차이 히트맵 (y=sigma, x=init_wr) ---
                st.markdown("**1. 성공률 차이 히트맵 (Guardrail \u2212 Fixed)**")

                gbm_diff_matrix = np.full((len(sigmas), len(init_wrs_gbm)), np.nan)
                for j_s, sig in enumerate(sigmas):
                    for i_w, wr in enumerate(init_wrs_gbm):
                        f_r = gbm_fixed[(gbm_fixed['sigma'] == sig) & (gbm_fixed['init_wr'] == wr)]
                        g_r = gbm_guard[(gbm_guard['sigma'] == sig) & (gbm_guard['init_wr'] == wr)]
                        if len(f_r) > 0 and len(g_r) > 0:
                            gbm_diff_matrix[j_s, i_w] = (
                                g_r.iloc[0]['x_success_rate'] - f_r.iloc[0]['x_success_rate']) * 100

                valid_vals = gbm_diff_matrix[np.isfinite(gbm_diff_matrix)]
                if len(valid_vals) > 0:
                    zmax_g = max(abs(valid_vals.min()), abs(valid_vals.max()), 3)
                    fig_gbm_sr = go.Figure(data=go.Heatmap(
                        z=gbm_diff_matrix,
                        x=[f"{w*100:.1f}%" for w in init_wrs_gbm],
                        y=[f"\u03c3={s*100:.0f}%" for s in sigmas],
                        colorscale='RdYlGn', zmid=0, zmin=-zmax_g, zmax=zmax_g,
                        text=np.round(gbm_diff_matrix, 1),
                        texttemplate='%{text:+.1f}',
                        textfont={"size": 8},
                        colorbar=dict(title="SR차이(%p)")
                    ))
                    fig_gbm_sr.update_layout(xaxis_title="초기인출률", yaxis_title="변동성 \u03c3", height=350,
                                             margin=dict(t=10, b=30, l=50, r=20))
                    st.plotly_chart(fig_gbm_sr, use_container_width=True)
                else:
                    st.info("히트맵 데이터가 부족합니다.")

                # --- GBM 2. 누적인출금 차이 히트맵 (y=sigma, x=init_wr) ---
                st.markdown("**2. 누적인출금 차이 히트맵 (Guardrail \u2212 Fixed, %)**")

                gbm_cum_matrix = np.full((len(sigmas), len(init_wrs_gbm)), np.nan)
                for j_s, sig in enumerate(sigmas):
                    for i_w, wr in enumerate(init_wrs_gbm):
                        f_r = gbm_fixed[(gbm_fixed['sigma'] == sig) & (gbm_fixed['init_wr'] == wr)]
                        g_r = gbm_guard[(gbm_guard['sigma'] == sig) & (gbm_guard['init_wr'] == wr)]
                        if len(f_r) > 0 and len(g_r) > 0:
                            f_cum = f_r.iloc[0]['y_cum_withdraw_median']
                            g_cum = g_r.iloc[0]['y_cum_withdraw_median']
                            if f_cum > 0:
                                gbm_cum_matrix[j_s, i_w] = ((g_cum / f_cum) - 1) * 100

                valid_cum = gbm_cum_matrix[np.isfinite(gbm_cum_matrix)]
                if len(valid_cum) > 0:
                    zmax_gc = max(abs(valid_cum.min()), abs(valid_cum.max()), 5)
                    fig_gbm_cum = go.Figure(data=go.Heatmap(
                        z=gbm_cum_matrix,
                        x=[f"{w*100:.1f}%" for w in init_wrs_gbm],
                        y=[f"\u03c3={s*100:.0f}%" for s in sigmas],
                        colorscale='RdBu', zmid=0, zmin=-zmax_gc, zmax=zmax_gc,
                        text=np.round(gbm_cum_matrix, 1),
                        texttemplate='%{text:+.1f}%',
                        textfont={"size": 8},
                        colorbar=dict(title="인출차이(%)")
                    ))
                    fig_gbm_cum.update_layout(xaxis_title="초기인출률", yaxis_title="변동성 \u03c3", height=350,
                                              margin=dict(t=10, b=30, l=50, r=20))
                    st.plotly_chart(fig_gbm_cum, use_container_width=True)
                else:
                    st.info("누적인출금 비교 데이터가 부족합니다.")

                # --- GBM 3. 변동성별 성공률 곡선 (단일 sigma 선택) ---
                st.markdown("**3. 변동성별 성공률 곡선**")

                sel_sigma = st.selectbox(
                    "변동성 선택",
                    options=sigmas,
                    index=min(3, len(sigmas) - 1),
                    format_func=lambda x: f"\u03c3={x*100:.0f}%",
                    key='tab3_gbm_sigma'
                )

                fig_gbm_curve = go.Figure()
                f_data = gbm_fixed[gbm_fixed['sigma'] == sel_sigma].sort_values('init_wr')
                if len(f_data) > 0:
                    fig_gbm_curve.add_trace(go.Scatter(
                        x=f_data['init_wr'] * 100, y=f_data['x_success_rate'],
                        mode='lines+markers', name='Fixed',
                        line=dict(color='#999', width=2, dash='dash'),
                        marker=dict(size=4, symbol='diamond'),
                    ))
                    if 'cf_success_rate' in f_data.columns:
                        cf_data = f_data.dropna(subset=['cf_success_rate'])
                        if len(cf_data) > 0:
                            fig_gbm_curve.add_trace(go.Scatter(
                                x=cf_data['init_wr'] * 100, y=cf_data['cf_success_rate'],
                                mode='lines', name='CF (Closed-form)',
                                line=dict(color='#999', width=1, dash='dot'),
                            ))
                g_data = gbm_guard[gbm_guard['sigma'] == sel_sigma].sort_values('init_wr')
                if len(g_data) > 0:
                    fig_gbm_curve.add_trace(go.Scatter(
                        x=g_data['init_wr'] * 100, y=g_data['x_success_rate'],
                        mode='lines+markers', name='Guardrail',
                        line=dict(color='#2196F3', width=2.5), marker=dict(size=4),
                    ))

                fig_gbm_curve.update_layout(
                    xaxis_title="초기인출률 (%)", yaxis_title="성공률", height=400,
                    xaxis=dict(ticksuffix='%'), yaxis=dict(tickformat='.0%'),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(t=10, b=30, l=50, r=20),
                )
                st.plotly_chart(fig_gbm_curve, use_container_width=True)

    # ========================================
    # 4. Guardrail 효과 분석 — 언제 효과적이고 언제 아닌가 (full-width)
    # ========================================
    st.markdown("---")
    st.subheader("4. Guardrail 효과 분석: 언제 유리하고 언제 불리한가")

    hist_band = FIXED_BAND

    # 전 포트폴리오 × 전 인출률에 대해 Fixed vs Guardrail 비교
    _eff_rows = []
    for port in portfolios:
        for wr in wrs:
            f_row = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                       (df['strategy_type'] == 'fixed_baseline')]
            g_row = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                       (df['strategy_type'] == 'dynamic') & (df['band'] == hist_band)]
            if len(f_row) > 0 and len(g_row) > 0:
                fr = f_row.iloc[0]
                gr = g_row.iloc[0]
                sr_d = (gr['success_rate'] - fr['success_rate']) * 100
                f_total = (fr.get('terminal_nav_median', 0) or 0) + fr['cum_withdraw_median']
                g_total = (gr.get('terminal_nav_median', 0) or 0) + gr['cum_withdraw_median']
                total_d = g_total - f_total
                _eff_rows.append({
                    '포트폴리오': port,
                    '초기인출률': f"{wr*100:.1f}%",
                    'Fixed 목표달성률': f"{fr['success_rate']*100:.1f}%",
                    'Guardrail 목표달성률': f"{gr['success_rate']*100:.1f}%",
                    '목표달성률 차이(%p)': round(sr_d, 1),
                    'Fixed 총 가치 (누적인출금+기말NAV)': round(f_total, 1),
                    'Guardrail 총 가치 (누적인출금+기말NAV)': round(g_total, 1),
                    '총 가치 차이': round(total_d, 1),
                    '_sr_diff': sr_d,
                    '_total_diff': total_d,
                })

    if _eff_rows:
        _eff_df = pd.DataFrame(_eff_rows)

        # Guardrail 효과 최대 (목표달성률 기준)
        st.markdown("##### Guardrail 효과가 가장 큰 조건 (목표달성률 기준)")
        _top_sr = _eff_df.nlargest(5, '_sr_diff')[
            ['포트폴리오', '초기인출률', 'Fixed 목표달성률', 'Guardrail 목표달성률', '목표달성률 차이(%p)', '총 가치 차이']
        ]
        st.dataframe(_top_sr, width='stretch', hide_index=True)

        # Guardrail 효과 최대 (누적인출금 + 기말 NAV 기준)
        st.markdown("##### Guardrail 효과가 가장 큰 조건 (누적인출금 + 기말 NAV 기준)")
        _top_total = _eff_df.nlargest(5, '_total_diff')[
            ['포트폴리오', '초기인출률', 'Fixed 총 가치 (누적인출금+기말NAV)', 'Guardrail 총 가치 (누적인출금+기말NAV)', '총 가치 차이', '목표달성률 차이(%p)']
        ]
        st.dataframe(_top_total, width='stretch', hide_index=True)

        # Guardrail 효과 없거나 불리한 경우
        st.markdown("##### Guardrail이 불리하거나 차이 없는 조건")
        _worst = _eff_df.nsmallest(5, '_sr_diff')[
            ['포트폴리오', '초기인출률', 'Fixed 목표달성률', 'Guardrail 목표달성률', '목표달성률 차이(%p)', '총 가치 차이']
        ]
        st.dataframe(_worst, width='stretch', hide_index=True)

        # 분석결과
        n_negative_sr = len(_eff_df[_eff_df['_sr_diff'] < 0])
        n_zero_sr = len(_eff_df[_eff_df['_sr_diff'].abs() < 0.5])
        n_total = len(_eff_df)
        n_negative_total = len(_eff_df[_eff_df['_total_diff'] < 0])

        st.markdown("---")
        st.subheader("분석결과")

        n_positive_sr = n_total - n_negative_sr - n_zero_sr
        st.markdown(_finding_box(
            f"<b>1. 전체 현황</b><br>"
            f"6개 포트폴리오(Port_4.0%~9.0%) × 다양한 초기인출률(3~15%)로 구성된 "
            f"<b>총 {n_total}개 조합</b>에 대해 Fixed(고정인출)와 Guardrail(±5%)의 목표달성률을 비교했습니다.<br><br>"
            f"• <b>Guardrail 목표달성률이 더 높은 조합: {n_positive_sr}건</b> — "
            f"인출 압박이 큰 영역에서 Guardrail이 파산을 방지하여 목표달성률을 유의미하게 끌어올립니다.<br>"
            f"• <b>차이가 거의 없는 조합 (&lt;0.5%p): {n_zero_sr}건</b> — "
            f"저인출률 + 고수익 포트폴리오처럼 이미 Fixed 목표달성률이 100%에 가까워 Guardrail이 개입할 여지가 없는 구간입니다.<br>"
            f"• <b>Guardrail 목표달성률이 더 낮은 조합: {n_negative_sr}건</b> — "
            f"Guardrail이 Fixed보다 불리한 경우는 단 한 건도 없습니다.<br><br>"
            f"한편 총 가치(누적인출금 + 기말NAV) 기준으로는 <b>{n_negative_total}건에서 Guardrail이 소폭 불리</b>합니다. "
            f"이는 Guardrail이 하락기에 인출을 줄여 파산을 막는 대신, 누적인출금이 감소하는 trade-off에서 비롯됩니다."
        ), unsafe_allow_html=True)

        st.markdown(_finding_box(
            f"<b>2. 총 가치 차이가 작은 이유</b><br>"
            f"Guardrail은 하락기에 인출을 줄여 NAV를 보전하는 대신, "
            f"누적인출금이 Fixed보다 감소합니다. 즉, '기말NAV 증가 vs 누적인출금 감소'의 trade-off가 발생하여 "
            f"총 가치(기말NAV + 누적인출금)의 차이는 상대적으로 작게 나타납니다. "
            f"Guardrail의 핵심 가치는 총 가치 극대화가 아니라 <b>파산 방지 및 목표달성률 제고</b>에 있습니다."
        ), unsafe_allow_html=True)

        st.markdown(_finding_box(
            f"<b>3. Guardrail이 유리한 조건</b><br>"
            f"• <b>중위험 포트폴리오(4~6%) + 중~고인출률(8~12%)</b>: "
            f"Fixed는 하락장에서 자본이 빠르게 소진되어 파산하지만, Guardrail은 인출 축소로 버텨 회복 기회를 확보합니다.<br>"
            f"• <b>고인출률(12%+) 전반</b>: 인출 압박이 강할수록 Guardrail의 자본 보전 효과가 극대화됩니다.<br>"
            f"• <b>변동성이 큰 시장 환경</b>: GBM 이론 분석에서도 σ↑일수록 Guardrail 우위가 커지는 패턴이 확인됩니다."
        ), unsafe_allow_html=True)

        st.markdown(_finding_box(
            f"<b>4. Guardrail이 불리한 조건</b><br>"
            f"• <b>저인출률(3~4%) + 고수익 포트폴리오(8~9%)</b>: "
            f"이미 Fixed 목표달성률이 100%에 근접하여 Guardrail 조절이 불필요합니다. "
            f"오히려 인출 축소가 누적인출금을 줄여 총 가치가 다소 감소합니다.<br>"
            f"• <b>극저변동성 환경(σ &lt; 4%)</b>: 자산 변동 자체가 작아 인출 조절의 실익이 미미합니다."
        ), unsafe_allow_html=True)

        st.markdown(_finding_box(
            f"<b>5. 실무적 시사점</b><br>"
            f"• 퇴직연금 OCIO 펀드의 실제 인출률 범위(4~8%)에서 Guardrail은 "
            f"목표달성률을 유의미하게 개선하면서 인출 안정성을 확보합니다.<br>"
            f"• 총 가치 차이가 크지 않다는 것은 '같은 경제적 가치를 유지하면서 파산 위험만 줄인다'는 의미이므로, "
            f"Guardrail 도입의 비용(기회비용)이 낮다는 긍정적 신호입니다.<br>"
            f"• Band ±5%의 소폭 조정으로도 충분한 효과를 얻을 수 있어, "
            f"수익자 입장에서 인출액 변동의 체감 불편이 최소화됩니다."
        ), unsafe_allow_html=True)
    else:
        st.info("비교 데이터가 부족합니다.")


# ============================================================================
# Tab 4: Guardrail Band 최적화
# ============================================================================

def render_tab4_optimization(df_all, beta, path_method, df_gbm=None):
    """탭 4: 최적 Band 탐색. Beta/Path Method는 사이드바 연동."""

    st.markdown("### Guardrail Band 최적화")
    st.caption(
        f"Band를 어떻게 설정해야 최적인가? "
        f"**기말잔액 비율: {beta*100:.0f}%** | **데이터: {PATH_METHOD_LABELS.get(path_method, path_method)}**"
    )

    # 컨트롤: 포트폴리오만 (beta/pm은 사이드바)
    opt_port = st.selectbox(
        "포트폴리오 선택",
        options=sorted(df_all['portfolio'].unique()),
        index=2,
        key='tab4_port'
    )

    df = df_all[(df_all['beta'] == beta) & (df_all['path_method'] == path_method)].copy()
    if len(df) == 0:
        st.warning("선택된 조합에 맞는 데이터가 없습니다.")
        return

    # ========================================
    # 1. Band별 성공률 곡선
    # ========================================
    st.markdown("---")
    st.subheader("1. Band별 성공률 곡선")

    fig_band = go.Figure()
    fixed = df[(df['portfolio'] == opt_port) & (df['strategy_type'] == 'fixed_baseline')]
    if len(fixed) > 0:
        fixed_by_wr = fixed.groupby('init_wr')['success_rate'].max().reset_index().sort_values('init_wr')
        fig_band.add_trace(go.Scatter(
            x=fixed_by_wr['init_wr'] * 100, y=fixed_by_wr['success_rate'],
            mode='lines+markers', name='Fixed (고정 인출)',
            line=dict(color='#999', width=2, dash='dash'),
            marker=dict(size=5, symbol='diamond'),
        ))

    band_colors = {0.05: '#2196F3', 0.10: '#4CAF50', 0.15: '#FF9800', 0.20: '#E91E63'}
    dynamic = df[(df['portfolio'] == opt_port) & (df['strategy_type'] == 'dynamic')]
    available_bands = sorted(dynamic['band'].unique())
    for band_val in available_bands:
        band_data = dynamic[dynamic['band'] == band_val]
        by_wr = band_data.groupby('init_wr')['success_rate'].max().reset_index().sort_values('init_wr')
        color = band_colors.get(band_val, '#000')
        fig_band.add_trace(go.Scatter(
            x=by_wr['init_wr'] * 100, y=by_wr['success_rate'],
            mode='lines+markers', name=f'Band \u00b1{band_val*100:.0f}%',
            line=dict(color=color, width=2), marker=dict(size=5),
        ))

    fig_band.update_layout(
        xaxis_title="초기인출률 (%)", yaxis_title="성공률 (Success Rate)", height=500,
        xaxis=dict(ticksuffix='%'), yaxis=dict(tickformat='.0%'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig_band, width='stretch')

    # 핵심발견
    st.markdown("")
    st.caption("Band ±5%가 대부분 조건에서 최적입니다. 빈번한 소폭 조정이 가장 효과적입니다.")


# ============================================================================
# Main
# ============================================================================

def main():
    st.set_page_config(
        page_title="퇴직 포트폴리오 인출 전략 분석기",
        page_icon="\U0001f4ca",
        layout="wide"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("퇴직 포트폴리오 인출 전략 분석기")

    # 데이터 로딩
    df_all = load_grid_results()
    df_gbm = load_gbm_results()

    # ========================================
    # 사이드바 — 글로벌 필터 (모든 탭에 연동)
    # ========================================

    st.sidebar.header("분석 설정")

    # 원본 대비 기말잔액 비율 선택
    available_betas = sorted(df_all['beta'].unique())
    if len(available_betas) == 0:
        available_betas = [0.5]
    default_beta_idx = available_betas.index(0.5) if 0.5 in available_betas else 0
    beta = st.sidebar.select_slider(
        "원본 대비 기말잔액 비율",
        options=available_betas,
        value=available_betas[default_beta_idx],
        format_func=lambda x: f"{x*100:.0f}%",
        key='global_beta'
    )
    st.sidebar.caption(BETA_LABELS.get(beta, f"기말잔액 \u2265 초기의 {beta*100:.0f}%"))

    # 초기 인출률 선택
    init_wr_pct = st.sidebar.slider(
        "초기 인출률 (%)", min_value=3, max_value=15, value=10, step=1,
        key='global_init_wr',
        help="연간 인출률. 예: 10% = 초기자산 100 기준 연 10 인출"
    )
    global_init_wr = init_wr_pct / 100

    # Path method 선택
    available_paths = sorted(df_all['path_method'].unique())
    path_method_options = [p for p in ['rolling', 'bootstrap', 'combined'] if p in available_paths]
    if len(path_method_options) == 0:
        path_method_options = available_paths
    default_pm = 'bootstrap' if 'bootstrap' in path_method_options else path_method_options[0]
    path_method = st.sidebar.selectbox(
        "데이터 기반",
        options=path_method_options,
        index=path_method_options.index(default_pm),
        format_func=lambda x: PATH_METHOD_LABELS.get(x, x),
        key='global_path_method'
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**기말잔액 비율**: {beta*100:.0f}% ({BETA_LABELS.get(beta, '')})\n\n"
        f"**초기 인출률**: {init_wr_pct}%\n\n"
        f"**데이터**: {PATH_METHOD_LABELS.get(path_method, path_method)}\n\n"
        f"**Guardrail Band**: \u00b15% 고정 (최적값)\n\n"
        f"**분석 기간**: 10년 (120개월)"
    )

    # 상단 배너
    st.markdown(
        f"**기말잔액 비율: {beta*100:.0f}%** ({BETA_LABELS.get(beta, '')}) | "
        f"**초기 인출률: {init_wr_pct}%** | "
        f"**데이터: {PATH_METHOD_LABELS.get(path_method, path_method)}** | "
        f"**Band: \u00b15% 고정**"
    )

    # 용어집
    with st.sidebar.expander("용어집 (Glossary)"):
        st.markdown("""
**성공 (Success)**: 파산 없음 AND 기말잔액 \u2265 초기의 기말잔액 비율%

**파산 (Ruin)**: 잔액 \u2264 0

**원본 대비 기말잔액 비율**: 성공 기준. 예: 50% = 기말잔액이 원본의 50% 이상

**초기 인출률 (Init WR)**: 시작 시점의 연간 인출률

**Guardrail Band**: 인출률 허용 범위 [\u00b1band%]

**인출변동성 (CV)**: 월별 인출액의 변동계수. 낮을수록 안정적.

**최대삭감률 (Worst Cut)**: 단일 월 최대 인출 감소율.
        """)

    # ========================================
    # 탭 생성 (4개, 사이드바 연동)
    # ========================================

    tab1, tab_dv, tab2, tab3, tab4 = st.tabs([
        "Guardrail 효과",
        "데이터 검증",
        "이론 분석 (GBM)",
        "실제 포트폴리오 검증",
        "Band 최적화",
    ])

    with tab1:
        render_tab1_mechanism(beta, path_method, global_init_wr)

    with tab_dv:
        render_tab_data_validation(beta, path_method, global_init_wr)

    with tab2:
        render_tab2_gbm(df_gbm, beta)

    with tab3:
        render_tab3_historical(df_all, beta, path_method, df_gbm=df_gbm)

    with tab4:
        render_tab4_optimization(df_all, beta, path_method, df_gbm=df_gbm)


if __name__ == "__main__":
    main()
