"""
퇴직 포트폴리오 인출 전략 분석기 (Streamlit Viewer v4 — Guardrail 스토리텔링)
============================================================================
Grid Search 결과를 6개 탭 스토리 흐름으로 시각화:
  1) Guardrail이란? — 개념 도입, 메커니즘 시각화
  2) 언제 Guardrail이 유리한가? — Fixed vs Guardrail 핵심 비교
  3) 최적 Band는? — Band별 분석
  4) 데이터 신뢰도 검증 — Rolling / Bootstrap / GBM 3종 비교
  5) 나의 전략 조합 — 사용자 선택형 탐색기
  6) 전략 상세 — NAV 경로 시뮬레이션 (on-demand)

v4 변경: 스토리텔링 기반 6탭 재구성, BETA_LABELS 5단계 확장
독립 실행: streamlit run viewer.py

UI 언어: 한국어 (Korean) + 영문 금융용어 병기.
  탭명, 레이블, 캡션, 용어집 등 사용자 대면 텍스트는 한국어로 작성하고,
  금융 전문용어(Success Rate, CV, Guardrail 등)는 괄호 안에 영문 병기.
  CLAUDE.md "한국어 주석/UI, 금융 전문용어는 영문 병기" 컨벤션을 따름.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
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

CUSTOM_CSS = """
<style>
/* 메트릭 카드 */
div[data-testid="stMetric"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 8px 12px;
}
/* 탭 헤더 크기 */
button[data-baseweb="tab"] {
    font-size: 1.05em;
}
/* 사이드바 용어집 */
.glossary-item {
    margin-bottom: 8px;
    padding: 6px 0;
    border-bottom: 1px solid #eee;
}
/* 핵심발견 박스 */
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
            path_returns=path,
            init_wr=init_wr,
            band=band,
            adj_on=False,
            beta=beta,
            W0=100.0,
            debug=False
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


def apply_global_filters(df, beta, path_method):
    """beta와 path_method로 글로벌 필터 적용"""
    return df[(df['beta'] == beta) & (df['path_method'] == path_method)].copy()


def filter_dataframe(df, portfolios, mode='All', frontier_only=False):
    """포트폴리오·전략모드·프론티어 필터 (테스트 호환용)"""
    filtered = df[df['portfolio'].isin(portfolios)].copy()
    if mode == 'Fixed':
        filtered = filtered[filtered['strategy_type'] == 'fixed_baseline']
    elif mode == 'Dynamic':
        filtered = filtered[filtered['strategy_type'] == 'dynamic']
    if frontier_only:
        filtered = filtered[filtered['is_frontier'] == True]
    return filtered


def _finding_box(text):
    """핵심발견 박스 HTML"""
    return f'<div class="finding-box">{text}</div>'


def _compute_crossover_data(df):
    """포트폴리오별 crossover 인출률 계산"""
    crossover_data = []
    for port in sorted(df['portfolio'].unique()):
        prev_gain = None
        for wr in sorted(df['init_wr'].unique()):
            f_row = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                       (df['strategy_type'] == 'fixed_baseline')]
            d_rows = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                        (df['strategy_type'] == 'dynamic')]
            if len(f_row) > 0 and len(d_rows) > 0:
                f_cum = f_row.iloc[0]['cum_withdraw_median']
                best_d = d_rows.loc[d_rows['success_rate'].idxmax()]
                d_cum = best_d['cum_withdraw_median']
                gain = ((d_cum / f_cum) - 1) * 100 if f_cum > 0 else 0
                if prev_gain is not None and prev_gain >= 0 and gain < 0:
                    crossover_data.append({
                        'portfolio': port,
                        'crossover_wr': wr * 100,
                        'fixed_sr': f_row.iloc[0]['success_rate'],
                        'dyn_sr': best_d['success_rate'],
                        'sr_gap': (best_d['success_rate'] - f_row.iloc[0]['success_rate']) * 100,
                        'cum_diff': gain,
                    })
                    break
                prev_gain = gain
    return crossover_data


# ============================================================================
# Tab 1: Guardrail이란? (도입)
# ============================================================================

def render_tab_intro(df, beta, path_method):
    """탭 1: Guardrail이란? — 개념과 메커니즘 시각적 설명"""

    st.markdown("### Guardrail이란?")
    st.info(
        "Guardrail은 시장 상황에 따라 인출액을 자동 조절하는 안전장치입니다. "
        "Band 5%는 인출률의 \u00b15% 범위에서 조정이 발생함을 의미합니다."
    )

    # ========================================
    # Guardrail 메커니즘 도해
    # ========================================

    st.markdown("---")
    st.subheader("고정 인출 vs Guardrail 월별 인출 패턴 비교")
    st.caption(
        "하락장에서 Guardrail은 인출을 줄이고, 상승장에서는 추가 인출합니다. "
        "핵심 파라미터: init_wr (초기인출률), band (조정 범위)"
    )

    # 예시 시나리오: 시장 상승 -> 하락 -> 회복
    months = list(range(25))
    # 자산 가치 시나리오 (100에서 시작)
    nav_scenario = [100]
    returns_scenario = [0.02, 0.03, 0.01, 0.02, 0.015,  # 상승기
                        -0.04, -0.06, -0.05, -0.03, -0.02,  # 하락기
                        0.01, 0.02, 0.03, 0.02, 0.01,  # 회복기
                        0.015, 0.02, -0.01, 0.01, 0.02,  # 안정기
                        0.015, 0.01, 0.02, 0.015]

    for r in returns_scenario:
        nav_scenario.append(nav_scenario[-1] * (1 + r))

    init_wr = 0.06
    W0 = 100.0
    base_monthly = W0 * init_wr / 12
    band = 0.10
    upper_wr = init_wr * (1 + band)
    lower_wr = init_wr * (1 - band)

    fixed_withdrawals = [base_monthly] * 24
    guardrail_withdrawals = []
    for t in range(24):
        W_t = nav_scenario[t + 1]
        if W_t > 0:
            current_wr = (base_monthly * 12) / W_t
            if current_wr > upper_wr:
                w = upper_wr * W_t / 12
            elif current_wr < lower_wr:
                w = lower_wr * W_t / 12
            else:
                w = base_monthly
        else:
            w = 0
        guardrail_withdrawals.append(w)

    fig_mech = go.Figure()
    fig_mech.add_trace(go.Scatter(
        x=months[1:], y=fixed_withdrawals,
        mode='lines', name='고정 인출 (Fixed)',
        line=dict(color='#E74C3C', width=2, dash='dash'),
    ))
    fig_mech.add_trace(go.Scatter(
        x=months[1:], y=guardrail_withdrawals,
        mode='lines', name=f'Guardrail (Band \u00b1{band*100:.0f}%)',
        line=dict(color='#2196F3', width=2.5),
    ))
    fig_mech.add_hline(y=base_monthly, line_dash="dot", line_color="#999",
                       annotation_text=f"기준 인출액: {base_monthly:.2f}")

    # 영역 구분
    fig_mech.add_vrect(x0=5, x1=10, fillcolor="rgba(231,76,60,0.08)",
                       layer="below", line_width=0,
                       annotation_text="하락장", annotation_position="top")
    fig_mech.add_vrect(x0=0, x1=5, fillcolor="rgba(39,174,96,0.08)",
                       layer="below", line_width=0,
                       annotation_text="상승장", annotation_position="top")

    fig_mech.update_layout(
        xaxis_title="월 (Month)",
        yaxis_title="월별 인출액",
        height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig_mech, use_container_width=True)
    st.caption(
        "해석: 하락장에서 Guardrail(파란 실선)은 인출을 줄여 자본을 보전하고, "
        "상승장에서는 추가 인출로 수혜를 누립니다. 고정 인출(빨간 점선)은 시장과 무관하게 동일합니다."
    )

    # ========================================
    # 이중 역할 소개: Surplus Harvesting vs Survival Mode
    # ========================================

    st.markdown("---")
    st.subheader("Guardrail의 이중 역할 (Surplus Harvesting \u2194 Survival Mode)")
    st.caption("0% 위 = Surplus Harvesting (더 많이 인출), 0% 아래 = Survival Mode (덜 인출하고 자본 보전)")

    fig_area = go.Figure()

    for port in sorted(df['portfolio'].unique()):
        gains = []
        wrs = sorted(df['init_wr'].unique())
        for wr in wrs:
            fixed_row = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                           (df['strategy_type'] == 'fixed_baseline')]
            dyn_rows = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                          (df['strategy_type'] == 'dynamic')]
            if len(fixed_row) > 0 and len(dyn_rows) > 0:
                f_cum = fixed_row.iloc[0]['cum_withdraw_median']
                best_dyn = dyn_rows.loc[dyn_rows['success_rate'].idxmax()]
                d_cum = best_dyn['cum_withdraw_median']
                gain = ((d_cum / f_cum) - 1) * 100 if f_cum > 0 else 0
                gains.append(gain)
            else:
                gains.append(0)

        color = PORT_COLORS.get(port, '#000000')
        fig_area.add_trace(go.Scatter(
            x=[w * 100 for w in wrs],
            y=gains,
            mode='lines+markers',
            name=port,
            line=dict(color=color, width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor=color.replace(')', ',0.08)').replace('rgb', 'rgba') if 'rgb' in color else None,
        ))

    fig_area.add_hline(y=0, line_width=2, line_color="black")
    fig_area.add_annotation(
        x=4, y=max(10, 5), text="Surplus Harvesting",
        showarrow=False, font=dict(size=12, color="green"), bgcolor="rgba(255,255,255,0.7)"
    )
    fig_area.add_annotation(
        x=12, y=min(-10, -5), text="Survival Mode",
        showarrow=False, font=dict(size=12, color="red"), bgcolor="rgba(255,255,255,0.7)"
    )
    fig_area.update_layout(
        xaxis_title="초기인출률 (%)",
        yaxis_title="Guardrail 추가 인출 효과 (%)",
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(ticksuffix='%'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', ticksuffix='%'),
    )
    st.plotly_chart(fig_area, use_container_width=True)
    st.caption(
        "해석: 저인출률 구간(왼쪽)에서는 Guardrail이 더 많이 인출 (Surplus Harvesting). "
        "고인출률 구간(오른쪽)에서는 인출을 줄여 생존 확률을 높임 (Survival Mode). "
        "각 선이 0%를 교차하는 지점이 전환점(Crossover Point)입니다."
    )

    st.markdown(_finding_box(
        "<b>Guardrail의 핵심 가치</b>: "
        "저인출률에서는 '무료 보너스' (성공률 동일 + 인출금 증가), "
        "고인출률에서는 '생존 보호' (인출금은 감소하지만 파산 위험 대폭 감소). "
        "어느 구간에서든 Guardrail은 가치가 있습니다."
    ), unsafe_allow_html=True)


# ============================================================================
# Tab 2: 언제 Guardrail이 유리한가? (핵심 비교)
# ============================================================================

def render_tab_comparison(df_all, beta, path_method):
    """탭 2: Fixed vs Guardrail 성공률 + 누적인출금 비교"""

    st.markdown("### 언제 Guardrail이 유리한가?")
    st.caption("Fixed vs Guardrail의 성공률 + 누적인출금 비교를 beta/포트폴리오별로 제시합니다.")

    # ========================================
    # 컨트롤
    # ========================================
    available_betas = sorted(df_all['beta'].unique())
    available_bands = sorted(df_all[df_all['strategy_type'] == 'dynamic']['band'].unique())

    col_beta, col_band = st.columns(2)
    with col_beta:
        selected_beta = st.select_slider(
            "Beta (성공 기준)",
            options=available_betas,
            value=beta,
            format_func=lambda x: f"{x*100:.0f}%",
            key='tab2_beta'
        )
    with col_band:
        if len(available_bands) > 0:
            band_options = available_bands
            default_band = band_options[0] if len(band_options) > 0 else 0.05
            selected_band = st.selectbox(
                "Guardrail Band",
                options=band_options,
                index=0,
                format_func=lambda x: f"\u00b1{x*100:.0f}%",
                key='tab2_band'
            )
        else:
            selected_band = 0.05

    # 해당 beta/path_method로 필터링
    df = df_all[(df_all['beta'] == selected_beta) & (df_all['path_method'] == path_method)].copy()

    if len(df) == 0:
        st.warning("선택된 Beta/Path Method 조합에 맞는 데이터가 없습니다.")
        return

    # ========================================
    # 성공률 차이 히트맵
    # ========================================
    st.markdown("---")
    st.subheader("성공률 차이 히트맵 (Guardrail - Fixed)")
    st.caption("녹색 영역이 Guardrail을 사용해야 하는 구간입니다.")

    sr_diff_data = []
    portfolios = sorted(df['portfolio'].unique())
    wrs = sorted(df['init_wr'].unique())

    for port in portfolios:
        for wr in wrs:
            fixed = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                       (df['strategy_type'] == 'fixed_baseline')]
            dyn = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                     (df['strategy_type'] == 'dynamic') & (df['band'] == selected_band)]
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
            colorscale='RdYlGn',
            zmid=0, zmin=-zmax, zmax=zmax,
            text=np.round(pivot_sr.values, 1),
            texttemplate='%{text:+.1f}',
            textfont={"size": 9},
            colorbar=dict(title="성공률 차이(%p)")
        ))
        fig_sr.update_layout(
            xaxis_title="초기인출률 (Init WR)",
            yaxis_title="포트폴리오",
            height=350,
        )
        st.plotly_chart(fig_sr, use_container_width=True)
    else:
        st.info("선택된 Band에 대한 데이터가 없습니다.")

    # ========================================
    # 누적인출금 차이 히트맵
    # ========================================
    st.markdown("---")
    st.subheader("누적인출금 차이 히트맵 (Guardrail - Fixed, %)")
    st.caption("파란색은 Guardrail이 더 많이 인출, 빨간색은 적게 인출함을 의미합니다.")

    cw_diff_data = []
    for port in portfolios:
        for wr in wrs:
            fixed = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                       (df['strategy_type'] == 'fixed_baseline')]
            dyn = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                     (df['strategy_type'] == 'dynamic') & (df['band'] == selected_band)]
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
            colorscale='RdBu',
            zmid=0, zmin=-zmax_cw, zmax=zmax_cw,
            text=np.round(pivot_cw.values, 1),
            texttemplate='%{text:+.1f}%',
            textfont={"size": 9},
            colorbar=dict(title="인출금 차이(%)")
        ))
        fig_cw.update_layout(
            xaxis_title="초기인출률 (Init WR)",
            yaxis_title="포트폴리오",
            height=350,
        )
        st.plotly_chart(fig_cw, use_container_width=True)

    # ========================================
    # 전환점 요약 테이블
    # ========================================
    st.markdown("---")
    st.subheader("Crossover Point 요약 (전환점)")
    st.caption("Guardrail이 Surplus Harvesting에서 Survival Mode로 전환되는 초기인출률.")

    crossover_data = _compute_crossover_data(df)
    if len(crossover_data) > 0:
        cp_df = pd.DataFrame(crossover_data)
        display_cp = pd.DataFrame({
            '포트폴리오': cp_df['portfolio'],
            'Crossover WR(%)': cp_df['crossover_wr'].apply(lambda x: f"{x:.1f}%"),
            'Fixed 성공률': cp_df['fixed_sr'].apply(lambda x: f"{x*100:.1f}%"),
            'Guardrail 성공률': cp_df['dyn_sr'].apply(lambda x: f"{x*100:.1f}%"),
            '성공률 Gap(%p)': cp_df['sr_gap'].apply(lambda x: f"+{x:.1f}"),
        })
        st.dataframe(display_cp, use_container_width=True, hide_index=True)
    else:
        st.info("Crossover point를 찾을 수 없습니다 (Guardrail이 모든 인출률에서 우위일 수 있음).")

    # ========================================
    # 핵심발견 박스
    # ========================================
    st.markdown("---")
    st.subheader("핵심 발견")
    findings = [
        "인출률 3~7%: Guardrail은 무료 보너스 (성공률 동일 + 인출금 +4~8% 증가)",
        "인출률 8%: Sweet Spot (Beta 50% 기준, 성공률 +8.9%p + 인출금 거의 동일)",
        "인출률 10%+: 성공률 대폭 향상(+10~33%p)이지만 인출금 감소(-8~20%)",
        "공격적 포트폴리오(Port_8~9%)는 순수 우위 구간이 더 넓음",
    ]
    for f in findings:
        st.markdown(_finding_box(f), unsafe_allow_html=True)


# ============================================================================
# Tab 3: 최적 Band는? (Band 분석)
# ============================================================================

def render_tab_band_analysis(df, beta, path_method):
    """탭 3: 어떤 Band가 가장 효과적인지 분석"""

    st.markdown("### 최적 Band는?")
    st.caption("어떤 Band가 가장 효과적인지 beta/인출률별로 제시합니다.")

    # ========================================
    # Band별 성공률 곡선
    # ========================================
    st.subheader("Band별 성공률 곡선")

    col_port, col_beta = st.columns(2)
    with col_port:
        band_port = st.selectbox(
            "포트폴리오 선택",
            options=sorted(df['portfolio'].unique()),
            index=2,
            key='tab3_port'
        )

    fig_band = go.Figure()

    # Fixed baseline (점선)
    fixed = df[(df['portfolio'] == band_port) & (df['strategy_type'] == 'fixed_baseline')]
    if len(fixed) > 0:
        fixed_by_wr = fixed.groupby('init_wr')['success_rate'].max().reset_index().sort_values('init_wr')
        fig_band.add_trace(go.Scatter(
            x=fixed_by_wr['init_wr'] * 100,
            y=fixed_by_wr['success_rate'],
            mode='lines+markers',
            name='Fixed (고정 인출)',
            line=dict(color='#999', width=2, dash='dash'),
            marker=dict(size=5, symbol='diamond'),
        ))

    # 각 Band별 실선
    band_colors = {0.05: '#2196F3', 0.10: '#4CAF50', 0.15: '#FF9800', 0.20: '#E91E63'}
    dynamic = df[(df['portfolio'] == band_port) & (df['strategy_type'] == 'dynamic')]
    available_bands = sorted(dynamic['band'].unique())

    for band_val in available_bands:
        band_data = dynamic[dynamic['band'] == band_val]
        by_wr = band_data.groupby('init_wr')['success_rate'].max().reset_index().sort_values('init_wr')
        color = band_colors.get(band_val, '#000')
        fig_band.add_trace(go.Scatter(
            x=by_wr['init_wr'] * 100,
            y=by_wr['success_rate'],
            mode='lines+markers',
            name=f'Band \u00b1{band_val*100:.0f}%',
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    fig_band.update_layout(
        xaxis_title="초기인출률 (%)",
        yaxis_title="성공률 (Success Rate)",
        height=500,
        xaxis=dict(ticksuffix='%'),
        yaxis=dict(tickformat='.0%'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig_band, use_container_width=True)

    # ========================================
    # Band 최적값 분포
    # ========================================
    st.markdown("---")
    st.subheader("Band 최적값 분포")
    st.caption("각 (포트폴리오, 인출률) 조합에서 어떤 Band가 가장 높은 성공률을 달성했는지.")

    optimal_band_counts = {}
    for port in sorted(df['portfolio'].unique()):
        for wr in sorted(df['init_wr'].unique()):
            dyn = df[(df['portfolio'] == port) & (df['init_wr'] == wr) &
                     (df['strategy_type'] == 'dynamic')]
            if len(dyn) > 0:
                best = dyn.loc[dyn['success_rate'].idxmax()]
                b = best['band']
                label = f"\u00b1{b*100:.0f}%"
                optimal_band_counts[label] = optimal_band_counts.get(label, 0) + 1

    if len(optimal_band_counts) > 0:
        total_combos = sum(optimal_band_counts.values())
        labels = list(optimal_band_counts.keys())
        values = list(optimal_band_counts.values())
        pcts = [v / total_combos * 100 for v in values]

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            textinfo='label+percent',
            marker=dict(colors=['#2196F3', '#4CAF50', '#FF9800', '#E91E63'][:len(labels)]),
        )])
        fig_pie.update_layout(height=350, title="최적 Band 비율")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ========================================
    # Band별 Trade-off 표
    # ========================================
    st.markdown("---")
    st.subheader("Band별 Trade-off 비교")
    st.caption("인출률 6%/8%/10%/12% 기준 포인트에서 Band별 성공률 변화와 인출금 변화를 비교합니다.")

    ref_wrs = [0.06, 0.08, 0.10, 0.12]
    tradeoff_rows = []

    for band_val in available_bands:
        for ref_wr in ref_wrs:
            dyn_row = df[(df['portfolio'] == band_port) & (df['init_wr'] == ref_wr) &
                         (df['strategy_type'] == 'dynamic') & (df['band'] == band_val)]
            fix_row = df[(df['portfolio'] == band_port) & (df['init_wr'] == ref_wr) &
                         (df['strategy_type'] == 'fixed_baseline')]
            if len(dyn_row) > 0 and len(fix_row) > 0:
                sr_diff = (dyn_row.iloc[0]['success_rate'] - fix_row.iloc[0]['success_rate']) * 100
                f_cum = fix_row.iloc[0]['cum_withdraw_median']
                d_cum = dyn_row.iloc[0]['cum_withdraw_median']
                cum_diff = ((d_cum / f_cum) - 1) * 100 if f_cum > 0 else 0
                tradeoff_rows.append({
                    'Band': f"\u00b1{band_val*100:.0f}%",
                    '인출률': f"{ref_wr*100:.0f}%",
                    '성공률 변화(%p)': f"{sr_diff:+.1f}",
                    '인출금 변화(%)': f"{cum_diff:+.1f}",
                })

    if len(tradeoff_rows) > 0:
        st.dataframe(pd.DataFrame(tradeoff_rows), use_container_width=True, hide_index=True)

    # ========================================
    # 핵심발견 박스
    # ========================================
    st.markdown("---")
    st.subheader("핵심 발견")
    findings = [
        "Band 5%가 압도적 최적. 좁은 Band일수록 빠른 조정 \u2192 강한 보호",
        "Beta 100%(원금 보전) 기준에서는 Band 5%만 소폭 개선, 나머지는 역효과",
        "넓은 Band(15~20%)는 인출금 감소가 적지만 보호 효과도 약함",
    ]
    for f in findings:
        st.markdown(_finding_box(f), unsafe_allow_html=True)


# ============================================================================
# Tab 4: 데이터 신뢰도 검증 (Rolling vs Bootstrap vs GBM)
# ============================================================================

def render_tab_validation(df_all, beta, path_method):
    """탭 4: 분석 결과의 강건성을 3종 방법론 비교로 검증"""

    st.markdown("### 데이터 신뢰도 검증")
    st.caption("분석 결과의 강건성을 Rolling / Bootstrap / GBM 3종 방법론 비교로 검증합니다.")

    # ========================================
    # 컨트롤
    # ========================================
    col_port, col_beta = st.columns(2)
    with col_port:
        val_port = st.selectbox(
            "포트폴리오 선택",
            options=sorted(df_all['portfolio'].unique()),
            index=2,
            key='tab4_port'
        )
    with col_beta:
        available_betas = sorted(df_all['beta'].unique())
        val_beta = st.select_slider(
            "Beta (성공 기준)",
            options=available_betas,
            value=beta,
            format_func=lambda x: f"{x*100:.0f}%",
            key='tab4_beta'
        )

    # ========================================
    # 3종 성공률 비교 곡선
    # ========================================
    st.markdown("---")
    st.subheader("3종 성공률 비교 곡선")
    st.caption("Rolling(실선) / Bootstrap(점선) / GBM(도트). 두 선의 차이가 클수록 방법론에 따라 결과가 민감합니다.")

    fig_3way = go.Figure()
    color = PORT_COLORS.get(val_port, '#2196F3')

    # Rolling
    rolling_df = df_all[(df_all['portfolio'] == val_port) & (df_all['beta'] == val_beta) &
                        (df_all['path_method'] == 'rolling') & (df_all['strategy_type'] == 'fixed_baseline')]
    if len(rolling_df) > 0:
        by_wr = rolling_df.groupby('init_wr')['success_rate'].max().reset_index().sort_values('init_wr')
        fig_3way.add_trace(go.Scatter(
            x=by_wr['init_wr'] * 100, y=by_wr['success_rate'],
            mode='lines+markers', name='Rolling (과거 실제)',
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
        ))

    # Bootstrap
    bootstrap_df = df_all[(df_all['portfolio'] == val_port) & (df_all['beta'] == val_beta) &
                          (df_all['path_method'] == 'bootstrap') & (df_all['strategy_type'] == 'fixed_baseline')]
    if len(bootstrap_df) > 0:
        by_wr = bootstrap_df.groupby('init_wr')['success_rate'].max().reset_index().sort_values('init_wr')
        fig_3way.add_trace(go.Scatter(
            x=by_wr['init_wr'] * 100, y=by_wr['success_rate'],
            mode='lines+markers', name='Bootstrap',
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(size=5, symbol='square'),
        ))

    # GBM
    port_info = PORTFOLIOS.get(val_port)
    if port_info:
        mu = port_info['target_return'] / 100
        sigma = port_info['target_risk'] / 100
        wr_range = np.arange(0.03, 0.155, 0.005)
        gbm_surv = [gbm_survival_probability(mu, sigma, wr, T=10, beta=val_beta) for wr in wr_range]
        fig_3way.add_trace(go.Scatter(
            x=wr_range * 100, y=gbm_surv,
            mode='markers', name='GBM (이론)',
            marker=dict(size=7, color=color, symbol='x', line=dict(width=1)),
            opacity=0.7,
        ))

    fig_3way.update_layout(
        xaxis_title="초기 인출률 (%)",
        yaxis_title="성공률 / 생존확률",
        height=500,
        xaxis=dict(ticksuffix='%'),
        yaxis=dict(tickformat='.0%'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig_3way, use_container_width=True)

    # ========================================
    # Rolling-Bootstrap 차이 히트맵
    # ========================================
    st.markdown("---")
    st.subheader("Rolling - Bootstrap 성공률 차이 히트맵")
    st.caption("차이가 큰 구간은 방법론에 따라 결과가 민감한 구간입니다.")

    rb_diff_data = []
    portfolios = sorted(df_all['portfolio'].unique())
    for port in portfolios:
        rolling = df_all[(df_all['portfolio'] == port) & (df_all['beta'] == val_beta) &
                         (df_all['path_method'] == 'rolling') & (df_all['strategy_type'] == 'fixed_baseline')]
        bootstrap = df_all[(df_all['portfolio'] == port) & (df_all['beta'] == val_beta) &
                           (df_all['path_method'] == 'bootstrap') & (df_all['strategy_type'] == 'fixed_baseline')]

        for wr in sorted(df_all['init_wr'].unique()):
            r_row = rolling[rolling['init_wr'] == wr]
            b_row = bootstrap[bootstrap['init_wr'] == wr]
            if len(r_row) > 0 and len(b_row) > 0:
                diff = (r_row.iloc[0]['success_rate'] - b_row.iloc[0]['success_rate']) * 100
                rb_diff_data.append({'portfolio': port, 'init_wr': wr, 'diff': diff})

    if len(rb_diff_data) > 0:
        rb_df = pd.DataFrame(rb_diff_data)
        pivot_rb = rb_df.pivot_table(index='portfolio', columns='init_wr', values='diff', aggfunc='mean')
        pivot_rb = pivot_rb.reindex(sorted(pivot_rb.index))

        zmax_rb = max(abs(pivot_rb.values[np.isfinite(pivot_rb.values)].min()),
                      abs(pivot_rb.values[np.isfinite(pivot_rb.values)].max()), 3)

        fig_rb = go.Figure(data=go.Heatmap(
            z=pivot_rb.values,
            x=[f"{x*100:.1f}%" for x in pivot_rb.columns],
            y=pivot_rb.index,
            colorscale='RdBu',
            zmid=0, zmin=-zmax_rb, zmax=zmax_rb,
            text=np.round(pivot_rb.values, 1),
            texttemplate='%{text:+.1f}',
            textfont={"size": 9},
            colorbar=dict(title="차이(%p)")
        ))
        fig_rb.update_layout(
            xaxis_title="초기인출률 (Init WR)",
            yaxis_title="포트폴리오",
            height=350,
        )
        st.plotly_chart(fig_rb, use_container_width=True)
    else:
        st.info("Rolling 또는 Bootstrap 데이터가 없습니다. 두 방법 모두 grid_search.py에서 생성해야 합니다.")

    # ========================================
    # GBM 괴리도 테이블
    # ========================================
    st.markdown("---")
    st.subheader("포트폴리오별 GBM 괴리도")

    gbm_comp_rows = []
    for port in portfolios:
        port_info = PORTFOLIOS.get(port)
        if not port_info:
            continue
        mu = port_info['target_return'] / 100
        sigma = port_info['target_risk'] / 100

        rolling = df_all[(df_all['portfolio'] == port) & (df_all['beta'] == val_beta) &
                         (df_all['path_method'] == 'rolling') & (df_all['strategy_type'] == 'fixed_baseline')]

        for _, r in rolling.iterrows():
            wr = r['init_wr']
            hist_sr = r['success_rate']
            gbm_sr = gbm_survival_probability(mu, sigma, wr, T=10, beta=val_beta)
            diff = (hist_sr - gbm_sr) * 100
            gbm_comp_rows.append({
                '포트폴리오': port,
                '인출률': f"{wr*100:.1f}%",
                'Rolling 성공률': f"{hist_sr*100:.1f}%",
                'GBM 생존확률': f"{gbm_sr*100:.1f}%",
                '차이(%p)': diff,
            })

    if len(gbm_comp_rows) > 0:
        gbm_table = pd.DataFrame(gbm_comp_rows)

        def _style_diff(val):
            if isinstance(val, (int, float)):
                if val > 5:
                    return 'color: #2980b9; font-weight: bold'
                elif val < -5:
                    return 'color: #e74c3c; font-weight: bold'
            return ''

        styled = gbm_table.style.map(_style_diff, subset=['차이(%p)']).format({'차이(%p)': '{:+.1f}'})
        st.dataframe(styled, use_container_width=True)
        st.caption("파란색(+): Rolling이 GBM보다 낙관적 | 빨간색(-): Rolling이 GBM보다 보수적")

    # ========================================
    # 핵심발견 박스
    # ========================================
    st.markdown("---")
    st.subheader("핵심 발견")
    findings = [
        "Bootstrap이 가장 보수적 \u2192 실무 보수 추정에 적합",
        "Beta 75%에서 GBM \u2248 Rolling 최적 정합",
        "Beta 100%에서 Rolling이 GBM보다 +22%p 높음 \u2192 과거 한국 시장이 GBM보다 우호적",
        "인출률 12%+에서 GBM은 비현실적으로 낙관적(높은 성공 예측 vs 실제 낮은 성공률)",
    ]
    for f in findings:
        st.markdown(_finding_box(f), unsafe_allow_html=True)


# ============================================================================
# Tab 5: 나의 전략 조합 (사용자 선택형 탐색기)
# ============================================================================

def render_tab_explorer(df_all, beta, path_method):
    """탭 5: 사용자가 구체적 전략 조합을 선택하고 모든 지표 확인"""

    st.markdown("### 나의 전략 조합")
    st.caption("구체적 전략 조합을 선택하면 Fixed vs Guardrail을 모든 지표에서 비교합니다.")

    # ========================================
    # 입력 파라미터
    # ========================================
    col1, col2, col3 = st.columns(3)
    with col1:
        exp_port = st.selectbox(
            "포트폴리오",
            options=sorted(df_all['portfolio'].unique()),
            index=2,
            key='tab5_port'
        )
    with col2:
        available_wrs = sorted(df_all['init_wr'].unique())
        default_wr_idx = min(3, len(available_wrs) - 1)
        exp_wr = st.selectbox(
            "초기 인출률",
            options=available_wrs,
            index=default_wr_idx,
            format_func=lambda x: f"{x*100:.1f}%",
            key='tab5_wr'
        )
    with col3:
        dyn_data = df_all[(df_all['portfolio'] == exp_port) & (df_all['strategy_type'] == 'dynamic')]
        available_bands = sorted(dyn_data['band'].unique()) if len(dyn_data) > 0 else [0.05]
        exp_band = st.selectbox(
            "Guardrail Band",
            options=available_bands,
            index=0,
            format_func=lambda x: f"\u00b1{x*100:.0f}%",
            key='tab5_band'
        )

    # ========================================
    # Beta별 성공률 비교 바 차트
    # ========================================
    st.markdown("---")
    st.subheader("Beta별 성공률 비교 (Fixed vs Guardrail)")

    available_betas = sorted(df_all['beta'].unique())
    beta_labels = [f"{b*100:.0f}%" for b in available_betas]

    fixed_srs = []
    guard_srs = []
    for b in available_betas:
        for pm in [path_method]:
            f_row = df_all[(df_all['portfolio'] == exp_port) & (df_all['init_wr'] == exp_wr) &
                           (df_all['beta'] == b) & (df_all['path_method'] == pm) &
                           (df_all['strategy_type'] == 'fixed_baseline')]
            g_row = df_all[(df_all['portfolio'] == exp_port) & (df_all['init_wr'] == exp_wr) &
                           (df_all['beta'] == b) & (df_all['path_method'] == pm) &
                           (df_all['strategy_type'] == 'dynamic') & (df_all['band'] == exp_band)]
            fixed_srs.append(f_row.iloc[0]['success_rate'] * 100 if len(f_row) > 0 else 0)
            guard_srs.append(g_row.iloc[0]['success_rate'] * 100 if len(g_row) > 0 else 0)

    fig_beta = go.Figure()
    fig_beta.add_trace(go.Bar(
        x=beta_labels, y=fixed_srs,
        name='고정 인출 (Fixed)', marker_color='#bdc3c7',
        text=[f"{v:.1f}%" for v in fixed_srs], textposition='outside',
    ))
    fig_beta.add_trace(go.Bar(
        x=beta_labels, y=guard_srs,
        name=f'Guardrail (\u00b1{exp_band*100:.0f}%)',
        marker_color=PORT_COLORS.get(exp_port, '#2196F3'),
        text=[f"{v:.1f}%" for v in guard_srs], textposition='outside',
    ))
    fig_beta.update_layout(
        barmode='group', height=420,
        xaxis_title="Beta (기말잔액 기준, %)",
        yaxis_title="성공률 (%)",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis=dict(range=[0, 110]),
    )
    st.plotly_chart(fig_beta, use_container_width=True)

    # ========================================
    # Path Method별 비교
    # ========================================
    st.markdown("---")
    st.subheader("Path Method별 성공률 비교")

    pm_data = []
    for pm in ['rolling', 'bootstrap', 'combined']:
        f_row = df_all[(df_all['portfolio'] == exp_port) & (df_all['init_wr'] == exp_wr) &
                       (df_all['beta'] == beta) & (df_all['path_method'] == pm) &
                       (df_all['strategy_type'] == 'fixed_baseline')]
        g_row = df_all[(df_all['portfolio'] == exp_port) & (df_all['init_wr'] == exp_wr) &
                       (df_all['beta'] == beta) & (df_all['path_method'] == pm) &
                       (df_all['strategy_type'] == 'dynamic') & (df_all['band'] == exp_band)]
        pm_data.append({
            'Path Method': PATH_METHOD_LABELS.get(pm, pm),
            'Fixed 성공률': f"{f_row.iloc[0]['success_rate']*100:.1f}%" if len(f_row) > 0 else "-",
            'Guardrail 성공률': f"{g_row.iloc[0]['success_rate']*100:.1f}%" if len(g_row) > 0 else "-",
        })
    st.dataframe(pd.DataFrame(pm_data), use_container_width=True, hide_index=True)

    # ========================================
    # 핵심 지표 카드 (Fixed vs Guardrail)
    # ========================================
    st.markdown("---")
    st.subheader("핵심 지표 비교")

    df = df_all[(df_all['beta'] == beta) & (df_all['path_method'] == path_method)].copy()
    fixed_row = df[(df['portfolio'] == exp_port) & (df['init_wr'] == exp_wr) &
                   (df['strategy_type'] == 'fixed_baseline')]
    guard_row = df[(df['portfolio'] == exp_port) & (df['init_wr'] == exp_wr) &
                   (df['strategy_type'] == 'dynamic') & (df['band'] == exp_band)]

    if len(fixed_row) > 0 and len(guard_row) > 0:
        fr = fixed_row.iloc[0]
        gr = guard_row.iloc[0]

        metrics = [
            ("성공률", f"{fr['success_rate']*100:.1f}%", f"{gr['success_rate']*100:.1f}%",
             (gr['success_rate'] - fr['success_rate']) * 100, True),
            ("파산확률", f"{fr['p_ruin']*100:.2f}%", f"{gr['p_ruin']*100:.2f}%",
             (gr['p_ruin'] - fr['p_ruin']) * 100, False),
            ("누적인출금", f"{fr['cum_withdraw_median']:.1f}", f"{gr['cum_withdraw_median']:.1f}",
             ((gr['cum_withdraw_median'] / fr['cum_withdraw_median']) - 1) * 100 if fr['cum_withdraw_median'] > 0 else 0, True),
            ("인출변동성(CV)", f"{fr['cv_median']:.3f}", f"{gr['cv_median']:.3f}",
             (gr['cv_median'] - fr['cv_median']), False),
            ("최대삭감률", f"{fr['worst_cut_median']*100:.1f}%", f"{gr['worst_cut_median']*100:.1f}%",
             (gr['worst_cut_median'] - fr['worst_cut_median']) * 100, False),
        ]

        # 지표별 2열 카드
        for label, fv, gv, delta, higher_good in metrics:
            c1, c2, c3 = st.columns([2, 2, 1.5])
            with c1:
                st.metric(f"Fixed: {label}", fv)
            with c2:
                st.metric(f"Guardrail: {label}", gv)
            with c3:
                if isinstance(delta, float):
                    is_better = (delta > 0) == higher_good
                    color = "#27ae60" if is_better else "#e74c3c"
                    sign = "+" if delta > 0 else ""
                    unit = "%p" if "률" in label or "확률" in label else "%"
                    st.markdown(
                        f'<div style="text-align:center; padding-top:12px;">'
                        f'<span style="font-size:1.3em; font-weight:700; color:{color};">'
                        f'{sign}{delta:.1f}{unit}</span></div>',
                        unsafe_allow_html=True
                    )

        # ========================================
        # GBM 이론값 대비
        # ========================================
        st.markdown("---")
        st.subheader("GBM 이론값 대비")

        port_info = PORTFOLIOS.get(exp_port)
        if port_info:
            mu = port_info['target_return'] / 100
            sigma = port_info['target_risk'] / 100
            gbm_sr = gbm_survival_probability(mu, sigma, exp_wr, T=10, beta=beta)

            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.metric("GBM 이론 생존확률", f"{gbm_sr*100:.1f}%")
            with col_g2:
                hist_sr = fr['success_rate']
                diff = (hist_sr - gbm_sr) * 100
                st.metric("Rolling 과거 성공률", f"{hist_sr*100:.1f}%",
                          delta=f"{diff:+.1f}%p vs GBM")
            with col_g3:
                guard_sr = gr['success_rate']
                diff_g = (guard_sr - gbm_sr) * 100
                st.metric("Guardrail 성공률", f"{guard_sr*100:.1f}%",
                          delta=f"{diff_g:+.1f}%p vs GBM")

        # ========================================
        # 실무적 해석 코멘트
        # ========================================
        st.markdown("---")
        sr_diff = (gr['success_rate'] - fr['success_rate']) * 100
        cum_diff = ((gr['cum_withdraw_median'] / fr['cum_withdraw_median']) - 1) * 100 if fr['cum_withdraw_median'] > 0 else 0

        comment = (
            f"이 조합은 Beta {beta*100:.0f}% 기준 Guardrail 성공률 **{gr['success_rate']*100:.1f}%**이며, "
            f"고정인출 대비 성공률이 **{sr_diff:+.1f}%p** "
            f"{'높습니다' if sr_diff >= 0 else '낮습니다'}. "
        )
        if cum_diff < 0:
            comment += f"인출금은 **{cum_diff:.1f}%** 감소하지만, 포트폴리오 생존 가능성이 향상됩니다."
        else:
            comment += f"인출금도 **+{cum_diff:.1f}%** 증가하여 Guardrail이 순수 우위입니다."

        st.success(comment)

    else:
        st.warning("선택된 조합에 대한 데이터가 없습니다. 필터를 확인하세요.")


# ============================================================================
# Tab 6: 전략 상세 (기존 Tab 3 유지 — NAV 경로 시뮬레이션)
# ============================================================================

def render_tab_detail(df, beta, path_method):
    """탭 6: 전략 상세 — 단일 전략 deep-dive + NAV 시뮬레이션"""

    st.header("전략 상세 분석")
    st.caption("개별 전략의 핵심 지표, NAV 시뮬레이션, Fixed/Dynamic 비교를 한 화면에서 확인합니다.")

    col_left, col_right = st.columns([1, 3])

    # ==================================================================
    # 좌측: Strategy Selector
    # ==================================================================
    with col_left:
        st.subheader("전략 선택")

        all_portfolios = sorted(df['portfolio'].unique())
        portfolio = st.selectbox(
            "포트폴리오",
            options=all_portfolios,
            index=0,
            key='detail_portfolio'
        )

        type_labels = {"dynamic": "Guardrail", "fixed_baseline": "고정 인출"}
        strategy_type = st.radio(
            "전략 유형",
            options=['dynamic', 'fixed_baseline'],
            format_func=lambda x: type_labels[x],
            index=0,
            key='detail_type'
        )

        filtered = df[
            (df['portfolio'] == portfolio) &
            (df['strategy_type'] == strategy_type)
        ]

        init_wr_options = sorted(filtered['init_wr'].unique())
        init_wr = st.selectbox(
            "초기 인출률 (Init WR)",
            options=init_wr_options,
            index=min(3, max(len(init_wr_options) - 1, 0)),
            format_func=lambda x: f"{x*100:.1f}%",
            key='detail_wr'
        )

        if strategy_type == 'dynamic':
            band_options = sorted(
                filtered[filtered['init_wr'] == init_wr]['band'].unique()
            )
            band = st.selectbox(
                "Guardrail 밴드 (Band)",
                options=band_options,
                index=0,
                format_func=lambda x: f"\u00b1{x*100:.0f}%",
                key='detail_band'
            )
        else:
            band = 99.0

        # 선택된 전략 row
        selected = filtered[
            (filtered['init_wr'] == init_wr) &
            (filtered['band'] == band)
        ]
        if len(selected) == 0:
            st.error("선택된 전략을 찾을 수 없습니다.")
            return
        row = selected.iloc[0]

    # ==================================================================
    # 우측: Metrics + Charts
    # ==================================================================
    with col_right:

        # ----------------------------------------------------------
        # 1) 색상 메트릭 카드
        # ----------------------------------------------------------
        st.subheader("핵심 지표")

        def _color(value, thresholds, reverse=False):
            good, caution = thresholds
            if reverse:
                if value <= good:
                    return "#27ae60", "#e8f8f0"
                elif value <= caution:
                    return "#e67e22", "#fef5e7"
                else:
                    return "#e74c3c", "#fdedec"
            else:
                if value >= good:
                    return "#27ae60", "#e8f8f0"
                elif value >= caution:
                    return "#e67e22", "#fef5e7"
                else:
                    return "#e74c3c", "#fdedec"

        cards = [
            ("성공률", f"{row['success_rate']*100:.1f}%", "Success Rate",
             _color(row['success_rate'], (0.90, 0.85))),
            ("누적인출금", f"{row['cum_withdraw_median']:.1f}", "Cum Withdraw",
             ("#2980b9", "#ebf5fb")),
            ("파산확률", f"{row['p_ruin']*100:.2f}%", "P(ruin)",
             _color(row['p_ruin'], (0.01, 0.03), reverse=True)),
            ("인출변동성", f"{row['cv_median']:.3f}", "CV",
             ("#8e44ad", "#f5eef8")),
            ("최대삭감률", f"{row['worst_cut_median']*100:.1f}%", "Worst Cut",
             ("#e67e22", "#fef5e7")),
        ]

        met_cols = st.columns(5)
        for i, (label_kr, value_str, label_en, (fg, bg)) in enumerate(cards):
            with met_cols[i]:
                st.markdown(f"""
                <div style="background:{bg}; border-left:4px solid {fg};
                            padding:12px 10px; border-radius:6px; text-align:center;">
                    <div style="font-size:0.8em; color:#555;">{label_kr}</div>
                    <div style="font-size:1.6em; font-weight:700; color:{fg};">{value_str}</div>
                    <div style="font-size:0.65em; color:#999;">{label_en}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ----------------------------------------------------------
        # 2) NAV Path Simulation — AUTO RUN
        # ----------------------------------------------------------
        st.subheader("잔액 경로 시뮬레이션")
        st.caption("녹색=성공 경로, 적색=실패 경로. 중앙값(검정)과 P5-P95 밴드(회색)로 전체 분포를 확인합니다.")

        sim_key = f"{portfolio}_{init_wr}_{band}_{beta}_{path_method}"

        if st.session_state.get('_detail_sim_key') != sim_key:
            with st.spinner("경로 시뮬레이션 실행 중..."):
                success_paths, failure_paths, all_withdraw_series = simulate_paths_for_strategy(
                    portfolio, init_wr, band, beta, path_method
                )
            st.session_state['_detail_sim_key'] = sim_key
            st.session_state['_detail_success'] = success_paths
            st.session_state['_detail_failure'] = failure_paths
            st.session_state['_detail_withdrawals'] = all_withdraw_series
        else:
            success_paths = st.session_state['_detail_success']
            failure_paths = st.session_state['_detail_failure']
            all_withdraw_series = st.session_state['_detail_withdrawals']

        max_display = 200
        rng = np.random.default_rng(42)
        disp_s = success_paths
        disp_f = failure_paths
        if len(disp_s) > max_display:
            idx = rng.choice(len(disp_s), max_display, replace=False)
            disp_s = [disp_s[i] for i in idx]
        if len(disp_f) > max_display:
            idx = rng.choice(len(disp_f), max_display, replace=False)
            disp_f = [disp_f[i] for i in idx]

        fig_nav = go.Figure()

        for ws in disp_s:
            fig_nav.add_trace(go.Scatter(
                x=list(range(len(ws))), y=ws, mode='lines',
                line=dict(color='rgba(39,174,96,0.12)', width=1),
                showlegend=False, hoverinfo='skip'
            ))
        for ws in disp_f:
            fig_nav.add_trace(go.Scatter(
                x=list(range(len(ws))), y=ws, mode='lines',
                line=dict(color='rgba(231,76,60,0.18)', width=1),
                showlegend=False, hoverinfo='skip'
            ))

        all_paths = success_paths + failure_paths
        if all_paths:
            max_len = max(len(p) for p in all_paths)
            padded = np.full((len(all_paths), max_len), np.nan)
            for i, p in enumerate(all_paths):
                padded[i, :len(p)] = p
            months = list(range(max_len))
            p50 = np.nanmedian(padded, axis=0)
            p05 = np.nanpercentile(padded, 5, axis=0)
            p95 = np.nanpercentile(padded, 95, axis=0)

            fig_nav.add_trace(go.Scatter(
                x=months, y=p95, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            fig_nav.add_trace(go.Scatter(
                x=months, y=p05, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(149,165,166,0.15)',
                name='P5-P95 밴드', hoverinfo='skip'
            ))
            fig_nav.add_trace(go.Scatter(
                x=months, y=p50, mode='lines',
                line=dict(color='black', width=2.5), name='중앙값 (Median)',
                hovertemplate='월 %{x}: %{y:.1f}<extra>Median</extra>'
            ))

        fig_nav.add_hline(y=100, line_dash="dot", line_color="#2980b9",
                          annotation_text="W0 = 100")
        fig_nav.add_hline(y=beta * 100, line_dash="dot", line_color="#95a5a6",
                          annotation_text=f"Terminal 기준 = {beta*100:.0f}")

        fig_nav.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='#27ae60', width=2),
            name=f'성공 ({len(success_paths)}경로)'
        ))
        fig_nav.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='#e74c3c', width=2),
            name=f'실패 ({len(failure_paths)}경로)'
        ))

        fig_nav.update_layout(
            title="잔액 경로 시뮬레이션 (NAV Path Fan Chart)",
            xaxis_title="월 (Month)", yaxis_title="포트폴리오 잔액",
            height=480,
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1)
        )
        st.plotly_chart(fig_nav, use_container_width=True)

        st.markdown("---")

        # ----------------------------------------------------------
        # 3) 인출률 시계열 Fan Chart
        # ----------------------------------------------------------
        if len(all_withdraw_series) > 0:
            st.subheader("인출률 시계열 (Fan Chart)")
            st.caption("시간에 따른 실제 인출률 분포. 짙은 색일수록 빈도가 높은 구간입니다.")

            ws_arr = np.array(all_withdraw_series)  # (n_paths, T)
            T = ws_arr.shape[1]

            wr_annual = (ws_arr * 12) / 100.0
            wr_median = np.median(wr_annual, axis=0)
            wr_p05 = np.percentile(wr_annual, 5, axis=0)
            wr_p10 = np.percentile(wr_annual, 10, axis=0)
            wr_p25 = np.percentile(wr_annual, 25, axis=0)
            wr_p75 = np.percentile(wr_annual, 75, axis=0)
            wr_p90 = np.percentile(wr_annual, 90, axis=0)
            wr_p95 = np.percentile(wr_annual, 95, axis=0)
            months_t = list(range(T))

            fig_wr_ts = go.Figure()

            fig_wr_ts.add_trace(go.Scatter(
                x=months_t, y=wr_p95 * 100, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            fig_wr_ts.add_trace(go.Scatter(
                x=months_t, y=wr_p05 * 100, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(52,152,219,0.10)',
                name='P5-P95', hoverinfo='skip'
            ))
            fig_wr_ts.add_trace(go.Scatter(
                x=months_t, y=wr_p90 * 100, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            fig_wr_ts.add_trace(go.Scatter(
                x=months_t, y=wr_p10 * 100, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(52,152,219,0.15)',
                name='P10-P90', hoverinfo='skip'
            ))
            fig_wr_ts.add_trace(go.Scatter(
                x=months_t, y=wr_p75 * 100, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            fig_wr_ts.add_trace(go.Scatter(
                x=months_t, y=wr_p25 * 100, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(52,152,219,0.25)',
                name='P25-P75', hoverinfo='skip'
            ))
            fig_wr_ts.add_trace(go.Scatter(
                x=months_t, y=wr_median * 100, mode='lines',
                line=dict(color='#2980b9', width=2.5), name='중앙값 (Median)',
                hovertemplate='월 %{x}: %{y:.2f}%<extra>Median WR</extra>'
            ))
            fig_wr_ts.add_hline(
                y=init_wr * 100, line_dash="dot", line_color="#2c3e50",
                annotation_text=f"기준 인출률: {init_wr*100:.1f}%"
            )
            fig_wr_ts.update_layout(
                title="인출률 시계열 Fan Chart (연환산 인출률, %)",
                xaxis_title="월 (Month)",
                yaxis_title="연간 인출률 (%)",
                height=420,
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1)
            )
            st.plotly_chart(fig_wr_ts, use_container_width=True)

            st.markdown("---")

        # ----------------------------------------------------------
        # 4) Fixed vs Dynamic 세로 막대 비교
        # ----------------------------------------------------------
        st.subheader("고정 인출 vs 현재 전략 비교")
        st.caption("같은 포트폴리오/인출률에서 Fixed Baseline과 선택 전략의 지표 차이를 시각적으로 비교합니다.")

        fixed_comp = df[
            (df['portfolio'] == portfolio) &
            (df['init_wr'] == init_wr) &
            (df['strategy_type'] == 'fixed_baseline')
        ]

        if len(fixed_comp) > 0:
            fr = fixed_comp.iloc[0]
            port_color = PORT_COLORS.get(portfolio, '#3498db')

            metrics_list = [
                ("성공률 (%)", fr['success_rate'] * 100, row['success_rate'] * 100, True),
                ("누적인출금", fr['cum_withdraw_median'], row['cum_withdraw_median'], True),
                ("파산확률 (%)", fr['p_ruin'] * 100, row['p_ruin'] * 100, False),
                ("인출변동성\n(CV)", fr['cv_median'], row['cv_median'], False),
                ("최대삭감률\n(%)", fr['worst_cut_median'] * 100, row['worst_cut_median'] * 100, False),
            ]

            metric_names = [m[0] for m in metrics_list]
            fixed_vals = [m[1] for m in metrics_list]
            sel_vals = [m[2] for m in metrics_list]

            fig_fvd = go.Figure()
            fig_fvd.add_trace(go.Bar(
                x=metric_names, y=fixed_vals,
                name='고정 인출 (Fixed)', marker_color='#bdc3c7', opacity=0.85
            ))
            fig_fvd.add_trace(go.Bar(
                x=metric_names, y=sel_vals,
                name='현재 전략 (Guardrail)' if strategy_type == 'dynamic' else '현재 전략',
                marker_color=port_color, opacity=0.85
            ))

            for i, (name, fv, sv, higher_good) in enumerate(metrics_list):
                delta = sv - fv
                if delta == 0:
                    continue
                color = "#27ae60" if (delta > 0) == higher_good else "#e74c3c"
                sign = "+" if delta > 0 else ""
                fig_fvd.add_annotation(
                    x=name,
                    y=max(fv, sv) * 1.08 + 0.5,
                    text=f"<b>{sign}{delta:.1f}</b>",
                    showarrow=False, font=dict(color=color, size=11)
                )

            fig_fvd.update_layout(
                barmode='group', height=400,
                title="Fixed Baseline vs 선택 전략",
                xaxis_title="지표", yaxis_title="값",
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1)
            )
            st.plotly_chart(fig_fvd, use_container_width=True)
        else:
            st.info("해당 포트폴리오/인출률의 Fixed Baseline 데이터가 없습니다.")

        st.markdown("---")

        # ----------------------------------------------------------
        # 5) Excel 다운로드
        # ----------------------------------------------------------
        st.subheader("결과 다운로드")

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            port_results = df[df['portfolio'] == portfolio]
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                port_results.to_excel(writer, sheet_name='전체결과', index=False)
                port_frontier = port_results[port_results['is_frontier']]
                if len(port_frontier) > 0:
                    port_frontier.to_excel(writer, sheet_name='Frontier', index=False)

            st.download_button(
                label=f"{portfolio} 전략 결과 (Excel)",
                data=buffer.getvalue(),
                file_name=f"{portfolio}_detail_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='tab6_download'
            )

        with dl_col2:
            all_paths_list = success_paths + failure_paths
            if len(all_paths_list) > 0:
                max_len = max(len(p) for p in all_paths_list)
                nav_dict = {}
                nav_dict['Month'] = list(range(max_len))
                for i, p in enumerate(all_paths_list):
                    label = f"성공_{i+1}" if i < len(success_paths) else f"실패_{i+1-len(success_paths)}"
                    padded = list(p) + [None] * (max_len - len(p))
                    nav_dict[label] = padded
                nav_df = pd.DataFrame(nav_dict)

                nav_buffer = io.BytesIO()
                with pd.ExcelWriter(nav_buffer, engine='openpyxl') as writer:
                    nav_df.to_excel(writer, sheet_name='NAV_Monthly', index=False)
                    if len(all_withdraw_series) > 0:
                        wd_dict = {'Month': list(range(len(all_withdraw_series[0])))}
                        for i, ws in enumerate(all_withdraw_series):
                            wd_dict[f"Path_{i+1}"] = list(ws)
                        wd_df = pd.DataFrame(wd_dict)
                        wd_df.to_excel(writer, sheet_name='Withdraw_Monthly', index=False)

                st.download_button(
                    label="NAV 경로 다운로드 (Excel)",
                    data=nav_buffer.getvalue(),
                    file_name=f"{portfolio}_nav_paths.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='tab6_nav_download'
                )
            else:
                st.caption("시뮬레이션 경로가 없습니다.")


# ============================================================================
# Main
# ============================================================================

def main():
    st.set_page_config(
        page_title="퇴직 포트폴리오 인출 전략 분석기",
        page_icon="\U0001f4ca",
        layout="wide"
    )

    # 커스텀 CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("퇴직 포트폴리오 인출 전략 분석기")

    # 데이터 로딩
    df_all = load_grid_results()

    # ========================================
    # 사이드바 — 글로벌 필터
    # ========================================

    st.sidebar.header("분석 설정")

    # Beta 선택
    available_betas = sorted(df_all['beta'].unique())
    if len(available_betas) == 0:
        available_betas = [0.5]
    default_beta_idx = available_betas.index(0.5) if 0.5 in available_betas else 0
    beta = st.sidebar.select_slider(
        "성공 기준 (기말잔액 비율, beta)",
        options=available_betas,
        value=available_betas[default_beta_idx],
        format_func=lambda x: f"{x*100:.0f}%",
        key='global_beta'
    )
    st.sidebar.caption(BETA_LABELS.get(beta, f"기말잔액 \u2265 초기의 {beta*100:.0f}%"))

    # Path method 선택
    available_paths = sorted(df_all['path_method'].unique())
    path_method_options = [p for p in ['rolling', 'bootstrap', 'combined'] if p in available_paths]
    if len(path_method_options) == 0:
        path_method_options = available_paths
    default_pm = 'rolling' if 'rolling' in path_method_options else path_method_options[0]
    path_method = st.sidebar.selectbox(
        "데이터 기반",
        options=path_method_options,
        index=path_method_options.index(default_pm),
        format_func=lambda x: PATH_METHOD_LABELS.get(x, x),
        key='global_path_method'
    )

    st.sidebar.markdown("---")

    # 글로벌 필터 적용
    df = apply_global_filters(df_all, beta, path_method)

    # 상단 배너
    st.markdown(f"**성공 기준: {BETA_LABELS.get(beta, '')}** | "
                f"데이터 기반: {PATH_METHOD_LABELS.get(path_method, path_method)}")

    st.sidebar.info(
        f"**전체 전략 수**: {len(df):,}\n\n"
        f"**포트폴리오**: {len(df['portfolio'].unique())}개\n\n"
        f"**Beta**: {beta}\n\n"
        f"**데이터**: {PATH_METHOD_LABELS.get(path_method, path_method)}\n\n"
        f"**분석 기간**: 10년 (120개월)"
    )

    st.sidebar.markdown("---")

    # 용어집 (한글화)
    with st.sidebar.expander("용어집 (Glossary)"):
        st.markdown("""
**성공 (Success)**
파산 없음 AND 기말 잔액 \u2265 초기의 beta%

**파산 (Ruin, P_ruin)**
운용 중 잔액이 0 이하로 떨어지는 경우

**기말 실패 (Terminal Fail)**
최종 잔액 < beta \u00d7 초기자산

**초기 인출률 (Init WR)**
시작 시점의 연간 인출률 (% 기준)

**Guardrail 밴드 (Band)**
인출률 허용 범위: [초기인출률 \u00d7 (1-band), 초기인출률 \u00d7 (1+band)]

**인출변동성 (CV, Coefficient of Variation)**
월별 인출액의 표준편차/평균. 낮을수록 안정적.

**최대삭감률 (Worst Cut)**
전체 기간 중 단일 월 최대 인출 감소율.

**최저월소득 (P5 Income)**
월별 인출액의 5번째 백분위 (최악 시나리오 소득).
        """)

    # ========================================
    # 탭 생성 (6개 스토리 순서)
    # ========================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Guardrail이란?",
        "언제 유리한가?",
        "최적 Band는?",
        "데이터 신뢰도",
        "나의 전략 조합",
        "전략 상세",
    ])

    with tab1:
        render_tab_intro(df, beta, path_method)

    with tab2:
        render_tab_comparison(df_all, beta, path_method)

    with tab3:
        render_tab_band_analysis(df, beta, path_method)

    with tab4:
        render_tab_validation(df_all, beta, path_method)

    with tab5:
        render_tab_explorer(df_all, beta, path_method)

    with tab6:
        render_tab_detail(df, beta, path_method)


if __name__ == "__main__":
    main()
