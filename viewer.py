"""
Streamlit Viewer for Dynamic Withdrawal Strategy Results
=========================================================
독립 실행형 시각화 뷰어: streamlit run viewer.py
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
from engine import daily_to_monthly_returns, generate_paths_rolling, simulate_withdrawal_on_path


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


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data
def load_grid_results():
    """grid_results_full.pkl → DataFrame 변환"""
    try:
        with open('grid_results_full.pkl', 'rb') as f:
            results = pickle.load(f)

        # list[dict] → DataFrame
        df = pd.DataFrame([
            {
                'portfolio': r['portfolio'],
                'path_method': r['path_method'],
                'strategy_type': r['strategy_type'],
                'init_wr': r['init_wr'],
                'band': r['band'],
                'adj_on': r['adj_on'],
                'lookback': r.get('lookback', 1),
                'thr_up': r.get('thr_up', 0.03),
                'thr_dn': r.get('thr_dn', -0.03),
                'adj_up': r.get('adj_up', 0.05),
                'adj_dn': r.get('adj_dn', -0.05),
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
                'is_frontier': r.get('is_frontier', False),
                'is_optimal': r.get('is_optimal', False),
            }
            for r in results
        ])

        return df

    except FileNotFoundError:
        st.error("❌ grid_results_full.pkl 파일을 찾을 수 없습니다. grid_search.py를 먼저 실행하세요.")
        st.stop()


@st.cache_data
def load_benchmark_data():
    """benchmark_data.pkl 로드"""
    try:
        with open('benchmark_data.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ benchmark_data.pkl 파일을 찾을 수 없습니다.")
        st.stop()


@st.cache_data
def simulate_paths_for_strategy(portfolio_name, init_wr, band, adj_on,
                                  lookback, thr_up, thr_dn, adj_up, adj_dn):
    """
    선택된 전략의 모든 rolling path에 대해 W_series를 계산

    Returns
    -------
    success_paths : list[np.ndarray]  — 성공한 path의 W_series
    failure_paths : list[np.ndarray]  — 실패한 path의 W_series
    """
    # 1. benchmark_data.pkl 로드
    benchmark_data = load_benchmark_data()

    # 2. DataPreprocessor
    preprocessor = DataPreprocessor(benchmark_data, add_portfolios=True)
    returns_df, month_starts = preprocessor.get_data()

    # 3. 해당 portfolio의 daily_returns → monthly_returns
    daily_returns = returns_df[portfolio_name]
    monthly_returns = daily_to_monthly_returns(daily_returns, month_starts)

    # 4. generate_paths_rolling
    rolling_paths = generate_paths_rolling(monthly_returns, T_months=120)

    # 5. 각 path에 시뮬레이션 적용
    success_paths = []
    failure_paths = []

    for path in rolling_paths:
        result = simulate_withdrawal_on_path(
            path_returns=path,
            init_wr=init_wr,
            band=band,
            adj_on=adj_on,
            lookback=lookback,
            thr_up=thr_up,
            thr_dn=thr_dn,
            adj_up=adj_up,
            adj_dn=adj_dn,
            beta=0.5,
            W0=100.0,
            debug=False
        )

        if result['success']:
            success_paths.append(result['W_series'])
        else:
            failure_paths.append(result['W_series'])

    return success_paths, failure_paths


# ============================================================================
# Helper Functions
# ============================================================================

def gbm_survival_probability(mu, sigma, wr, T=10, beta=0.5):
    """
    GBM 보존확률 계산

    Parameters
    ----------
    mu : float — 연간 기대수익률
    sigma : float — 연간 변동성
    wr : float — 연간 인출률
    T : float — 기간(년)
    beta : float — terminal threshold V_T >= beta*V0

    Returns
    -------
    float — 보존확률 (0~1)
    """
    if wr <= 0:
        return 1.0
    if sigma <= 0:
        remaining = 1.0 - wr * T + mu * T
        return 1.0 if remaining >= beta else 0.0

    dt = 1/12
    n_steps = int(T / dt)
    c_monthly = wr / 12  # V0=1 정규화

    # 1차 모멘트 전파
    EV = 1.0
    # 2차 모멘트 전파
    EV2 = 1.0

    for _ in range(n_steps):
        new_EV = EV * np.exp(mu * dt) - c_monthly
        new_EV2 = EV2 * np.exp((2*mu + sigma**2) * dt) \
                  - 2 * c_monthly * EV * np.exp(mu * dt) \
                  + c_monthly**2
        EV = new_EV
        EV2 = max(new_EV2, 0)
        if EV <= 0:
            return 0.0

    VarV = max(EV2 - EV**2, 1e-12)
    z = (beta - EV) / np.sqrt(VarV)
    survival = 1.0 - norm.cdf(z)
    return np.clip(survival, 0.0, 1.0)


def filter_dataframe(df, portfolios, strategy_type, frontier_only):
    """글로벌 필터 적용"""
    filtered = df[df['portfolio'].isin(portfolios)].copy()

    if strategy_type == 'Fixed':
        filtered = filtered[filtered['strategy_type'] == 'fixed_baseline']
    elif strategy_type == 'Dynamic':
        filtered = filtered[filtered['strategy_type'] == 'dynamic']

    if frontier_only:
        filtered = filtered[filtered['is_frontier'] == True]

    return filtered


# ============================================================================
# Tab 1: Product Selection
# ============================================================================

def render_product_tab(df):
    """탭 1: 상품 추천"""
    st.header("📊 Product Selection (상품 추천)")

    # ========================================
    # 섹션 1: Fixed vs Dynamic 킬러 비교
    # ========================================

    st.subheader("💡 Fixed vs Dynamic: Guardrail의 가치")
    st.markdown("**같은 성공확률에서 Dynamic Guardrail이 더 많이 인출 가능**")

    success_threshold = st.slider(
        "Success Rate Threshold (%)",
        min_value=80, max_value=100, value=90, step=1
    ) / 100

    # 포트폴리오별로 Fixed vs Dynamic 최적 찾기
    comparison_data = []

    for portfolio in df['portfolio'].unique():
        # Fixed baseline
        fixed_subset = df[
            (df['portfolio'] == portfolio) &
            (df['strategy_type'] == 'fixed_baseline') &
            (df['success_rate'] >= success_threshold)
        ]

        if len(fixed_subset) > 0:
            fixed_best = fixed_subset.loc[fixed_subset['cum_withdraw_median'].idxmax()]
            fixed_wr = fixed_best['init_wr']
            fixed_cum = fixed_best['cum_withdraw_median']
        else:
            fixed_wr = None
            fixed_cum = 0

        # Dynamic
        dynamic_subset = df[
            (df['portfolio'] == portfolio) &
            (df['strategy_type'] == 'dynamic') &
            (df['success_rate'] >= success_threshold)
        ]

        if len(dynamic_subset) > 0:
            dynamic_best = dynamic_subset.loc[dynamic_subset['cum_withdraw_median'].idxmax()]
            dynamic_wr = dynamic_best['init_wr']
            dynamic_cum = dynamic_best['cum_withdraw_median']
        else:
            dynamic_wr = None
            dynamic_cum = 0

        comparison_data.append({
            'Portfolio': portfolio,
            'Fixed_WR': fixed_wr,
            'Fixed_Cum': fixed_cum,
            'Dynamic_WR': dynamic_wr,
            'Dynamic_Cum': dynamic_cum,
            'Gain': dynamic_cum - fixed_cum if (fixed_cum > 0 and dynamic_cum > 0) else 0,
            'Gain_Pct': (dynamic_cum / fixed_cum - 1) * 100 if fixed_cum > 0 else 0
        })

    # Grouped Bar Chart
    comp_df = pd.DataFrame(comparison_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp_df['Portfolio'],
        y=comp_df['Fixed_Cum'],
        name='Fixed Baseline',
        marker_color='lightcoral'
    ))
    fig.add_trace(go.Bar(
        x=comp_df['Portfolio'],
        y=comp_df['Dynamic_Cum'],
        name='Dynamic Guardrail',
        marker_color='lightgreen'
    ))

    fig.update_layout(
        title=f"Cumulative Withdrawal Comparison (Success Rate >= {success_threshold*100:.0f}%)",
        xaxis_title="Portfolio",
        yaxis_title="Cumulative Withdrawal",
        barmode='group',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 비교 테이블
    st.dataframe(
        comp_df.style.format({
            'Fixed_WR': lambda x: f"{x*100:.1f}%" if x else "N/A",
            'Fixed_Cum': '{:.1f}',
            'Dynamic_WR': lambda x: f"{x*100:.1f}%" if x else "N/A",
            'Dynamic_Cum': '{:.1f}',
            'Gain': '{:+.1f}',
            'Gain_Pct': '{:+.1f}%'
        }),
        use_container_width=True
    )

    st.markdown("---")

    # ========================================
    # 섹션 2: 3개 고객 프로파일 최적 선택
    # ========================================

    st.subheader("👥 Customer Profile: 최적 상품 추천")

    col1, col2, col3 = st.columns(3)

    profiles = {
        'Conservative': {
            'col': col1,
            'color': '#2196F3',
            'defaults': {'min_success': 0.95, 'max_p_ruin': 0.005, 'max_cv': 0.10, 'max_worst_cut': 0.10}
        },
        'Balanced': {
            'col': col2,
            'color': '#4CAF50',
            'defaults': {'min_success': 0.90, 'max_p_ruin': 0.01, 'max_cv': 0.15, 'max_worst_cut': 0.20}
        },
        'Aggressive': {
            'col': col3,
            'color': '#E91E63',
            'defaults': {'min_success': 0.85, 'max_p_ruin': 0.02, 'max_cv': 0.25, 'max_worst_cut': 0.30}
        }
    }

    profile_results = {}

    for profile_name, profile_info in profiles.items():
        with profile_info['col']:
            st.markdown(f"### {profile_name}")

            # 제약 슬라이더
            min_success = st.number_input(
                "Min Success Rate",
                min_value=0.0, max_value=1.0,
                value=profile_info['defaults']['min_success'],
                step=0.01,
                key=f"{profile_name}_success"
            )

            max_p_ruin = st.number_input(
                "Max P(ruin)",
                min_value=0.0, max_value=0.1,
                value=profile_info['defaults']['max_p_ruin'],
                step=0.001,
                format="%.3f",
                key=f"{profile_name}_ruin"
            )

            max_cv = st.number_input(
                "Max CV",
                min_value=0.0, max_value=0.5,
                value=profile_info['defaults']['max_cv'],
                step=0.01,
                key=f"{profile_name}_cv"
            )

            max_worst_cut = st.number_input(
                "Max Worst Cut",
                min_value=0.0, max_value=0.5,
                value=profile_info['defaults']['max_worst_cut'],
                step=0.01,
                key=f"{profile_name}_cut"
            )

            # 제약 통과 후 최적 선택
            dynamic = df[df['strategy_type'] == 'dynamic']
            candidates = dynamic[
                (dynamic['success_rate'] >= min_success) &
                (dynamic['p_ruin'] <= max_p_ruin) &
                (dynamic['cv_median'] <= max_cv) &
                (dynamic['worst_cut_median'] <= max_worst_cut)
            ]

            if len(candidates) > 0:
                optimal = candidates.loc[candidates['cum_withdraw_median'].idxmax()]
                profile_results[profile_name] = optimal

                # 카드 표시
                st.success(f"**{optimal['portfolio']}**")
                st.metric("Init WR", f"{optimal['init_wr']*100:.1f}%")
                st.metric("Band", f"±{optimal['band']*100:.0f}%")
                st.metric("Adj On", "Yes" if optimal['adj_on'] else "No")
                st.metric("Cum Withdraw", f"{optimal['cum_withdraw_median']:.1f}")
                st.metric("Success Rate", f"{optimal['success_rate']*100:.1f}%")
                st.metric("P(ruin)", f"{optimal['p_ruin']*100:.2f}%")
            else:
                st.warning("제약 조건을 만족하는 전략이 없습니다")
                profile_results[profile_name] = None

    st.markdown("---")

    # ========================================
    # 섹션 3: Frontier 산점도 + 3개 최적 점 하이라이트
    # ========================================

    st.subheader("🎯 Frontier + Optimal Products")

    frontier = df[df['is_frontier'] == True]

    fig = go.Figure()

    # Frontier 점
    for portfolio in frontier['portfolio'].unique():
        subset = frontier[frontier['portfolio'] == portfolio]
        fig.add_trace(go.Scatter(
            x=subset['success_rate'],
            y=subset['cum_withdraw_median'],
            mode='markers+lines',
            name=portfolio,
            marker=dict(size=8, color=PORT_COLORS.get(portfolio, '#000000')),
            line=dict(color=PORT_COLORS.get(portfolio, '#000000'), width=1),
            opacity=0.7
        ))

    # 3개 프로파일 최적 점 하이라이트
    for profile_name, optimal in profile_results.items():
        if optimal is not None:
            fig.add_trace(go.Scatter(
                x=[optimal['success_rate']],
                y=[optimal['cum_withdraw_median']],
                mode='markers+text',
                name=f"{profile_name} Optimal",
                marker=dict(size=20, symbol='star', color=profiles[profile_name]['color'], line=dict(width=2, color='white')),
                text=[profile_name[0]],  # C, B, A
                textposition='middle center',
                textfont=dict(size=10, color='white'),
                showlegend=True
            ))

    fig.update_layout(
        title="Pareto Frontier with Optimal Products Highlighted",
        xaxis_title="Success Rate",
        yaxis_title="Cumulative Withdrawal (Median)",
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(tickformat='.0%'),
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ========================================
    # 섹션 4: 3개 상품 비교 테이블
    # ========================================

    st.subheader("📋 Product Comparison Table")

    if len(profile_results) > 0 and any(v is not None for v in profile_results.values()):
        comparison = []
        for profile_name, optimal in profile_results.items():
            if optimal is not None:
                comparison.append({
                    'Profile': profile_name,
                    'Portfolio': optimal['portfolio'],
                    'Init WR': f"{optimal['init_wr']*100:.1f}%",
                    'Band': f"±{optimal['band']*100:.0f}%",
                    'Adj On': 'Yes' if optimal['adj_on'] else 'No',
                    'Success Rate': f"{optimal['success_rate']*100:.1f}%",
                    'Cum Withdraw': f"{optimal['cum_withdraw_median']:.1f}",
                    'P(ruin)': f"{optimal['p_ruin']*100:.2f}%",
                    'CV': f"{optimal['cv_median']:.3f}",
                    'Worst Cut': f"{optimal['worst_cut_median']*100:.1f}%",
                    'P5 Income': f"{optimal['p5_monthly_income']:.3f}"
                })

        comp_table = pd.DataFrame(comparison)
        st.dataframe(comp_table, use_container_width=True)


# ============================================================================
# Tab 2: Frontier Exploration
# ============================================================================

def render_frontier_tab(df):
    """탭 2: Frontier 탐색"""
    st.header("🔍 Frontier Exploration (탐색)")

    col1, col2 = st.columns(2)

    # ========================================
    # 차트 1: 전체 산점도
    # ========================================

    with col1:
        st.subheader("Success Rate vs Cumulative Withdrawal")

        fig = go.Figure()

        # 비-frontier 점 (투명하게)
        non_frontier = df[df['is_frontier'] == False]
        for portfolio in non_frontier['portfolio'].unique():
            subset = non_frontier[non_frontier['portfolio'] == portfolio]
            fig.add_trace(go.Scatter(
                x=subset['success_rate'],
                y=subset['cum_withdraw_median'],
                mode='markers',
                name=f"{portfolio} (all)",
                marker=dict(size=4, color=PORT_COLORS.get(portfolio, '#000000')),
                opacity=0.3,
                showlegend=False
            ))

        # Frontier 점 (진하게)
        frontier = df[df['is_frontier'] == True]
        for portfolio in frontier['portfolio'].unique():
            subset = frontier[frontier['portfolio'] == portfolio]
            fig.add_trace(go.Scatter(
                x=subset['success_rate'],
                y=subset['cum_withdraw_median'],
                mode='markers+lines',
                name=portfolio,
                marker=dict(size=10, color=PORT_COLORS.get(portfolio, '#000000')),
                line=dict(color=PORT_COLORS.get(portfolio, '#000000'), width=2)
            ))

        # 90% success rate 세로 점선
        fig.add_vline(x=0.90, line_dash="dot", line_color="gray", annotation_text="90% SR")

        fig.update_layout(
            xaxis_title="Success Rate",
            yaxis_title="Cumulative Withdrawal (Median)",
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis=dict(tickformat='.0%')
        )

        st.plotly_chart(fig, use_container_width=True)

    # ========================================
    # 차트 2: 인출률별 성공확률 곡선 (Fixed baseline만)
    # ========================================

    with col2:
        st.subheader("Init WR vs Success Rate (Fixed)")

        fixed = df[df['strategy_type'] == 'fixed_baseline']

        fig = go.Figure()

        for portfolio in fixed['portfolio'].unique():
            subset = fixed[fixed['portfolio'] == portfolio].sort_values('init_wr')
            fig.add_trace(go.Scatter(
                x=subset['init_wr'] * 100,
                y=subset['success_rate'],
                mode='lines+markers',
                name=portfolio,
                line=dict(color=PORT_COLORS.get(portfolio, '#000000'), width=2),
                marker=dict(size=6)
            ))

        # 90% success rate 가로 점선
        fig.add_hline(y=0.90, line_dash="dot", line_color="gray", annotation_text="90%")

        fig.update_layout(
            xaxis_title="Init WR (%)",
            yaxis_title="Success Rate",
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis=dict(tickformat='.0%')
        )

        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # ========================================
    # 차트 3: 인출률별 누적인출 곡선 (Fixed baseline만)
    # ========================================

    with col3:
        st.subheader("Init WR vs Cum Withdraw (Fixed)")

        fig = go.Figure()

        for portfolio in fixed['portfolio'].unique():
            subset = fixed[fixed['portfolio'] == portfolio].sort_values('init_wr')
            fig.add_trace(go.Scatter(
                x=subset['init_wr'] * 100,
                y=subset['cum_withdraw_median'],
                mode='lines+markers',
                name=portfolio,
                line=dict(color=PORT_COLORS.get(portfolio, '#000000'), width=2),
                marker=dict(size=6)
            ))

        fig.update_layout(
            xaxis_title="Init WR (%)",
            yaxis_title="Cumulative Withdrawal (Median)",
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    # ========================================
    # 차트 4: 파라미터 히트맵
    # ========================================

    with col4:
        st.subheader("Parameter Heatmap")

        # 필터
        heatmap_portfolio = st.selectbox(
            "Portfolio (Heatmap)",
            options=df['portfolio'].unique(),
            key='heatmap_port'
        )

        heatmap_adj = st.radio(
            "Adj On (Heatmap)",
            options=[False, True],
            format_func=lambda x: "Off" if not x else "On",
            key='heatmap_adj',
            horizontal=True
        )

        # 데이터 필터링
        dynamic = df[
            (df['portfolio'] == heatmap_portfolio) &
            (df['strategy_type'] == 'dynamic') &
            (df['adj_on'] == heatmap_adj)
        ]

        if len(dynamic) > 0:
            # pivot table
            pivot = dynamic.pivot_table(
                index='band',
                columns='init_wr',
                values='success_rate',
                aggfunc='mean'
            )

            fig = go.Figure(data=go.Heatmap(
                z=pivot.values * 100,
                x=[f"{x*100:.1f}%" for x in pivot.columns],
                y=[f"±{y*100:.0f}%" for y in pivot.index],
                colorscale='RdYlGn',
                zmin=50,
                zmax=100,
                text=pivot.values * 100,
                texttemplate='%{text:.1f}%',
                textfont={"size": 10},
                colorbar=dict(title="Success Rate (%)")
            ))

            fig.update_layout(
                title=f"Success Rate: {heatmap_portfolio} (Adj {'On' if heatmap_adj else 'Off'})",
                xaxis_title="Init WR",
                yaxis_title="Band",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data for this combination")


# ============================================================================
# Tab 3: Strategy Detail
# ============================================================================

def render_detail_tab(df):
    """탭 3: 전략 상세"""
    st.header("🔬 Strategy Detail (상세 비교)")

    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.subheader("Select Strategy")

        # Portfolio
        portfolio = st.selectbox(
            "Portfolio",
            options=df['portfolio'].unique()
        )

        # Strategy Type
        strategy_type = st.radio(
            "Strategy Type",
            options=['dynamic', 'fixed_baseline'],
            format_func=lambda x: "Dynamic" if x == 'dynamic' else "Fixed Baseline"
        )

        # 필터링된 데이터
        filtered = df[
            (df['portfolio'] == portfolio) &
            (df['strategy_type'] == strategy_type)
        ]

        # Init WR
        init_wr_options = sorted(filtered['init_wr'].unique())
        init_wr = st.selectbox(
            "Init WR (%)",
            options=init_wr_options,
            format_func=lambda x: f"{x*100:.1f}%"
        )

        # Band (dynamic일 때만)
        if strategy_type == 'dynamic':
            band_options = sorted(filtered[filtered['init_wr'] == init_wr]['band'].unique())
            band = st.selectbox(
                "Band",
                options=band_options,
                format_func=lambda x: f"±{x*100:.0f}%"
            )

            # Adj On
            adj_options = sorted(filtered[
                (filtered['init_wr'] == init_wr) &
                (filtered['band'] == band)
            ]['adj_on'].unique())
            adj_on = st.selectbox(
                "Adj On",
                options=adj_options,
                format_func=lambda x: "Yes" if x else "No"
            )
        else:
            band = 99.0
            adj_on = False

        # 선택된 전략 찾기
        selected = filtered[
            (filtered['init_wr'] == init_wr) &
            (filtered['band'] == band) &
            (filtered['adj_on'] == adj_on)
        ]

        if len(selected) == 0:
            st.error("선택된 전략을 찾을 수 없습니다")
            return

        selected_row = selected.iloc[0]

    with col_right:
        st.subheader("Strategy Metrics")

        # 메트릭 카드 (5열)
        met_cols = st.columns(5)

        with met_cols[0]:
            st.metric("Success Rate", f"{selected_row['success_rate']*100:.1f}%")
        with met_cols[1]:
            st.metric("Cum Withdraw", f"{selected_row['cum_withdraw_median']:.1f}")
        with met_cols[2]:
            st.metric("P(ruin)", f"{selected_row['p_ruin']*100:.2f}%")
        with met_cols[3]:
            st.metric("CV", f"{selected_row['cv_median']:.3f}")
        with met_cols[4]:
            st.metric("Worst Cut", f"{selected_row['worst_cut_median']*100:.1f}%")

        st.markdown("---")

        # NAV Path 차트
        st.subheader("NAV Path Simulation")

        if st.button("🚀 Run Simulation", key='run_sim'):
            with st.spinner("Simulating paths..."):
                success_paths, failure_paths = simulate_paths_for_strategy(
                    portfolio,
                    init_wr,
                    band,
                    adj_on,
                    selected_row['lookback'],
                    selected_row['thr_up'],
                    selected_row['thr_dn'],
                    selected_row['adj_up'],
                    selected_row['adj_dn']
                )

                # 최대 200개만 표시 (랜덤 샘플링)
                max_paths = 200
                if len(success_paths) > max_paths:
                    success_paths = np.random.choice(success_paths, max_paths, replace=False).tolist()
                if len(failure_paths) > max_paths:
                    failure_paths = np.random.choice(failure_paths, max_paths, replace=False).tolist()

                fig = go.Figure()

                # 성공 paths (초록)
                for W_series in success_paths:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(W_series))),
                        y=W_series,
                        mode='lines',
                        line=dict(color='rgba(0,128,0,0.2)', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # 실패 paths (빨강)
                for W_series in failure_paths:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(W_series))),
                        y=W_series,
                        mode='lines',
                        line=dict(color='rgba(255,0,0,0.3)', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                # W0=100 기준선
                fig.add_hline(y=100, line_dash="dot", line_color="blue", annotation_text="W0=100")

                # beta*W0=50 기준선
                fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Terminal Threshold=50")

                # Legend (더미 trace로 추가)
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color='green', width=2),
                    name=f'Success ({len(success_paths)} paths)'
                ))
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'Failure ({len(failure_paths)} paths)'
                ))

                fig.update_layout(
                    title="NAV Paths (Green=Success, Red=Failure)",
                    xaxis_title="Month",
                    yaxis_title="Portfolio Value",
                    height=500,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )

                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Fixed vs Dynamic 비교 테이블
        st.subheader("Fixed vs Dynamic Comparison")

        # 같은 포트폴리오/init_wr에서 fixed와 현재 dynamic 비교
        fixed_comp = df[
            (df['portfolio'] == portfolio) &
            (df['init_wr'] == init_wr) &
            (df['strategy_type'] == 'fixed_baseline')
        ]

        if len(fixed_comp) > 0:
            fixed_row = fixed_comp.iloc[0]

            comp_data = {
                'Strategy': ['Fixed Baseline', 'Selected Dynamic'],
                'Band': [f"±{fixed_row['band']*100:.0f}%", f"±{selected_row['band']*100:.0f}%"],
                'Success Rate': [f"{fixed_row['success_rate']*100:.1f}%", f"{selected_row['success_rate']*100:.1f}%"],
                'Cum Withdraw': [f"{fixed_row['cum_withdraw_median']:.1f}", f"{selected_row['cum_withdraw_median']:.1f}"],
                'P(ruin)': [f"{fixed_row['p_ruin']*100:.2f}%", f"{selected_row['p_ruin']*100:.2f}%"],
                'CV': [f"{fixed_row['cv_median']:.3f}", f"{selected_row['cv_median']:.3f}"],
                'Worst Cut': [f"{fixed_row['worst_cut_median']*100:.1f}%", f"{selected_row['worst_cut_median']*100:.1f}%"],
                'P5 Income': [f"{fixed_row['p5_monthly_income']:.3f}", f"{selected_row['p5_monthly_income']:.3f}"]
            }

            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        st.markdown("---")

        # Adj On 효과 비교 (dynamic일 때만)
        if strategy_type == 'dynamic':
            st.subheader("Adj On Effect")

            adj_variants = df[
                (df['portfolio'] == portfolio) &
                (df['init_wr'] == init_wr) &
                (df['band'] == band) &
                (df['strategy_type'] == 'dynamic')
            ]

            if len(adj_variants) > 1:
                adj_comp = adj_variants[[
                    'adj_on', 'success_rate', 'cum_withdraw_median', 'p_ruin', 'cv_median', 'worst_cut_median'
                ]].copy()

                adj_comp.columns = ['Adj On', 'Success Rate', 'Cum Withdraw', 'P(ruin)', 'CV', 'Worst Cut']
                adj_comp['Adj On'] = adj_comp['Adj On'].apply(lambda x: 'Yes' if x else 'No')

                st.dataframe(
                    adj_comp.style.format({
                        'Success Rate': '{:.1%}',
                        'Cum Withdraw': '{:.1f}',
                        'P(ruin)': '{:.3f}',
                        'CV': '{:.3f}',
                        'Worst Cut': '{:.3f}'
                    }),
                    use_container_width=True
                )

        st.markdown("---")

        # CV vs 누적인출 Scatter
        st.subheader("CV vs Cumulative Withdrawal")

        port_dynamic = df[
            (df['portfolio'] == portfolio) &
            (df['strategy_type'] == 'dynamic')
        ]

        if len(port_dynamic) > 0:
            fig = px.scatter(
                port_dynamic,
                x='cv_median',
                y='cum_withdraw_median',
                color='band',
                symbol='adj_on',
                labels={
                    'cv_median': 'CV (Median)',
                    'cum_withdraw_median': 'Cumulative Withdrawal (Median)',
                    'band': 'Band',
                    'adj_on': 'Adj On'
                },
                title=f"{portfolio}: CV vs Cum Withdraw",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Worst Cut by Band Box plot
        st.subheader("Worst Cut by Band")

        if len(port_dynamic) > 0:
            fig = px.box(
                port_dynamic,
                x='band',
                y='worst_cut_median',
                color='adj_on',
                labels={
                    'band': 'Band',
                    'worst_cut_median': 'Worst Cut (Median)',
                    'adj_on': 'Adj On'
                },
                title=f"{portfolio}: Worst Cut Distribution",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Excel 다운로드
        st.subheader("📥 Download Results")

        port_results = df[df['portfolio'] == portfolio]

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            port_results.to_excel(writer, sheet_name='All_Results', index=False)

            port_frontier = port_results[port_results['is_frontier'] == True]
            if len(port_frontier) > 0:
                port_frontier.to_excel(writer, sheet_name='Frontier', index=False)

        st.download_button(
            label=f"Download {portfolio} Results (Excel)",
            data=buffer.getvalue(),
            file_name=f"{portfolio}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ============================================================================
# Tab 4: GBM Comparison
# ============================================================================

def render_gbm_tab(df):
    """탭 4: GBM 비교"""
    st.header("📈 GBM Comparison (Historical vs Theoretical)")

    # GBM 수식 설명
    with st.expander("📚 GBM Formula"):
        st.latex(r"""
        \text{GBM with Constant Withdrawal:}
        """)
        st.latex(r"""
        dV = (\mu V - c)dt + \sigma V dZ
        """)
        st.markdown("""
        Where:
        - V = Portfolio value
        - μ = Annual expected return
        - σ = Annual volatility
        - c = WR × V0 = Annual withdrawal (fixed)

        **Survival Probability Calculation**: Iterative moment propagation
        - dt = 1/12 (monthly)
        - E[V_{t+1}] = E[V_t] × exp(μ×dt) - c/12
        - E[V²_{t+1}] = E[V²_t] × exp((2μ+σ²)×dt) - 2×(c/12)×E[V_t]×exp(μ×dt) + (c/12)²
        - Var[V_T] = E[V²_T] - E[V_T]²
        - P(V_T ≥ β×V0) = 1 - Φ((β×V0 - E[V_T]) / √Var[V_T])
        """)

    # 파라미터 조정 슬라이더
    col1, col2 = st.columns(2)

    with col1:
        mu_adj = st.slider(
            "μ Adjustment (%)",
            min_value=-2.0, max_value=2.0, value=0.0, step=0.1
        ) / 100

    with col2:
        sigma_adj = st.slider(
            "σ Adjustment (%)",
            min_value=-2.0, max_value=2.0, value=0.0, step=0.1
        ) / 100

    st.markdown("---")

    # Historical vs GBM 곡선
    st.subheader("Historical vs GBM: Success/Survival Rate")

    # Fixed baseline 데이터
    fixed = df[df['strategy_type'] == 'fixed_baseline']

    # init_wr 범위
    wr_range = np.arange(0.03, 0.155, 0.005)

    fig = go.Figure()

    for portfolio in sorted(fixed['portfolio'].unique()):
        # Historical
        port_data = fixed[fixed['portfolio'] == portfolio].sort_values('init_wr')

        fig.add_trace(go.Scatter(
            x=port_data['init_wr'] * 100,
            y=port_data['success_rate'],
            mode='lines+markers',
            name=f"{portfolio} (Historical)",
            line=dict(color=PORT_COLORS.get(portfolio, '#000000'), width=2),
            marker=dict(size=6)
        ))

        # GBM
        port_info = PORTFOLIOS[portfolio]
        mu = port_info['target_return'] / 100 + mu_adj
        sigma = port_info['target_risk'] / 100 + sigma_adj

        gbm_survival = [gbm_survival_probability(mu, sigma, wr, T=10, beta=0.5) for wr in wr_range]

        fig.add_trace(go.Scatter(
            x=wr_range * 100,
            y=gbm_survival,
            mode='lines',
            name=f"{portfolio} (GBM)",
            line=dict(color=PORT_COLORS.get(portfolio, '#000000'), width=2, dash='dot'),
            opacity=0.7
        ))

    fig.update_layout(
        xaxis_title="Init WR (%)",
        yaxis_title="Success/Survival Rate",
        height=500,
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02),
        yaxis=dict(tickformat='.0%')
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 단일 포트폴리오 상세 비교
    st.subheader("Detailed Comparison (Single Portfolio)")

    comp_portfolio = st.selectbox(
        "Select Portfolio",
        options=sorted(fixed['portfolio'].unique())
    )

    port_data = fixed[fixed['portfolio'] == comp_portfolio].sort_values('init_wr')
    port_info = PORTFOLIOS[comp_portfolio]
    mu = port_info['target_return'] / 100 + mu_adj
    sigma = port_info['target_risk'] / 100 + sigma_adj

    comparison = []
    for _, row in port_data.iterrows():
        wr = row['init_wr']
        hist_sr = row['success_rate']
        gbm_sr = gbm_survival_probability(mu, sigma, wr, T=10, beta=0.5)

        comparison.append({
            'Init WR': f"{wr*100:.1f}%",
            'Historical SR': f"{hist_sr*100:.1f}%",
            'GBM SR': f"{gbm_sr*100:.1f}%",
            'Difference': f"{(hist_sr - gbm_sr)*100:+.1f}%"
        })

    comp_df = pd.DataFrame(comparison)
    st.dataframe(comp_df, use_container_width=True)

    st.info(f"""
    **Portfolio**: {comp_portfolio}
    - μ (adjusted): {mu*100:.2f}%
    - σ (adjusted): {sigma*100:.2f}%
    - Period: 10 years (120 months)
    - Terminal Threshold: β = 0.5 (50% of initial)
    """)


# ============================================================================
# Main
# ============================================================================

def main():
    st.set_page_config(
        page_title="Withdrawal Strategy Viewer",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Dynamic Withdrawal Strategy Viewer")
    st.markdown("**Grid Search 결과 시각화 및 분석**")

    # 데이터 로딩
    df = load_grid_results()

    st.sidebar.header("🎛️ Global Filters")

    # 포트폴리오 멀티셀렉트
    portfolios = st.sidebar.multiselect(
        "Portfolios",
        options=sorted(df['portfolio'].unique()),
        default=sorted(df['portfolio'].unique())
    )

    # Strategy Type 라디오
    strategy_type_filter = st.sidebar.radio(
        "Strategy Type",
        options=['All', 'Fixed', 'Dynamic'],
        index=0
    )

    # Frontier Only 토글
    frontier_only = st.sidebar.checkbox("Frontier Only", value=False)

    # 필터 적용
    filtered_df = filter_dataframe(df, portfolios, strategy_type_filter, frontier_only)

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Filtered Results**: {len(filtered_df)} / {len(df)}")

    if len(filtered_df) == 0:
        st.warning("⚠️ 선택된 필터 조건에 맞는 데이터가 없습니다.")
        return

    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Product Selection",
        "🔍 Frontier Exploration",
        "🔬 Strategy Detail",
        "📈 GBM Comparison"
    ])

    with tab1:
        render_product_tab(filtered_df)

    with tab2:
        render_frontier_tab(filtered_df)

    with tab3:
        render_detail_tab(df)  # 필터 없이 전체 데이터 사용

    with tab4:
        render_gbm_tab(df)  # 필터 없이 전체 데이터 사용


if __name__ == "__main__":
    main()
