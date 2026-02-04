"""
Dynamic Withdrawal Strategy UI Components
==========================================
Streamlit UI components for dynamic withdrawal strategy optimization and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from withdrawal_backtest import DataPreprocessor, PORTFOLIOS
from optimizer import WithdrawalOptimizer


def render_dynamic_strategies_tab(benchmark_data):
    """
    Main UI for Dynamic Withdrawal Strategies tab

    Parameters
    ----------
    benchmark_data : pd.DataFrame
        Benchmark data with DatetimeIndex and asset columns
    """
    st.header("🎯 Dynamic Withdrawal Strategies")

    st.markdown("""
    다양한 인출 전략을 비교하고 최적의 인출률을 찾아보세요:
    - **Fixed Rate**: 고정 인출률 + 인플레이션 조정
    - **Guardrails**: NAV 기준 즉각 반응 방식
    - **Guyton-Klinger**: 고정 비율 단계적 조정
    """)

    # Create tabs for different sections
    opt_tab, comp_tab, detail_tab = st.tabs(
        ["🔍 최적화", "📊 비교 분석", "📈 상세 분석"]
    )

    # ========================================================================
    # Tab 1: Optimization
    # ========================================================================
    with opt_tab:
        render_optimization_section(benchmark_data)

    # ========================================================================
    # Tab 2: Comparison
    # ========================================================================
    with comp_tab:
        render_comparison_section()

    # ========================================================================
    # Tab 3: Detailed Analysis
    # ========================================================================
    with detail_tab:
        render_detailed_analysis_section()


def render_optimization_section(benchmark_data):
    """
    Optimization section: Find optimal withdrawal rates
    """
    st.subheader("최적 인출률 탐색")

    with st.form("optimization_form"):
        st.markdown("### 최적화 설정")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**제약 조건**")
            max_volatility = st.number_input(
                "최대 YoY 변동성 (%)",
                min_value=5.0,
                max_value=30.0,
                value=15.0,
                step=1.0,
                help="연간 인출액 변화의 최대 표준편차"
            )

        with col2:
            max_failure = st.number_input(
                "최대 실패율 (%)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=1.0,
                help="포트폴리오 고갈 확률의 최대값"
            )

        with col3:
            min_terminal = st.number_input(
                "최소 최종 NAV (%)",
                min_value=20.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                help="기초자산 대비 최소 남은 자산 비율"
            )

        st.markdown("### 인출률 범위")
        col1, col2, col3 = st.columns(3)

        with col1:
            min_wr = st.number_input(
                "최소 인출률 (%)",
                min_value=1.0,
                max_value=15.0,
                value=3.0,
                step=0.5
            )

        with col2:
            max_wr = st.number_input(
                "최대 인출률 (%)",
                min_value=1.0,
                max_value=15.0,
                value=10.0,
                step=0.5
            )

        with col3:
            wr_step = st.number_input(
                "단계 크기 (%)",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )

        st.markdown("### 포트폴리오 선택")
        portfolio_names = list(PORTFOLIOS.keys())
        selected_portfolios = st.multiselect(
            "분석할 포트폴리오",
            options=portfolio_names,
            default=portfolio_names[:3]
        )

        submitted = st.form_submit_button("🚀 최적화 실행", use_container_width=True)

    if submitted:
        if not selected_portfolios:
            st.error("최소 1개 이상의 포트폴리오를 선택해주세요.")
            return

        with st.spinner("데이터 전처리 중..."):
            try:
                # 데이터 전처리
                preprocessor = DataPreprocessor(benchmark_data, add_portfolios=True)
                returns, month_starts = preprocessor.get_data()

                # 최적화 객체 생성
                optimizer = WithdrawalOptimizer(returns, month_starts)

                # 인출률 배열 생성
                withdrawal_rates = np.arange(min_wr/100, max_wr/100 + wr_step/100, wr_step/100)

                constraints = {
                    'max_yoy_volatility': max_volatility / 100,
                    'max_failure_rate': max_failure / 100,
                    'min_terminal_nav': min_terminal / 100
                }

                st.info(f"🔄 {len(selected_portfolios)}개 포트폴리오 × 3개 전략 × {len(withdrawal_rates)}개 인출률 = "
                       f"{len(selected_portfolios) * 3 * len(withdrawal_rates):,}개 시나리오 실행 중...")

                # 최적화 실행
                results_df = optimizer.optimize_all_portfolios(
                    selected_portfolios,
                    withdrawal_rates,
                    horizon_years=10,
                    constraints=constraints,
                    v0=100.0
                )

                # 세션에 결과 저장
                st.session_state['optimization_results'] = results_df
                st.session_state['optimization_constraints'] = constraints
                st.session_state['optimization_portfolios'] = selected_portfolios

                st.success("✅ 최적화 완료!")

            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                return

    # 결과 표시
    if 'optimization_results' in st.session_state:
        results_df = st.session_state['optimization_results']
        constraints = st.session_state['optimization_constraints']

        st.divider()
        st.subheader("📊 최적화 결과")

        # 최적 인출률 요약
        st.markdown("#### 전략별 최적 인출률")

        summary_data = []
        for portfolio in results_df['Portfolio'].unique():
            df_port = results_df[results_df['Portfolio'] == portfolio]

            for strategy in ['fixed', 'guardrails', 'guyton_klinger']:
                df_strat = df_port[df_port['Strategy'] == strategy]
                df_feasible = df_strat[df_strat['Feasible'] == True]

                if not df_feasible.empty:
                    best = df_feasible.loc[df_feasible['Total_Withdrawal'].idxmax()]
                    summary_data.append({
                        '포트폴리오': portfolio,
                        '전략': strategy.replace('_', ' ').title(),
                        '최적 인출률': f"{best['WR']:.2%}",
                        '총 인출액': f"{best['Total_Withdrawal']:.0f}",
                        'YoY 변동성': f"{best['YoY_Volatility']:.2%}",
                        '실패율': f"{best['Failure_Rate']:.2%}",
                        '최종 NAV': f"{best['Terminal_NAV']:.2%}"
                    })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.warning("제약 조건을 만족하는 솔루션이 없습니다.")

        st.divider()

        # Pareto Frontier 차트
        st.markdown("#### Pareto Frontier 분석")

        col1, col2 = st.columns([1, 3])

        with col1:
            selected_portfolio_chart = st.selectbox(
                "포트폴리오 선택",
                options=results_df['Portfolio'].unique(),
                key="pareto_portfolio"
            )

        fig_pareto = create_pareto_frontier_chart(results_df, selected_portfolio_chart)
        if fig_pareto:
            st.plotly_chart(fig_pareto, use_container_width=True)

        st.divider()

        # 전체 결과 테이블
        st.markdown("#### 전체 결과 (필터링 가능)")

        # 필터링 옵션
        col1, col2 = st.columns(2)

        with col1:
            filter_portfolio = st.multiselect(
                "포트폴리오 필터",
                options=results_df['Portfolio'].unique(),
                default=results_df['Portfolio'].unique(),
                key="filter_portfolio"
            )

        with col2:
            filter_strategy = st.multiselect(
                "전략 필터",
                options=results_df['Strategy'].unique(),
                default=results_df['Strategy'].unique(),
                key="filter_strategy"
            )

        filtered_results = results_df[
            (results_df['Portfolio'].isin(filter_portfolio)) &
            (results_df['Strategy'].isin(filter_strategy))
        ]

        st.dataframe(
            filtered_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "WR": st.column_config.NumberColumn(format="%.2%%"),
                "YoY_Volatility": st.column_config.NumberColumn(format="%.2%%"),
                "Failure_Rate": st.column_config.NumberColumn(format="%.2%%"),
                "Terminal_NAV": st.column_config.NumberColumn(format="%.2%%"),
            }
        )

        st.divider()

        # 다운로드 버튼
        csv = filtered_results.to_csv(index=False)
        st.download_button(
            label="📥 결과 CSV 다운로드",
            data=csv,
            file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )


def render_comparison_section():
    """
    Comparison section: Compare strategies side by side
    """
    st.subheader("전략 비교")

    if 'optimization_results' not in st.session_state:
        st.info("먼저 최적화를 실행해주세요.")
        return

    results_df = st.session_state['optimization_results']

    # 포트폴리오 선택
    selected_portfolio = st.selectbox(
        "포트폴리오 선택",
        options=results_df['Portfolio'].unique(),
        key="comparison_portfolio"
    )

    # 해당 포트폴리오의 데이터
    df_port = results_df[results_df['Portfolio'] == selected_portfolio]

    st.markdown(f"### {selected_portfolio} - 전략 비교")

    # 메트릭 비교
    col1, col2, col3 = st.columns(3)

    for idx, strategy in enumerate(['fixed', 'guardrails', 'guyton_klinger']):
        df_strat = df_port[df_port['Strategy'] == strategy]
        df_feasible = df_strat[df_strat['Feasible'] == True]

        if not df_feasible.empty:
            best = df_feasible.loc[df_feasible['Total_Withdrawal'].idxmax()]

            if idx % 3 == 0:
                col = col1
            elif idx % 3 == 1:
                col = col2
            else:
                col = col3

            with col:
                st.markdown(f"**{strategy.replace('_', ' ').title()}**")
                st.metric("최적 인출률", f"{best['WR']:.2%}")
                st.metric("총 인출액", f"{best['Total_Withdrawal']:.0f}")
                st.metric("YoY 변동성", f"{best['YoY_Volatility']:.2%}")
                st.metric("실패율", f"{best['Failure_Rate']:.2%}")
                st.metric("최종 NAV", f"{best['Terminal_NAV']:.2%}")
        else:
            col_obj = [col1, col2, col3][idx % 3]
            with col_obj:
                st.markdown(f"**{strategy.replace('_', ' ').title()}**")
                st.warning("제약 조건을 만족하는 솔루션 없음")

    st.divider()

    # 메트릭별 비교 차트
    st.markdown("### 메트릭별 비교")

    # WR vs Total Withdrawal
    fig1 = create_strategy_comparison_chart(df_port, metric_x='WR', metric_y='Total_Withdrawal')
    if fig1:
        st.plotly_chart(fig1, use_container_width=True)

    # WR vs YoY Volatility
    fig2 = create_strategy_comparison_chart(df_port, metric_x='WR', metric_y='YoY_Volatility')
    if fig2:
        st.plotly_chart(fig2, use_container_width=True)

    # Total Withdrawal vs Failure Rate
    fig3 = create_strategy_comparison_chart(df_port, metric_x='Total_Withdrawal', metric_y='Failure_Rate')
    if fig3:
        st.plotly_chart(fig3, use_container_width=True)


def render_detailed_analysis_section():
    """
    Detailed Analysis section: Advanced analysis
    """
    st.subheader("상세 분석")

    if 'optimization_results' not in st.session_state:
        st.info("먼저 최적화를 실행해주세요.")
        return

    results_df = st.session_state['optimization_results']

    st.markdown("""
    ### 주요 지표 설명

    - **WR (Withdrawal Rate)**: 초기 자산 대비 연간 인출 비율
    - **Total Withdrawal**: 10년 동안의 총 인출액
    - **YoY Volatility**: 연간 인출액 변화의 표준편차 (낮을수록 안정적)
    - **Failure Rate**: 포트폴리오가 고갈되는 비율
    - **Terminal NAV**: 기초자산 대비 최종 남은 자산 비율

    """)

    # 데이터 통계
    st.markdown("### 전체 데이터 통계")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "총 시나리오 수",
            f"{len(results_df):,}"
        )

        feasible_count = len(results_df[results_df['Feasible'] == True])
        st.metric(
            "제약 조건 만족 시나리오",
            f"{feasible_count:,}",
            f"{feasible_count/len(results_df)*100:.1f}%"
        )

    with col2:
        st.metric(
            "평균 총 인출액",
            f"{results_df['Total_Withdrawal'].mean():.0f}"
        )

        st.metric(
            "평균 실패율",
            f"{results_df['Failure_Rate'].mean():.2%}"
        )

    st.divider()

    # 고급 필터링
    st.markdown("### 고급 필터링")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_wr_filter = st.slider(
            "최소 인출률",
            min_value=float(results_df['WR'].min()),
            max_value=float(results_df['WR'].max()),
            value=float(results_df['WR'].min()),
            step=0.005,
            format="%.2%%"
        )

    with col2:
        max_failure_filter = st.slider(
            "최대 실패율",
            min_value=0.0,
            max_value=float(results_df['Failure_Rate'].max()),
            value=float(results_df['Failure_Rate'].max()),
            step=0.01,
            format="%.2%%"
        )

    with col3:
        min_withdrawal_filter = st.slider(
            "최소 총 인출액",
            min_value=float(results_df['Total_Withdrawal'].min()),
            max_value=float(results_df['Total_Withdrawal'].max()),
            value=float(results_df['Total_Withdrawal'].min()),
            step=100.0
        )

    # 필터링 적용
    filtered_df = results_df[
        (results_df['WR'] >= min_wr_filter) &
        (results_df['Failure_Rate'] <= max_failure_filter) &
        (results_df['Total_Withdrawal'] >= min_withdrawal_filter)
    ]

    st.markdown(f"### 필터링 결과: {len(filtered_df):,} 행")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# Chart Functions
# ============================================================================

def create_pareto_frontier_chart(results_df, portfolio_name):
    """
    Create Pareto Frontier chart: Total Withdrawal vs YoY Volatility
    """
    df_port = results_df[results_df['Portfolio'] == portfolio_name]

    fig = go.Figure()

    strategies = ['fixed', 'guardrails', 'guyton_klinger']
    colors = {'fixed': '#3498db', 'guardrails': '#2ecc71', 'guyton_klinger': '#e74c3c'}
    symbols = {'fixed': 'circle', 'guardrails': 'square', 'guyton_klinger': 'diamond'}

    for strategy in strategies:
        df_strat = df_port[df_port['Strategy'] == strategy]

        # Feasible points
        feasible = df_strat[df_strat['Feasible'] == True]

        if not feasible.empty:
            fig.add_trace(go.Scatter(
                x=feasible['YoY_Volatility'],
                y=feasible['Total_Withdrawal'],
                mode='markers+lines',
                name=strategy.replace('_', ' ').title(),
                marker=dict(
                    size=10,
                    color=colors[strategy],
                    symbol=symbols[strategy],
                    line=dict(color='white', width=2)
                ),
                text=[f"WR: {wr:.2%}<br>Vol: {vol:.2%}<br>Withdrawal: {w:.0f}"
                      for wr, vol, w in zip(feasible['WR'], feasible['YoY_Volatility'],
                                           feasible['Total_Withdrawal'])],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))

        # Infeasible points
        infeasible = df_strat[df_strat['Feasible'] == False]

        if not infeasible.empty:
            fig.add_trace(go.Scatter(
                x=infeasible['YoY_Volatility'],
                y=infeasible['Total_Withdrawal'],
                mode='markers',
                name=f"{strategy.replace('_', ' ').title()} (제약 위반)",
                marker=dict(
                    size=5,
                    color=colors[strategy],
                    symbol='x',
                    opacity=0.3
                ),
                hoverinfo='skip'
            ))

    fig.update_layout(
        title=f"Pareto Frontier - {portfolio_name}",
        xaxis_title="YoY Volatility (변동성)",
        yaxis_title="Total Withdrawal (총 인출액)",
        hovermode='closest',
        height=500,
        showlegend=True
    )

    return fig


def create_strategy_comparison_chart(df_port, metric_x, metric_y):
    """
    Create comparison chart between strategies
    """
    fig = go.Figure()

    strategies = ['fixed', 'guardrails', 'guyton_klinger']
    colors = {'fixed': '#3498db', 'guardrails': '#2ecc71', 'guyton_klinger': '#e74c3c'}

    for strategy in strategies:
        df_strat = df_port[df_port['Strategy'] == strategy]
        df_feasible = df_strat[df_strat['Feasible'] == True]

        if not df_feasible.empty:
            fig.add_trace(go.Scatter(
                x=df_feasible[metric_x],
                y=df_feasible[metric_y],
                mode='lines+markers',
                name=strategy.replace('_', ' ').title(),
                line=dict(color=colors[strategy], width=3),
                marker=dict(size=8)
            ))

    fig.update_layout(
        title=f"{metric_x} vs {metric_y}",
        xaxis_title=metric_x,
        yaxis_title=metric_y,
        hovermode='x unified',
        height=400
    )

    return fig
