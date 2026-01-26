"""
Streamlit Withdrawal Rate Backtesting Application
=================================================
Retirement portfolio withdrawal rate backtesting tool
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
import io
import os

# Import portfolio definitions
from withdrawal_backtest import PORTFOLIOS, BENCHMARK_MAPPING

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="인출률 백테스팅",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# Data Loading with Caching
# ============================================================================
@st.cache_data
def load_benchmark_data():
    """Load benchmark data from pickle file"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'benchmark_data.pkl')
        data = pd.read_pickle(data_path)
        return data
    except FileNotFoundError:
        st.error("benchmark_data.pkl 파일을 찾을 수 없습니다. 파일이 앱과 동일한 디렉토리에 있는지 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None


@st.cache_data
def get_portfolio_names():
    """Get list of portfolio names"""
    return list(PORTFOLIOS.keys())


# ============================================================================
# Backtest Function (Placeholder)
# ============================================================================
def run_backtest(data, weights, withdrawal_rate, start_date, end_date,
                 period, upper_guard, lower_guard, failure_threshold):
    """
    Run withdrawal rate backtest for a portfolio

    Parameters
    ----------
    data : pd.DataFrame
        Benchmark price data
    weights : dict
        Portfolio weights (%)
    withdrawal_rate : float
        Annual withdrawal rate (%)
    start_date : date
        Backtest start date
    end_date : date
        Backtest end date
    period : int
        Withdrawal period (years)
    upper_guard : float or None
        Upper guardrail (%), None if no constraint
    lower_guard : float or None
        Lower guardrail (%), None if no constraint
    failure_threshold : float
        Failure threshold (% of initial value)

    Returns
    -------
    dict
        Backtest results containing success_rate, final_values, monthly_data, statistics
    """
    # TODO: Implement actual backtesting logic
    # This is a placeholder implementation

    # Convert weights to decimal
    weights_decimal = {k: v/100 for k, v in weights.items()}

    # Filter data by date range
    if isinstance(data.index, pd.DatetimeIndex):
        mask = (data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))
        filtered_data = data.loc[mask]
    else:
        filtered_data = data

    # Calculate portfolio returns
    returns = filtered_data.pct_change().dropna()

    # Map asset names to benchmark names
    portfolio_returns = pd.Series(0.0, index=returns.index)
    for asset_name, weight in weights_decimal.items():
        benchmark_name = BENCHMARK_MAPPING.get(asset_name, asset_name)
        if benchmark_name in returns.columns:
            portfolio_returns += returns[benchmark_name] * weight

    # Identify month starts
    month_starts = portfolio_returns.index.to_series().groupby(
        portfolio_returns.index.to_period('M')
    ).first()

    # Calculate number of possible simulations
    total_months = len(month_starts)
    period_months = period * 12

    if total_months < period_months:
        return {
            'success_rate': 0.0,
            'failure_rate': 100.0,
            'final_values': [],
            'monthly_data': pd.DataFrame(),
            'statistics': {},
            'start_months': [],
            'success_count': 0,
            'failure_count': 0,
            'total_simulations': 0
        }

    # Run rolling window simulations
    v0 = 100.0  # Initial portfolio value
    monthly_wr = (withdrawal_rate / 100) * v0 / 12  # Monthly withdrawal amount
    failure_value = v0 * (failure_threshold / 100)  # Failure threshold value

    final_values = []
    start_months_list = []
    success_flags = []

    # Simulate for each possible start month
    valid_start_months = list(month_starts.values)[:-period_months] if len(month_starts) > period_months else []

    for i, start_month in enumerate(valid_start_months):
        V = v0
        failed = False

        # Get returns for this simulation period
        sim_start_idx = portfolio_returns.index.get_loc(start_month)

        # Find the index period_months later
        end_month_idx = min(i + period_months, len(month_starts) - 1)
        if end_month_idx < len(month_starts):
            end_month = month_starts.values[i + period_months] if i + period_months < len(month_starts) else portfolio_returns.index[-1]
        else:
            continue

        # Simple monthly simulation
        current_month_starts = month_starts.values[i:i+period_months+1]

        for j in range(len(current_month_starts) - 1):
            # Apply withdrawal at month start
            if upper_guard is not None and upper_guard > 0:
                adjusted_wr = min(monthly_wr, V * (upper_guard / 100) / 12)
            elif lower_guard is not None and lower_guard > 0:
                adjusted_wr = max(monthly_wr, V * (lower_guard / 100) / 12)
            else:
                adjusted_wr = monthly_wr

            V = max(V - adjusted_wr, 0)

            if V <= 0:
                failed = True
                break

            # Get monthly return
            month_start = current_month_starts[j]
            month_end = current_month_starts[j + 1]

            try:
                month_mask = (portfolio_returns.index >= month_start) & (portfolio_returns.index < month_end)
                month_returns = portfolio_returns.loc[month_mask]
                cumulative_return = (1 + month_returns).prod()
                V = V * cumulative_return
            except:
                pass

            # Check failure condition
            if V < failure_value:
                failed = True
                break

        final_values.append(V)
        start_months_list.append(start_month)
        success_flags.append(not failed and V >= failure_value)

    # Calculate statistics
    if len(final_values) > 0:
        success_count = sum(success_flags)
        failure_count = len(success_flags) - success_count
        success_rate = (success_count / len(success_flags)) * 100

        final_values_arr = np.array(final_values)
        statistics = {
            'mean_final_value': float(np.mean(final_values_arr)),
            'median_final_value': float(np.median(final_values_arr)),
            'std_final_value': float(np.std(final_values_arr)),
            'min_final_value': float(np.min(final_values_arr)),
            'max_final_value': float(np.max(final_values_arr)),
            'percentile_5': float(np.percentile(final_values_arr, 5)),
            'percentile_25': float(np.percentile(final_values_arr, 25)),
            'percentile_75': float(np.percentile(final_values_arr, 75)),
            'percentile_95': float(np.percentile(final_values_arr, 95)),
        }
    else:
        success_count = 0
        failure_count = 0
        success_rate = 0.0
        statistics = {}

    # Create monthly data for Excel export
    monthly_data = pd.DataFrame({
        'Start Month': start_months_list,
        'Final Value': final_values,
        'Success': success_flags
    })

    return {
        'success_rate': success_rate,
        'failure_rate': 100 - success_rate,
        'final_values': final_values,
        'monthly_data': monthly_data,
        'statistics': statistics,
        'start_months': start_months_list,
        'success_count': success_count,
        'failure_count': failure_count,
        'total_simulations': len(final_values)
    }


# ============================================================================
# Excel Download Functions
# ============================================================================
def create_single_portfolio_excel(portfolio_name, result_data, params):
    """Create Excel file for single portfolio results"""
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Monthly data sheet
        if not result_data['monthly_data'].empty:
            result_data['monthly_data'].to_excel(writer, sheet_name='Monthly_Data', index=False)

        # Statistics sheet
        stats_df = pd.DataFrame([result_data['statistics']])
        stats_df['success_rate'] = result_data['success_rate']
        stats_df['failure_rate'] = result_data['failure_rate']
        stats_df['total_simulations'] = result_data['total_simulations']
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        # Parameters sheet
        params_df = pd.DataFrame([{
            'Portfolio': portfolio_name,
            'Withdrawal Rate (%)': params['withdrawal_rate'],
            'Start Date': str(params['start_date']),
            'End Date': str(params['end_date']),
            'Period (Years)': params['withdrawal_period'],
            'Upper Guard (%)': params['upper_guard'] if params['upper_guard'] > 0 else 'None',
            'Lower Guard (%)': params['lower_guard'] if params['lower_guard'] > 0 else 'None',
            'Failure Threshold (%)': params['failure_threshold']
        }])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

    return buffer.getvalue()


def create_comparison_excel(selected_portfolios, results, params):
    """Create Excel file for portfolio comparison"""
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Summary comparison sheet
        summary_data = []
        for port_name in selected_portfolios:
            if port_name in results:
                result = results[port_name]
                summary_data.append({
                    'Portfolio': port_name,
                    'Success Rate (%)': result['success_rate'],
                    'Failure Rate (%)': result['failure_rate'],
                    'Total Simulations': result['total_simulations'],
                    'Mean Final Value': result['statistics'].get('mean_final_value', 0),
                    'Median Final Value': result['statistics'].get('median_final_value', 0),
                    'Min Final Value': result['statistics'].get('min_final_value', 0),
                    'Max Final Value': result['statistics'].get('max_final_value', 0)
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Individual sheets for each portfolio
        for port_name in selected_portfolios:
            if port_name in results:
                result = results[port_name]
                if not result['monthly_data'].empty:
                    # Shorten sheet name if too long
                    sheet_name = port_name[:31] if len(port_name) > 31 else port_name
                    result['monthly_data'].to_excel(writer, sheet_name=sheet_name, index=False)

        # Parameters sheet
        params_df = pd.DataFrame([{
            'Withdrawal Rate (%)': params['withdrawal_rate'],
            'Start Date': str(params['start_date']),
            'End Date': str(params['end_date']),
            'Period (Years)': params['withdrawal_period'],
            'Upper Guard (%)': params['upper_guard'] if params['upper_guard'] > 0 else 'None',
            'Lower Guard (%)': params['lower_guard'] if params['lower_guard'] > 0 else 'None',
            'Failure Threshold (%)': params['failure_threshold']
        }])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

    return buffer.getvalue()


# ============================================================================
# Visualization Functions
# ============================================================================
def create_success_failure_stacked_chart(result_data, portfolio_name):
    """Create stacked bar chart for success/failure by start month"""
    monthly_data = result_data['monthly_data']

    if monthly_data.empty:
        return None

    # Group by month
    monthly_data['Year_Month'] = pd.to_datetime(monthly_data['Start Month']).dt.to_period('M')
    monthly_grouped = monthly_data.groupby('Year_Month')['Success'].agg(['sum', 'count']).reset_index()
    monthly_grouped.columns = ['Year_Month', 'Success_Count', 'Total_Count']
    monthly_grouped['Failure_Count'] = monthly_grouped['Total_Count'] - monthly_grouped['Success_Count']
    monthly_grouped['Date'] = monthly_grouped['Year_Month'].dt.to_timestamp()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Success',
        x=monthly_grouped['Date'],
        y=monthly_grouped['Success_Count'],
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        name='Failure',
        x=monthly_grouped['Date'],
        y=monthly_grouped['Failure_Count'],
        marker_color='red'
    ))

    fig.update_layout(
        barmode='stack',
        title=f'Success/Failure by Start Month - {portfolio_name}',
        xaxis_title='Start Month',
        yaxis_title='Count',
        legend_title='Result',
        hovermode='x unified'
    )

    return fig


def create_comparison_bar_chart(results, selected_portfolios):
    """Create bar chart comparing success rates across portfolios"""
    data = []
    for port_name in selected_portfolios:
        if port_name in results:
            data.append({
                'Portfolio': port_name,
                'Success Rate (%)': results[port_name]['success_rate']
            })

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x='Portfolio',
        y='Success Rate (%)',
        labels={'Portfolio': 'Portfolio', 'Success Rate (%)': 'Success Rate (%)'},
        title='Portfolio Success Rate Comparison',
        color='Success Rate (%)',
        color_continuous_scale='RdYlGn'
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )

    return fig


def create_comparison_table(results, selected_portfolios):
    """Create comparison table for selected portfolios"""
    data = []
    for port_name in selected_portfolios:
        if port_name in results:
            result = results[port_name]
            stats = result['statistics']
            data.append({
                'Portfolio': port_name,
                'Success Rate (%)': f"{result['success_rate']:.2f}",
                'Total Simulations': result['total_simulations'],
                'Mean Final Value': f"{stats.get('mean_final_value', 0):.2f}",
                'Median Final Value': f"{stats.get('median_final_value', 0):.2f}",
                '5th Percentile': f"{stats.get('percentile_5', 0):.2f}",
                '95th Percentile': f"{stats.get('percentile_95', 0):.2f}"
            })

    return pd.DataFrame(data)


# ============================================================================
# Main Application
# ============================================================================
def main():
    st.title("📊 인출률 백테스팅 애플리케이션")
    st.markdown("은퇴 포트폴리오의 인출률 백테스팅을 수행합니다.")

    st.divider()

    # Load data
    data = load_benchmark_data()

    if data is None:
        st.stop()

    # Get portfolio names
    portfolio_names = get_portfolio_names()

    # ========================================================================
    # Input Parameters Form
    # ========================================================================
    st.subheader("📝 백테스팅 파라미터 설정")

    with st.form(key="backtest_form"):
        col1, col2 = st.columns(2)

        with col1:
            withdrawal_rate = st.number_input(
                "연간 인출률 (%)",
                min_value=1.0,
                max_value=10.0,
                value=4.0,
                step=0.1,
                help="연간 인출률을 입력하세요 (1.0% ~ 10.0%)"
            )

            start_date = st.date_input(
                "백테스팅 시작일",
                value=date(2001, 1, 3),
                help="백테스팅을 시작할 날짜"
            )

            end_date = st.date_input(
                "백테스팅 종료일",
                value=date(2025, 12, 31),
                help="백테스팅을 종료할 날짜"
            )

            withdrawal_period = st.number_input(
                "인출 기간 (년)",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                help="인출 기간을 년 단위로 입력하세요"
            )

        with col2:
            upper_guard = st.number_input(
                "상한 가드레일 (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                help="0이면 제약 없음"
            )

            lower_guard = st.number_input(
                "하한 가드레일 (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                help="0이면 제약 없음"
            )

            failure_threshold = st.number_input(
                "실패 조건 (초기 자산 대비 %)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                help="예: 50 입력 시 초기 자산의 50% 이하로 떨어지면 실패로 간주"
            )

        submit_button = st.form_submit_button(
            "🚀 모든 포트폴리오 계산 시작",
            use_container_width=True
        )

    # ========================================================================
    # Run Backtest on Form Submit
    # ========================================================================
    if submit_button:
        # Save parameters to session state
        st.session_state['params'] = {
            'withdrawal_rate': withdrawal_rate,
            'start_date': start_date,
            'end_date': end_date,
            'withdrawal_period': withdrawal_period,
            'upper_guard': upper_guard,
            'lower_guard': lower_guard,
            'failure_threshold': failure_threshold
        }

        # Convert guardrails (0 means no constraint)
        upper_guard_val = upper_guard if upper_guard > 0 else None
        lower_guard_val = lower_guard if lower_guard > 0 else None

        # Run backtest for all portfolios
        with st.spinner("백테스팅 실행 중..."):
            results = {}
            progress_bar = st.progress(0)

            for i, port_name in enumerate(portfolio_names):
                port_config = PORTFOLIOS[port_name]

                result = run_backtest(
                    data=data,
                    weights=port_config['weights'],
                    withdrawal_rate=withdrawal_rate,
                    start_date=start_date,
                    end_date=end_date,
                    period=withdrawal_period,
                    upper_guard=upper_guard_val,
                    lower_guard=lower_guard_val,
                    failure_threshold=failure_threshold
                )

                results[port_name] = result
                progress_bar.progress((i + 1) / len(portfolio_names))

            # Save results to session state
            st.session_state['results'] = results

        st.success("✅ 계산 완료!")

    # ========================================================================
    # Display Results (only if results exist in session state)
    # ========================================================================
    if 'results' in st.session_state and st.session_state['results']:
        st.divider()
        st.subheader("📈 백테스팅 결과")

        results = st.session_state['results']
        params = st.session_state['params']

        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 개별 포트폴리오 분석", "🔄 포트폴리오 비교"])

        # ====================================================================
        # Tab 1: Individual Portfolio Analysis
        # ====================================================================
        with tab1:
            selected_portfolio = st.selectbox(
                "포트폴리오 선택",
                options=portfolio_names,
                key="individual_portfolio_select"
            )

            if selected_portfolio in results:
                result = results[selected_portfolio]

                # Display success rate prominently
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="성공률",
                        value=f"{result['success_rate']:.2f}%",
                        delta=f"{result['success_count']} / {result['total_simulations']}"
                    )

                with col2:
                    st.metric(
                        label="실패율",
                        value=f"{result['failure_rate']:.2f}%",
                        delta=f"{result['failure_count']} / {result['total_simulations']}"
                    )

                with col3:
                    st.metric(
                        label="총 시뮬레이션 수",
                        value=f"{result['total_simulations']:,}"
                    )

                st.divider()

                # Stacked bar chart
                fig = create_success_failure_stacked_chart(result, selected_portfolio)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Statistics table
                st.markdown("#### 주요 통계량")
                if result['statistics']:
                    stats_df = pd.DataFrame([{
                        'Mean Final Value': f"{result['statistics'].get('mean_final_value', 0):.2f}",
                        'Median Final Value': f"{result['statistics'].get('median_final_value', 0):.2f}",
                        'Std Dev': f"{result['statistics'].get('std_final_value', 0):.2f}",
                        'Min': f"{result['statistics'].get('min_final_value', 0):.2f}",
                        'Max': f"{result['statistics'].get('max_final_value', 0):.2f}",
                        '5th Percentile': f"{result['statistics'].get('percentile_5', 0):.2f}",
                        '25th Percentile': f"{result['statistics'].get('percentile_25', 0):.2f}",
                        '75th Percentile': f"{result['statistics'].get('percentile_75', 0):.2f}",
                        '95th Percentile': f"{result['statistics'].get('percentile_95', 0):.2f}"
                    }])
                    st.dataframe(stats_df, use_container_width=True)

                # Excel download button
                st.divider()
                excel_data = create_single_portfolio_excel(selected_portfolio, result, params)
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"{selected_portfolio}_backtest_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        # ====================================================================
        # Tab 2: Portfolio Comparison
        # ====================================================================
        with tab2:
            selected_portfolios = st.multiselect(
                "비교할 포트폴리오 선택",
                options=portfolio_names,
                default=portfolio_names,
                key="comparison_portfolio_select"
            )

            if selected_portfolios:
                # Comparison bar chart
                fig = create_comparison_bar_chart(results, selected_portfolios)
                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                # Comparison table
                st.markdown("#### 포트폴리오별 주요 지표 비교")
                comparison_df = create_comparison_table(results, selected_portfolios)
                st.dataframe(comparison_df, use_container_width=True)

                # Excel download button for comparison
                st.divider()
                comparison_excel = create_comparison_excel(selected_portfolios, results, params)
                st.download_button(
                    label="📥 Download Comparison Excel",
                    data=comparison_excel,
                    file_name="portfolio_comparison.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("비교할 포트폴리오를 선택해주세요.")


if __name__ == "__main__":
    main()
