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
from plotly.subplots import make_subplots
import io
import os
 
# Import portfolio definitions
from withdrawal_backtest import PORTFOLIOS, BENCHMARK_MAPPING
from dynamic_strategy_ui import render_dynamic_strategies_tab
 
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
# Portfolio Info Functions
# ============================================================================
def get_portfolio_info(portfolio_name):
    """Get portfolio information including weights, target return, and risk"""
    if portfolio_name not in PORTFOLIOS:
        return None
 
    port_config = PORTFOLIOS[portfolio_name]
    return {
        'name': portfolio_name,
        'target_return': port_config['target_return'],
        'target_risk': port_config['target_risk'],
        'weights': port_config['weights']
    }
 
 
def create_portfolio_weights_chart(portfolio_name):
    """Create pie chart for portfolio weights"""
    info = get_portfolio_info(portfolio_name)
    if info is None:
        return None
 
    weights = info['weights']
 
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    )])
 
    fig.update_layout(
        # title=f'Asset Allocation - {portfolio_name}',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
 
    return fig
 
 
def create_portfolio_info_table(portfolio_names):
    """Create table with portfolio info (target return, risk, weights)"""
    data = []
    for port_name in portfolio_names:
        info = get_portfolio_info(port_name)
        if info:
            row = {
                'Portfolio': port_name,
                'Target Return (%)': info['target_return'],
                'Target Risk (%)': info['target_risk'],
            }
            # Add weights
            for asset, weight in info['weights'].items():
                row[asset] = f"{weight:.2f}%"
            data.append(row)
 
    return pd.DataFrame(data)
 
 
# ============================================================================
# Backtest Function
# ============================================================================
def run_backtest(data, weights, withdrawal_rate, start_date, end_date,
                 period, upper_guard, lower_guard, failure_threshold):
    """
    Run withdrawal rate backtest for a portfolio
 
    Returns results including daily NAV data for each simulation
    """
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
 
    # Map asset names to benchmark names and store mapping
    asset_to_benchmark = {}
    portfolio_returns = pd.Series(0.0, index=returns.index)
    for asset_name, weight in weights_decimal.items():
        benchmark_name = BENCHMARK_MAPPING.get(asset_name, asset_name)
        if benchmark_name in returns.columns:
            portfolio_returns += returns[benchmark_name] * weight
            asset_to_benchmark[asset_name] = benchmark_name
 
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
            'total_simulations': 0,
            'nav_paths': [],
            'nav_paths_success': [],
            'nav_paths_failure': [],
            'daily_data': {},
            'filtered_data': filtered_data,
            'asset_to_benchmark': asset_to_benchmark
        }
 
    # Run rolling window simulations
    v0 = 100.0  # Initial portfolio value
    monthly_wr = (withdrawal_rate / 100) * v0 / 12  # Monthly withdrawal amount
    failure_value = v0 * (failure_threshold / 100)  # Failure threshold value
 
    final_values = []
    start_months_list = []
    success_flags = []
    nav_paths = []  # Store all NAV paths (monthly)
    nav_paths_success = []
    nav_paths_failure = []
    daily_data = {}  # Store daily data for each start month
 
    # Simulate for each possible start month
    valid_start_months = list(month_starts.values)[:-period_months] if len(month_starts) > period_months else []
 
    for i, start_month in enumerate(valid_start_months):
        V = v0
        failed = False
        path = [V]  # Monthly NAV path
 
        # Get the end date for this simulation
        end_month_idx = i + period_months
        if end_month_idx < len(month_starts):
            sim_end_date = month_starts.values[end_month_idx]
        else:
            continue
 
        # Get daily data for this simulation period
        daily_mask = (filtered_data.index >= start_month) & (filtered_data.index < sim_end_date)
        sim_daily_data = filtered_data.loc[daily_mask].copy()
 
        # Calculate daily NAV
        daily_nav = []
        daily_dates = []
        current_V = v0
 
        # Simple monthly simulation
        current_month_starts = month_starts.values[i:i+period_months+1]
 
        for j in range(len(current_month_starts) - 1):
            month_start = current_month_starts[j]
            month_end = current_month_starts[j + 1]
 
            # Apply withdrawal at month start
            if upper_guard is not None and upper_guard > 0:
                adjusted_wr = min(monthly_wr, current_V * (upper_guard / 100) / 12)
            elif lower_guard is not None and lower_guard > 0:
                adjusted_wr = max(monthly_wr, current_V * (lower_guard / 100) / 12)
            else:
                adjusted_wr = monthly_wr
 
            current_V = max(current_V - adjusted_wr, 0)
 
            if current_V <= 0:
                failed = True
                path.append(0)
                break
 
            # Get daily returns for this month
            month_mask = (portfolio_returns.index >= month_start) & (portfolio_returns.index < month_end)
            month_daily_returns = portfolio_returns.loc[month_mask]
 
            # Calculate daily NAV within this month
            for idx, daily_ret in month_daily_returns.items():
                current_V = current_V * (1 + daily_ret)
                daily_nav.append(current_V)
                daily_dates.append(idx)
 
            V = current_V
            path.append(V)
 
            # Check failure condition
            if V < failure_value:
                failed = True
                break
 
        final_values.append(V)
        start_months_list.append(start_month)
        is_success = not failed and V >= failure_value
        success_flags.append(is_success)
        nav_paths.append(path)
 
        if is_success:
            nav_paths_success.append(path)
        else:
            nav_paths_failure.append(path)
 
        # Store daily data for this simulation
        if daily_dates:
            sim_daily_df = pd.DataFrame({
                'Date': daily_dates,
                'Portfolio_NAV': daily_nav
            })
            # Add benchmark indices (normalized to 100)
            for asset_name, benchmark_name in asset_to_benchmark.items():
                if benchmark_name in filtered_data.columns:
                    benchmark_data = filtered_data.loc[daily_dates, benchmark_name]
                    # Normalize to 100 at start
                    if len(benchmark_data) > 0:
                        normalized = (benchmark_data / benchmark_data.iloc[0]) * 100
                        sim_daily_df[asset_name] = normalized.values
 
            daily_data[start_month] = sim_daily_df
 
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
        'total_simulations': len(final_values),
        'nav_paths': nav_paths,
        'nav_paths_success': nav_paths_success,
        'nav_paths_failure': nav_paths_failure,
        'period_months': period_months,
        'daily_data': daily_data,
        'filtered_data': filtered_data,
        'asset_to_benchmark': asset_to_benchmark
    }
 
 
# ============================================================================
# Excel Download Functions
# ============================================================================
def create_daily_nav_excel(portfolio_name, selected_start_date, daily_data, params):
    """Create Excel file with daily NAV data for selected start date"""
    buffer = io.BytesIO()
 
    # Get portfolio info
    info = get_portfolio_info(portfolio_name)
 
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Get the worksheet
        workbook = writer.book
 
        # Create main sheet
        sheet_name = 'Daily_NAV'
 
        # Prepare portfolio info dataframe (for top section)
        if info:
            # Portfolio info table
            info_data = {
                'Portfolio': [portfolio_name],
                'Target Return (%)': [info['target_return']],
                'Target Risk (%)': [info['target_risk']],
            }
            for asset, weight in info['weights'].items():
                info_data[f'{asset} (%)'] = [weight]
 
            info_df = pd.DataFrame(info_data)
 
            # Write portfolio info at top
            info_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
 
            # Add parameters
            params_data = {
                'Withdrawal Rate (%)': [params['withdrawal_rate']],
                'Withdrawal Start Date': [str(selected_start_date)],
                'Period (Years)': [params['withdrawal_period']],
                'Upper Guard (%)': [params['upper_guard'] if params['upper_guard'] > 0 else 'None'],
                'Lower Guard (%)': [params['lower_guard'] if params['lower_guard'] > 0 else 'None'],
                'Failure Threshold (%)': [params['failure_threshold']]
            }
            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=3)
 
            # Write daily data below (with gap)
            start_row = 7
 
        else:
            start_row = 0
 
        # Get daily data for selected start date
        if selected_start_date in daily_data:
            daily_df = daily_data[selected_start_date].copy()
            # Reorder columns: Date first, then Portfolio_NAV, then assets
            cols = ['Date', 'Portfolio_NAV'] + [c for c in daily_df.columns if c not in ['Date', 'Portfolio_NAV']]
            daily_df = daily_df[cols]
            daily_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
 
    return buffer.getvalue()
 
 
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
 
        # Portfolio info sheet
        info = get_portfolio_info(portfolio_name)
        if info:
            info_df = pd.DataFrame([{
                'Portfolio': portfolio_name,
                'Target Return (%)': info['target_return'],
                'Target Risk (%)': info['target_risk'],
                **{f'{k} (%)': v for k, v in info['weights'].items()}
            }])
            info_df.to_excel(writer, sheet_name='Portfolio_Info', index=False)
 
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
 
        # Portfolio info sheet
        portfolio_info_df = create_portfolio_info_table(selected_portfolios)
        portfolio_info_df.to_excel(writer, sheet_name='Portfolio_Info', index=False)
 
        # Individual sheets for each portfolio
        for port_name in selected_portfolios:
            if port_name in results:
                result = results[port_name]
                if not result['monthly_data'].empty:
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
def create_nav_paths_chart(result_data, portfolio_name, max_paths=200):
    """Create line chart showing all NAV simulation paths"""
    nav_paths_success = result_data.get('nav_paths_success', [])
    nav_paths_failure = result_data.get('nav_paths_failure', [])
 
    if not nav_paths_success and not nav_paths_failure:
        return None
 
    fig = go.Figure()
 
    # Sample paths if too many
    if len(nav_paths_failure) > max_paths // 2:
        sample_idx = np.random.choice(len(nav_paths_failure), max_paths // 2, replace=False)
        sampled_failure = [nav_paths_failure[i] for i in sample_idx]
    else:
        sampled_failure = nav_paths_failure
 
    if len(nav_paths_success) > max_paths // 2:
        sample_idx = np.random.choice(len(nav_paths_success), max_paths // 2, replace=False)
        sampled_success = [nav_paths_success[i] for i in sample_idx]
    else:
        sampled_success = nav_paths_success
 
    # Plot failure paths (red)
    for i, path in enumerate(sampled_failure):
        fig.add_trace(go.Scatter(
            x=list(range(len(path))),
            y=path,
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
            showlegend=(i == 0),
            name='Failure',
            hoverinfo='skip'
        ))
 
    # Plot success paths (green)
    for i, path in enumerate(sampled_success):
        fig.add_trace(go.Scatter(
            x=list(range(len(path))),
            y=path,
            mode='lines',
            line=dict(color='rgba(0, 128, 0, 0.2)', width=1),
            showlegend=(i == 0),
            name='Success',
            hoverinfo='skip'
        ))
 
    # Add initial value reference line
    fig.add_hline(y=100, line_dash="dash", line_color="blue",
                  annotation_text="Initial Value (100)")
 
    fig.update_layout(
        title=f'Simulation Paths - {portfolio_name}',
        xaxis_title='Month',
        yaxis_title='Portfolio Value',
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
 
    return fig
 
 
def create_terminal_values_chart(result_data, portfolio_name):
    """Create scatter chart showing terminal values by start date"""
    monthly_data = result_data['monthly_data']
 
    if monthly_data.empty:
        return None
 
    fig = go.Figure()
 
    # Separate success and failure
    success_mask = monthly_data['Success']
 
    # Failure points (red)
    fig.add_trace(go.Scatter(
        x=monthly_data.loc[~success_mask, 'Start Month'],
        y=monthly_data.loc[~success_mask, 'Final Value'],
        mode='markers',
        marker=dict(color='red', size=5, opacity=0.5),
        name='Failure'
    ))
 
    # Success points (green)
    fig.add_trace(go.Scatter(
        x=monthly_data.loc[success_mask, 'Start Month'],
        y=monthly_data.loc[success_mask, 'Final Value'],
        mode='markers',
        marker=dict(color='green', size=5, opacity=0.3),
        name='Success'
    ))
 
    # Add reference lines
    fig.add_hline(y=100, line_dash="dash", line_color="blue",
                  annotation_text="Initial Value (100)")
 
    median_val = monthly_data['Final Value'].median()
    fig.add_hline(y=median_val, line_dash="dot", line_color="orange",
                  annotation_text=f"Median ({median_val:.1f})")
 
    fig.update_layout(
        title=f'Terminal Values by Start Date - {portfolio_name}',
        xaxis_title='Start Date',
        yaxis_title='Final Portfolio Value',
        hovermode='closest',
        showlegend=True
    )
 
    return fig
 
 
def create_success_failure_stacked_chart(result_data, portfolio_name):
    """Create stacked bar chart for success/failure by start month"""
    monthly_data = result_data['monthly_data'].copy()
 
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
 
 
def create_monthly_failure_rate_chart(result_data, portfolio_name):
    """Create line chart showing failure rate by start month"""
    monthly_data = result_data['monthly_data'].copy()
 
    if monthly_data.empty:
        return None
 
    # Group by month
    monthly_data['Year_Month'] = pd.to_datetime(monthly_data['Start Month']).dt.to_period('M')
    monthly_grouped = monthly_data.groupby('Year_Month')['Success'].agg(['sum', 'count']).reset_index()
    monthly_grouped.columns = ['Year_Month', 'Success_Count', 'Total_Count']
    monthly_grouped['Failure_Rate'] = (1 - monthly_grouped['Success_Count'] / monthly_grouped['Total_Count']) * 100
    monthly_grouped['Date'] = monthly_grouped['Year_Month'].dt.to_timestamp()
 
    fig = go.Figure()
 
    fig.add_trace(go.Scatter(
        x=monthly_grouped['Date'],
        y=monthly_grouped['Failure_Rate'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='darkred', width=2),
        fillcolor='rgba(255, 0, 0, 0.3)',
        name='Failure Rate'
    ))
 
    # Add major financial events
    events = [
        ('2008-09-15', 'Lehman Brothers'),
        ('2020-03-15', 'COVID-19'),
    ]
 
    for event_date, event_name in events:
        event_dt = pd.to_datetime(event_date)
        if monthly_grouped['Date'].min() <= event_dt <= monthly_grouped['Date'].max():
            fig.add_vline(x=event_dt, line_dash="dash", line_color="black", opacity=0.5)
            fig.add_annotation(x=event_dt, y=monthly_grouped['Failure_Rate'].max() * 0.9,
                             text=event_name, showarrow=False)
 
    fig.update_layout(
        title=f'Monthly Failure Rate - {portfolio_name}',
        xaxis_title='Start Month',
        yaxis_title='Failure Rate (%)',
        hovermode='x unified'
    )
 
    return fig
 
 
def create_daily_nav_chart(daily_data, selected_start_date, portfolio_name):
    """Create daily NAV chart for selected start date"""
    if selected_start_date not in daily_data:
        return None
 
    df = daily_data[selected_start_date]
 
    fig = go.Figure()
 
    # Add Portfolio NAV
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Portfolio_NAV'],
        mode='lines',
        name='Portfolio NAV',
        line=dict(color='blue', width=2)
    ))
 
    # Add asset indices
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    color_idx = 0
    for col in df.columns:
        if col not in ['Date', 'Portfolio_NAV']:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=colors[color_idx % len(colors)], width=1, dash='dot'),
                opacity=0.7
            ))
            color_idx += 1
 
    fig.add_hline(y=100, line_dash="dash", line_color="gray",
                  annotation_text="Initial Value (100)")
 
    fig.update_layout(
        title=f'Daily NAV - {portfolio_name} (Start: {selected_start_date})',
        xaxis_title='Date',
        yaxis_title='Value (Normalized to 100)',
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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
            width='stretch'
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
        tab1, tab2, tab3 = st.tabs(["📊 개별 포트폴리오 분석", "🔄 포트폴리오 비교", "🎯 Dynamic 전략"])
 
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
 
                # ============================================================
                # Portfolio Information Section
                # ============================================================
                st.markdown("### 📋 포트폴리오 정보")
 
                port_info = get_portfolio_info(selected_portfolio)
                if port_info:
                    col1, col2 = st.columns([1, 2])
 
                    with col1:
                        st.metric("기대 수익률", f"{port_info['target_return']:.2f}%")
                        st.metric("기대 변동성", f"{port_info['target_risk']:.2f}%")
                        # Weights table
                        weights_df = pd.DataFrame([
                            {'Asset': k, 'Weight (%)': f"{v:.2f}"}
                            for k, v in port_info['weights'].items()
                        ])
                        st.dataframe(weights_df, width='stretch', hide_index=True)                        
 
                    with col2:
                        # Weights pie chart
                        fig_weights = create_portfolio_weights_chart(selected_portfolio)
                        if fig_weights:
                            st.plotly_chart(fig_weights, width='stretch')
 

 
                st.divider()
 
                # ============================================================
                # Backtest Results Section
                # ============================================================
                st.markdown("### 📈 백테스팅 결과")
 
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
 
                # ============================================================
                # Visualization Section
                # ============================================================
                st.markdown("### 📉 시각화")
 
                # NAV Paths Chart
                st.markdown("#### Simulation Paths (NAV)")
                fig_paths = create_nav_paths_chart(result, selected_portfolio)
                if fig_paths:
                    st.plotly_chart(fig_paths, width='stretch')
 
                # Terminal Values Chart
                st.markdown("#### Terminal Values by Start Date")
                fig_terminal = create_terminal_values_chart(result, selected_portfolio)
                if fig_terminal:
                    st.plotly_chart(fig_terminal, width='stretch')
 
                # Success/Failure Stacked Chart
                st.markdown("#### Success/Failure by Start Month")
                fig_stacked = create_success_failure_stacked_chart(result, selected_portfolio)
                if fig_stacked:
                    st.plotly_chart(fig_stacked, width='stretch')
 
                # Monthly Failure Rate Chart
                st.markdown("#### Monthly Failure Rate")
                fig_failure_rate = create_monthly_failure_rate_chart(result, selected_portfolio)
                if fig_failure_rate:
                    st.plotly_chart(fig_failure_rate, width='stretch')
 
                st.divider()
 
                # Statistics table
                st.markdown("### 📊 주요 통계량")
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
                    st.dataframe(stats_df, width='stretch')
 
                st.divider()
 
                # ============================================================
                # Daily NAV Excel Download Section
                # ============================================================
                st.markdown("### 📥 일별 NAV 데이터 다운로드")
 
                daily_data = result.get('daily_data', {})
 
                if daily_data:
                    # Get available start dates
                    available_dates = sorted(daily_data.keys())
 
                    # Convert to date objects for date_input
                    available_dates_dt = [pd.Timestamp(d).date() for d in available_dates]
 
                    col1, col2 = st.columns([2, 1])
 
                    with col1:
                        # Date picker for withdrawal start date
                        selected_start_date = st.date_input(
                            "인출 시작일 선택",
                            value=available_dates_dt[0] if available_dates_dt else None,
                            min_value=available_dates_dt[0] if available_dates_dt else None,
                            max_value=available_dates_dt[-1] if available_dates_dt else None,
                            help="일별 NAV 데이터를 다운로드할 인출 시작일을 선택하세요",
                            key="daily_nav_date_select"
                        )
 
                    # Find matching timestamp in daily_data
                    selected_ts = None
                    for ts in available_dates:
                        if pd.Timestamp(ts).date() == selected_start_date:
                            selected_ts = ts
                            break
 
                    # If exact date not found, find closest
                    if selected_ts is None and available_dates:
                        selected_ts = min(available_dates, key=lambda x: abs(pd.Timestamp(x).date() - selected_start_date))
                        st.info(f"선택한 날짜에 데이터가 없어 가장 가까운 날짜({pd.Timestamp(selected_ts).date()})를 사용합니다.")
 
                    if selected_ts:
                        # Show daily NAV chart for selected date
                        st.markdown("#### Daily NAV Chart")
                        fig_daily = create_daily_nav_chart(daily_data, selected_ts, selected_portfolio)
                        if fig_daily:
                            st.plotly_chart(fig_daily, width='stretch')
 
                        # Excel download button
                        with col2:
                            excel_data = create_daily_nav_excel(
                                selected_portfolio,
                                selected_ts,
                                daily_data,
                                params
                            )
                            st.download_button(
                                label="📥 일별 NAV 엑셀 다운로드",
                                data=excel_data,
                                file_name=f"{selected_portfolio}_daily_nav_{pd.Timestamp(selected_ts).strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                else:
                    st.warning("일별 데이터가 없습니다.")
 
                st.divider()
 
                # Summary Excel download button
                st.markdown("### 📥 전체 결과 다운로드")
                excel_data = create_single_portfolio_excel(selected_portfolio, result, params)
                st.download_button(
                    label="📥 전체 결과 엑셀 다운로드",
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
                # ============================================================
                # Portfolio Information Comparison
                # ============================================================
                st.markdown("### 📋 포트폴리오 정보 비교")
 
                portfolio_info_df = create_portfolio_info_table(selected_portfolios)
                st.dataframe(portfolio_info_df, width='stretch', hide_index=True)
 
                st.divider()
 
                # ============================================================
                # Backtest Results Comparison
                # ============================================================
                st.markdown("### 📈 백테스팅 결과 비교")
 
                # Comparison bar chart
                fig = create_comparison_bar_chart(results, selected_portfolios)
                st.plotly_chart(fig, width='stretch')
 
                st.divider()
 
                # Comparison table
                st.markdown("#### 포트폴리오별 주요 지표 비교")
                comparison_df = create_comparison_table(results, selected_portfolios)
                st.dataframe(comparison_df, width='stretch', hide_index=True)
 
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

        # ====================================================================
        # Tab 3: Dynamic Withdrawal Strategies
        # ====================================================================
        with tab3:
            render_dynamic_strategies_tab(data)


if __name__ == "__main__":
    main()
