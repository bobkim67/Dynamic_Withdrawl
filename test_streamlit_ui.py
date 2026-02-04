"""
Simple test script to debug Streamlit UI issues
Run this without Streamlit to identify problems
"""

import pickle
import pandas as pd
import numpy as np
from withdrawal_backtest import DataPreprocessor, PORTFOLIOS
from dynamic_simulator import DynamicWithdrawalSimulator
from optimizer import WithdrawalOptimizer
from metrics_calculator import MetricsCalculator

def test_basic_flow():
    """Test the complete flow without Streamlit"""

    print("\n" + "="*70)
    print("TESTING DYNAMIC WITHDRAWAL STRATEGY FLOW")
    print("="*70)

    # Step 1: Load data
    print("\n[Step 1] Loading benchmark data...")
    try:
        with open('benchmark_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Data loaded: {data.shape}")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return

    # Step 2: Preprocess data
    print("\n[Step 2] Preprocessing data...")
    try:
        preprocessor = DataPreprocessor(data, add_portfolios=True)
        returns_df, month_starts = preprocessor.get_data()
        print(f"✓ Returns shape: {returns_df.shape}")
        print(f"✓ Month starts: {np.sum(month_starts)}")
    except Exception as e:
        print(f"✗ Failed to preprocess: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Initialize optimizer
    print("\n[Step 3] Initializing optimizer...")
    try:
        optimizer = WithdrawalOptimizer(returns_df, month_starts)
        print("✓ Optimizer initialized")
    except Exception as e:
        print(f"✗ Failed to initialize optimizer: {e}")
        return

    # Step 4: Quick optimization test
    print("\n[Step 4] Running quick optimization test...")
    print("    Portfolio: Port_5.0%")
    print("    Withdrawal rates: 4%, 5%, 6%")

    try:
        constraints = {
            'max_yoy_volatility': 0.20,
            'max_failure_rate': 0.10,
            'min_terminal_nav': 0.40
        }

        withdrawal_rates = np.array([0.04, 0.05, 0.06])

        portfolio_results = optimizer.optimize_single_portfolio(
            portfolio_name='Port_5.0%',
            withdrawal_rates=withdrawal_rates,
            horizon_years=10,
            constraints=constraints,
            v0=100.0,
            verbose=False
        )

        print("✓ Optimization completed")
        print(f"\n  Results structure:")
        for wr in withdrawal_rates:
            print(f"    WR {wr:.1%}:")
            for strategy, metrics in portfolio_results[wr].items():
                feasible = "✓" if metrics.get('is_feasible', False) else "✗"
                total_w = metrics.get('total_withdrawal_mean', 0)
                print(f"      {strategy:20s} {feasible} TotalW={total_w:,.2f}")

    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Multi-portfolio optimization
    print("\n[Step 5] Running multi-portfolio optimization...")
    print("    Portfolios: Port_4.0%, Port_5.0%")
    print("    Withdrawal rates: 4%, 5%, 6%")

    try:
        test_portfolios = ['Port_4.0%', 'Port_5.0%']
        test_wrs = np.array([0.04, 0.05, 0.06])

        results_df = optimizer.optimize_all_portfolios(
            portfolio_names=test_portfolios,
            withdrawal_rates=test_wrs,
            horizon_years=10,
            constraints=constraints,
            v0=100.0
        )

        print(f"✓ Multi-portfolio optimization completed")
        print(f"  Results shape: {results_df.shape}")
        print(f"  Feasible solutions: {len(results_df[results_df['Feasible']==True])}")

        # Show summary
        print("\n  Optimal solutions by portfolio:")
        for portfolio in results_df['Portfolio'].unique():
            df_port = results_df[results_df['Portfolio'] == portfolio]
            for strategy in ['fixed', 'guardrails', 'guyton_klinger']:
                df_strat = df_port[df_port['Strategy'] == strategy]
                df_feasible = df_strat[df_strat['Feasible'] == True]
                if not df_feasible.empty:
                    best = df_feasible.loc[df_feasible['Total_Withdrawal'].idxmax()]
                    print(f"    {portfolio:12s} {strategy:20s}: WR={best['WR']:.2%}, Vol={best['YoY_Volatility']:.2%}")

    except Exception as e:
        print(f"✗ Multi-portfolio optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 6: Test UI components
    print("\n[Step 6] Testing UI components...")
    try:
        from dynamic_strategy_ui import (
            create_pareto_frontier_chart,
            create_strategy_comparison_chart
        )

        # Test Pareto frontier
        fig = create_pareto_frontier_chart(results_df, 'Port_5.0%')
        print(f"✓ Pareto frontier chart created: {type(fig)}")

        # Test strategy comparison
        df_port = results_df[results_df['Portfolio'] == 'Port_5.0%']
        fig2 = create_strategy_comparison_chart(df_port, 'WR', 'Total_Withdrawal')
        print(f"✓ Strategy comparison chart created: {type(fig2)}")

    except Exception as e:
        print(f"✗ UI component test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nThe Dynamic Withdrawal Strategy system is working correctly.")
    print("If you see errors in Streamlit, it's likely a UI/session state issue.")


if __name__ == "__main__":
    test_basic_flow()
