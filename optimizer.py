"""
Withdrawal Strategy Optimizer
=============================
Grid search optimization for Fixed Rate, Guardrails, and Guyton-Klinger strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from withdrawal_backtest import WithdrawalSimulator, PORTFOLIOS
from dynamic_simulator import DynamicWithdrawalSimulator
from metrics_calculator import MetricsCalculator


class WithdrawalOptimizer:
    """
    Fixed Rate, Guardrails, Guyton-Klinger 전략 최적화
    """

    def __init__(self,
                 returns_df: pd.DataFrame,
                 month_starts: pd.Series,
                 portfolios: Dict = None):
        """
        Parameters
        ----------
        returns_df : pd.DataFrame
            일별 수익률 (벤치마크/포트폴리오 × 날짜)
        month_starts : pd.Series
            월초 거래일 여부 (boolean)
        portfolios : dict
            포트폴리오 정의 (None이면 PORTFOLIOS 사용)
        """
        self.returns_df = returns_df
        self.month_starts = month_starts
        self.portfolios = portfolios if portfolios is not None else PORTFOLIOS
        self.fixed_simulator = WithdrawalSimulator(returns_df, month_starts)
        self.dynamic_simulator = DynamicWithdrawalSimulator(returns_df, month_starts)
        self.metrics_calc = MetricsCalculator()

    def optimize_single_portfolio(self,
                                 portfolio_name: str,
                                 withdrawal_rates: np.ndarray,
                                 horizon_years: int = 10,
                                 constraints: Dict = None,
                                 v0: float = 100.0,
                                 verbose: bool = True) -> Dict:
        """
        단일 포트폴리오에 대해 3가지 전략 최적화

        Parameters
        ----------
        portfolio_name : str
            포트폴리오 이름
        withdrawal_rates : np.ndarray
            테스트할 인출률 배열
        horizon_years : int
            투자 기간 (년)
        constraints : dict
            제약 조건
            {
                'max_yoy_volatility': 0.15,
                'max_failure_rate': 0.05,
                'min_terminal_nav': 0.50
            }
        v0 : float
            초기 포트폴리오 가치
        verbose : bool
            진행 상황 출력 여부

        Returns
        -------
        dict : {wr: {strategy: metrics, ...}, ...}
        """
        if constraints is None:
            constraints = {
                'max_yoy_volatility': 0.15,
                'max_failure_rate': 0.05,
                'min_terminal_nav': 0.50
            }

        results = {}

        if verbose:
            print(f"\n{'='*70}")
            print(f"포트폴리오: {portfolio_name}")
            print(f"{'='*70}")

        # 각 인출률 테스트
        n_wrs = len(withdrawal_rates)
        iterator = enumerate(withdrawal_rates)
        if TQDM_AVAILABLE and verbose:
            iterator = tqdm(enumerate(withdrawal_rates), total=n_wrs,
                          desc=f"{portfolio_name}", leave=True)

        for wr_idx, wr in iterator:
            wr_results = {}

            # ===== Fixed Rate =====
            try:
                fixed_vt = self.fixed_simulator.run_rolling_backtest(
                    portfolio_name,
                    horizon_years,
                    wr,
                    v0,
                    verbose=False
                )

                # 인출액 계산 (Fixed Rate는 withdrawal_backtest에 없으므로 계산)
                monthly_withdrawal = wr * v0 / 12
                annual_withdrawals = [monthly_withdrawal * 12]
                for year in range(1, 10):
                    annual_withdrawals.append(annual_withdrawals[0] * (1.02 ** year))

                # 모든 rolling window에 대해 동일한 인출 패턴 생성
                withdrawal_paths = []
                total_withdrawals = []
                for _ in range(len(fixed_vt)):
                    w_path = np.array([monthly_withdrawal * (1.02 ** (i//12)) for i in range(120)])
                    withdrawal_paths.append(w_path)
                    total_withdrawals.append(np.sum(w_path))

                fixed_results_df = pd.DataFrame({
                    'terminal_nav': fixed_vt,
                    'is_fail': fixed_vt < v0,
                    'withdrawal_path': withdrawal_paths,
                    'total_withdrawal': total_withdrawals,  # 필수 컬럼 추가
                    'nav_path': [np.full(121, v0) for _ in range(len(fixed_vt))]  # 근사값
                })

                fixed_metrics = self.metrics_calc.calculate_optimization_metrics(
                    fixed_results_df, v0
                )
                fixed_metrics['is_feasible'] = self.metrics_calc.evaluate_feasibility(
                    fixed_metrics, constraints
                )
                wr_results['fixed'] = fixed_metrics

            except Exception as e:
                if verbose:
                    print(f"    ⚠️  Fixed Rate 오류: {e}")
                wr_results['fixed'] = {'is_feasible': False, 'failure_rate': 1.0}

            # ===== Guardrails =====
            try:
                guardrails_results_df = self.dynamic_simulator.run_guardrails_backtest(
                    portfolio_name,
                    horizon_years,
                    wr,
                    guardrail_width=0.20,
                    inflation_rate=0.02,
                    v0=v0,
                    verbose=False
                )

                guardrails_metrics = self.metrics_calc.calculate_optimization_metrics(
                    guardrails_results_df, v0
                )
                guardrails_metrics['is_feasible'] = self.metrics_calc.evaluate_feasibility(
                    guardrails_metrics, constraints
                )
                wr_results['guardrails'] = guardrails_metrics

            except Exception as e:
                if verbose:
                    print(f"    ⚠️  Guardrails 오류: {e}")
                wr_results['guardrails'] = {'is_feasible': False, 'failure_rate': 1.0}

            # ===== Guyton-Klinger =====
            try:
                gk_results_df = self.dynamic_simulator.run_guyton_klinger_backtest(
                    portfolio_name,
                    horizon_years,
                    wr,
                    guardrail_width=0.20,
                    adjustment_pct=0.10,
                    freeze_threshold=-0.10,
                    inflation_rate=0.02,
                    v0=v0,
                    verbose=False
                )

                gk_metrics = self.metrics_calc.calculate_optimization_metrics(
                    gk_results_df, v0
                )
                gk_metrics['is_feasible'] = self.metrics_calc.evaluate_feasibility(
                    gk_metrics, constraints
                )
                wr_results['guyton_klinger'] = gk_metrics

            except Exception as e:
                if verbose:
                    print(f"    ⚠️  Guyton-Klinger 오류: {e}")
                wr_results['guyton_klinger'] = {'is_feasible': False, 'failure_rate': 1.0}

            results[wr] = wr_results

        return results

    def optimize_all_portfolios(self,
                               portfolio_names: List[str],
                               withdrawal_rates: np.ndarray,
                               horizon_years: int = 10,
                               constraints: Dict = None,
                               v0: float = 100.0) -> pd.DataFrame:
        """
        모든 포트폴리오 최적화

        Parameters
        ----------
        portfolio_names : List[str]
            포트폴리오 이름 리스트
        withdrawal_rates : np.ndarray
            테스트할 인출률 배열
        horizon_years : int
            투자 기간
        constraints : dict
            제약 조건
        v0 : float
            초기 포트폴리오 가치

        Returns
        -------
        pd.DataFrame : 모든 결과 요약
            Columns: Portfolio, Strategy, WR, Feasible, Total_Withdrawal,
                    YoY_Volatility, Failure_Rate, Terminal_NAV
        """
        all_results = {}

        for portfolio_name in portfolio_names:
            portfolio_results = self.optimize_single_portfolio(
                portfolio_name,
                withdrawal_rates,
                horizon_years,
                constraints,
                v0,
                verbose=True
            )

            for wr, strategy_results in portfolio_results.items():
                for strategy, metrics in strategy_results.items():
                    key = (portfolio_name, strategy, wr)
                    all_results[key] = metrics

        # DataFrame으로 변환
        return self.metrics_calc.create_summary_dataframe(all_results)

    @staticmethod
    def print_optimal_summary(results_df: pd.DataFrame,
                             constraints: Dict = None):
        """
        최적 인출률 요약 출력

        Parameters
        ----------
        results_df : pd.DataFrame
            최적화 결과 데이터프레임
        constraints : dict
            제약 조건 (출력용)
        """
        print(f"\n{'='*100}")
        print(f"{'최적 인출률 요약':^100}")
        print(f"{'='*100}")

        if constraints:
            print(f"\n제약 조건:")
            print(f"  • 최대 YoY 변동성: {constraints.get('max_yoy_volatility', 0.15)*100:.1f}%")
            print(f"  • 최대 실패율: {constraints.get('max_failure_rate', 0.05)*100:.1f}%")
            print(f"  • 최소 최종 NAV: {constraints.get('min_terminal_nav', 0.50)*100:.1f}%")

        print(f"\n{'포트폴리오':<15} {'전략':<20} {'인출률':<10} {'총 인출액':<15} "
              f"{'YoY 변동성':<15} {'실패율':<10} {'최종 NAV':<10}")
        print(f"{'-'*100}")

        for portfolio in results_df['Portfolio'].unique():
            df_port = results_df[results_df['Portfolio'] == portfolio]

            for strategy in ['fixed', 'guardrails', 'guyton_klinger']:
                df_strat = df_port[df_port['Strategy'] == strategy]
                df_feasible = df_strat[df_strat['Feasible'] == True]

                if not df_feasible.empty:
                    # 총 인출액 최대
                    best = df_feasible.loc[df_feasible['Total_Withdrawal'].idxmax()]

                    print(f"{portfolio:<15} {strategy:<20} {best['WR']:.1%}"
                          f"{'':<5} {best['Total_Withdrawal']:<15,.0f} "
                          f"{best['YoY_Volatility']:<15.2%} {best['Failure_Rate']:<10.2%} "
                          f"{best['Terminal_NAV']:<10.2%}")
                else:
                    print(f"{portfolio:<15} {strategy:<20} {'N/A':<10} "
                          f"{'No feasible solution':<15}")

        print(f"{'='*100}\n")

    @staticmethod
    def export_results(results_df: pd.DataFrame,
                      filepath: str):
        """
        결과를 CSV로 저장

        Parameters
        ----------
        results_df : pd.DataFrame
            최적화 결과
        filepath : str
            저장 경로
        """
        # 퍼센트와 숫자 포맷팅
        def format_value(val):
            if pd.isna(val):
                return "N/A"
            if isinstance(val, bool):
                return "Yes" if val else "No"
            return val

        export_df = results_df.copy()

        # 숫자 포맷팅
        for col in ['WR', 'YoY_Volatility', 'Failure_Rate', 'Terminal_NAV']:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(
                    lambda x: f"{x:.2%}" if isinstance(x, (int, float)) and x is not None else "N/A"
                )

        for col in ['Total_Withdrawal']:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(
                    lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and x is not None else "N/A"
                )

        export_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n✅ 결과를 저장했습니다: {filepath}")


# ============================================================================
# 예제 및 테스트 용도
# ============================================================================

if __name__ == "__main__":
    """
    사용 예제:

    from withdrawal_backtest import DataPreprocessor
    import pickle

    # 데이터 로드
    with open('benchmark_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # 전처리
    preprocessor = DataPreprocessor(data, add_portfolios=True)
    returns, month_starts = preprocessor.get_data()

    # 최적화
    optimizer = WithdrawalOptimizer(returns, month_starts)
    withdrawal_rates = np.arange(0.03, 0.10, 0.005)

    constraints = {
        'max_yoy_volatility': 0.15,
        'max_failure_rate': 0.05,
        'min_terminal_nav': 0.50
    }

    results = optimizer.optimize_all_portfolios(
        list(PORTFOLIOS.keys()),
        withdrawal_rates,
        horizon_years=10,
        constraints=constraints
    )

    # 결과 출력 및 저장
    WithdrawalOptimizer.print_optimal_summary(results, constraints)
    WithdrawalOptimizer.export_results(results, 'optimization_results.csv')
    """
    pass
