"""
Metrics Calculator for Dynamic Withdrawal Analysis
===================================================
Computes optimization metrics and behavioral characteristics for withdrawal strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    최적화 및 비교를 위한 메트릭 계산
    """

    @staticmethod
    def calculate_yoy_volatility(withdrawal_path: np.ndarray) -> float:
        """
        Year-over-Year (YoY) 변동성 계산

        Parameters
        ----------
        withdrawal_path : np.ndarray
            120개월 인출액 배열 [W_0, W_1, ..., W_119]

        Returns
        -------
        float : YoY 변동성 (표준편차)
            - 연간 인출액을 계산하고
            - 전년 대비 변화율을 계산
            - 그 변화율들의 표준편차를 반환
        """
        if len(withdrawal_path) < 24:  # 최소 2년 필요
            return np.nan

        # 연간 인출액 계산 (12개월씩 묶어서 합산)
        annual_withdrawals = []
        for i in range(0, len(withdrawal_path), 12):
            if i + 12 <= len(withdrawal_path):
                annual_sum = np.sum(withdrawal_path[i:i+12])
                annual_withdrawals.append(annual_sum)

        if len(annual_withdrawals) < 2:
            return np.nan

        # YoY 변화율 계산
        yoy_changes = []
        for i in range(len(annual_withdrawals) - 1):
            if annual_withdrawals[i] > 0:
                change = (annual_withdrawals[i+1] - annual_withdrawals[i]) / annual_withdrawals[i]
                yoy_changes.append(change)

        if len(yoy_changes) == 0:
            return np.nan

        # 표준편차 (변동성)
        return np.std(yoy_changes)

    @staticmethod
    def calculate_optimization_metrics(results_df: pd.DataFrame,
                                       v0: float = 100.0) -> Dict:
        """
        모든 롤링 윈도우 결과 집계 및 메트릭 계산

        Parameters
        ----------
        results_df : pd.DataFrame
            각 rolling window의 결과
            Columns: ['start_date', 'withdrawal_path', 'total_withdrawal',
                     'terminal_nav', 'nav_path', 'is_fail']
        v0 : float
            초기 포트폴리오 가치

        Returns
        -------
        dict : 집계 메트릭
            - total_withdrawal_*: 총 인출액 관련
            - yoy_volatility_*: YoY 변동성 관련
            - failure_rate: 실패율
            - terminal_nav_*: 최종 NAV 관련
        """
        if len(results_df) == 0:
            return {
                'total_withdrawal_mean': 0,
                'total_withdrawal_worst': 0,
                'total_withdrawal_best': 0,
                'yoy_volatility_mean': np.nan,
                'yoy_volatility_90pct': np.nan,
                'yoy_volatility_max': np.nan,
                'failure_rate': 1.0,
                'terminal_nav_median': 0,
                'terminal_nav_10pct': 0,
                'terminal_nav_mean': 0,
                'total_paths': 0
            }

        # YoY Volatility 계산
        volatilities = []
        for withdrawal_path in results_df['withdrawal_path']:
            vol = MetricsCalculator.calculate_yoy_volatility(withdrawal_path)
            if not np.isnan(vol):
                volatilities.append(vol)

        volatilities = np.array(volatilities) if volatilities else np.array([np.nan])

        return {
            # Objective: 총 인출액
            'total_withdrawal_mean': results_df['total_withdrawal'].mean(),
            'total_withdrawal_worst': results_df['total_withdrawal'].min(),
            'total_withdrawal_best': results_df['total_withdrawal'].max(),

            # Constraint 1: YoY Volatility
            'yoy_volatility_mean': np.nanmean(volatilities),
            'yoy_volatility_90pct': np.nanpercentile(volatilities, 90),
            'yoy_volatility_max': np.nanmax(volatilities),

            # Constraint 2: Failure Rate
            'failure_rate': results_df['is_fail'].mean(),

            # Constraint 3: Terminal NAV
            'terminal_nav_median': results_df['terminal_nav'].median(),
            'terminal_nav_10pct': results_df['terminal_nav'].quantile(0.1),
            'terminal_nav_mean': results_df['terminal_nav'].mean(),

            # Meta
            'total_paths': len(results_df)
        }

    @staticmethod
    def calculate_behavioral_metrics(withdrawal_path: np.ndarray,
                                     nav_path: np.ndarray,
                                     initial_withdrawal: float) -> Dict:
        """
        Dynamic 전략의 행동 특성 분석

        Parameters
        ----------
        withdrawal_path : np.ndarray
            120개월 인출액 배열
        nav_path : np.ndarray
            120개월 NAV 배열
        initial_withdrawal : float
            초기 연간 인출액

        Returns
        -------
        dict : 행동 메트릭
            - adjustment_count: 조정 발생 횟수 (5% 이상 변화)
            - adjustment_frequency: 조정 발생 비율
            - cut_count: 감액 발생 횟수
            - avg_cut_size: 평균 감액 크기
            - max_cut_size: 최대 감액 크기
            - increase_count: 증액 발생 횟수
            - avg_increase_size: 평균 증액 크기
            - max_increase_size: 최대 증액 크기
        """
        # 연간 인출액
        annual_withdrawals = []
        for i in range(0, len(withdrawal_path), 12):
            if i + 12 <= len(withdrawal_path):
                annual_sum = np.sum(withdrawal_path[i:i+12])
                annual_withdrawals.append(annual_sum)

        if len(annual_withdrawals) < 2:
            return {
                'adjustment_count': 0,
                'adjustment_frequency': 0,
                'cut_count': 0,
                'avg_cut_size': 0,
                'max_cut_size': 0,
                'increase_count': 0,
                'avg_increase_size': 0,
                'max_increase_size': 0,
            }

        # 조정 발생 여부 (전년 대비 5% 이상 변화)
        adjustments = []
        cuts = []
        increases = []

        for i in range(len(annual_withdrawals) - 1):
            if annual_withdrawals[i] > 0:
                change_pct = abs(annual_withdrawals[i+1] - annual_withdrawals[i]) / annual_withdrawals[i]

                if change_pct > 0.05:  # 5% 이상 변화
                    adjustments.append(True)
                else:
                    adjustments.append(False)

                # 감액 vs 증액
                if annual_withdrawals[i+1] < annual_withdrawals[i]:
                    cut_pct = (annual_withdrawals[i] - annual_withdrawals[i+1]) / annual_withdrawals[i]
                    cuts.append(cut_pct)
                elif annual_withdrawals[i+1] > annual_withdrawals[i]:
                    increase_pct = (annual_withdrawals[i+1] - annual_withdrawals[i]) / annual_withdrawals[i]
                    increases.append(increase_pct)

        return {
            'adjustment_count': sum(adjustments),
            'adjustment_frequency': sum(adjustments) / len(adjustments),

            'cut_count': len(cuts),
            'avg_cut_size': np.mean(cuts) if cuts else 0,
            'max_cut_size': max(cuts) if cuts else 0,

            'increase_count': len(increases),
            'avg_increase_size': np.mean(increases) if increases else 0,
            'max_increase_size': max(increases) if increases else 0,
        }

    @staticmethod
    def aggregate_behavioral_metrics(results_df: pd.DataFrame) -> Dict:
        """
        모든 롤링 윈도우의 행동 메트릭 집계

        Parameters
        ----------
        results_df : pd.DataFrame
            각 rolling window의 결과

        Returns
        -------
        dict : 집계된 행동 메트릭
        """
        all_behavioral = []

        for idx, row in results_df.iterrows():
            nav_path = row.get('nav_path', np.array([]))
            withdrawal_path = row.get('withdrawal_path', np.array([]))
            initial_wr = withdrawal_path[0] * 12 if len(withdrawal_path) > 0 else 0

            behavioral = MetricsCalculator.calculate_behavioral_metrics(
                withdrawal_path,
                nav_path,
                initial_wr
            )
            all_behavioral.append(behavioral)

        if not all_behavioral:
            return {
                'adjustment_count_mean': 0,
                'adjustment_frequency_mean': 0,
                'cut_count_mean': 0,
                'avg_cut_size_mean': 0,
                'max_cut_size_mean': 0,
                'increase_count_mean': 0,
                'avg_increase_size_mean': 0,
                'max_increase_size_mean': 0,
            }

        # 각 메트릭의 평균 계산
        behavioral_df = pd.DataFrame(all_behavioral)

        return {
            'adjustment_count_mean': behavioral_df['adjustment_count'].mean(),
            'adjustment_frequency_mean': behavioral_df['adjustment_frequency'].mean(),
            'cut_count_mean': behavioral_df['cut_count'].mean(),
            'avg_cut_size_mean': behavioral_df['avg_cut_size'].mean(),
            'max_cut_size_mean': behavioral_df['max_cut_size'].mean(),
            'increase_count_mean': behavioral_df['increase_count'].mean(),
            'avg_increase_size_mean': behavioral_df['avg_increase_size'].mean(),
            'max_increase_size_mean': behavioral_df['max_increase_size'].mean(),
        }

    @staticmethod
    def evaluate_feasibility(metrics: Dict,
                            constraints: Dict) -> bool:
        """
        제약 조건 만족 여부 판단

        Parameters
        ----------
        metrics : dict
            계산된 메트릭
        constraints : dict
            제약 조건
            - max_yoy_volatility: 최대 YoY 변동성
            - max_failure_rate: 최대 실패율
            - min_terminal_nav: 최소 최종 NAV

        Returns
        -------
        bool : 모든 제약을 만족하면 True
        """
        return (
            metrics['yoy_volatility_mean'] <= constraints.get('max_yoy_volatility', 0.20) and
            metrics['failure_rate'] <= constraints.get('max_failure_rate', 0.05) and
            metrics['terminal_nav_median'] >= constraints.get('min_terminal_nav', 0.50)
        )

    @staticmethod
    def create_summary_dataframe(results_dict: Dict) -> pd.DataFrame:
        """
        최적화 결과를 정렬된 요약 테이블로 변환

        Parameters
        ----------
        results_dict : dict
            {(portfolio, strategy, wr): metrics, ...}

        Returns
        -------
        pd.DataFrame : 정렬된 결과 테이블
        """
        rows = []

        for (portfolio, strategy, wr), metrics in results_dict.items():
            rows.append({
                'Portfolio': portfolio,
                'Strategy': strategy,
                'WR': wr,
                'Feasible': metrics.get('is_feasible', False),
                'Total_Withdrawal': metrics.get('total_withdrawal_mean', 0),
                'YoY_Volatility': metrics.get('yoy_volatility_mean', np.nan),
                'Failure_Rate': metrics.get('failure_rate', 1.0),
                'Terminal_NAV': metrics.get('terminal_nav_median', 0),
            })

        df = pd.DataFrame(rows)

        # 정렬: 포트폴리오 > 전략 > 인출률
        strategy_order = {'fixed': 0, 'guardrails': 1, 'guyton_klinger': 2}
        df['strategy_order'] = df['Strategy'].map(strategy_order)

        df = df.sort_values(
            by=['Portfolio', 'strategy_order', 'WR'],
            ascending=[True, True, True]
        )

        df = df.drop('strategy_order', axis=1)

        return df

    @staticmethod
    def find_optimal_withdrawals(results_dict: Dict,
                                 constraints: Dict) -> Dict:
        """
        각 (포트폴리오, 전략) 조합의 최적 인출률 찾기

        Parameters
        ----------
        results_dict : dict
            {(portfolio, strategy, wr): metrics, ...}
        constraints : dict
            제약 조건

        Returns
        -------
        dict : {(portfolio, strategy): (optimal_wr, metrics), ...}
        """
        optimal = {}

        # 포트폴리오 × 전략 조합별로 그룹화
        grouped = {}
        for (portfolio, strategy, wr), metrics in results_dict.items():
            key = (portfolio, strategy)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((wr, metrics))

        # 각 그룹에서 최적 인출률 찾기
        for (portfolio, strategy), candidates in grouped.items():
            # 제약을 만족하는 후보만 필터링
            feasible = [
                (wr, metrics) for wr, metrics in candidates
                if MetricsCalculator.evaluate_feasibility(metrics, constraints)
            ]

            if feasible:
                # 총 인출액 최대화
                best_wr, best_metrics = max(
                    feasible,
                    key=lambda x: x[1].get('total_withdrawal_mean', 0)
                )
                optimal[(portfolio, strategy)] = (best_wr, best_metrics)

        return optimal
