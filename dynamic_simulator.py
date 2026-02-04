"""
Dynamic Withdrawal Strategies - Guardrails and Guyton-Klinger
==============================================================
Implements two dynamic withdrawal strategies that adjust based on portfolio performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# tqdm 선택적 import
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class GuardrailsWithdrawal:
    """
    Guardrails 전략: NAV 기준 즉각 반응 방식

    핵심 원리:
    - 인출률이 상한/하한을 벗어나면 NAV × Guardrail rate로 즉시 재계산
    - 한 번의 조정으로 정상 범위 복귀
    - 시장 변화에 빠르게 반응
    """

    def __init__(self,
                 initial_wr: float,
                 guardrail_width: float = 0.20,
                 inflation_rate: float = 0.02):
        """
        Parameters
        ----------
        initial_wr : float
            초기 연간 인출률 (예: 0.05 = 5%)
        guardrail_width : float
            Guardrail 폭 (예: 0.20 = ±20%)
            상한 = initial_wr × (1 + width)
            하한 = initial_wr × (1 - width)
        inflation_rate : float
            연간 인플레이션율 (기본값 2%)
        """
        self.initial_wr = initial_wr
        self.upper_guardrail = initial_wr * (1 + guardrail_width)
        self.lower_guardrail = initial_wr * (1 - guardrail_width)
        self.inflation_rate = inflation_rate
        self.current_wr = initial_wr  # 추적용

    def calculate_withdrawal(self,
                           current_nav: float,
                           previous_withdrawal: float,
                           month_idx: int) -> float:
        """
        매월 인출액 계산

        Parameters
        ----------
        current_nav : float
            현재 포트폴리오 가치
        previous_withdrawal : float
            직전 월 인출액
        month_idx : int
            현재 월 인덱스 (0-119)

        Returns
        -------
        float : 이번 달 인출액
        """
        # 첫 달
        if month_idx == 0:
            self.current_wr = self.initial_wr
            return current_nav * self.initial_wr / 12

        # 연간 인출 시점에만 조정 체크 (매년 초)
        if month_idx % 12 == 0:
            # 작년 연간 인출액
            annual_withdrawal_last_year = previous_withdrawal * 12

            # 현재 인출률 = 작년 연간 인출액 / 현재 NAV
            current_wr = annual_withdrawal_last_year / current_nav if current_nav > 0 else 0

            if current_wr > self.upper_guardrail:
                # 상한 위반: NAV 기준 재계산
                new_annual_withdrawal = current_nav * self.upper_guardrail
                self.current_wr = self.upper_guardrail
                return new_annual_withdrawal / 12

            elif current_wr < self.lower_guardrail:
                # 하한 위반: NAV 기준 재계산
                new_annual_withdrawal = current_nav * self.lower_guardrail
                self.current_wr = self.lower_guardrail
                return new_annual_withdrawal / 12

            else:
                # 정상 범위: 인플레이션 조정
                inflation_adjustment = (1 + self.inflation_rate) ** (month_idx / 12)
                self.current_wr = self.initial_wr * inflation_adjustment
                return (current_nav * self.initial_wr / 12) * inflation_adjustment

        else:
            # 월 중에는 이전 인출액 유지
            return previous_withdrawal


class GuytonKlingerWithdrawal:
    """
    Guyton-Klinger 전략: 고정 비율 단계적 조정 + 추가 규칙

    핵심 원리:
    - Guardrail 위반 시 이전 인출액 × (1 ± adjustment_pct) 고정 비율 조정
    - 여러 번에 걸쳐 단계적 조정 가능
    - 포트폴리오 수익률 < -10% 시 동결 (인플레이션 조정 안함)

    규칙:
    1. Guardrail Rule: 인출률이 상한/하한 벗어나면 고정 비율로 조정
    2. Portfolio Management Rule: 작년 수익률 < -10%이면 동결
    3. Normal: 정상 범위 & 정상 수익 → 인플레이션 조정
    """

    def __init__(self,
                 initial_wr: float,
                 guardrail_width: float = 0.20,
                 adjustment_pct: float = 0.10,
                 freeze_threshold: float = -0.10,
                 inflation_rate: float = 0.02):
        """
        Parameters
        ----------
        initial_wr : float
            초기 연간 인출률 (예: 0.05 = 5%)
        guardrail_width : float
            Guardrail 폭 (예: 0.20 = ±20%)
        adjustment_pct : float
            조정 크기 (예: 0.10 = ±10%)
        freeze_threshold : float
            동결 기준 (예: -0.10 = -10%)
            작년 포트폴리오 수익률이 이 값 아래면 동결
        inflation_rate : float
            연간 인플레이션율 (기본값 2%)
        """
        self.initial_wr = initial_wr
        self.upper_guardrail = initial_wr * (1 + guardrail_width)
        self.lower_guardrail = initial_wr * (1 - guardrail_width)
        self.adjustment_pct = adjustment_pct
        self.freeze_threshold = freeze_threshold
        self.inflation_rate = inflation_rate
        self.previous_nav = None
        self.current_wr = initial_wr  # 추적용

    def calculate_withdrawal(self,
                           current_nav: float,
                           previous_withdrawal: float,
                           month_idx: int,
                           portfolio_return: float = 0.0) -> float:
        """
        매월 인출액 계산

        Parameters
        ----------
        current_nav : float
            현재 포트폴리오 가치
        previous_withdrawal : float
            직전 월 인출액
        month_idx : int
            현재 월 인덱스 (0-119)
        portfolio_return : float
            지난 12개월 포트폴리오 수익률 (month_idx % 12 == 0일 때 사용)

        Returns
        -------
        float : 이번 달 인출액
        """
        # 첫 달
        if month_idx == 0:
            self.previous_nav = current_nav
            self.current_wr = self.initial_wr
            return current_nav * self.initial_wr / 12

        # 연간 인출 시점에만 조정 체크 (매년 초)
        if month_idx % 12 == 0:
            annual_withdrawal_last_year = previous_withdrawal * 12
            current_wr = annual_withdrawal_last_year / current_nav if current_nav > 0 else 0

            # Rule 1: Guardrail 위반 체크
            if current_wr > self.upper_guardrail:
                # 고정 비율로 감액
                new_annual_withdrawal = annual_withdrawal_last_year * (1 - self.adjustment_pct)
                self.current_wr = new_annual_withdrawal / current_nav if current_nav > 0 else 0
                self.previous_nav = current_nav
                return new_annual_withdrawal / 12

            elif current_wr < self.lower_guardrail:
                # 고정 비율로 증액
                new_annual_withdrawal = annual_withdrawal_last_year * (1 + self.adjustment_pct)
                self.current_wr = new_annual_withdrawal / current_nav if current_nav > 0 else 0
                self.previous_nav = current_nav
                return new_annual_withdrawal / 12

            # Rule 2: Portfolio Management Rule
            elif portfolio_return < self.freeze_threshold:
                # 큰 손실 발생 → 동결 (인플레이션 조정 없음)
                self.current_wr = annual_withdrawal_last_year / current_nav if current_nav > 0 else 0
                self.previous_nav = current_nav
                return previous_withdrawal

            else:
                # 정상 범위 + 정상 수익 → 인플레이션 조정
                inflation_adjustment = (1 + self.inflation_rate) ** (month_idx / 12)
                self.current_wr = self.initial_wr * inflation_adjustment
                self.previous_nav = current_nav
                return (current_nav * self.initial_wr / 12) * inflation_adjustment

        else:
            # 월 중에는 이전 인출액 유지
            return previous_withdrawal


class DynamicWithdrawalSimulator:
    """
    Dynamic 전략을 위한 롤링 백테스트 엔진
    """

    def __init__(self, returns: pd.DataFrame, month_starts: pd.Series):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            일별 수익률 (벤치마크/포트폴리오 × 날짜)
        month_starts : pd.Series
            월초 거래일 여부 (boolean)
        """
        self.returns = returns
        self.month_starts = month_starts
        self.dates = returns.index
        self.month_start_indices = np.where(month_starts.values)[0]

    def _get_end_index(self, start_idx: int, horizon_years: int) -> int:
        """
        시작 인덱스로부터 정확히 horizon_years 후의 인덱스 찾기
        """
        start_date = self.dates[start_idx]
        target_date = start_date + pd.DateOffset(years=horizon_years)
        valid_dates = self.dates[self.dates >= target_date]

        if len(valid_dates) == 0:
            return None

        end_date = valid_dates[0]
        end_idx = self.dates.get_loc(end_date)
        return end_idx

    def simulate_guardrails_path(self,
                                benchmark: str,
                                start_idx: int,
                                end_idx: int,
                                initial_wr: float,
                                guardrail_width: float = 0.20,
                                inflation_rate: float = 0.02,
                                v0: float = 100.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Guardrails 전략 단일 경로 시뮬레이션

        Returns
        -------
        nav_path : np.ndarray
            월별 NAV 배열
        withdrawal_path : np.ndarray
            월별 인출액 배열
        terminal_nav : float
            최종 NAV
        """
        strategy = GuardrailsWithdrawal(initial_wr, guardrail_width, inflation_rate)

        V = v0
        withdrawal = v0 * initial_wr / 12

        nav_path = [V]
        withdrawal_path = [withdrawal]

        period_returns = self.returns[benchmark].iloc[start_idx:end_idx+1].values
        period_month_starts = self.month_starts.iloc[start_idx:end_idx+1].values
        month_indices = np.where(period_month_starts)[0]

        for month_in_simulation in range(1, len(month_indices)):
            month_start_idx = month_indices[month_in_simulation]
            prev_month_start_idx = month_indices[month_in_simulation - 1]

            # NAV 계산 (직전 월 말)
            month_returns = period_returns[prev_month_start_idx:month_start_idx]
            cumulative_return = np.prod(1 + month_returns)
            V = V * cumulative_return
            V = max(V, 0)

            if V == 0:
                nav_path.append(0)
                withdrawal_path.append(0)
                continue

            # 인출액 계산 (이번 월 초)
            withdrawal = strategy.calculate_withdrawal(
                V,
                withdrawal_path[-1],
                month_in_simulation
            )

            # 인출 실행
            V = max(V - withdrawal, 0)

            nav_path.append(V)
            withdrawal_path.append(withdrawal)

        return np.array(nav_path), np.array(withdrawal_path), V

    def simulate_guyton_klinger_path(self,
                                     benchmark: str,
                                     start_idx: int,
                                     end_idx: int,
                                     initial_wr: float,
                                     guardrail_width: float = 0.20,
                                     adjustment_pct: float = 0.10,
                                     freeze_threshold: float = -0.10,
                                     inflation_rate: float = 0.02,
                                     v0: float = 100.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Guyton-Klinger 전략 단일 경로 시뮬레이션

        Returns
        -------
        nav_path : np.ndarray
            월별 NAV 배열
        withdrawal_path : np.ndarray
            월별 인출액 배열
        terminal_nav : float
            최종 NAV
        """
        strategy = GuytonKlingerWithdrawal(
            initial_wr,
            guardrail_width,
            adjustment_pct,
            freeze_threshold,
            inflation_rate
        )

        V = v0
        withdrawal = v0 * initial_wr / 12
        prev_nav = v0

        nav_path = [V]
        withdrawal_path = [withdrawal]

        period_returns = self.returns[benchmark].iloc[start_idx:end_idx+1].values
        period_month_starts = self.month_starts.iloc[start_idx:end_idx+1].values
        month_indices = np.where(period_month_starts)[0]

        for month_in_simulation in range(1, len(month_indices)):
            month_start_idx = month_indices[month_in_simulation]
            prev_month_start_idx = month_indices[month_in_simulation - 1]

            # NAV 계산 (직전 월 말)
            month_returns = period_returns[prev_month_start_idx:month_start_idx]
            cumulative_return = np.prod(1 + month_returns)
            V = V * cumulative_return
            V = max(V, 0)

            if V == 0:
                nav_path.append(0)
                withdrawal_path.append(0)
                continue

            # 지난 12개월 포트폴리오 수익률 계산
            if month_in_simulation >= 12:
                portfolio_return = (V - prev_nav) / prev_nav
            else:
                portfolio_return = 0.0

            # 인출액 계산 (이번 월 초)
            withdrawal = strategy.calculate_withdrawal(
                V,
                withdrawal_path[-1],
                month_in_simulation,
                portfolio_return
            )

            # 인출 실행
            V = max(V - withdrawal, 0)

            # 연간 주기마다 NAV 업데이트 (수익률 계산용)
            if month_in_simulation % 12 == 0:
                prev_nav = V

            nav_path.append(V)
            withdrawal_path.append(withdrawal)

        return np.array(nav_path), np.array(withdrawal_path), V

    def run_guardrails_backtest(self,
                               benchmark: str,
                               horizon_years: int,
                               initial_wr: float,
                               guardrail_width: float = 0.20,
                               inflation_rate: float = 0.02,
                               v0: float = 100.0,
                               verbose: bool = True) -> pd.DataFrame:
        """
        Guardrails 전략 롤링 백테스트

        Returns
        -------
        DataFrame with columns:
            - start_date
            - terminal_nav
            - total_withdrawal
            - withdrawal_path (array)
            - nav_path (array)
        """
        results = []

        # 유효한 시작 인덱스 찾기
        valid_start_indices = []
        for start_idx in range(len(self.dates)):
            end_idx = self._get_end_index(start_idx, horizon_years)
            if end_idx is not None:
                valid_start_indices.append((start_idx, end_idx))

        n_paths = len(valid_start_indices)

        if verbose:
            print(f"\nGuardrails 백테스트: {benchmark}, {horizon_years}년, WR={initial_wr*100:.1f}%")
            print(f"총 경로 수: {n_paths:,}")

        iterator = tqdm(valid_start_indices, desc="Guardrails", leave=False) if TQDM_AVAILABLE else valid_start_indices

        for start_idx, end_idx in iterator:
            nav_path, withdrawal_path, terminal_nav = self.simulate_guardrails_path(
                benchmark, start_idx, end_idx, initial_wr, guardrail_width, inflation_rate, v0
            )

            results.append({
                'start_date': self.dates[start_idx],
                'terminal_nav': terminal_nav,
                'total_withdrawal': np.sum(withdrawal_path),
                'withdrawal_path': withdrawal_path,
                'nav_path': nav_path,
                'is_fail': terminal_nav < v0
            })

        if verbose:
            print(f"완료: {n_paths:,}개 경로")

        return pd.DataFrame(results)

    def run_guyton_klinger_backtest(self,
                                   benchmark: str,
                                   horizon_years: int,
                                   initial_wr: float,
                                   guardrail_width: float = 0.20,
                                   adjustment_pct: float = 0.10,
                                   freeze_threshold: float = -0.10,
                                   inflation_rate: float = 0.02,
                                   v0: float = 100.0,
                                   verbose: bool = True) -> pd.DataFrame:
        """
        Guyton-Klinger 전략 롤링 백테스트

        Returns
        -------
        DataFrame with columns:
            - start_date
            - terminal_nav
            - total_withdrawal
            - withdrawal_path (array)
            - nav_path (array)
        """
        results = []

        # 유효한 시작 인덱스 찾기
        valid_start_indices = []
        for start_idx in range(len(self.dates)):
            end_idx = self._get_end_index(start_idx, horizon_years)
            if end_idx is not None:
                valid_start_indices.append((start_idx, end_idx))

        n_paths = len(valid_start_indices)

        if verbose:
            print(f"\nGuyton-Klinger 백테스트: {benchmark}, {horizon_years}년, WR={initial_wr*100:.1f}%")
            print(f"총 경로 수: {n_paths:,}")

        iterator = tqdm(valid_start_indices, desc="G-K", leave=False) if TQDM_AVAILABLE else valid_start_indices

        for start_idx, end_idx in iterator:
            nav_path, withdrawal_path, terminal_nav = self.simulate_guyton_klinger_path(
                benchmark, start_idx, end_idx, initial_wr, guardrail_width,
                adjustment_pct, freeze_threshold, inflation_rate, v0
            )

            results.append({
                'start_date': self.dates[start_idx],
                'terminal_nav': terminal_nav,
                'total_withdrawal': np.sum(withdrawal_path),
                'withdrawal_path': withdrawal_path,
                'nav_path': nav_path,
                'is_fail': terminal_nav < v0
            })

        if verbose:
            print(f"완료: {n_paths:,}개 경로")

        return pd.DataFrame(results)
