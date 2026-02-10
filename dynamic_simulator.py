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

from withdrawal_backtest import PORTFOLIOS, BENCHMARK_MAPPING

# tqdm 선택적 import
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class FixedWithdrawal:
    """
    Fixed 전략: 완전 고정 인출액

    핵심 원리:
    - 첫 달: 초기 NAV × 초기 인출률로 인출액 결정
    - 이후: 동일한 금액을 영구적으로 인출 (절대 변경 안 됨)
    - Guardrail 없음, 재계산 없음
    - 진짜 "Fixed" - 인출액이 고정됨
    """

    def __init__(self, initial_wr: float, inflation_rate: float = 0.02):
        """
        Parameters
        ----------
        initial_wr : float
            초기 연간 인출률 (예: 0.05 = 5%)
        inflation_rate : float
            연간 인플레이션율 (기본값 2%)
        """
        self.initial_wr = initial_wr
        self.inflation_rate = inflation_rate
        # Guardrail 없음 (무한대/0으로 설정)
        self.upper_guardrail = float('inf')
        self.lower_guardrail = 0.0
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
            현재 포트폴리오 가치 (사용하지 않음)
        previous_withdrawal : float
            직전 월 인출액
        month_idx : int
            현재 월 인덱스 (0-119)

        Returns
        -------
        float : 이번 달 인출액
        """
        # 첫 달: 초기 인출액 설정
        if month_idx == 0:
            self.current_wr = self.initial_wr
            return current_nav * self.initial_wr / 12

        # 이후 모든 달: 고정 인출액 유지 (절대 변경 안 됨)
        return previous_withdrawal


class GuardrailsWithdrawal:
    """
    Guardrails 전략: 인출액 한도 제한 방식

    핵심 원리:
    - 매월 기본 인출액은 동일하게 유지
    - 인출액이 Guardrail 한도를 초과하면 한도로 제한 (캡핑)
    - 상한 = NAV × upper_guardrail / 12
    - 하한 = NAV × lower_guardrail / 12
    - 매월 NAV 기준으로 한도를 재계산하여 포트폴리오 보호
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
        self.base_withdrawal = None  # 첫 달 인출액 저장용 (고정)
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
        # 첫 달: 초기 인출액 설정 및 저장
        if month_idx == 0:
            self.base_withdrawal = current_nav * self.initial_wr / 12
            return self.base_withdrawal

        # 원래 인출액 사용 (캡핑되어도 변경되지 않음)
        base = self.base_withdrawal

        # 매월 Guardrail 한도 계산
        max_withdrawal = current_nav * self.upper_guardrail / 12
        min_withdrawal = current_nav * self.lower_guardrail / 12

        # 한도 적용 (단순 캡핑, base는 변경 안 됨)
        if base > max_withdrawal:
            # 상한 초과: 상한으로 제한
            return max_withdrawal
        elif base < min_withdrawal:
            # 하한 미만: 하한으로 제한
            return min_withdrawal
        else:
            # 정상 범위: 원래 인출액 유지
            return base


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
    3. Normal: 정상 범위 & 정상 수익 → NAV 기준 인출
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
                # 정상 범위 + 정상 수익 → NAV 기준 인출 (인플레이션 조정 제거)
                self.current_wr = self.initial_wr
                self.previous_nav = current_nav
                return current_nav * self.initial_wr / 12

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

    def simulate_fixed_path(self,
                           benchmark: str,
                           start_idx: int,
                           end_idx: int,
                           initial_wr: float,
                           inflation_rate: float = 0.02,
                           v0: float = 100.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fixed Rate 전략 단일 경로 시뮬레이션

        Returns
        -------
        nav_path : np.ndarray
            월별 NAV 배열
        withdrawal_path : np.ndarray
            월별 인출액 배열
        terminal_nav : float
            최종 NAV
        """
        strategy = FixedWithdrawal(initial_wr, inflation_rate)

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

    def run_fixed_backtest(self,
                          benchmark: str,
                          horizon_years: int,
                          initial_wr: float,
                          inflation_rate: float = 0.02,
                          v0: float = 100.0,
                          verbose: bool = True) -> pd.DataFrame:
        """
        Fixed Rate 전략 롤링 백테스트

        Returns
        -------
        DataFrame with columns:
            - start_date
            - terminal_nav
            - total_withdrawal
            - withdrawal_path (array)
            - nav_path (array)
            - is_fail
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
            print(f"\nFixed Rate 백테스트: {benchmark}, {horizon_years}년, WR={initial_wr*100:.1f}%")
            print(f"총 경로 수: {n_paths:,}")

        iterator = tqdm(valid_start_indices, desc="Fixed", leave=False) if TQDM_AVAILABLE else valid_start_indices

        for start_idx, end_idx in iterator:
            nav_path, withdrawal_path, terminal_nav = self.simulate_fixed_path(
                benchmark, start_idx, end_idx, initial_wr, inflation_rate, v0
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

    # ============================================================
    # 특정 경로의 일별 상세 데이터 반환
    # ============================================================

    @staticmethod
    def _build_asset_groups(portfolio: str) -> List[Tuple[str, float, str]]:
        """
        포트폴리오 가중치를 8개 개별 자산으로 반환

        개별 자산:
          - 한국주식 (Korean_Equity)
          - 미국성장주 (US_Growth)
          - 한국종합채권 (Korean_Bond_Composite)
          - 한국국고채10년 (Korean_Bond_10Y)
          - 신흥국달러채권 (EM_Dollar_Bond)
          - 미국채권 (US_Bond)
          - 미국외글로벌채권 (Global_ex_US_Bond)
          - 금 (Gold)

        Returns
        -------
        List[Tuple[str, float, str]]
            각 튜플: (mapped_column_name, weight_fraction, display_name)
        """
        if portfolio not in PORTFOLIOS:
            raise ValueError(f"Unknown portfolio: '{portfolio}'. "
                             f"Available: {list(PORTFOLIOS.keys())}")

        raw_weights = PORTFOLIOS[portfolio]['weights']  # % 단위

        # 자산 매핑: 원본 키 → (컬럼명, 표시명)
        ASSET_MAP = {
            '한국주식':        ('국내주식', 'Korean_Equity'),
            '미국성장주':      ('미국성장주', 'US_Growth'),
            '한국종합채권':    ('국내중기채', 'Korean_Bond_Composite'),
            '한국국고채10년':  ('국내장기채', 'Korean_Bond_10Y'),
            '신흥국달러채권':  ('신흥국달러채권', 'EM_Dollar_Bond'),
            '미국채권':        ('미국국채', 'US_Bond'),
            '미국외글로벌채권': ('미국외국채', 'Global_ex_US_Bond'),
            '금':             ('금', 'Gold'),
        }

        # 8개 자산 초기화 (가중치 0)
        ALL_ASSETS = [
            ('국내주식', 'Korean_Equity'),
            ('미국성장주', 'US_Growth'),
            ('국내중기채', 'Korean_Bond_Composite'),
            ('국내장기채', 'Korean_Bond_10Y'),
            ('신흥국달러채권', 'EM_Dollar_Bond'),
            ('미국국채', 'US_Bond'),
            ('미국외국채', 'Global_ex_US_Bond'),
            ('금', 'Gold'),
        ]

        asset_weights = {display_name: 0.0 for _, display_name in ALL_ASSETS}

        # 포트폴리오 가중치 적용
        for asset_key, weight_pct in raw_weights.items():
            if asset_key not in ASSET_MAP:
                raise ValueError(f"Unknown asset '{asset_key}' in portfolio '{portfolio}'")
            col_name, display_name = ASSET_MAP[asset_key]
            asset_weights[display_name] = weight_pct / 100.0  # % → 비율

        # 결과 리스트 생성 (순서 유지)
        result = []
        for col_name, display_name in ALL_ASSETS:
            weight = asset_weights[display_name]
            result.append((col_name, weight, display_name))

        return result

    # 자산 표시명 → 원본 data 컬럼명 매핑 (가격 지수용)
    PRICE_COL_MAP = {
        'Korean_Equity':          '국내주식',
        'US_Growth':              '미국성장주',
        'Korean_Bond_Composite':  '국내중기채',
        'Korean_Bond_10Y':        '국내장기채',
        'EM_Dollar_Bond':         '신흥국달러채권',
        'US_Bond':                '미국국채',
        'Global_ex_US_Bond':      '미국외국채',
        'Gold':                   '금',
    }

    # 포트폴리오 가중치 매핑 (한글 → 영문)
    WEIGHT_MAP = {
        '한국주식': 'Korean_Equity',
        '미국성장주': 'US_Growth',
        '한국종합채권': 'Korean_Bond_Composite',
        '한국국고채10년': 'Korean_Bond_10Y',
        '신흥국달러채권': 'EM_Dollar_Bond',
        '미국채권': 'US_Bond',
        '미국외글로벌채권': 'Global_ex_US_Bond',
        '금': 'Gold',
    }

    def get_single_path_detail(self,
                               portfolio: str,
                               start_date,
                               strategy: str,
                               horizon_years: int = 10,
                               initial_wr: float = 0.05,
                               guardrail_width: float = 0.20,
                               adjustment_pct: float = 0.10,
                               freeze_threshold: float = -0.10,
                               inflation_rate: float = 0.02,
                               v0: float = 100.0,
                               price_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        특정 시작일의 일별 경로 데이터 반환

        기존 백테스트와 동일한 로직 사용:
        - portfolio_returns = self.returns[portfolio] (수익률의 가중평균)
        - Total_NAV만 추적 (개별 자산 NAV 제거)
        - Price Index 및 Weight 컬럼 포함 (Excel 검증용)

        Parameters
        ----------
        portfolio : str
            포트폴리오 이름 (예: 'Port_5.0%')
        start_date : str or pd.Timestamp
            시뮬레이션 시작일 (예: '2008-01-02')
        strategy : str
            'fixed', 'guardrails', 'guyton_klinger'
        horizon_years : int
            시뮬레이션 기간 (년)
        initial_wr : float
            초기 인출률
        guardrail_width : float
            Guardrail 폭 (fixed에서는 무시)
        adjustment_pct : float
            조정 비율 (Guyton-Klinger only)
        freeze_threshold : float
            동결 기준 (Guyton-Klinger only)
        inflation_rate : float
            인플레이션율
        v0 : float
            초기 NAV
        price_data : pd.DataFrame, optional
            원본 가격 지수 데이터 (Price_* 컬럼 추가용).
            None이면 Price_* 컬럼 생략.

        Returns
        -------
        pd.DataFrame
            일별 데이터 (Price Index, Weight 포함)
        """
        # ----------------------------------------------------------
        # 입력 검증 및 인덱스 설정
        # ----------------------------------------------------------
        start_date = pd.Timestamp(start_date)
        if start_date not in self.dates:
            valid = self.dates[self.dates >= start_date]
            if len(valid) == 0:
                raise ValueError(f"시작일 {start_date} 이후 거래일이 없습니다.")
            start_date = valid[0]

        start_idx = self.dates.get_loc(start_date)
        end_idx = self._get_end_index(start_idx, horizon_years)
        if end_idx is None:
            raise ValueError(
                f"시작일 {start_date.date()}로부터 {horizon_years}년 후 데이터가 없습니다."
            )

        # ----------------------------------------------------------
        # 자산 가중치 정보 준비 (Price/Weight 컬럼용)
        # ----------------------------------------------------------
        assets_list = self._build_asset_groups(portfolio)  # List[Tuple[col, weight, name]]

        # 포트폴리오 수익률 (백테스트와 동일)
        n_days = end_idx - start_idx + 1
        portfolio_returns = self.returns[portfolio].iloc[start_idx:end_idx + 1].values

        # ----------------------------------------------------------
        # 전략 객체 생성
        # ----------------------------------------------------------
        if strategy == 'fixed':
            strategy_obj = FixedWithdrawal(initial_wr, inflation_rate)
        elif strategy == 'guardrails':
            strategy_obj = GuardrailsWithdrawal(initial_wr, guardrail_width, inflation_rate)
        elif strategy == 'guyton_klinger':
            strategy_obj = GuytonKlingerWithdrawal(
                initial_wr, guardrail_width, adjustment_pct,
                freeze_threshold, inflation_rate
            )
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'. Use 'fixed', 'guardrails', or 'guyton_klinger'.")

        # ----------------------------------------------------------
        # 일별 시뮬레이션
        # ----------------------------------------------------------
        period_month_starts = self.month_starts.iloc[start_idx:end_idx + 1].values
        period_dates = self.dates[start_idx:end_idx + 1]

        Total_NAV = v0
        withdrawal = v0 * initial_wr / 12
        month_counter = 0
        prev_nav = v0  # Guyton-Klinger 연간 수익률 추적용
        nav_no_withdrawal = v0  # 인출 없는 순수 포트폴리오 NAV 추적용

        daily_data = []

        for day_idx in range(n_days):
            date = period_dates[day_idx]
            is_month_start = bool(period_month_starts[day_idx])
            is_year_start = (
                is_month_start
                and day_idx > 0
                and date.month == 1
            )

            # --------------------------------------------------------
            # 월초 처리: 인출 먼저 실행
            # --------------------------------------------------------
            if is_month_start:
                if strategy == 'guyton_klinger':
                    portfolio_return = (
                        (Total_NAV - prev_nav) / prev_nav
                        if prev_nav > 0 else 0.0
                    )
                    withdrawal = strategy_obj.calculate_withdrawal(
                        Total_NAV, withdrawal, month_counter, portfolio_return
                    )
                else:
                    withdrawal = strategy_obj.calculate_withdrawal(
                        Total_NAV, withdrawal, month_counter
                    )

                # 인출 실행
                if Total_NAV > 0:
                    Total_NAV = max(Total_NAV - withdrawal, 0.0)

                # Guyton-Klinger: 연간 주기마다 prev_nav 갱신
                if strategy == 'guyton_klinger' and month_counter > 0 and month_counter % 12 == 0:
                    prev_nav = Total_NAV

                month_counter += 1

            # --------------------------------------------------------
            # 수익률 적용 (첫 날 제외)
            # --------------------------------------------------------
            if day_idx > 0:
                daily_ret = portfolio_returns[day_idx]
                Total_NAV = Total_NAV * (1 + daily_ret)
                Total_NAV = max(Total_NAV, 0.0)

                # 인출 없는 순수 포트폴리오 NAV도 업데이트
                nav_no_withdrawal = nav_no_withdrawal * (1 + daily_ret)
                nav_no_withdrawal = max(nav_no_withdrawal, 0.0)

            # --------------------------------------------------------
            # 행 기록
            # --------------------------------------------------------
            cumulative_return = (Total_NAV - v0) / v0 if v0 != 0 else 0.0
            portfolio_return_no_withdrawal = (nav_no_withdrawal - v0) / v0 if v0 != 0 else 0.0
            current_wr = (withdrawal * 12) / Total_NAV if Total_NAV > 0 else 0.0

            if current_wr > strategy_obj.upper_guardrail:
                status = 'Upper_Breach'
            elif current_wr < strategy_obj.lower_guardrail:
                status = 'Lower_Breach'
            else:
                status = 'Normal'

            row = {
                'Date': date,
                'Day_Index': day_idx,
                'Total_NAV': round(Total_NAV, 8),
                'Cumulative_Return': round(cumulative_return, 8),
                'Portfolio_Return_No_Withdrawal': round(portfolio_return_no_withdrawal, 8),
                'Withdrawal_Amount': round(withdrawal, 8) if is_month_start else 0.0,
                'Monthly_Withdrawal': round(withdrawal, 8),
                'Current_WR': round(current_wr, 8),
                'Upper_Guardrail': strategy_obj.upper_guardrail,
                'Lower_Guardrail': strategy_obj.lower_guardrail,
                'Is_Month_Start': is_month_start,
                'Is_Year_Start': is_year_start,
                'Guardrail_Status': status,
                'Year_Month': f"{date.year}-{date.month:02d}",
            }

            # Price Index 컬럼 (원본 가격 데이터가 있을 때만)
            if price_data is not None:
                for display_name, data_col in self.PRICE_COL_MAP.items():
                    row[f'Price_{display_name}'] = price_data.loc[date, data_col]

            # Weight 컬럼 (% 단위, 고정 가중치)
            for _, weight, display_name in assets_list:
                row[f'Weight_{display_name}'] = round(weight * 100, 2)

            daily_data.append(row)

        return pd.DataFrame(daily_data)
