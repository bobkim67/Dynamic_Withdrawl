"""
Dynamic Withdrawal Optimization Engine - Step 1: Path Generation
=================================================================
새 최적화 엔진의 Path 생성 모듈
- Rolling Historical Window
- Block Bootstrap
"""

import numpy as np
import pandas as pd
import pickle
from typing import List
from withdrawal_backtest import DataPreprocessor, PORTFOLIOS


# ============================================================================
# Path 생성을 위한 데이터 변환 함수
# ============================================================================

def daily_to_monthly_returns(daily_returns: pd.Series, month_starts: pd.Series) -> np.ndarray:
    """
    일별 수익률을 월간 수익률로 변환

    각 월의 일별 수익률을 누적 곱하여 월간 수익률 계산:
    monthly_return = (1+r1) * (1+r2) * ... * (1+rn) - 1

    Parameters
    ----------
    daily_returns : pd.Series
        일별 수익률 시계열
    month_starts : pd.Series
        월초 거래일 boolean Series

    Returns
    -------
    monthly_returns : np.ndarray
        월간 수익률 배열
    """
    # 월 구분을 위해 period 인덱스 생성
    periods = daily_returns.index.to_period('M')

    # 각 월별로 그룹화하여 누적 수익률 계산
    monthly_returns = []

    for period in periods.unique():
        # 해당 월의 일별 수익률 추출
        mask = periods == period
        month_daily_returns = daily_returns[mask]

        # 누적 수익률 계산: (1+r1)*(1+r2)*...*(1+rn) - 1
        cumulative_return = (1 + month_daily_returns).prod() - 1
        monthly_returns.append(cumulative_return)

    return np.array(monthly_returns)


# ============================================================================
# Path 생성 함수
# ============================================================================

def generate_paths_rolling(monthly_returns: np.ndarray, T_months: int) -> List[np.ndarray]:
    """
    Rolling Historical Window 방식으로 Path 생성

    실제 연속 구간을 시작점을 1개월씩 이동하며 추출.
    각 시작점마다 T_months 길이의 path 생성.

    Parameters
    ----------
    monthly_returns : np.ndarray
        포트폴리오의 월간 수익률 시계열 (1차원 배열)
    T_months : int
        path 길이 (월 단위, 예: 360=30년, 120=10년)

    Returns
    -------
    paths : List[np.ndarray]
        각 path는 길이 T_months의 월간 수익률 배열
    """
    n_total = len(monthly_returns)

    # 생성 가능한 path 수
    n_paths = n_total - T_months + 1

    if n_paths <= 0:
        raise ValueError(f"데이터 길이({n_total}개월)가 path 길이({T_months}개월)보다 짧습니다")

    paths = []

    # 시작점을 1개월씩 이동하며 연속 구간 추출
    for start_idx in range(n_paths):
        end_idx = start_idx + T_months
        path = monthly_returns[start_idx:end_idx]
        paths.append(path)

    return paths


def generate_paths_bootstrap(monthly_returns: np.ndarray,
                              T_months: int,
                              block_length: int = 12,
                              n_paths: int = 5000,
                              seed: int = 42) -> List[np.ndarray]:
    """
    Block Bootstrap 방식으로 Path 생성

    월간 수익률에서 block_length 길이의 연속 블록을 랜덤 추출하여
    T_months가 될 때까지 이어붙임.

    Parameters
    ----------
    monthly_returns : np.ndarray
        포트폴리오의 월간 수익률 시계열
    T_months : int
        path 길이 (월 단위)
    block_length : int
        블록 크기 (월 단위, 기본 12개월)
    n_paths : int
        생성할 path 수
    seed : int
        랜덤 시드 (재현성)

    Returns
    -------
    paths : List[np.ndarray]
        각 path는 길이 T_months의 월간 수익률 배열
    """
    n_total = len(monthly_returns)

    if block_length > n_total:
        raise ValueError(f"블록 길이({block_length})가 데이터 길이({n_total})보다 깁니다")

    # 랜덤 시드 설정
    rng = np.random.RandomState(seed)

    paths = []

    for _ in range(n_paths):
        path = []

        # T_months가 될 때까지 블록 추출하여 이어붙이기
        while len(path) < T_months:
            # 랜덤 시작점 선택 (블록이 데이터 범위를 벗어나지 않도록)
            max_start = n_total - block_length
            start_idx = rng.randint(0, max_start + 1)

            # 블록 추출
            block = monthly_returns[start_idx:start_idx + block_length]
            path.extend(block)

        # 정확히 T_months 길이로 자르기
        path = np.array(path[:T_months])
        paths.append(path)

    return paths


# ============================================================================
# 검증 코드
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Path 생성 모듈 검증")
    print("=" * 70)

    # ========================================
    # 1. 데이터 로딩 및 전처리
    # ========================================

    print("\n[1단계] 데이터 로딩 및 전처리")
    print("-" * 70)

    # benchmark_data.pkl 로드
    with open('benchmark_data.pkl', 'rb') as f:
        benchmark_data = pickle.load(f)

    print(f"벤치마크 데이터 로드 완료")
    print(f"  기간: {benchmark_data.index[0].date()} ~ {benchmark_data.index[-1].date()}")
    print(f"  거래일 수: {len(benchmark_data):,}일")
    print(f"  자산: {list(benchmark_data.columns)}")

    # DataPreprocessor로 일별 수익률 및 월초 거래일 계산
    preprocessor = DataPreprocessor(benchmark_data, add_portfolios=True)
    returns_df, month_starts = preprocessor.get_data()

    # ========================================
    # 2. Port_5.0% 월간 수익률 변환
    # ========================================

    print("\n[2단계] Port_5.0% 월간 수익률 변환")
    print("-" * 70)

    # Port_5.0% 일별 수익률 추출
    portfolio_name = 'Port_5.0%'
    daily_returns = returns_df[portfolio_name]

    # 월간 수익률로 변환
    monthly_returns = daily_to_monthly_returns(daily_returns, month_starts)

    print(f"✅ 월간 수익률 변환 완료")
    print(f"  길이: {len(monthly_returns)}개월")

    # 기간 계산
    first_date = daily_returns.index[0]
    last_date = daily_returns.index[-1]
    years = (last_date - first_date).days / 365.25

    print(f"  기간: {first_date.date()} ~ {last_date.date()} ({years:.1f}년)")

    # 기본 통계
    monthly_mean = np.mean(monthly_returns)
    monthly_std = np.std(monthly_returns, ddof=1)
    annual_return = (1 + monthly_mean) ** 12 - 1
    annual_vol = monthly_std * np.sqrt(12)

    print(f"\n  월 평균 수익률: {monthly_mean*100:.3f}%")
    print(f"  월 표준편차: {monthly_std*100:.3f}%")
    print(f"  연환산 수익률: {annual_return*100:.2f}%")
    print(f"  연환산 변동성: {annual_vol*100:.2f}%")

    # ========================================
    # 3. Rolling Path 생성
    # ========================================

    print("\n[3단계] Rolling Historical Window Path 생성")
    print("-" * 70)

    T_months = 120  # 10년
    rolling_paths = generate_paths_rolling(monthly_returns, T_months)

    print(f"✅ Rolling path 생성 완료")
    print(f"  Path 길이: {T_months}개월 (10년)")
    print(f"  생성된 path 수: {len(rolling_paths)}개")

    # 첫 번째 path의 누적 수익률 계산
    first_path_cumulative = (1 + rolling_paths[0]).prod() - 1
    print(f"\n  첫 번째 path 누적 수익률: {first_path_cumulative*100:.2f}%")
    print(f"  연평균 수익률: {((1 + first_path_cumulative)**(1/10) - 1)*100:.2f}%")

    # ========================================
    # 4. Bootstrap Path 생성
    # ========================================

    print("\n[4단계] Block Bootstrap Path 생성")
    print("-" * 70)

    bootstrap_paths = generate_paths_bootstrap(
        monthly_returns,
        T_months=T_months,
        block_length=12,
        n_paths=1000,
        seed=42
    )

    print(f"✅ Bootstrap path 생성 완료")
    print(f"  Path 길이: {T_months}개월 (10년)")
    print(f"  블록 길이: 12개월")
    print(f"  생성된 path 수: {len(bootstrap_paths)}개")

    # 누적 수익률 분포 계산
    bootstrap_cumulative = np.array([(1 + path).prod() - 1 for path in bootstrap_paths])

    print(f"\n  누적 수익률 분포:")
    print(f"    평균: {np.mean(bootstrap_cumulative)*100:.2f}%")
    print(f"    중앙값: {np.median(bootstrap_cumulative)*100:.2f}%")
    print(f"    5th percentile: {np.percentile(bootstrap_cumulative, 5)*100:.2f}%")
    print(f"    95th percentile: {np.percentile(bootstrap_cumulative, 95)*100:.2f}%")

    # ========================================
    # 5. 두 방식 비교
    # ========================================

    print("\n[5단계] Rolling vs Bootstrap 비교")
    print("-" * 70)

    # Rolling paths의 누적 수익률 분포
    rolling_cumulative = np.array([(1 + path).prod() - 1 for path in rolling_paths])

    print(f"Rolling Historical Window:")
    print(f"  평균 누적 수익률: {np.mean(rolling_cumulative)*100:.2f}%")
    print(f"  중앙값: {np.median(rolling_cumulative)*100:.2f}%")
    print(f"  표준편차: {np.std(rolling_cumulative)*100:.2f}%p")
    print(f"  최소: {np.min(rolling_cumulative)*100:.2f}%")
    print(f"  최대: {np.max(rolling_cumulative)*100:.2f}%")

    print(f"\nBlock Bootstrap:")
    print(f"  평균 누적 수익률: {np.mean(bootstrap_cumulative)*100:.2f}%")
    print(f"  중앙값: {np.median(bootstrap_cumulative)*100:.2f}%")
    print(f"  표준편차: {np.std(bootstrap_cumulative)*100:.2f}%p")
    print(f"  최소: {np.min(bootstrap_cumulative)*100:.2f}%")
    print(f"  최대: {np.max(bootstrap_cumulative)*100:.2f}%")

    print("\n" + "=" * 70)
    print("검증 완료")
    print("=" * 70)
