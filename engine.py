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
# 인출 시뮬레이션 함수
# ============================================================================

def simulate_withdrawal_on_path(path_returns: np.ndarray,
                                 init_wr: float,
                                 band: float,
                                 adj_on: bool = False,
                                 lookback: int = 1,
                                 thr_up: float = 0.03,
                                 thr_dn: float = -0.03,
                                 adj_up: float = 0.05,
                                 adj_dn: float = -0.05,
                                 beta: float = 0.5,
                                 W0: float = 100.0) -> dict:
    """
    월간 수익률 path에 인출 전략을 적용하여 시뮬레이션

    인출 순서 (매월 반복, 순서 고정):
    1. 수익률 적용: W_t = W_{t-1} * (1 + r_t)
    2. Base 인출액 계산: base = W0 * init_wr / 12
    3. 수익률 기반 보정 (adj_on=True일 때)
    4. Guardrail 캡 적용
    5. 인출 실행: W_t = W_t - withdraw_final

    Parameters
    ----------
    path_returns : np.ndarray
        월간 수익률 배열 (길이 T_months)
    init_wr : float
        초기 연간 인출률 (예: 0.05 = 5%)
    band : float
        guardrail 폭 (예: 0.15 = ±15%)
        upper = init_wr * (1 + band)
        lower = init_wr * (1 - band)
    adj_on : bool
        수익률 기반 보정 사용 여부
    lookback : int
        보정 시 참조 기간 (1=전월, 3=최근 3개월 누적)
    thr_up : float
        상향 보정 트리거 (월간/누적 수익률이 이 값 초과 시)
    thr_dn : float
        하향 보정 트리거 (월간/누적 수익률이 이 값 미만 시)
    adj_up : float
        상향 보정 비율 (예: 0.05 = +5%)
    adj_dn : float
        하향 보정 비율 (예: -0.05 = -5%, 음수로 입력)
    beta : float
        terminal 실패 기준 (W_T < beta * W0이면 실패)
    W0 : float
        초기 자산

    Returns
    -------
    result : dict
        'W_series': np.ndarray - 월말 NAV 시계열 (길이 T+1, W_0 포함)
        'withdraw_series': np.ndarray - 월별 인출액 (길이 T)
        'cum_withdraw': float - 총 누적 인출액
        'terminal_nav': float - 최종 NAV (W_T)
        'ruin_flag': bool - min(W_t) <= 0 여부
        'terminal_fail_flag': bool - W_T < beta * W0 여부
        'success': bool - not ruin AND not terminal_fail
    """
    T = len(path_returns)

    # 결과 저장 배열 초기화
    W_series = np.zeros(T + 1)  # W_0부터 W_T까지
    withdraw_series = np.zeros(T)

    # 초기 자산
    W_series[0] = W0

    # Base 인출액 (고정, 매월 동일)
    base_withdraw = W0 * init_wr / 12

    # Guardrail 경계
    upper_wr = init_wr * (1 + band)
    lower_wr = init_wr * (1 - band)

    # 매월 시뮬레이션
    for t in range(T):
        # Step 1: 수익률 적용
        W_t = W_series[t] * (1 + path_returns[t])

        # Step 2: Base 인출액
        base = base_withdraw

        # Step 3: 수익률 기반 보정 (adj_on=True일 때)
        if adj_on and t >= lookback:
            # lookback 기간의 수익률 계산
            if lookback == 1:
                prev_return = path_returns[t - 1]
            else:  # lookback == 3
                # 최근 lookback 개월의 누적 수익률
                prev_return = (1 + path_returns[t - lookback:t]).prod() - 1

            # 보정 적용
            if prev_return > thr_up:
                withdraw_adj = base * (1 + adj_up)
            elif prev_return < thr_dn:
                withdraw_adj = base * (1 + adj_dn)  # adj_dn은 음수
            else:
                withdraw_adj = base
        else:
            # 보정 없음 (adj_on=False 또는 lookback 데이터 부족)
            withdraw_adj = base

        # Step 4: Guardrail 캡 적용
        if W_t > 0:
            current_wr = (withdraw_adj * 12) / W_t

            if current_wr > upper_wr:
                withdraw_final = upper_wr * W_t / 12
            elif current_wr < lower_wr:
                withdraw_final = lower_wr * W_t / 12
            else:
                withdraw_final = withdraw_adj
        else:
            # 이미 파산 상태
            withdraw_final = 0

        # Step 5: 인출 실행
        W_t = W_t - withdraw_final
        W_t = max(W_t, 0)  # 음수 방지

        # 결과 저장
        W_series[t + 1] = W_t
        withdraw_series[t] = withdraw_final

    # 집계 지표 계산
    cum_withdraw = np.sum(withdraw_series)
    terminal_nav = W_series[-1]
    ruin_flag = np.any(W_series <= 0)
    terminal_fail_flag = terminal_nav < beta * W0
    success = (not ruin_flag) and (not terminal_fail_flag)

    return {
        'W_series': W_series,
        'withdraw_series': withdraw_series,
        'cum_withdraw': cum_withdraw,
        'terminal_nav': terminal_nav,
        'ruin_flag': ruin_flag,
        'terminal_fail_flag': terminal_fail_flag,
        'success': success
    }


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

    # ========================================
    # 6. 인출 시뮬레이션 테스트
    # ========================================

    print("\n" + "=" * 70)
    print("인출 시뮬레이션 검증")
    print("=" * 70)

    # 테스트용 path: 첫 번째 rolling path 사용
    test_path = rolling_paths[0]

    # ========================================
    # 테스트 1: 기본 케이스 (보정 없음)
    # ========================================

    print("\n[테스트 1] 기본 케이스 (adj_on=False)")
    print("-" * 70)

    result1 = simulate_withdrawal_on_path(
        path_returns=test_path,
        init_wr=0.05,
        band=0.15,
        adj_on=False,
        W0=100.0
    )

    print(f"초기 자산: 100.0")
    print(f"초기 인출률: 5.0%")
    print(f"Guardrail 밴드: ±15% (상한 5.75%, 하한 4.25%)")
    print(f"\n결과:")
    print(f"  최종 NAV: {result1['terminal_nav']:.2f}")
    print(f"  총 누적 인출액: {result1['cum_withdraw']:.2f}")
    print(f"  Ruin 발생: {'예' if result1['ruin_flag'] else '아니오'}")
    print(f"  Terminal 실패: {'예' if result1['terminal_fail_flag'] else '아니오'} (기준: NAV < 50.0)")
    print(f"  성공 여부: {'✅ 성공' if result1['success'] else '❌ 실패'}")

    # ========================================
    # 테스트 2: 높은 인출률 (파산 유도)
    # ========================================

    print("\n[테스트 2] 높은 인출률 (파산 유도)")
    print("-" * 70)

    result2 = simulate_withdrawal_on_path(
        path_returns=test_path,
        init_wr=0.15,
        band=0.15,
        adj_on=False,
        W0=100.0
    )

    print(f"초기 자산: 100.0")
    print(f"초기 인출률: 15.0% (매우 높음)")
    print(f"Guardrail 밴드: ±15%")
    print(f"\n결과:")
    print(f"  최종 NAV: {result2['terminal_nav']:.2f}")
    print(f"  총 누적 인출액: {result2['cum_withdraw']:.2f}")
    print(f"  Ruin 발생: {'예' if result2['ruin_flag'] else '아니오'}")
    print(f"  Terminal 실패: {'예' if result2['terminal_fail_flag'] else '아니오'}")
    print(f"  성공 여부: {'✅ 성공' if result2['success'] else '❌ 실패'}")

    if result2['ruin_flag']:
        # 파산 발생 시점 찾기
        ruin_month = np.where(result2['W_series'] <= 0)[0][0]
        print(f"  파산 발생 시점: {ruin_month}개월째")

    # ========================================
    # 테스트 3: 수익률 보정 ON
    # ========================================

    print("\n[테스트 3] 수익률 보정 ON")
    print("-" * 70)

    result3 = simulate_withdrawal_on_path(
        path_returns=test_path,
        init_wr=0.05,
        band=0.15,
        adj_on=True,
        lookback=1,
        thr_up=0.03,
        thr_dn=-0.03,
        adj_up=0.05,
        adj_dn=-0.05,
        W0=100.0
    )

    print(f"초기 자산: 100.0")
    print(f"초기 인출률: 5.0%")
    print(f"Guardrail 밴드: ±15%")
    print(f"보정 설정:")
    print(f"  - lookback: 1개월 (전월 수익률)")
    print(f"  - 상향 트리거: >3%, 조정 +5%")
    print(f"  - 하향 트리거: <-3%, 조정 -5%")
    print(f"\n결과:")
    print(f"  최종 NAV: {result3['terminal_nav']:.2f}")
    print(f"  총 누적 인출액: {result3['cum_withdraw']:.2f}")
    print(f"  Ruin 발생: {'예' if result3['ruin_flag'] else '아니오'}")
    print(f"  Terminal 실패: {'예' if result3['terminal_fail_flag'] else '아니오'}")
    print(f"  성공 여부: {'✅ 성공' if result3['success'] else '❌ 성공'}")

    print(f"\n테스트 1 대비 차이:")
    print(f"  총 인출액 차이: {result3['cum_withdraw'] - result1['cum_withdraw']:.2f} "
          f"({(result3['cum_withdraw'] / result1['cum_withdraw'] - 1)*100:+.2f}%)")
    print(f"  최종 NAV 차이: {result3['terminal_nav'] - result1['terminal_nav']:.2f} "
          f"({(result3['terminal_nav'] / result1['terminal_nav'] - 1)*100:+.2f}%)")

    # ========================================
    # 테스트 4: 인출 순서 검증 (짧은 path)
    # ========================================

    print("\n[테스트 4] 인출 순서 검증 (Step-by-step)")
    print("-" * 70)

    # 짧은 path 생성 (3개월)
    short_path = np.array([0.02, -0.03, 0.01])

    result4 = simulate_withdrawal_on_path(
        path_returns=short_path,
        init_wr=0.06,
        band=0.20,
        adj_on=False,
        W0=100.0
    )

    print(f"짧은 path: [2.0%, -3.0%, 1.0%]")
    print(f"초기 자산: 100.0")
    print(f"초기 인출률: 6.0% (월 0.5)")
    print(f"Guardrail: 상한 7.2%, 하한 4.8%")

    print("\n월별 계산 과정:")
    W_prev = 100.0
    base = 100.0 * 0.06 / 12

    for t in range(3):
        print(f"\n--- {t}개월째 ---")
        r_t = short_path[t]

        # Step 1: 수익률 적용
        W_after_return = W_prev * (1 + r_t)
        print(f"  1) 수익률 적용: {W_prev:.4f} × (1 + {r_t:.4f}) = {W_after_return:.4f}")

        # Step 2: Base 인출액
        print(f"  2) Base 인출액: {base:.4f}")

        # Step 4: Guardrail 캡
        current_wr = (base * 12) / W_after_return
        upper_wr = 0.06 * 1.20
        lower_wr = 0.06 * 0.80

        print(f"  3) Current WR: ({base:.4f} × 12) / {W_after_return:.4f} = {current_wr:.4f} ({current_wr*100:.2f}%)")

        if current_wr > upper_wr:
            withdraw = upper_wr * W_after_return / 12
            print(f"     ⚠️  상한 초과 → 캡 적용: {upper_wr:.4f} × {W_after_return:.4f} / 12 = {withdraw:.4f}")
        elif current_wr < lower_wr:
            withdraw = lower_wr * W_after_return / 12
            print(f"     ⚠️  하한 미달 → 캡 적용: {lower_wr:.4f} × {W_after_return:.4f} / 12 = {withdraw:.4f}")
        else:
            withdraw = base
            print(f"     ✅ 범위 내 → Base 사용: {withdraw:.4f}")

        # Step 5: 인출 실행
        W_after_withdraw = W_after_return - withdraw
        print(f"  4) 인출 후 NAV: {W_after_return:.4f} - {withdraw:.4f} = {W_after_withdraw:.4f}")

        # 실제 결과와 비교
        actual_W = result4['W_series'][t + 1]
        actual_withdraw = result4['withdraw_series'][t]

        print(f"\n  검증: NAV {W_after_withdraw:.4f} vs {actual_W:.4f} "
              f"({'✅ 일치' if abs(W_after_withdraw - actual_W) < 1e-6 else '❌ 불일치'})")
        print(f"        인출 {withdraw:.4f} vs {actual_withdraw:.4f} "
              f"({'✅ 일치' if abs(withdraw - actual_withdraw) < 1e-6 else '❌ 불일치'})")

        W_prev = W_after_withdraw

    print("\n" + "=" * 70)
    print("검증 완료")
    print("=" * 70)
