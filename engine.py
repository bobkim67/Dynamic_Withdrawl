"""
Dynamic Withdrawal Optimization Engine - Step 1: Path Generation
=================================================================
새 최적화 엔진의 Path 생성 모듈
- Rolling Historical Window
- Block Bootstrap
- 인출 시뮬레이션
- 전략 평가 집계
- Pareto Frontier
"""

import numpy as np
import pandas as pd
import pickle
from typing import List
from withdrawal_backtest import DataPreprocessor, PORTFOLIOS

try:
    from tqdm import tqdm
except ImportError:
    # tqdm이 없으면 더미 함수 사용
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# Path 생성을 위한 데이터 변환 함수
# ============================================================================

def daily_to_monthly_returns(daily_returns: pd.Series) -> np.ndarray:
    """
    일별 수익률을 월간 수익률로 변환

    각 월의 일별 수익률을 누적 곱하여 월간 수익률 계산:
    monthly_return = (1+r1) * (1+r2) * ... * (1+rn) - 1

    Parameters
    ----------
    daily_returns : pd.Series
        일별 수익률 시계열

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
                                 W0: float = 100.0,
                                 vol_adj: bool = False,
                                 sigma_target: float = 0.0,
                                 vol_lookback: int = 12,
                                 vol_floor: float = 0.5,
                                 vol_cap: float = 2.0,
                                 debug: bool = False) -> dict:
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
    vol_adj : bool
        변동성 조정 ON/OFF (True일 때 sigma_target 필수)
    sigma_target : float
        목표 변동성 (연율). Historical→전체 표본 σ, GBM→생성 파라미터 σ
    vol_lookback : int
        실현 변동성 추정 window (월, 기본 12)
    vol_floor : float
        밴드 축소 하한 배수 (기본 0.5)
    vol_cap : float
        밴드 확대 상한 배수 (기본 2.0)
    debug : bool
        True일 때 중간 계산 과정을 DataFrame으로 반환

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
        'debug_df': pd.DataFrame - (debug=True일 때만) 월별 상세 계산 과정
    """
    T = len(path_returns)

    # 결과 저장 배열 초기화
    W_series = np.zeros(T + 1)  # W_0부터 W_T까지
    withdraw_series = np.zeros(T)

    # 초기 자산
    W_series[0] = W0

    # 초기 인출액 (이후 동적으로 변동)
    target_ratio = init_wr / 12               # 목표 W/NAV 월비율
    prev_withdraw = W0 * target_ratio         # 이전 달 인출액 (초기값)

    # Guardrail 경계 (W/NAV 월비율 기준)
    upper_ratio = target_ratio * (1 + band)   # 밴드 상한
    lower_ratio = target_ratio * (1 - band)   # 밴드 하한

    # Debug용 중간 계산 기록
    if debug:
        debug_records = []

    # 매월 시뮬레이션 — Annuity Due (월초 인출 → 잔액에 수익률 적용)
    # W_series[t]는 t월 월초 NAV (= 전월 잔액에 전월 수익률 적용된 값)
    for t in range(T):
        # Step 1: 월초 NAV (인출 전)
        W_t = W_series[t]  # 이미 전월 수익률이 적용된 상태

        # Step 2: 인출 시도액 결정
        base = prev_withdraw

        # Step 3: 수익률 기반 보정 (adj_on=True일 때)
        if adj_on and t >= lookback:
            if lookback == 1:
                prev_return = path_returns[t - 1]
            else:
                prev_return = (1 + path_returns[t - lookback:t]).prod() - 1

            if prev_return > thr_up:
                withdraw_adj = base * (1 + adj_up)
            elif prev_return < thr_dn:
                withdraw_adj = base * (1 + adj_dn)
            else:
                withdraw_adj = base
        else:
            withdraw_adj = base

        # Step 3.5: 변동성 밴드 조정 (vol_adj=True일 때)
        if vol_adj and sigma_target > 0 and t >= vol_lookback:
            recent = path_returns[t - vol_lookback + 1 : t + 1]
            sigma_realized = np.std(recent, ddof=1) * np.sqrt(12)
            sigma_ratio = np.clip(sigma_realized / sigma_target, vol_floor, vol_cap)
            adj_upper = target_ratio * (1 + band * sigma_ratio)
            adj_lower = target_ratio * (1 - band * sigma_ratio)
        else:
            adj_upper = upper_ratio
            adj_lower = lower_ratio

        # Step 4: Guardrail — W/NAV 비율 밴드 적용
        cap_applied = 'None'
        if W_t > 0:
            ratio = withdraw_adj / W_t

            if ratio > adj_upper:
                withdraw_final = adj_upper * W_t
                cap_applied = 'Upper'
            elif ratio < adj_lower:
                withdraw_final = adj_lower * W_t
                cap_applied = 'Lower'
            else:
                withdraw_final = withdraw_adj
        else:
            ratio = 0
            withdraw_final = 0
            cap_applied = 'Ruin'

        # Step 5: 인출 실행 → 잔액에 수익률 적용
        W_after_wd = max(W_t - withdraw_final, 0)
        W_next = W_after_wd * (1 + path_returns[t])  # 잔액에 당월 수익률

        prev_withdraw = withdraw_final

        # W_series[t+1] = 다음 월초 NAV (수익률 반영 완료)
        W_series[t + 1] = W_next
        withdraw_series[t] = withdraw_final

        if debug:
            cum_withdraw_t = np.sum(withdraw_series[:t+1])
            debug_records.append({
                'Month': t,
                'Monthly_Return': path_returns[t],
                'NAV_Before_Withdrawal': W_t,
                'Prev_Withdrawal': base,
                'Adj_Withdrawal': withdraw_adj,
                'W_NAV_Ratio': ratio,
                'Upper_Ratio': adj_upper,
                'Lower_Ratio': adj_lower,
                'Cap_Applied': cap_applied,
                'Final_Withdrawal': withdraw_final,
                'NAV_After_Withdrawal': W_after_wd,
                'NAV_After_Return': W_next,
                'Cum_Withdrawal': cum_withdraw_t
            })

    # 집계 지표 계산
    cum_withdraw = np.sum(withdraw_series)
    terminal_nav = W_series[-1]
    ruin_flag = np.any(W_series <= 0)
    terminal_fail_flag = terminal_nav < beta * W0
    success = (not ruin_flag) and (not terminal_fail_flag)

    result = {
        'W_series': W_series,
        'withdraw_series': withdraw_series,
        'cum_withdraw': cum_withdraw,
        'terminal_nav': terminal_nav,
        'ruin_flag': ruin_flag,
        'terminal_fail_flag': terminal_fail_flag,
        'success': success
    }

    # Debug DataFrame 추가
    if debug:
        result['debug_df'] = pd.DataFrame(debug_records)

    return result


# ============================================================================
# 전략 평가 집계 함수
# ============================================================================

def evaluate_strategy(paths: List[np.ndarray],
                      init_wr: float,
                      band: float,
                      adj_on: bool = False,
                      lookback: int = 1,
                      thr_up: float = 0.03,
                      thr_dn: float = -0.03,
                      adj_up: float = 0.05,
                      adj_dn: float = -0.05,
                      beta: float = 0.5,
                      W0: float = 100.0,
                      vol_adj: bool = False,
                      sigma_target: float = 0.0,
                      vol_lookback: int = 12,
                      vol_floor: float = 0.5,
                      vol_cap: float = 2.0) -> dict:
    """
    여러 path에 대해 인출 전략을 시뮬레이션하고 결과를 집계

    Parameters
    ----------
    paths : List[np.ndarray]
        월간 수익률 path들의 리스트
    (나머지 파라미터는 simulate_withdrawal_on_path와 동일)

    Returns
    -------
    result : dict
        파라미터 기록 + 집계 메트릭
    """
    n_paths = len(paths)

    # 각 path별 결과 저장
    success_list = []
    cum_withdraw_list = []
    ruin_list = []
    terminal_fail_list = []
    terminal_nav_list = []
    cv_list = []
    worst_cut_list = []
    all_withdrawals = []  # 모든 path의 모든 월 인출액

    # 각 path에 대해 시뮬레이션
    for path in paths:
        result = simulate_withdrawal_on_path(
            path_returns=path,
            init_wr=init_wr,
            band=band,
            adj_on=adj_on,
            lookback=lookback,
            thr_up=thr_up,
            thr_dn=thr_dn,
            adj_up=adj_up,
            adj_dn=adj_dn,
            beta=beta,
            W0=W0,
            vol_adj=vol_adj,
            sigma_target=sigma_target,
            vol_lookback=vol_lookback,
            vol_floor=vol_floor,
            vol_cap=vol_cap,
            debug=False
        )

        # 기본 지표
        success_list.append(1 if result['success'] else 0)
        cum_withdraw_list.append(result['cum_withdraw'])
        ruin_list.append(1 if result['ruin_flag'] else 0)
        terminal_fail_list.append(1 if result['terminal_fail_flag'] else 0)
        terminal_nav_list.append(result['terminal_nav'])

        # 인출액 수집 (p5 계산용)
        all_withdrawals.extend(result['withdraw_series'])

        # CV 계산 (각 path별)
        withdraw_series = result['withdraw_series']
        nonzero_withdrawals = withdraw_series[withdraw_series > 0]
        if len(nonzero_withdrawals) > 1:
            cv = np.std(nonzero_withdrawals, ddof=1) / np.mean(nonzero_withdrawals)
            cv_list.append(cv)
        else:
            cv_list.append(0)

        # worst_cut 계산 (각 path별 최대 월간 인출 감소율)
        worst_cut = 0
        for t in range(1, len(withdraw_series)):
            if withdraw_series[t-1] > 0:
                cut_rate = (withdraw_series[t-1] - withdraw_series[t]) / withdraw_series[t-1]
                worst_cut = max(worst_cut, cut_rate)
        worst_cut_list.append(worst_cut)

    # 집계
    return {
        # 파라미터 기록
        'init_wr': init_wr,
        'band': band,
        'adj_on': adj_on,
        'lookback': lookback,
        'thr_up': thr_up,
        'thr_dn': thr_dn,
        'adj_up': adj_up,
        'adj_dn': adj_dn,

        # 프론티어 축
        'x_success_rate': np.mean(success_list),
        'y_cum_withdraw_median': np.median(cum_withdraw_list),
        'y_cum_withdraw_mean': np.mean(cum_withdraw_list),

        # 실패 분해
        'p_ruin': np.mean(ruin_list),
        'p_terminal_fail': np.mean(terminal_fail_list),
        'p_fail': 1 - np.mean(success_list),

        # 보조 지표 - 인출 변동성
        'cv_median': np.median(cv_list),
        'cv_mean': np.mean(cv_list),

        # 보조 지표 - worst cut
        'worst_cut_median': np.median(worst_cut_list),
        'worst_cut_mean': np.mean(worst_cut_list),

        # 보조 지표 - p5 월 인출액
        'p5_monthly_income': np.percentile(all_withdrawals, 5),

        # 기말잔액 (terminal NAV)
        'terminal_nav_median': np.median(terminal_nav_list),
        'terminal_nav_mean': np.mean(terminal_nav_list),

        # 메타
        'n_paths': n_paths,
    }


def pareto_frontier(results: List[dict],
                    x_col: str = 'x_success_rate',
                    y_col: str = 'y_cum_withdraw_median') -> List[dict]:
    """
    Pareto frontier 계산 (dominated 점 제거)

    A가 B를 지배: x_A >= x_B AND y_A >= y_B (하나 이상 strict inequality)

    Parameters
    ----------
    results : List[dict]
        evaluate_strategy 반환값들의 리스트
    x_col : str
        x축 컬럼명
    y_col : str
        y축 컬럼명

    Returns
    -------
    frontier : List[dict]
        Pareto frontier 위의 점들 (is_frontier=True 추가)
    """
    n = len(results)
    is_dominated = [False] * n

    # 각 점에 대해 지배 여부 확인
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            x_i, y_i = results[i][x_col], results[i][y_col]
            x_j, y_j = results[j][x_col], results[j][y_col]

            # j가 i를 지배하는지 확인
            if x_j >= x_i and y_j >= y_i and (x_j > x_i or y_j > y_i):
                is_dominated[i] = True
                break

    # Frontier 위의 점들만 선택
    frontier = []
    for i in range(n):
        if not is_dominated[i]:
            result_copy = results[i].copy()
            result_copy['is_frontier'] = True
            frontier.append(result_copy)

    return frontier


def select_optimal(results: List[dict],
                   x_min: float = 0.90,
                   q_ruin: float = 0.01) -> dict:
    """
    제약 조건 하에서 최적 전략 선택

    제약:
    - x_success_rate >= x_min
    - p_ruin <= q_ruin

    목표:
    - y_cum_withdraw_median 최대화

    Parameters
    ----------
    results : List[dict]
        evaluate_strategy 반환값들의 리스트
    x_min : float
        최소 성공확률 (기본 0.90)
    q_ruin : float
        최대 허용 파산확률 (기본 0.01)

    Returns
    -------
    optimal : dict or None
        최적 전략 (is_optimal=True 추가), 후보 없으면 None
    """
    # 제약 조건 필터링
    candidates = [
        r for r in results
        if r['x_success_rate'] >= x_min and r['p_ruin'] <= q_ruin
    ]

    if len(candidates) == 0:
        return None

    # y_cum_withdraw_median 최대인 것 선택
    optimal = max(candidates, key=lambda r: r['y_cum_withdraw_median'])
    optimal_copy = optimal.copy()
    optimal_copy['is_optimal'] = True

    return optimal_copy


# ============================================================================
# GBM 경로 생성 및 벡터화 시뮬레이션
# ============================================================================

def generate_paths_gbm(mu, sigma, n_paths=1000, T_months=120, seed=42):
    """
    GBM (Geometric Brownian Motion)으로 월간 수익률 경로 생성

    r_t = exp((mu/12 - 0.5*(sigma/sqrt(12))^2) + sigma/sqrt(12)*Z) - 1

    Parameters
    ----------
    mu : float
        연간 기대수익률 (예: 0.06 = 6%)
    sigma : float
        연간 변동성 (예: 0.10 = 10%)
    n_paths : int
        생성할 경로 수
    T_months : int
        경로 길이 (월 단위)
    seed : int
        랜덤 시드

    Returns
    -------
    paths : np.ndarray
        shape (n_paths, T_months) 월간 수익률 행렬
    """
    rng = np.random.RandomState(seed)

    mu_monthly = mu / 12
    sigma_monthly = sigma / np.sqrt(12)

    Z = rng.standard_normal((n_paths, T_months))
    log_returns = (mu_monthly - 0.5 * sigma_monthly**2) + sigma_monthly * Z
    monthly_returns = np.exp(log_returns) - 1

    return monthly_returns


def simulate_withdrawal_on_paths_vectorized(paths_matrix, init_wr, band, W0=100.0,
                                              vol_adj=False, sigma_target=0.0,
                                              vol_lookback=12, vol_floor=0.5, vol_cap=2.0):
    """
    벡터화된 인출 시뮬레이션 (다수 경로 동시 처리)

    기존 simulate_withdrawal_on_path와 동일 로직을 numpy 행렬 연산으로 벡터화.
    매월 순서: [수익률 적용 → base 인출 → (변동성 밴드 조정) → guardrail 캡 → 인출 실행]
    adj_on 미지원 (기존 grid_search에서도 adj_on=True 제거됨)

    Parameters
    ----------
    paths_matrix : np.ndarray
        shape (n_paths, T_months) 월간 수익률 행렬
    init_wr : float
        초기 연간 인출률
    band : float
        guardrail 밴드 (99.0이면 사실상 fixed)
    W0 : float
        초기 자산
    vol_adj : bool
        변동성 조정 ON/OFF
    sigma_target : float
        목표 변동성 (연율)
    vol_lookback : int
        실현 변동성 추정 window (월)
    vol_floor : float
        밴드 축소 하한 배수
    vol_cap : float
        밴드 확대 상한 배수

    Returns
    -------
    result : dict
        W_terminal: (n_paths,) 최종 NAV
        cum_withdraw: (n_paths,) 경로별 누적 인출액
        withdraw_matrix: (n_paths, T_months) 월별 인출액
        ruin_flags: (n_paths,) 파산 여부 (bool)
    """
    n_paths, T_months = paths_matrix.shape

    W = np.full(n_paths, W0, dtype=np.float64)
    withdraw_matrix = np.zeros((n_paths, T_months), dtype=np.float64)
    ever_ruined = np.zeros(n_paths, dtype=bool)

    # W/NAV 월비율 기준 밴드
    target_ratio = init_wr / 12
    upper_ratio = target_ratio * (1 + band)
    lower_ratio = target_ratio * (1 - band)

    # 이전 달 인출액 (경로별, 동적 변동)
    prev_withdraw = np.full(n_paths, W0 * target_ratio, dtype=np.float64)

    for t in range(T_months):
        # Step 1: 수익률 적용
        W = W * (1 + paths_matrix[:, t])

        # Step 2: 인출 시도액 = 이전 달 인출액 유지
        withdraw = prev_withdraw.copy()

        # Step 3.5: 변동성 밴드 조정 (vol_adj=True일 때)
        if vol_adj and sigma_target > 0 and t >= vol_lookback:
            window = paths_matrix[:, t - vol_lookback + 1 : t + 1]  # (n_paths, vol_lookback)
            sigma_realized = np.std(window, axis=1, ddof=1) * np.sqrt(12)  # (n_paths,)
            sigma_ratio = np.clip(sigma_realized / sigma_target, vol_floor, vol_cap)
            cur_upper = target_ratio * (1 + band * sigma_ratio)  # (n_paths,)
            cur_lower = target_ratio * (1 - band * sigma_ratio)  # (n_paths,)
        else:
            cur_upper = upper_ratio  # scalar broadcast
            cur_lower = lower_ratio

        # Step 4: Guardrail — W/NAV 비율 밴드 적용
        alive = W > 0
        safe_W = np.where(alive, W, 1.0)
        ratio = np.where(alive, withdraw / safe_W, 0.0)

        # Upper: 비율 과다 → 인출 축소
        upper_mask = alive & (ratio > cur_upper)
        withdraw = np.where(upper_mask, cur_upper * W, withdraw)

        # Lower: 비율 과소 → 인출 확대
        lower_mask = alive & (ratio < cur_lower)
        withdraw = np.where(lower_mask, cur_lower * W, withdraw)

        # 파산 경로: 인출 0
        withdraw = np.where(alive, withdraw, 0.0)

        # Step 5: 인출 실행
        W = W - withdraw
        W = np.maximum(W, 0.0)

        # 이전 인출액 갱신
        prev_withdraw = withdraw.copy()

        withdraw_matrix[:, t] = withdraw
        ever_ruined |= (W <= 0)

    return {
        'W_terminal': W.copy(),
        'cum_withdraw': withdraw_matrix.sum(axis=1),
        'withdraw_matrix': withdraw_matrix,
        'ruin_flags': ever_ruined,
    }


def evaluate_gbm_strategy(sim_result, betas=(0.1, 0.25, 0.5, 0.75, 1.0), W0=100.0):
    """
    GBM 시뮬레이션 결과를 다중 beta로 평가 (시뮬레이션 1회로 5개 beta 처리)

    Parameters
    ----------
    sim_result : dict
        simulate_withdrawal_on_paths_vectorized 반환값
    betas : tuple/list
        terminal 실패 기준 beta 목록
    W0 : float
        초기 자산

    Returns
    -------
    results : list[dict]
        beta별 집계 메트릭
    """
    W_terminal = sim_result['W_terminal']
    cum_withdraw = sim_result['cum_withdraw']
    ruin_flags = sim_result['ruin_flags']
    withdraw_matrix = sim_result['withdraw_matrix']
    n_paths = len(W_terminal)

    # CV 계산 (벡터화: 전체 row 기준)
    row_means = withdraw_matrix.mean(axis=1)
    row_stds = withdraw_matrix.std(axis=1, ddof=1)
    cv_array = np.where(row_means > 0, row_stds / row_means, 0.0)

    results = []
    for beta in betas:
        terminal_fail = W_terminal < beta * W0
        success = ~ruin_flags & ~terminal_fail

        results.append({
            'x_success_rate': float(success.mean()),
            'y_cum_withdraw_median': float(np.median(cum_withdraw)),
            'y_cum_withdraw_mean': float(np.mean(cum_withdraw)),
            'p_ruin': float(ruin_flags.mean()),
            'p_terminal_fail': float(terminal_fail.mean()),
            'p_fail': float(1 - success.mean()),
            'cv_median': float(np.median(cv_array)),
            'terminal_nav_median': float(np.median(W_terminal)),
            'terminal_nav_mean': float(np.mean(W_terminal)),
            'n_paths': n_paths,
        })

    return results


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
    monthly_returns = daily_to_monthly_returns(daily_returns)

    print(f"[OK] 월간 수익률 변환 완료")
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

    print(f"[OK] Rolling path 생성 완료")
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

    print(f"[OK] Bootstrap path 생성 완료")
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
    print(f"  성공 여부: {'[OK] 성공' if result1['success'] else '[FAIL] 실패'}")

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
    print(f"  성공 여부: {'[OK] 성공' if result2['success'] else '[FAIL] 실패'}")

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
    print(f"  성공 여부: {'[OK] 성공' if result3['success'] else '[FAIL] 실패'}")

    print(f"\n테스트 1 대비 차이:")
    print(f"  총 인출액 차이: {result3['cum_withdraw'] - result1['cum_withdraw']:.2f} "
          f"({(result3['cum_withdraw'] / result1['cum_withdraw'] - 1)*100:+.2f}%)")
    print(f"  최종 NAV 차이: {result3['terminal_nav'] - result1['terminal_nav']:.2f} "
          f"({(result3['terminal_nav'] / result1['terminal_nav'] - 1)*100:+.2f}%)")

    # ========================================
    # 테스트 4: 인출 순서 검증 (짧은 path)
    # ========================================

    print("\n[테스트 4] 인출 순서 검증 (Step-by-step) - W/NAV 비율 밴드 방식")
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
    target_r = 0.06 / 12
    upper_r = target_r * 1.20
    lower_r = target_r * 0.80
    print(f"Guardrail: W/NAV 비율 밴드 {lower_r:.6f} ~ {upper_r:.6f}")

    print("\n월별 계산 과정:")
    W_prev = 100.0
    prev_wd = 100.0 * target_r   # 초기 인출액

    for t in range(3):
        print(f"\n--- {t}개월째 ---")
        r_t = short_path[t]

        # Step 1: 수익률 적용
        W_after_return = W_prev * (1 + r_t)
        print(f"  1) 수익률 적용: {W_prev:.4f} x (1 + {r_t:.4f}) = {W_after_return:.4f}")

        # Step 2: 인출 시도액 = 이전 인출액
        print(f"  2) 이전 인출액(시도): {prev_wd:.4f}")

        # Step 4: W/NAV 비율 밴드
        ratio = prev_wd / W_after_return
        print(f"  3) W/NAV 비율: {prev_wd:.4f} / {W_after_return:.4f} = {ratio:.6f}")
        print(f"     밴드: [{lower_r:.6f}, {upper_r:.6f}]")

        if ratio > upper_r:
            withdraw = upper_r * W_after_return
            print(f"     [WARN] 상한 초과 -> 축소: {upper_r:.6f} x {W_after_return:.4f} = {withdraw:.4f}")
        elif ratio < lower_r:
            withdraw = lower_r * W_after_return
            print(f"     [WARN] 하한 미달 -> 확대: {lower_r:.6f} x {W_after_return:.4f} = {withdraw:.4f}")
        else:
            withdraw = prev_wd
            print(f"     [OK] 밴드 내 -> 이전 인출액 유지: {withdraw:.4f}")

        # Step 5: 인출 실행
        W_after_withdraw = W_after_return - withdraw
        print(f"  4) 인출 후 NAV: {W_after_return:.4f} - {withdraw:.4f} = {W_after_withdraw:.4f}")

        # 실제 결과와 비교
        actual_W = result4['W_series'][t + 1]
        actual_withdraw = result4['withdraw_series'][t]

        print(f"\n  검증: NAV {W_after_withdraw:.4f} vs {actual_W:.4f} "
              f"({'[OK] 일치' if abs(W_after_withdraw - actual_W) < 1e-6 else '[FAIL] 불일치'})")
        print(f"        인출 {withdraw:.4f} vs {actual_withdraw:.4f} "
              f"({'[OK] 일치' if abs(withdraw - actual_withdraw) < 1e-6 else '[FAIL] 불일치'})")

        prev_wd = withdraw
        W_prev = W_after_withdraw

    # ========================================
    # Excel 내보내기
    # ========================================

    print("\n" + "=" * 70)
    print("Excel 내보내기")
    print("=" * 70)

    # 테스트 1, 2, 3을 debug=True로 재실행
    result1_debug = simulate_withdrawal_on_path(
        path_returns=test_path,
        init_wr=0.05,
        band=0.15,
        adj_on=False,
        W0=100.0,
        debug=True
    )

    result2_debug = simulate_withdrawal_on_path(
        path_returns=test_path,
        init_wr=0.15,
        band=0.15,
        adj_on=False,
        W0=100.0,
        debug=True
    )

    result3_debug = simulate_withdrawal_on_path(
        path_returns=test_path,
        init_wr=0.05,
        band=0.15,
        adj_on=True,
        lookback=1,
        thr_up=0.03,
        thr_dn=-0.03,
        adj_up=0.05,
        adj_dn=-0.05,
        W0=100.0,
        debug=True
    )

    # Excel 파일로 저장
    excel_filename = 'engine_verification.xlsx'

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        result1_debug['debug_df'].to_excel(writer, sheet_name='Test1_Base', index=False)
        result2_debug['debug_df'].to_excel(writer, sheet_name='Test2_HighWR', index=False)
        result3_debug['debug_df'].to_excel(writer, sheet_name='Test3_AdjOn', index=False)

    print(f"\n[OK] Excel 파일 생성 완료: {excel_filename}")
    print(f"  시트 1: Test1_Base (init_wr=0.05, band=0.15, adj_on=False)")
    print(f"  시트 2: Test2_HighWR (init_wr=0.15, band=0.15, adj_on=False)")
    print(f"  시트 3: Test3_AdjOn (init_wr=0.05, band=0.15, adj_on=True)")
    print(f"\n파일 크기: {len(result1_debug['debug_df'])}행 × 3시트")

    # ========================================
    # 7. 전략 평가 집계 및 Pareto Frontier 테스트
    # ========================================

    print("\n" + "=" * 70)
    print("전략 평가 집계 및 Pareto Frontier 검증")
    print("=" * 70)

    # 소규모 그리드 서치
    print("\n[소규모 그리드 서치]")
    print("-" * 70)

    # 파라미터 그리드
    init_wr_grid = [0.04, 0.05, 0.06, 0.07, 0.08]
    band_grid = [0.10, 0.15, 0.20]
    adj_on_grid = [False]  # 보정 없이 기본만 테스트

    # 전체 조합 수
    total_combinations = len(init_wr_grid) * len(band_grid) * len(adj_on_grid)

    print(f"파라미터 그리드:")
    print(f"  init_wr: {init_wr_grid}")
    print(f"  band: {band_grid}")
    print(f"  adj_on: {adj_on_grid}")
    print(f"  총 조합: {total_combinations}개")
    print(f"  사용 paths: {len(rolling_paths)}개 (Port_5.0% rolling)")

    # 그리드 서치 실행
    grid_results = []

    print("\n그리드 서치 실행 중...")
    for init_wr in tqdm(init_wr_grid, desc="init_wr"):
        for band in band_grid:
            for adj_on in adj_on_grid:
                result = evaluate_strategy(
                    paths=rolling_paths,
                    init_wr=init_wr,
                    band=band,
                    adj_on=adj_on,
                    W0=100.0
                )
                grid_results.append(result)

    print(f"\n[OK] 그리드 서치 완료: {len(grid_results)}개 조합 평가")

    # 결과 테이블 출력
    print("\n" + "=" * 70)
    print("그리드 서치 결과 테이블")
    print("=" * 70)

    # DataFrame으로 변환
    df_results = pd.DataFrame([
        {
            'init_wr': r['init_wr'],
            'band': r['band'],
            'success_rate': r['x_success_rate'],
            'cum_withdraw': r['y_cum_withdraw_median'],
            'p_ruin': r['p_ruin'],
            'cv_median': r['cv_median'],
        }
        for r in grid_results
    ])

    # 테이블 출력
    print(df_results.to_string(index=False, float_format='%.4f'))

    # Pareto frontier 계산
    print("\n" + "=" * 70)
    print("Pareto Frontier")
    print("=" * 70)

    frontier_results = pareto_frontier(grid_results)

    print(f"\n[OK] Frontier 위의 점: {len(frontier_results)}/{len(grid_results)}개")
    print("\nFrontier 포인트:")
    df_frontier = pd.DataFrame([
        {
            'init_wr': r['init_wr'],
            'band': r['band'],
            'success_rate': r['x_success_rate'],
            'cum_withdraw': r['y_cum_withdraw_median'],
            'p_ruin': r['p_ruin'],
        }
        for r in frontier_results
    ])
    print(df_frontier.to_string(index=False, float_format='%.4f'))

    # 최적 전략 선택
    print("\n" + "=" * 70)
    print("최적 전략 선택")
    print("=" * 70)

    optimal = select_optimal(grid_results, x_min=0.90, q_ruin=0.01)

    if optimal:
        print(f"\n[OK] 최적 전략 발견 (제약: 성공률 ≥ 90%, 파산률 ≤ 1%)")
        print(f"\n파라미터:")
        print(f"  init_wr: {optimal['init_wr']:.4f} ({optimal['init_wr']*100:.1f}%)")
        print(f"  band: {optimal['band']:.4f} (±{optimal['band']*100:.0f}%)")
        print(f"  adj_on: {optimal['adj_on']}")
        print(f"\n성과:")
        print(f"  성공확률: {optimal['x_success_rate']:.4f} ({optimal['x_success_rate']*100:.2f}%)")
        print(f"  누적 인출액 (중앙값): {optimal['y_cum_withdraw_median']:.2f}")
        print(f"  파산확률: {optimal['p_ruin']:.4f} ({optimal['p_ruin']*100:.2f}%)")
        print(f"  Terminal 실패확률: {optimal['p_terminal_fail']:.4f} ({optimal['p_terminal_fail']*100:.2f}%)")
        print(f"  CV (중앙값): {optimal['cv_median']:.4f}")
        print(f"  Worst Cut (중앙값): {optimal['worst_cut_median']:.4f} ({optimal['worst_cut_median']*100:.2f}%)")
        print(f"  P5 월 인출액: {optimal['p5_monthly_income']:.4f}")
    else:
        print("\n[FAIL] 제약 조건을 만족하는 전략이 없습니다")

    # 결과 저장
    print("\n" + "=" * 70)
    print("결과 저장")
    print("=" * 70)

    with open('grid_test_results.pkl', 'wb') as f:
        pickle.dump(grid_results, f)

    print(f"\n[OK] 결과 저장 완료: grid_test_results.pkl ({len(grid_results)}개 조합)")

    # ========================================
    # 8. GBM 벡터화 함수 검증
    # ========================================

    print("\n" + "=" * 70)
    print("GBM 벡터화 함수 검증")
    print("=" * 70)

    # GBM 경로 생성 테스트
    print("\n[테스트 5] GBM 경로 생성")
    print("-" * 70)

    gbm_paths = generate_paths_gbm(mu=0.06, sigma=0.10, n_paths=1000, T_months=120, seed=42)
    print(f"[OK] GBM 경로 생성 완료: shape={gbm_paths.shape}")
    print(f"  월 평균 수익률: {gbm_paths.mean()*100:.3f}%")
    print(f"  월 표준편차: {gbm_paths.std()*100:.3f}%")
    ann_ret = (1 + gbm_paths.mean())**12 - 1
    ann_vol = gbm_paths.std() * np.sqrt(12)
    print(f"  연환산 수익률: {ann_ret*100:.2f}% (목표: 6.00%)")
    print(f"  연환산 변동성: {ann_vol*100:.2f}% (목표: 10.00%)")

    # 벡터화 vs Scalar 비교 테스트
    print("\n[테스트 6] 벡터화 vs Scalar 일치 확인")
    print("-" * 70)

    test_n_paths = 10
    test_T = 24
    test_paths_gbm = generate_paths_gbm(mu=0.07, sigma=0.12, n_paths=test_n_paths, T_months=test_T, seed=123)

    test_params = [
        {'init_wr': 0.05, 'band': 99.0},   # fixed
        {'init_wr': 0.08, 'band': 0.10},    # guardrail ±10%
        {'init_wr': 0.10, 'band': 0.15},    # guardrail ±15%
    ]

    all_match = True
    for params in test_params:
        init_wr_t = params['init_wr']
        band_t = params['band']

        # 벡터화 버전
        vec_result = simulate_withdrawal_on_paths_vectorized(test_paths_gbm, init_wr_t, band_t, W0=100.0)

        # Scalar 버전 (각 path별)
        max_diff_W = 0.0
        max_diff_cum = 0.0
        ruin_match_flag = True

        for i in range(test_n_paths):
            scalar_result = simulate_withdrawal_on_path(
                path_returns=test_paths_gbm[i],
                init_wr=init_wr_t, band=band_t, adj_on=False,
                beta=0.5, W0=100.0, debug=False
            )

            diff_W = abs(vec_result['W_terminal'][i] - scalar_result['terminal_nav'])
            diff_cum = abs(vec_result['cum_withdraw'][i] - scalar_result['cum_withdraw'])
            max_diff_W = max(max_diff_W, diff_W)
            max_diff_cum = max(max_diff_cum, diff_cum)

            if vec_result['ruin_flags'][i] != scalar_result['ruin_flag']:
                ruin_match_flag = False

        tol = 1e-8
        match = max_diff_W < tol and max_diff_cum < tol and ruin_match_flag
        status = "[OK]" if match else "[FAIL]"
        if not match:
            all_match = False

        band_label = "fixed" if band_t == 99.0 else f"\u00b1{band_t*100:.0f}%"
        print(f"  {status} init_wr={init_wr_t*100:.0f}%, band={band_label}: "
              f"max_diff(W)={max_diff_W:.2e}, max_diff(cum)={max_diff_cum:.2e}, "
              f"ruin_match={'OK' if ruin_match_flag else 'FAIL'}")

    print(f"\n{'[OK] 벡터화 검증 통과' if all_match else '[FAIL] 벡터화 검증 실패'}")

    # evaluate_gbm_strategy 테스트
    print("\n[테스트 7] evaluate_gbm_strategy 다중 beta 평가")
    print("-" * 70)

    gbm_test_paths_eval = generate_paths_gbm(mu=0.06, sigma=0.10, n_paths=500, T_months=120, seed=42)
    sim_eval = simulate_withdrawal_on_paths_vectorized(gbm_test_paths_eval, init_wr=0.05, band=0.10, W0=100.0)
    eval_results = evaluate_gbm_strategy(sim_eval, betas=[0.1, 0.5, 1.0], W0=100.0)

    for idx_eval, beta_val in enumerate([0.1, 0.5, 1.0]):
        r_eval = eval_results[idx_eval]
        print(f"  Beta={beta_val}: 성공률={r_eval['x_success_rate']*100:.1f}%, "
              f"파산률={r_eval['p_ruin']*100:.2f}%, "
              f"누적인출={r_eval['y_cum_withdraw_median']:.1f}")

    # ========================================
    # 9. vol_adj 검증
    # ========================================

    print("\n" + "=" * 70)
    print("Volatility-Adjusted Guardrail 검증")
    print("=" * 70)

    # 테스트 8: vol_adj=False → 기존 결과와 100% 일치
    print("\n[테스트 8] vol_adj=False → 기존 결과 동일 확인")
    print("-" * 70)

    result_no_vol = simulate_withdrawal_on_path(
        path_returns=test_path,
        init_wr=0.05, band=0.15, adj_on=False, W0=100.0,
        vol_adj=False, sigma_target=0.10,
    )
    nav_match = np.allclose(result_no_vol['W_series'], result1['W_series'], atol=1e-10)
    cum_match = abs(result_no_vol['cum_withdraw'] - result1['cum_withdraw']) < 1e-10
    print(f"  NAV 시계열 일치: {'[OK]' if nav_match else '[FAIL]'}")
    print(f"  누적인출 일치: {'[OK]' if cum_match else '[FAIL]'}")

    # 테스트 9: vol_adj=True → 결과가 달라지는지 확인
    print("\n[테스트 9] vol_adj=True → 변동성 조정 효과 확인")
    print("-" * 70)

    sigma_full = np.std(monthly_returns, ddof=1) * np.sqrt(12)
    print(f"  전체 표본 연환산 σ: {sigma_full*100:.2f}%")

    result_vol = simulate_withdrawal_on_path(
        path_returns=test_path,
        init_wr=0.05, band=0.15, adj_on=False, W0=100.0,
        vol_adj=True, sigma_target=sigma_full, vol_lookback=12,
    )
    nav_differs = not np.allclose(result_vol['W_series'], result1['W_series'], atol=1e-6)
    print(f"  vol_adj=True 결과가 기존과 다름: {'[OK]' if nav_differs else '[WARN] 동일 (σ_target ≈ σ_realized이면 가능)'}")
    print(f"  기존 누적인출: {result1['cum_withdraw']:.2f}")
    print(f"  vol_adj 누적인출: {result_vol['cum_withdraw']:.2f}")
    print(f"  기존 기말NAV: {result1['terminal_nav']:.2f}")
    print(f"  vol_adj 기말NAV: {result_vol['terminal_nav']:.2f}")

    # 테스트 10: 벡터화 vs scalar vol_adj 일치
    print("\n[테스트 10] 벡터화 vs Scalar vol_adj 일치 확인")
    print("-" * 70)

    test_paths_vol = generate_paths_gbm(mu=0.07, sigma=0.12, n_paths=10, T_months=24, seed=123)
    sigma_t = 0.12

    vol_test_params = [
        {'init_wr': 0.05, 'band': 0.10},
        {'init_wr': 0.08, 'band': 0.15},
    ]

    all_vol_match = True
    for params in vol_test_params:
        iw, bd = params['init_wr'], params['band']

        # 벡터화
        vec_r = simulate_withdrawal_on_paths_vectorized(
            test_paths_vol, iw, bd, W0=100.0,
            vol_adj=True, sigma_target=sigma_t, vol_lookback=12,
        )

        max_d_W, max_d_cum = 0.0, 0.0
        for i in range(10):
            sc_r = simulate_withdrawal_on_path(
                path_returns=test_paths_vol[i], init_wr=iw, band=bd,
                adj_on=False, beta=0.5, W0=100.0,
                vol_adj=True, sigma_target=sigma_t, vol_lookback=12,
            )
            max_d_W = max(max_d_W, abs(vec_r['W_terminal'][i] - sc_r['terminal_nav']))
            max_d_cum = max(max_d_cum, abs(vec_r['cum_withdraw'][i] - sc_r['cum_withdraw']))

        match = max_d_W < 1e-8 and max_d_cum < 1e-8
        if not match:
            all_vol_match = False
        status = "[OK]" if match else "[FAIL]"
        print(f"  {status} init_wr={iw*100:.0f}%, band=±{bd*100:.0f}%: "
              f"max_diff(W)={max_d_W:.2e}, max_diff(cum)={max_d_cum:.2e}")

    print(f"\n{'[OK] vol_adj 벡터화 검증 통과' if all_vol_match else '[FAIL] vol_adj 벡터화 검증 실패'}")

    print("\n" + "=" * 70)
    print("검증 완료")
    print("=" * 70)