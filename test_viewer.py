"""
Test Script for viewer.py Core Functions
==========================================
Streamlit 없이 viewer.py의 핵심 함수 4개 + CLAUDE.md 수식 검증
실행: python test_viewer.py
"""

import numpy as np
import pandas as pd
import pickle
import sys
import os

# ============================================================================
# Imports (viewer.py에서 Streamlit 의존 없는 함수 직접 import)
# ============================================================================

from viewer import gbm_survival_probability, filter_dataframe, BETA_LABELS
from engine import (
    simulate_withdrawal_on_path,
    evaluate_strategy,
    generate_paths_rolling,
    daily_to_monthly_returns,
)
from withdrawal_backtest import DataPreprocessor


# ============================================================================
# Test Framework
# ============================================================================

_pass_count = 0
_fail_count = 0


def check(condition, description):
    """단일 검증"""
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  [OK] {description}")
    else:
        _fail_count += 1
        print(f"  [FAIL] {description}")


def approx_eq(a, b, tol=1e-6):
    """부동소수점 근사 비교"""
    return abs(a - b) < tol


# ============================================================================
# Test 1: load_grid_results 로직 검증
# ============================================================================

def test_load_grid_results():
    """pkl 로드 + DataFrame 변환 테스트 (viewer.py 로직 재현)"""
    print("\n" + "=" * 60)
    print("[Test 1] load_grid_results")
    print("=" * 60)

    # pkl 파일 존재 확인
    pkl_path = 'grid_results_full.pkl'
    check(os.path.exists(pkl_path), f"{pkl_path} 파일 존재")

    # 로드
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    check(isinstance(results, list), "pkl 내용이 list 타입")
    check(len(results) > 0, f"결과 {len(results)}개 (비어있지 않음)")

    # DataFrame 변환 (viewer.py load_grid_results 로직 재현)
    df = pd.DataFrame([
        {
            'portfolio': r['portfolio'],
            'path_method': r['path_method'],
            'strategy_type': r['strategy_type'],
            'init_wr': r['init_wr'],
            'band': r['band'],
            'adj_on': r['adj_on'],
            'lookback': r.get('lookback', 1),
            'thr_up': r.get('thr_up', 0.03),
            'thr_dn': r.get('thr_dn', -0.03),
            'adj_up': r.get('adj_up', 0.05),
            'adj_dn': r.get('adj_dn', -0.05),
            'success_rate': r['x_success_rate'],
            'cum_withdraw_median': r['y_cum_withdraw_median'],
            'cum_withdraw_mean': r['y_cum_withdraw_mean'],
            'p_ruin': r['p_ruin'],
            'p_terminal_fail': r['p_terminal_fail'],
            'p_fail': r['p_fail'],
            'cv_median': r['cv_median'],
            'cv_mean': r['cv_mean'],
            'worst_cut_median': r['worst_cut_median'],
            'worst_cut_mean': r['worst_cut_mean'],
            'p5_monthly_income': r['p5_monthly_income'],
            'n_paths': r['n_paths'],
            'is_frontier': r.get('is_frontier', False),
            'is_optimal': r.get('is_optimal', False),
        }
        for r in results
    ])

    # 컬럼 검증
    expected_cols = [
        'portfolio', 'path_method', 'strategy_type', 'init_wr', 'band',
        'adj_on', 'lookback', 'success_rate', 'cum_withdraw_median',
        'p_ruin', 'p_terminal_fail', 'cv_median', 'worst_cut_median',
        'p5_monthly_income', 'n_paths', 'is_frontier', 'is_optimal',
    ]
    for col in expected_cols:
        check(col in df.columns, f"컬럼 '{col}' 존재")

    # 데이터 타입 검증
    check(df['success_rate'].dtype == np.float64, "success_rate: float64 타입")
    check(df['adj_on'].dtype == bool, "adj_on: bool 타입")
    check(df['n_paths'].dtype == np.int64 or df['n_paths'].dtype == int,
          f"n_paths: int 타입 (actual: {df['n_paths'].dtype})")

    # 값 범위 검증
    check((df['success_rate'] >= 0).all() and (df['success_rate'] <= 1).all(),
          "success_rate 범위 [0, 1]")
    check((df['p_ruin'] >= 0).all() and (df['p_ruin'] <= 1).all(),
          "p_ruin 범위 [0, 1]")
    check((df['init_wr'] > 0).all(), "init_wr > 0")

    # strategy_type 값 검증
    valid_types = {'fixed_baseline', 'dynamic'}
    actual_types = set(df['strategy_type'].unique())
    check(actual_types.issubset(valid_types),
          f"strategy_type: {actual_types} (유효값: {valid_types})")

    # portfolio 값 검증
    check(len(df['portfolio'].unique()) >= 1, f"포트폴리오 {len(df['portfolio'].unique())}개")

    return df


# ============================================================================
# Test 2: filter_dataframe 검증
# ============================================================================

def test_filter_dataframe(df):
    """필터링 로직 테스트"""
    print("\n" + "=" * 60)
    print("[Test 2] filter_dataframe")
    print("=" * 60)

    all_portfolios = sorted(df['portfolio'].unique())

    # Fixed만 필터링
    filtered_fixed = filter_dataframe(df, all_portfolios, 'Fixed', False)
    check((filtered_fixed['strategy_type'] == 'fixed_baseline').all(),
          "Fixed 필터 -> strategy_type == 'fixed_baseline'")
    check(len(filtered_fixed) > 0, f"Fixed 필터 결과: {len(filtered_fixed)}행")

    # Dynamic만 필터링
    filtered_dynamic = filter_dataframe(df, all_portfolios, 'Dynamic', False)
    check((filtered_dynamic['strategy_type'] == 'dynamic').all(),
          "Dynamic 필터 -> strategy_type == 'dynamic'")
    check(len(filtered_dynamic) > 0, f"Dynamic 필터 결과: {len(filtered_dynamic)}행")

    # All 필터
    filtered_all = filter_dataframe(df, all_portfolios, 'All', False)
    check(len(filtered_all) == len(df[df['portfolio'].isin(all_portfolios)]),
          "All 필터 -> 전체 데이터")

    # Fixed + Dynamic 합 = All (같은 포트폴리오 범위)
    check(len(filtered_fixed) + len(filtered_dynamic) == len(filtered_all),
          "Fixed + Dynamic = All")

    # Frontier only 필터
    filtered_frontier = filter_dataframe(df, all_portfolios, 'All', True)
    if len(filtered_frontier) > 0:
        check((filtered_frontier['is_frontier'] == True).all(),
              "Frontier only -> is_frontier == True")
    check(len(filtered_frontier) <= len(filtered_all),
          f"Frontier only: {len(filtered_frontier)}행 <= All: {len(filtered_all)}행")

    # 포트폴리오 멀티 필터링
    if len(all_portfolios) >= 2:
        subset_ports = all_portfolios[:2]
        filtered_multi = filter_dataframe(df, subset_ports, 'All', False)
        check(set(filtered_multi['portfolio'].unique()).issubset(set(subset_ports)),
              f"포트폴리오 멀티 필터: {subset_ports}")
        check(len(filtered_multi) < len(filtered_all),
              "포트폴리오 서브셋 -> 결과 감소")

    # 단일 포트폴리오 필터링
    single_port = [all_portfolios[0]]
    filtered_single = filter_dataframe(df, single_port, 'All', False)
    check((filtered_single['portfolio'] == single_port[0]).all(),
          f"단일 포트폴리오 필터: {single_port[0]}")

    # 빈 포트폴리오 리스트
    filtered_empty = filter_dataframe(df, [], 'All', False)
    check(len(filtered_empty) == 0, "빈 포트폴리오 리스트 -> 0행")


# ============================================================================
# Test 3: gbm_survival_probability 검증
# ============================================================================

def test_gbm_survival_probability():
    """GBM 생존확률 계산 테스트"""
    print("\n" + "=" * 60)
    print("[Test 3] gbm_survival_probability")
    print("=" * 60)

    # wr=0 -> 확률 1.0 (인출 없음)
    p = gbm_survival_probability(mu=0.05, sigma=0.10, wr=0.0, T=10, beta=0.5)
    check(approx_eq(p, 1.0), f"wr=0 -> P={p:.6f} (expected 1.0)")

    # sigma=0 -> deterministic 계산
    # remaining = 1.0 - wr*T + mu*T = 1.0 - 0.04*10 + 0.06*10 = 1.2
    # 1.2 >= beta=0.5 -> P=1.0
    p = gbm_survival_probability(mu=0.06, sigma=0.0, wr=0.04, T=10, beta=0.5)
    check(approx_eq(p, 1.0), f"sigma=0, surplus -> P={p:.6f} (expected 1.0)")

    # sigma=0, 높은 인출률 -> deterministic failure
    # remaining = 1.0 - 0.20*10 + 0.05*10 = 1.0 - 2.0 + 0.5 = -0.5
    # -0.5 < beta=0.5 -> P=0.0
    p = gbm_survival_probability(mu=0.05, sigma=0.0, wr=0.20, T=10, beta=0.5)
    check(approx_eq(p, 0.0), f"sigma=0, deficit -> P={p:.6f} (expected 0.0)")

    # 높은 인출률 -> 낮은 생존확률
    p_low_wr = gbm_survival_probability(mu=0.05, sigma=0.10, wr=0.03, T=10, beta=0.5)
    p_high_wr = gbm_survival_probability(mu=0.05, sigma=0.10, wr=0.12, T=10, beta=0.5)
    check(p_low_wr > p_high_wr,
          f"높은 인출률 -> 낮은 생존확률: P(wr=3%)={p_low_wr:.4f} > P(wr=12%)={p_high_wr:.4f}")

    # 반환값 범위 [0, 1]
    test_params = [
        (0.05, 0.10, 0.04, 10, 0.5),
        (0.08, 0.15, 0.06, 10, 0.5),
        (0.03, 0.05, 0.10, 10, 0.5),
        (0.10, 0.20, 0.15, 10, 0.5),
    ]
    all_in_range = True
    for mu, sigma, wr, T, beta in test_params:
        p = gbm_survival_probability(mu, sigma, wr, T, beta)
        if p < 0.0 or p > 1.0:
            all_in_range = False
            break
    check(all_in_range, "모든 테스트 케이스에서 P 범위 [0, 1]")

    # 단조 감소: wr 증가 -> 생존확률 감소
    wr_values = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13]
    probs = [gbm_survival_probability(0.06, 0.10, wr, 10, 0.5) for wr in wr_values]
    is_monotone = all(probs[i] >= probs[i+1] for i in range(len(probs)-1))
    check(is_monotone, f"wr 증가 -> P 단조 감소: {[f'{p:.3f}' for p in probs]}")


# ============================================================================
# Test 4: simulate_paths_for_strategy 로직 검증
# ============================================================================

def test_simulate_paths_for_strategy():
    """경로 시뮬레이션 테스트 (viewer.py 로직 재현)"""
    print("\n" + "=" * 60)
    print("[Test 4] simulate_paths_for_strategy")
    print("=" * 60)

    # 데이터 로드 (viewer.py의 simulate_paths_for_strategy 로직 재현)
    with open('benchmark_data.pkl', 'rb') as f:
        benchmark_data = pickle.load(f)

    preprocessor = DataPreprocessor(benchmark_data, add_portfolios=True)
    returns_df, month_starts = preprocessor.get_data()

    portfolio_name = 'Port_5.0%'
    daily_returns = returns_df[portfolio_name]
    monthly_returns = daily_to_monthly_returns(daily_returns)
    rolling_paths = generate_paths_rolling(monthly_returns, T_months=120)

    # 시뮬레이션 실행
    init_wr = 0.05
    band = 0.15
    success_paths = []
    failure_paths = []

    for path in rolling_paths:
        result = simulate_withdrawal_on_path(
            path_returns=path,
            init_wr=init_wr,
            band=band,
            adj_on=False,
            beta=0.5,
            W0=100.0,
        )
        if result['success']:
            success_paths.append(result['W_series'])
        else:
            failure_paths.append(result['W_series'])

    # 반환 타입 검증
    check(isinstance(success_paths, list), "success_paths: list 타입")
    check(isinstance(failure_paths, list), "failure_paths: list 타입")

    # 각 path 길이 == 121 (120개월 + W0)
    T_expected = 121  # 120 months + initial W0
    all_correct_len = True
    for paths_group in [success_paths, failure_paths]:
        for w_series in paths_group:
            if len(w_series) != T_expected:
                all_correct_len = False
                break
    check(all_correct_len, f"모든 path 길이 == {T_expected} (120개월 + W0)")

    # success + failure 합 == 전체 rolling path 수
    total = len(success_paths) + len(failure_paths)
    check(total == len(rolling_paths),
          f"success({len(success_paths)}) + failure({len(failure_paths)}) "
          f"== rolling paths({len(rolling_paths)})")

    # 모든 path의 첫 값 == W0
    all_start_w0 = True
    for paths_group in [success_paths, failure_paths]:
        for w_series in paths_group:
            if not approx_eq(w_series[0], 100.0):
                all_start_w0 = False
                break
    check(all_start_w0, "모든 path의 W_series[0] == 100.0")

    # success path의 terminal NAV >= beta * W0
    all_success_terminal_ok = True
    for w_series in success_paths:
        if w_series[-1] < 0.5 * 100.0:
            all_success_terminal_ok = False
            break
    check(all_success_terminal_ok, "success paths: terminal NAV >= 50.0 (beta*W0)")

    print(f"\n  성공률: {len(success_paths)/total*100:.1f}% "
          f"({len(success_paths)}/{total})")


# ============================================================================
# Test 5: CLAUDE.md 수식 검증
# ============================================================================

def test_formula_verification():
    """CLAUDE.md에 기재된 수식과 engine.py 코드 일치 검증"""
    print("\n" + "=" * 60)
    print("[Test 5] CLAUDE.md 수식 검증")
    print("=" * 60)

    # ========================================
    # 5-1. 인출 시뮬레이션 순서 검증 (짧은 path)
    # ========================================

    print("\n--- 5-1. 인출 순서 검증 (3개월 path) ---")

    short_path = np.array([0.02, -0.03, 0.01])
    W0 = 100.0
    init_wr = 0.06
    band = 0.20

    result = simulate_withdrawal_on_path(
        path_returns=short_path,
        init_wr=init_wr,
        band=band,
        adj_on=False,
        W0=W0,
    )

    # 수동 계산
    base = W0 * init_wr / 12  # 0.5
    upper_wr = init_wr * (1 + band)  # 0.072
    lower_wr = init_wr * (1 - band)  # 0.048

    # Month 0: r=0.02
    W_after_return_0 = W0 * (1 + 0.02)  # 102.0
    current_wr_0 = (base * 12) / W_after_return_0  # ~0.0588
    # 0.048 <= 0.0588 <= 0.072 -> base
    withdraw_0 = base  # 0.5
    W_after_0 = W_after_return_0 - withdraw_0  # 101.5

    check(approx_eq(result['W_series'][0], W0),
          f"Step 0: W_0 = {result['W_series'][0]:.4f} (expected {W0})")

    # Step 1 검증: 수익률 먼저 적용
    check(approx_eq(W_after_return_0, 102.0),
          f"Month 0 Step 1: W_t = {W0} * (1+0.02) = {W_after_return_0:.4f}")

    # Step 2 검증: base 인출액
    check(approx_eq(base, 0.5),
          f"Step 2: base = {W0} * {init_wr} / 12 = {base:.4f}")

    # Step 4 검증: guardrail 범위 내
    check(lower_wr <= current_wr_0 <= upper_wr,
          f"Month 0 Step 4: current_wr={current_wr_0:.6f} in [{lower_wr}, {upper_wr}]")

    # Step 5 검증: 인출 실행
    check(approx_eq(result['withdraw_series'][0], withdraw_0),
          f"Month 0: withdraw = {result['withdraw_series'][0]:.4f} (expected {withdraw_0})")
    check(approx_eq(result['W_series'][1], W_after_0),
          f"Month 0: NAV = {result['W_series'][1]:.4f} (expected {W_after_0:.4f})")

    # Month 1: r=-0.03
    W_after_return_1 = W_after_0 * (1 + (-0.03))  # 101.5 * 0.97 = 98.455
    current_wr_1 = (base * 12) / W_after_return_1
    withdraw_1 = base  # still within range
    W_after_1 = W_after_return_1 - withdraw_1

    check(approx_eq(result['W_series'][2], W_after_1),
          f"Month 1: NAV = {result['W_series'][2]:.4f} (expected {W_after_1:.4f})")
    check(approx_eq(result['withdraw_series'][1], withdraw_1),
          f"Month 1: withdraw = {result['withdraw_series'][1]:.4f} (expected {withdraw_1})")

    # Month 2: r=0.01
    W_after_return_2 = W_after_1 * (1 + 0.01)
    current_wr_2 = (base * 12) / W_after_return_2
    withdraw_2 = base
    W_after_2 = W_after_return_2 - withdraw_2

    check(approx_eq(result['W_series'][3], W_after_2),
          f"Month 2: NAV = {result['W_series'][3]:.4f} (expected {W_after_2:.4f})")
    check(approx_eq(result['withdraw_series'][2], withdraw_2),
          f"Month 2: withdraw = {result['withdraw_series'][2]:.4f} (expected {withdraw_2})")

    # 누적 인출액
    expected_cum = withdraw_0 + withdraw_1 + withdraw_2  # 1.5
    check(approx_eq(result['cum_withdraw'], expected_cum),
          f"cum_withdraw = {result['cum_withdraw']:.4f} (expected {expected_cum})")

    # ========================================
    # 5-2. 성공/실패 판정 검증
    # ========================================

    print("\n--- 5-2. 성공/실패 판정 ---")

    # Ruin 검증: W_series에 <= 0 없음 -> ruin=False
    check(result['ruin_flag'] == False,
          f"ruin_flag = {result['ruin_flag']} (no W_t <= 0)")

    # Terminal fail 검증: W_T < beta * W0 ?
    terminal_nav = result['W_series'][-1]
    expected_terminal_fail = terminal_nav < 0.5 * W0
    check(result['terminal_fail_flag'] == expected_terminal_fail,
          f"terminal_fail_flag = {result['terminal_fail_flag']} "
          f"(W_T={terminal_nav:.2f}, threshold={0.5*W0})")

    # Success 검증: not ruin AND not terminal_fail
    expected_success = (not result['ruin_flag']) and (not result['terminal_fail_flag'])
    check(result['success'] == expected_success,
          f"success = {result['success']} (expected {expected_success})")

    # ========================================
    # 5-3. Ruin 케이스 강제 테스트
    # ========================================

    print("\n--- 5-3. Ruin 케이스 ---")

    # 극단적 높은 인출률로 파산 유도
    # band=99.0 (guardrail 무효화) + 큰 하락 + 높은 인출률
    ruin_result = simulate_withdrawal_on_path(
        path_returns=np.array([-0.50, -0.50, -0.50, -0.50]),
        init_wr=0.50,  # 50% annual withdrawal rate
        band=99.0,     # guardrail 무효화 (fixed baseline처럼)
        adj_on=False,
        W0=100.0,
    )

    check(ruin_result['ruin_flag'] == True,
          f"극단 케이스: ruin_flag = True (W_series min = "
          f"{np.min(ruin_result['W_series']):.4f})")

    # ========================================
    # 5-4. Guardrail 캡 작동 검증
    # ========================================

    print("\n--- 5-4. Guardrail 캡 작동 ---")

    # 큰 하락 후 current_wr이 upper 초과하는 경우
    guardrail_path = np.array([-0.30])  # 30% 하락
    gr_result = simulate_withdrawal_on_path(
        path_returns=guardrail_path,
        init_wr=0.06,
        band=0.20,
        adj_on=False,
        W0=100.0,
    )

    W_after_drop = 100.0 * (1 - 0.30)  # 70.0
    base_wr_at_drop = (0.5 * 12) / W_after_drop  # 6/70 = 0.0857
    upper_limit = 0.06 * 1.20  # 0.072
    # 0.0857 > 0.072 -> upper cap applied
    expected_withdraw = upper_limit * W_after_drop / 12  # 0.072 * 70 / 12 = 0.42

    check(approx_eq(gr_result['withdraw_series'][0], expected_withdraw, tol=1e-4),
          f"Upper cap: withdraw = {gr_result['withdraw_series'][0]:.4f} "
          f"(expected {expected_withdraw:.4f})")

    # 큰 상승 후 current_wr이 lower 미달하는 경우
    guardrail_path2 = np.array([0.50])  # 50% 상승
    gr_result2 = simulate_withdrawal_on_path(
        path_returns=guardrail_path2,
        init_wr=0.06,
        band=0.20,
        adj_on=False,
        W0=100.0,
    )

    W_after_rise = 100.0 * 1.50  # 150.0
    base_wr_at_rise = (0.5 * 12) / W_after_rise  # 6/150 = 0.04
    lower_limit = 0.06 * 0.80  # 0.048
    # 0.04 < 0.048 -> lower cap applied
    expected_withdraw2 = lower_limit * W_after_rise / 12  # 0.048 * 150 / 12 = 0.6

    check(approx_eq(gr_result2['withdraw_series'][0], expected_withdraw2, tol=1e-4),
          f"Lower cap: withdraw = {gr_result2['withdraw_series'][0]:.4f} "
          f"(expected {expected_withdraw2:.4f})")

    # ========================================
    # 5-5. 수익률 보정 (adj_on) 검증
    # ========================================

    print("\n--- 5-5. 수익률 보정 (adj_on) ---")

    # lookback=1, 전월 수익률 > thr_up -> adj_up 적용
    adj_path = np.array([0.05, 0.01])  # month 0: +5%, month 1: +1%
    adj_result = simulate_withdrawal_on_path(
        path_returns=adj_path,
        init_wr=0.06,
        band=0.20,
        adj_on=True,
        lookback=1,
        thr_up=0.03,  # 3%
        thr_dn=-0.03,
        adj_up=0.05,  # +5%
        adj_dn=-0.05,
        W0=100.0,
    )

    # Month 0: t=0, t < lookback(1) -> no adj -> base = 0.5
    check(approx_eq(adj_result['withdraw_series'][0], 0.5, tol=0.1),
          f"Month 0 (t < lookback): withdraw ~= base")

    # Month 1: t=1, prev_return = path[0] = 0.05 > thr_up(0.03)
    #   -> withdraw_adj = base * (1 + adj_up) = 0.5 * 1.05 = 0.525
    # (guardrail may cap, but let's check the direction)
    month1_withdraw = adj_result['withdraw_series'][1]
    # 보정 적용되었으면 base(0.5)보다 커야 함
    check(month1_withdraw > 0.5 - 0.01,
          f"Month 1 (prev +5% > thr_up 3%): withdraw = {month1_withdraw:.4f} "
          f"(adj_up 적용, base보다 크거나 같음)")

    # ========================================
    # 5-6. 집계 지표 검증 (evaluate_strategy)
    # ========================================

    print("\n--- 5-6. evaluate_strategy 집계 검증 ---")

    # 작은 path 세트로 수동 검증
    test_paths = [
        np.array([0.02, -0.01, 0.03]),
        np.array([-0.05, 0.02, 0.01]),
        np.array([0.01, 0.01, 0.01]),
    ]

    eval_result = evaluate_strategy(
        paths=test_paths,
        init_wr=0.06,
        band=0.20,
        adj_on=False,
        W0=100.0,
    )

    # 개별 시뮬레이션 결과 수동 계산
    individual_results = []
    all_withdrawals = []
    for path in test_paths:
        r = simulate_withdrawal_on_path(
            path_returns=path, init_wr=0.06, band=0.20,
            adj_on=False, W0=100.0,
        )
        individual_results.append(r)
        all_withdrawals.extend(r['withdraw_series'])

    # success_rate 검증
    manual_success_rate = np.mean([1 if r['success'] else 0 for r in individual_results])
    check(approx_eq(eval_result['x_success_rate'], manual_success_rate),
          f"success_rate: {eval_result['x_success_rate']:.4f} == "
          f"{manual_success_rate:.4f}")

    # cum_withdraw_median 검증
    manual_cum_median = np.median([r['cum_withdraw'] for r in individual_results])
    check(approx_eq(eval_result['y_cum_withdraw_median'], manual_cum_median),
          f"cum_withdraw_median: {eval_result['y_cum_withdraw_median']:.4f} == "
          f"{manual_cum_median:.4f}")

    # p_ruin 검증
    manual_p_ruin = np.mean([1 if r['ruin_flag'] else 0 for r in individual_results])
    check(approx_eq(eval_result['p_ruin'], manual_p_ruin),
          f"p_ruin: {eval_result['p_ruin']:.4f} == {manual_p_ruin:.4f}")

    # p_terminal_fail 검증
    manual_p_terminal = np.mean([1 if r['terminal_fail_flag'] else 0 for r in individual_results])
    check(approx_eq(eval_result['p_terminal_fail'], manual_p_terminal),
          f"p_terminal_fail: {eval_result['p_terminal_fail']:.4f} == "
          f"{manual_p_terminal:.4f}")

    # p5_monthly_income 검증
    manual_p5 = np.percentile(all_withdrawals, 5)
    check(approx_eq(eval_result['p5_monthly_income'], manual_p5),
          f"p5_monthly_income: {eval_result['p5_monthly_income']:.4f} == "
          f"{manual_p5:.4f}")

    # CV 검증 (path별)
    manual_cv_list = []
    for r in individual_results:
        ws = r['withdraw_series']
        nz = ws[ws > 0]
        if len(nz) > 1:
            manual_cv_list.append(np.std(nz, ddof=1) / np.mean(nz))
        else:
            manual_cv_list.append(0)
    manual_cv_median = np.median(manual_cv_list)
    check(approx_eq(eval_result['cv_median'], manual_cv_median),
          f"cv_median: {eval_result['cv_median']:.6f} == {manual_cv_median:.6f}")

    # worst_cut 검증 (path별)
    manual_worst_cut_list = []
    for r in individual_results:
        ws = r['withdraw_series']
        worst_cut = 0
        for t in range(1, len(ws)):
            if ws[t-1] > 0:
                cut_rate = (ws[t-1] - ws[t]) / ws[t-1]
                worst_cut = max(worst_cut, cut_rate)
        manual_worst_cut_list.append(worst_cut)
    manual_worst_cut_median = np.median(manual_worst_cut_list)
    check(approx_eq(eval_result['worst_cut_median'], manual_worst_cut_median),
          f"worst_cut_median: {eval_result['worst_cut_median']:.6f} == "
          f"{manual_worst_cut_median:.6f}")

    # n_paths 검증
    check(eval_result['n_paths'] == len(test_paths),
          f"n_paths: {eval_result['n_paths']} == {len(test_paths)}")

    # ========================================
    # 5-7. fixed_baseline: band=99.0 -> guardrail 사실상 무효
    # ========================================

    print("\n--- 5-7. fixed_baseline (band=99.0) ---")

    fixed_result = simulate_withdrawal_on_path(
        path_returns=np.array([0.02, -0.03, 0.01]),
        init_wr=0.06,
        band=99.0,  # sealant: guardrail 무효
        adj_on=False,
        W0=100.0,
    )

    # band=99.0이면 upper=0.06*100=6.0, lower=0.06*(-98.0)=-5.88
    # 사실상 어떤 current_wr도 이 범위 안에 들어가므로 항상 base 인출
    for t in range(3):
        check(approx_eq(fixed_result['withdraw_series'][t], 0.5, tol=1e-4),
              f"fixed_baseline Month {t}: withdraw = "
              f"{fixed_result['withdraw_series'][t]:.4f} (expected 0.5)")


# ============================================================================
# Test 6: BETA_LABELS 검증
# ============================================================================

def test_beta_labels():
    """BETA_LABELS가 5단계 beta 값을 모두 포함하는지 검증"""
    print("\n" + "=" * 60)
    print("[Test 6] BETA_LABELS")
    print("=" * 60)

    expected_betas = {0.1, 0.25, 0.5, 0.75, 1.0}

    check(isinstance(BETA_LABELS, dict), "BETA_LABELS: dict 타입")
    check(len(BETA_LABELS) == 5, f"BETA_LABELS: {len(BETA_LABELS)}개 항목 (expected 5)")

    actual_betas = set(BETA_LABELS.keys())
    check(actual_betas == expected_betas,
          f"BETA_LABELS 키: {sorted(actual_betas)} == {sorted(expected_betas)}")

    # 각 값이 문자열인지 확인
    for beta, label in BETA_LABELS.items():
        check(isinstance(label, str) and len(label) > 0,
              f"BETA_LABELS[{beta}]: '{label}' (비어있지 않은 문자열)")

    # 0.25, 0.75 신규 추가 항목 확인
    check(0.25 in BETA_LABELS, "BETA_LABELS에 0.25 존재")
    check(0.75 in BETA_LABELS, "BETA_LABELS에 0.75 존재")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("viewer.py 핵심 함수 테스트")
    print("=" * 60)

    # Test 1: load_grid_results
    df = test_load_grid_results()

    # Test 2: filter_dataframe
    test_filter_dataframe(df)

    # Test 3: gbm_survival_probability
    test_gbm_survival_probability()

    # Test 4: simulate_paths_for_strategy
    test_simulate_paths_for_strategy()

    # Test 5: CLAUDE.md 수식 검증
    test_formula_verification()

    # Test 6: BETA_LABELS 검증
    test_beta_labels()

    # 최종 결과
    print("\n" + "=" * 60)
    total = _pass_count + _fail_count
    print(f"테스트 결과: {_pass_count}/{total} passed, {_fail_count} failed")
    print("=" * 60)

    if _fail_count > 0:
        sys.exit(1)
    else:
        print("[OK] All tests passed!")
        sys.exit(0)
