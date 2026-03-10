"""
Grid Search Script for Dynamic Withdrawal Strategy Optimization
================================================================
본격적인 파라미터 그리드 서치 실행 및 결과 저장
"""

import numpy as np
import pandas as pd
import pickle
import time
from typing import List, Dict
from itertools import product

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from withdrawal_backtest import DataPreprocessor
from engine import (
    daily_to_monthly_returns,
    generate_paths_rolling,
    generate_paths_bootstrap,
    evaluate_strategy,
    pareto_frontier,
    select_optimal
)


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # 대상 포트폴리오
    'portfolios': ['Port_4.0%', 'Port_5.0%', 'Port_6.0%', 'Port_7.0%', 'Port_8.0%', 'Port_9.0%'],

    # Path 설정
    'T_months': 120,           # 10년
    'path_methods': ['rolling', 'bootstrap'],   # bootstrap 활성화
    'bootstrap_block': 12,
    'bootstrap_n_sample': 181,   # rolling과 동일 수로 샘플링
    'bootstrap_n_generate': 5000,  # 생성 후 181개 랜덤 추출
    'bootstrap_seed': 42,

    # 파라미터 그리드
    'init_wr_range': np.arange(0.03, 0.155, 0.005),  # 3% ~ 15%, 0.5%p 간격
    'band_range': [0.05, 0.10, 0.15, 0.20],

    # 수익률 보정 그리드 (adj_on=True 제거, 영향 미미)
    'adj_configs': [
        # 고정 인출 baseline (band=99.0으로 강제, band_range 순회 안 함)
        {'adj_on': False, 'fixed_baseline': True},
        # Guardrail만 적용 (보정 없음)
        {'adj_on': False},
        # Volatility-Adjusted Guardrail (밴드 폭을 실현 변동성으로 동적 조정)
        {'adj_on': False, 'vol_adj': True},
    ],

    # 다중 beta 값 (성공 기준 기말잔액 비율)
    'betas': [0.1, 0.25, 0.5, 0.75, 1.0],
    'W0': 100.0,

    # 최적 선택 기준
    'x_min': 0.90,
    'q_ruin': 0.01,
}


# ============================================================================
# Step 1: 데이터 로딩 및 Path 생성
# ============================================================================

def load_and_generate_paths():
    """
    벤치마크 데이터 로드 및 포트폴리오별 Path 생성

    Returns
    -------
    portfolio_paths : dict
        {portfolio_name: {'rolling': [...], 'bootstrap': [...], 'combined': [...]}}
    """
    print("=" * 70)
    print("Step 1: 데이터 로딩 및 Path 생성")
    print("=" * 70)

    # 벤치마크 데이터 로드
    print("\n[데이터 로딩]")
    with open('benchmark_data.pkl', 'rb') as f:
        benchmark_data = pickle.load(f)

    print(f"[OK] 벤치마크 데이터 로드 완료")
    print(f"  기간: {benchmark_data.index[0].date()} ~ {benchmark_data.index[-1].date()}")
    print(f"  거래일 수: {len(benchmark_data):,}일")

    # DataPreprocessor
    preprocessor = DataPreprocessor(benchmark_data, add_portfolios=True)
    returns_df, month_starts = preprocessor.get_data()

    # 포트폴리오별 Path 생성
    print(f"\n[Path 생성]")
    print(f"  T_months: {CONFIG['T_months']} (10년)")
    print(f"  Path 방식: Rolling + Bootstrap + Combined")

    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    portfolio_paths = {}

    for portfolio_name in tqdm(CONFIG['portfolios'], desc="포트폴리오"):
        # 일별 → 월간 수익률 변환
        daily_returns = returns_df[portfolio_name]
        monthly_returns = daily_to_monthly_returns(daily_returns)

        # Rolling paths 생성
        rolling_paths = generate_paths_rolling(monthly_returns, CONFIG['T_months'])

        # Bootstrap paths 생성 (5000개 생성 후 181개 랜덤 샘플링)
        all_bootstrap = generate_paths_bootstrap(
            monthly_returns,
            T_months=CONFIG['T_months'],
            block_length=CONFIG['bootstrap_block'],
            n_paths=CONFIG['bootstrap_n_generate'],
            seed=CONFIG['bootstrap_seed']
        )
        sample_idx = rng.choice(
            CONFIG['bootstrap_n_generate'],
            CONFIG['bootstrap_n_sample'],
            replace=False
        )
        bootstrap_paths = [all_bootstrap[i] for i in sample_idx]

        # Combined: rolling + bootstrap 합산
        combined_paths = rolling_paths + bootstrap_paths

        # 전체 표본 연환산 변동성 (vol_adjusted 전략용)
        sigma_full = float(np.std(monthly_returns, ddof=1) * np.sqrt(12))

        portfolio_paths[portfolio_name] = {
            'rolling': rolling_paths,
            'bootstrap': bootstrap_paths,
            'combined': combined_paths,
            'sigma_target': sigma_full,
        }

    print(f"\n[OK] Path 생성 완료")
    for portfolio_name in CONFIG['portfolios']:
        n_rolling = len(portfolio_paths[portfolio_name]['rolling'])
        n_bootstrap = len(portfolio_paths[portfolio_name]['bootstrap'])
        n_combined = len(portfolio_paths[portfolio_name]['combined'])
        sigma_t = portfolio_paths[portfolio_name]['sigma_target']
        print(f"  {portfolio_name}: Rolling {n_rolling}개, Bootstrap {n_bootstrap}개, Combined {n_combined}개, σ={sigma_t*100:.2f}%")

    return portfolio_paths


# ============================================================================
# Step 2: 그리드 서치 실행
# ============================================================================

def run_grid_search(portfolio_paths):
    """
    파라미터 그리드 서치 실행

    Parameters
    ----------
    portfolio_paths : dict
        포트폴리오별 path 데이터

    Returns
    -------
    all_results : list[dict]
        전체 평가 결과
    """
    print("\n" + "=" * 70)
    print("Step 2: 그리드 서치 실행")
    print("=" * 70)

    # 파라미터 조합 생성
    init_wr_list = CONFIG['init_wr_range']
    band_list = CONFIG['band_range']
    adj_configs = CONFIG['adj_configs']
    betas = CONFIG['betas']
    path_methods = CONFIG['path_methods'] + ['combined']  # rolling, bootstrap, combined

    # fixed_baseline / dynamic / vol_adjusted 조합 수 계산
    n_fixed_baseline = len(init_wr_list) * 1  # band=99.0 하나만
    n_dynamic = len(init_wr_list) * len(band_list) * 1  # adj_on=False, no vol_adj
    n_vol_adjusted = len(init_wr_list) * len(band_list) * 1  # vol_adj=True
    n_combinations = n_fixed_baseline + n_dynamic + n_vol_adjusted

    n_portfolios = len(CONFIG['portfolios'])
    n_path_methods = len(path_methods)
    n_betas = len(betas)
    total_runs = n_combinations * n_portfolios * n_path_methods * n_betas

    print(f"\n포트폴리오: {n_portfolios}개")
    print(f"Path 방식: {', '.join(path_methods)}")
    print(f"Beta 값: {betas}")
    print(f"파라미터 조합: {n_combinations}개")
    print(f"  - Fixed baseline: {n_fixed_baseline}개 (init_wr {len(init_wr_list)} × band 1)")
    print(f"  - Dynamic: {n_dynamic}개 (init_wr {len(init_wr_list)} × band {len(band_list)})")
    print(f"  - Vol-Adjusted: {n_vol_adjusted}개 (init_wr {len(init_wr_list)} × band {len(band_list)})")
    print(f"총 실행 수: {total_runs:,}개 ({n_combinations} × {n_portfolios} portfolios × {n_path_methods} paths × {n_betas} betas)")

    estimated_time_sec = total_runs * 0.01
    estimated_time_min = estimated_time_sec / 60
    print(f"예상 소요시간: ~{estimated_time_min:.1f}분")

    # 그리드 서치 실행
    all_results = []
    start_time = time.time()

    # 포트폴리오 × path 방식 × beta 반복
    for portfolio_name in CONFIG['portfolios']:
        for path_method in path_methods:
            paths = portfolio_paths[portfolio_name][path_method]

            for beta in betas:
                method_start_time = time.time()
                n_processed = 0

                # sigma_target for vol_adjusted
                sigma_target = portfolio_paths[portfolio_name]['sigma_target']

                # 파라미터 조합 반복
                for init_wr in init_wr_list:
                    for adj_config in adj_configs:
                        # fixed_baseline 처리
                        if adj_config.get('fixed_baseline', False):
                            band_list_to_use = [99.0]  # band=99.0만 사용
                            strategy_type = 'fixed_baseline'
                        elif adj_config.get('vol_adj', False):
                            band_list_to_use = band_list
                            strategy_type = 'vol_adjusted'
                        else:
                            band_list_to_use = band_list
                            strategy_type = 'dynamic'

                        for band in band_list_to_use:
                            try:
                                # evaluate_strategy 호출
                                result = evaluate_strategy(
                                    paths=paths,
                                    init_wr=init_wr,
                                    band=band,
                                    adj_on=adj_config.get('adj_on', False),
                                    lookback=adj_config.get('lookback', 1),
                                    thr_up=adj_config.get('thr_up', 0.03),
                                    thr_dn=adj_config.get('thr_dn', -0.03),
                                    adj_up=adj_config.get('adj_up', 0.05),
                                    adj_dn=adj_config.get('adj_dn', -0.05),
                                    beta=beta,
                                    W0=CONFIG['W0'],
                                    vol_adj=adj_config.get('vol_adj', False),
                                    sigma_target=sigma_target if adj_config.get('vol_adj', False) else 0.0,
                                )

                                # 메타데이터 추가
                                result['portfolio'] = portfolio_name
                                result['path_method'] = path_method
                                result['strategy_type'] = strategy_type
                                result['beta'] = beta
                                result['vol_adj'] = adj_config.get('vol_adj', False)

                                all_results.append(result)
                                n_processed += 1

                            except Exception as e:
                                print(f"\n[WARN]  오류 발생 (스킵): {portfolio_name}/{path_method}/beta={beta}, "
                                      f"init_wr={init_wr:.3f}, band={band:.2f}, strategy_type={strategy_type}")
                                print(f"   에러: {str(e)}")
                                continue

                method_elapsed = time.time() - method_start_time
                print(f"[OK] {portfolio_name} / {path_method} / beta={beta} 완료 ({n_processed}개 조합, {method_elapsed:.1f}초)")

    total_elapsed = time.time() - start_time
    total_minutes = int(total_elapsed // 60)
    total_seconds = int(total_elapsed % 60)

    print(f"\n" + "=" * 70)
    print(f"Grid Search 완료")
    print(f"=" * 70)
    print(f"총 소요시간: {total_minutes}분 {total_seconds}초")
    print(f"총 결과: {len(all_results)}개")

    return all_results


# ============================================================================
# Step 3: 결과 저장
# ============================================================================

def save_results(all_results):
    """
    결과 저장 및 요약

    Parameters
    ----------
    all_results : list[dict]
        전체 평가 결과
    """
    print("\n" + "=" * 70)
    print("Step 3: 결과 저장")
    print("=" * 70)

    path_methods = CONFIG['path_methods'] + ['combined']

    # Pareto frontier 계산 (포트폴리오 × path 방식 × beta별로)
    print("\n[Pareto Frontier 계산]")

    for portfolio_name in CONFIG['portfolios']:
        for path_method in path_methods:
            for beta in CONFIG['betas']:
                # 해당 조합의 결과만 필터링
                subset = [r for r in all_results
                         if r['portfolio'] == portfolio_name
                         and r['path_method'] == path_method
                         and r['beta'] == beta]

                if len(subset) == 0:
                    continue

                # Pareto frontier 계산
                frontier = pareto_frontier(subset)

                # is_frontier 플래그 업데이트
                frontier_keys = set()
                for f in frontier:
                    key = (f['init_wr'], f['band'], f['adj_on'])
                    frontier_keys.add(key)

                for r in all_results:
                    if (r['portfolio'] == portfolio_name
                        and r['path_method'] == path_method
                        and r['beta'] == beta):
                        key = (r['init_wr'], r['band'], r['adj_on'])
                        r['is_frontier'] = key in frontier_keys

                print(f"  {portfolio_name}/{path_method}/beta={beta}: {len(frontier)}개 frontier 점")

    # 최적 전략 선택 (포트폴리오 × path 방식 × beta별로)
    print("\n[최적 전략 선택]")
    print(f"제약: 성공률 ≥ {CONFIG['x_min']*100:.0f}%, 파산률 ≤ {CONFIG['q_ruin']*100:.2f}%")
    print()

    for portfolio_name in CONFIG['portfolios']:
        for path_method in path_methods:
            for beta in CONFIG['betas']:
                print(f"\n{portfolio_name} / {path_method} / beta={beta}:")

                # Fixed baseline 최적해
                subset_fixed = [r for r in all_results
                               if r['portfolio'] == portfolio_name and r['path_method'] == path_method
                               and r['beta'] == beta and r['strategy_type'] == 'fixed_baseline']

                optimal_fixed = select_optimal(subset_fixed, x_min=CONFIG['x_min'], q_ruin=CONFIG['q_ruin'])

                if optimal_fixed:
                    for r in all_results:
                        if (r['portfolio'] == portfolio_name and r['path_method'] == path_method
                            and r['beta'] == beta and r['strategy_type'] == 'fixed_baseline'
                            and r['init_wr'] == optimal_fixed['init_wr']):
                            r['is_optimal'] = True

                    print(f"  [Fixed baseline]  init_wr={optimal_fixed['init_wr']*100:4.1f}%, "
                          f"누적인출={optimal_fixed['y_cum_withdraw_median']:5.1f}")
                else:
                    print(f"  [Fixed baseline]  최적 전략 없음 (제약 조건 불만족)")

                # Dynamic 최적해
                subset_dynamic = [r for r in all_results
                                 if r['portfolio'] == portfolio_name and r['path_method'] == path_method
                                 and r['beta'] == beta and r['strategy_type'] == 'dynamic']

                optimal_dynamic = select_optimal(subset_dynamic, x_min=CONFIG['x_min'], q_ruin=CONFIG['q_ruin'])

                if optimal_dynamic:
                    for r in all_results:
                        if (r['portfolio'] == portfolio_name and r['path_method'] == path_method
                            and r['beta'] == beta and r['strategy_type'] == 'dynamic'
                            and r['init_wr'] == optimal_dynamic['init_wr']
                            and r['band'] == optimal_dynamic['band']):
                            r['is_optimal'] = True

                    print(f"  [Dynamic]         init_wr={optimal_dynamic['init_wr']*100:4.1f}%, "
                          f"band={optimal_dynamic['band']*100:2.0f}%, "
                          f"누적인출={optimal_dynamic['y_cum_withdraw_median']:5.1f}")
                else:
                    print(f"  [Dynamic]         최적 전략 없음 (제약 조건 불만족)")

                # Vol-Adjusted 최적해
                subset_vol = [r for r in all_results
                             if r['portfolio'] == portfolio_name and r['path_method'] == path_method
                             and r['beta'] == beta and r['strategy_type'] == 'vol_adjusted']

                optimal_vol = select_optimal(subset_vol, x_min=CONFIG['x_min'], q_ruin=CONFIG['q_ruin'])

                if optimal_vol:
                    for r in all_results:
                        if (r['portfolio'] == portfolio_name and r['path_method'] == path_method
                            and r['beta'] == beta and r['strategy_type'] == 'vol_adjusted'
                            and r['init_wr'] == optimal_vol['init_wr']
                            and r['band'] == optimal_vol['band']):
                            r['is_optimal'] = True

                    print(f"  [Vol-Adjusted]    init_wr={optimal_vol['init_wr']*100:4.1f}%, "
                          f"band={optimal_vol['band']*100:2.0f}%, "
                          f"누적인출={optimal_vol['y_cum_withdraw_median']:5.1f}")
                else:
                    print(f"  [Vol-Adjusted]    최적 전략 없음 (제약 조건 불만족)")

    # 전체 결과 저장 (pkl)
    print("\n[파일 저장]")

    with open('grid_results_full.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"[OK] grid_results_full.pkl ({len(all_results)}개 결과)")

    # 요약 DataFrame 생성
    df_summary = pd.DataFrame([
        {
            'portfolio': r['portfolio'],
            'path_method': r['path_method'],
            'strategy_type': r['strategy_type'],
            'beta': r['beta'],
            'init_wr': r['init_wr'],
            'band': r['band'],
            'adj_on': r['adj_on'],
            'success_rate': r['x_success_rate'],
            'cum_withdraw_median': r['y_cum_withdraw_median'],
            'cum_withdraw_mean': r['y_cum_withdraw_mean'],
            'p_ruin': r['p_ruin'],
            'p_terminal_fail': r['p_terminal_fail'],
            'cv_median': r['cv_median'],
            'worst_cut_median': r['worst_cut_median'],
            'p5_monthly_income': r['p5_monthly_income'],
            'terminal_nav_median': r.get('terminal_nav_median', None),
            'terminal_nav_mean': r.get('terminal_nav_mean', None),
            'vol_adj': r.get('vol_adj', False),
            'is_frontier': r.get('is_frontier', False),
            'is_optimal': r.get('is_optimal', False),
        }
        for r in all_results
    ])

    # Excel 저장
    with pd.ExcelWriter('grid_results_summary.xlsx', engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='All_Results', index=False)

        # Frontier만 별도 시트
        df_frontier = df_summary[df_summary['is_frontier'] == True]
        df_frontier.to_excel(writer, sheet_name='Frontier', index=False)

        # Optimal만 별도 시트
        df_optimal = df_summary[df_summary['is_optimal'] == True]
        df_optimal.to_excel(writer, sheet_name='Optimal', index=False)

    print(f"[OK] grid_results_summary.xlsx")
    print(f"   - All_Results: {len(df_summary)}행")
    print(f"   - Frontier: {len(df_frontier)}행")
    print(f"   - Optimal: {len(df_optimal)}행")


# ============================================================================
# Main
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("=" * 70)
    print("Dynamic Withdrawal Strategy - Grid Search")
    print("=" * 70)
    print()

    # Step 1: Path 생성
    portfolio_paths = load_and_generate_paths()

    # Step 2: 그리드 서치
    all_results = run_grid_search(portfolio_paths)

    # Step 3: 결과 저장
    save_results(all_results)

    print("\n" + "=" * 70)
    print("전체 완료")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()