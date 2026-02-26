"""
GBM Grid Search for Volatility-based Guardrail Analysis
========================================================
GBM Monte Carlo 시뮬레이션으로 변동성(sigma)별 Guardrail 효과 분석.
mu-sigma 연속 축에서 Fixed vs Guardrail 비교를 수행.

출력: gbm_results.pkl, gbm_results_summary.xlsx
실행: python gbm_grid_search.py (~30초)
"""

import numpy as np
import pandas as pd
import pickle
import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from engine import (
    generate_paths_gbm,
    simulate_withdrawal_on_paths_vectorized,
    evaluate_gbm_strategy,
)
from scipy.stats import norm


# ============================================================================
# Closed-form GBM 생존확률 (Fixed 인출 전용)
# ============================================================================

def gbm_survival_probability(mu, sigma, wr, T=10, beta=0.5):
    """GBM Closed-form 생존확률 — 연속 배당 근사 (Fixed only)"""
    if wr <= 0:
        return 1.0
    if sigma <= 0:
        remaining = 1.0 - wr * T + mu * T
        return 1.0 if remaining >= beta else 0.0
    z = -(np.log(beta) - (mu - wr - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    survival = norm.cdf(z)
    return float(np.clip(survival, 0.0, 1.0))


# ============================================================================
# CONFIG
# ============================================================================

GBM_CONFIG = {
    'mu_range':     np.arange(0.02, 0.13, 0.01),    # 2%~12%, 11개
    'sigma_range':  np.arange(0.02, 0.21, 0.01),     # 2%~20%, 19개
    'init_wr_range': np.arange(0.03, 0.155, 0.005),  # 3%~15%, 25개
    'band_configs': [99.0, 0.05, 0.10, 0.15, 0.20],  # fixed + guardrail 4종
    'betas':        [0.1, 0.25, 0.5, 0.75, 1.0],
    'n_paths': 1000,
    'T_months': 120,
    'seed': 42,
}


# ============================================================================
# Grid Search 실행
# ============================================================================

def run_gbm_grid_search():
    """
    GBM 파라미터 그리드 서치 실행

    루프 구조 (효율 최적화):
        for (mu, sigma):           # 경로 1회 생성
            for init_wr:
                for band:          # 시뮬레이션 1회
                    evaluate(betas)  # 5개 beta 한번에

    Returns
    -------
    all_results : list[dict]
    """
    mu_range = GBM_CONFIG['mu_range']
    sigma_range = GBM_CONFIG['sigma_range']
    init_wr_range = GBM_CONFIG['init_wr_range']
    band_configs = GBM_CONFIG['band_configs']
    betas = GBM_CONFIG['betas']
    n_paths = GBM_CONFIG['n_paths']
    T_months = GBM_CONFIG['T_months']
    seed = GBM_CONFIG['seed']

    total_mu_sigma = len(mu_range) * len(sigma_range)
    total_sims = total_mu_sigma * len(init_wr_range) * len(band_configs)
    total_results = total_sims * len(betas)

    print(f"\n그리드 설정:")
    print(f"  mu: {len(mu_range)}개 ({mu_range[0]*100:.0f}%~{mu_range[-1]*100:.0f}%)")
    print(f"  sigma: {len(sigma_range)}개 ({sigma_range[0]*100:.0f}%~{sigma_range[-1]*100:.0f}%)")
    print(f"  init_wr: {len(init_wr_range)}개 ({init_wr_range[0]*100:.0f}%~{init_wr_range[-1]*100:.1f}%)")
    print(f"  band: {len(band_configs)}개 ({band_configs})")
    print(f"  betas: {betas}")
    print(f"  n_paths: {n_paths}, T_months: {T_months}")
    print(f"\n총 (mu,sigma) 쌍: {total_mu_sigma}")
    print(f"총 시뮬레이션 (벡터화): {total_sims:,}")
    print(f"총 결과 레코드: {total_results:,}")

    all_results = []
    start_time = time.time()

    mu_sigma_pairs = [(float(mu), float(sigma)) for mu in mu_range for sigma in sigma_range]

    for mu, sigma in tqdm(mu_sigma_pairs, desc="(mu,sigma)"):
        # 동일 (mu, sigma)에 대해 경로 1회만 생성
        paths = generate_paths_gbm(mu, sigma, n_paths, T_months, seed)

        for init_wr in init_wr_range:
            init_wr_f = float(init_wr)

            for band in band_configs:
                band_f = float(band)
                strategy_type = 'fixed_baseline' if band_f == 99.0 else 'dynamic'

                # 벡터화 시뮬레이션 1회
                sim = simulate_withdrawal_on_paths_vectorized(paths, init_wr_f, band_f)

                # 5개 beta 한번에 평가
                eval_results = evaluate_gbm_strategy(sim, betas)

                for i, beta in enumerate(betas):
                    result = eval_results[i].copy()
                    result['mu'] = mu
                    result['sigma'] = sigma
                    result['init_wr'] = init_wr_f
                    result['band'] = band_f
                    result['beta'] = float(beta)
                    result['strategy_type'] = strategy_type
                    # Fixed의 경우 Closed-form 확률도 계산
                    if strategy_type == 'fixed_baseline':
                        result['cf_success_rate'] = gbm_survival_probability(
                            mu, sigma, init_wr_f,
                            T=T_months / 12, beta=float(beta))
                    else:
                        result['cf_success_rate'] = None
                    all_results.append(result)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n[OK] 그리드 서치 완료: {len(all_results):,}개 결과, {minutes}분 {seconds}초")

    return all_results


# ============================================================================
# 결과 저장
# ============================================================================

def save_gbm_results(all_results):
    """결과를 pkl + xlsx로 저장"""
    print("\n[파일 저장]")

    # pkl 저장
    with open('gbm_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"[OK] gbm_results.pkl ({len(all_results):,}개 결과)")

    # 요약 DataFrame → Excel
    df = pd.DataFrame(all_results)

    with pd.ExcelWriter('gbm_results_summary.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Results', index=False)

        # 주요 통계 요약 시트
        summary = df.groupby(['mu', 'sigma', 'strategy_type', 'beta']).agg(
            mean_success=('x_success_rate', 'mean'),
            count=('x_success_rate', 'count'),
        ).reset_index()
        summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"[OK] gbm_results_summary.xlsx ({len(df):,}행)")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("GBM Grid Search for Volatility-based Guardrail Analysis")
    print("=" * 70)

    all_results = run_gbm_grid_search()
    save_gbm_results(all_results)

    # 간단한 검증 출력
    df = pd.DataFrame(all_results)
    print(f"\n[검증] 결과 DataFrame: {df.shape}")
    print(f"  mu 범위: {df['mu'].min():.2f} ~ {df['mu'].max():.2f}")
    print(f"  sigma 범위: {df['sigma'].min():.2f} ~ {df['sigma'].max():.2f}")
    print(f"  init_wr 범위: {df['init_wr'].min():.3f} ~ {df['init_wr'].max():.3f}")
    print(f"  strategy_type 분포: {df['strategy_type'].value_counts().to_dict()}")

    # 샘플: mu=6%, sigma=10%, beta=0.5 기준 Fixed vs Guardrail 비교
    sample = df[(df['mu'] == 0.06) & (df['sigma'] == 0.10) & (df['beta'] == 0.5)]
    if len(sample) > 0:
        print(f"\n[샘플] mu=6%, sigma=10%, beta=50%:")
        for st in ['fixed_baseline', 'dynamic']:
            sub = sample[sample['strategy_type'] == st]
            if len(sub) > 0:
                print(f"  {st}: {len(sub)}개 조합")
                for _, row in sub.head(3).iterrows():
                    band_label = "fixed" if row['band'] == 99.0 else f"\u00b1{row['band']*100:.0f}%"
                    print(f"    init_wr={row['init_wr']*100:.1f}%, band={band_label}, "
                          f"성공률={row['x_success_rate']*100:.1f}%")

    print("\n" + "=" * 70)
    print("완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
