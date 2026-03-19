"""
신규 펀드 Grid Search — grid_results_full.pkl에 합산
====================================================
product_paths.pkl의 3개 펀드에 대해 기존과 동일한 파라미터 그리드 서치 실행.
결과를 기존 grid_results_full.pkl에 append.

실행: python grid_search_product.py
"""

import numpy as np
import pandas as pd
import pickle
import time
import sys, io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from engine import evaluate_strategy, pareto_frontier, select_optimal


# ============================================================================
# CONFIG (grid_search.py와 동일)
# ============================================================================

CONFIG = {
    'init_wr_range': np.arange(0.03, 0.155, 0.005),
    'band_range': [0.05, 0.10, 0.15, 0.20],
    'adj_configs': [
        {'adj_on': False, 'fixed_baseline': True},
        {'adj_on': False},
        {'adj_on': False, 'vol_adj': True},
    ],
    'betas': [0.1, 0.25, 0.5, 0.75, 1.0],
    'W0': 100.0,
    'x_min': 0.90,
    'q_ruin': 0.01,
    'path_methods': ['rolling', 'bootstrap'],
}


# ============================================================================
# 데이터 로딩
# ============================================================================

print("1. 데이터 로딩...")
with open('product_paths.pkl', 'rb') as f:
    data = pickle.load(f)

funds = data['funds']
fund_paths = data['fund_paths']
print(f"   펀드: {funds}")


# ============================================================================
# Grid Search 실행
# ============================================================================

print("\n2. Grid Search 실행...")

init_wr_list = CONFIG['init_wr_range']
band_list = CONFIG['band_range']
betas = CONFIG['betas']
path_methods = CONFIG['path_methods']

all_results = []

for fund in funds:
    fp = fund_paths[fund]
    mr = fp['monthly_returns']
    # sigma_target: 연변동성 (월간 std * sqrt(12))
    sigma_target = mr.std() * np.sqrt(12)

    paths_dict = {
        'rolling': fp['rolling_paths'],
        'bootstrap': fp['bootstrap_paths'],
    }

    for pm in path_methods:
        paths = paths_dict[pm]
        n_paths = len(paths)

        for adj_config in CONFIG['adj_configs']:
            is_fixed = adj_config.get('fixed_baseline', False)
            is_vol_adj = adj_config.get('vol_adj', False)

            if is_fixed:
                strategy_type = 'fixed_baseline'
                bands = [99.0]
            elif is_vol_adj:
                strategy_type = 'vol_adjusted'
                bands = band_list
            else:
                strategy_type = 'dynamic'
                bands = band_list

            for band in bands:
                for init_wr in init_wr_list:
                    for beta in betas:
                        result = evaluate_strategy(
                            paths=paths,
                            init_wr=init_wr,
                            band=band,
                            adj_on=adj_config.get('adj_on', False),
                            beta=beta,
                            W0=CONFIG['W0'],
                            vol_adj=is_vol_adj,
                            sigma_target=sigma_target if is_vol_adj else 0.0,
                        )

                        result['portfolio'] = fund
                        result['path_method'] = pm
                        result['strategy_type'] = strategy_type
                        result['beta'] = beta
                        result['vol_adj'] = is_vol_adj
                        result['is_frontier'] = False
                        result['is_optimal'] = False
                        all_results.append(result)

    print(f"   {fund}: 완료 ({len([r for r in all_results if r['portfolio'] == fund])}건)")

print(f"\n   총 {len(all_results)}건 생성")


# ============================================================================
# 기존 pkl에 합산
# ============================================================================

print("\n3. 기존 grid_results_full.pkl에 합산...")

with open('grid_results_full.pkl', 'rb') as f:
    existing = pickle.load(f)

# 기존에서 신규 펀드 제거 (재실행 대비)
existing_clean = [r for r in existing if r['portfolio'] not in funds]
print(f"   기존: {len(existing)}건 -> 신규펀드 제거 후: {len(existing_clean)}건")

merged = existing_clean + all_results
print(f"   합산: {len(merged)}건")

with open('grid_results_full.pkl', 'wb') as f:
    pickle.dump(merged, f)

print("   grid_results_full.pkl 저장 완료")

# 포트폴리오 목록 확인
portfolios = sorted(set(r['portfolio'] for r in merged))
print(f"\n   포트폴리오 목록: {portfolios}")
for p in portfolios:
    cnt = sum(1 for r in merged if r['portfolio'] == p)
    print(f"     {p}: {cnt}건")

print("\n완료!")
