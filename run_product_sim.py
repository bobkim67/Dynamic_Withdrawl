"""
product_paths.pkl 기반 인출 시뮬레이션
======================================
3개 펀드 x Rolling 경로 -> Fixed vs Guardrail 비교
12% 인출, +-5% band, W0=1.2억

실행: python run_product_sim.py
출력: product_sim_results.pkl
"""

import pandas as pd
import numpy as np
import pickle
import sys, io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from engine import simulate_withdrawal_on_path

# 파라미터
W0 = 1.2e8
INIT_WR = 0.12
BAND = 0.05
BETA = 0.25  # 총가치 KPI용 (느슨한 기준)

print("1. 데이터 로딩...")
with open('product_paths.pkl', 'rb') as f:
    data = pickle.load(f)

funds = data['funds']
fund_paths = data['fund_paths']

print(f"   펀드: {funds}")
for fund in funds:
    print(f"   {fund}: {len(fund_paths[fund]['paths'])}개 경로")

# 시뮬레이션
print("\n2. 인출 시뮬레이션...")

results = {}

for fund in funds:
    paths = fund_paths[fund]['paths']
    start_dates = fund_paths[fund]['start_dates']
    n = len(paths)

    fund_results = []
    for i, path in enumerate(paths):
        # Fixed (band=99 → 사실상 고정 인출)
        res_f = simulate_withdrawal_on_path(path, init_wr=INIT_WR, band=99.0, beta=BETA, W0=W0)
        # Guardrail
        res_g = simulate_withdrawal_on_path(path, init_wr=INIT_WR, band=BAND, beta=BETA, W0=W0)

        tv_f = res_f['terminal_nav'] + res_f['cum_withdraw']
        tv_g = res_g['terminal_nav'] + res_g['cum_withdraw']

        fund_results.append({
            'start_date': start_dates[i],
            'fixed': {
                'W_series': res_f['W_series'],
                'withdraw_series': res_f['withdraw_series'],
                'terminal_nav': res_f['terminal_nav'],
                'cum_withdraw': res_f['cum_withdraw'],
                'total_value': tv_f,
                'ruin_flag': res_f['ruin_flag'],
                'success': res_f['success'],
            },
            'guard': {
                'W_series': res_g['W_series'],
                'withdraw_series': res_g['withdraw_series'],
                'terminal_nav': res_g['terminal_nav'],
                'cum_withdraw': res_g['cum_withdraw'],
                'total_value': tv_g,
                'ruin_flag': res_g['ruin_flag'],
                'success': res_g['success'],
            },
        })

    results[fund] = fund_results

    # 요약 통계
    tv_fixed = [r['fixed']['total_value'] for r in fund_results]
    tv_guard = [r['guard']['total_value'] for r in fund_results]
    ruin_f = sum(1 for r in fund_results if r['fixed']['ruin_flag'])
    ruin_g = sum(1 for r in fund_results if r['guard']['ruin_flag'])
    guard_wins = sum(1 for f, g in zip(tv_fixed, tv_guard) if g > f)

    CM = 1e7
    print(f"\n   === {fund} ({n}개 경로) ===")
    print(f"   총가치 중앙값: Fixed {np.median(tv_fixed)/CM:,.1f}천만 / Guard {np.median(tv_guard)/CM:,.1f}천만")
    print(f"   총가치 평균:   Fixed {np.mean(tv_fixed)/CM:,.1f}천만 / Guard {np.mean(tv_guard)/CM:,.1f}천만")
    print(f"   파산 경로:     Fixed {ruin_f}/{n} / Guard {ruin_g}/{n}")
    print(f"   Guard 우위:    {guard_wins}/{n} ({guard_wins/n*100:.0f}%)")

    # 실효 인출률
    eff_fixed = [r['fixed']['cum_withdraw'] / (W0 * 10) * 100 for r in fund_results]
    eff_guard = [r['guard']['cum_withdraw'] / (W0 * 10) * 100 for r in fund_results]
    print(f"   실효인출률:    Fixed {np.median(eff_fixed):.1f}% / Guard {np.median(eff_guard):.1f}% (중앙값)")

# 저장
print("\n3. 저장...")
output = {
    'funds': funds,
    'results': results,
    'params': {
        'W0': W0, 'init_wr': INIT_WR, 'band': BAND, 'beta': BETA,
        'T_months': 120,
    },
}

with open('product_sim_results.pkl', 'wb') as f:
    pickle.dump(output, f)

print("   product_sim_results.pkl 저장 완료")
print("\n완료!")
