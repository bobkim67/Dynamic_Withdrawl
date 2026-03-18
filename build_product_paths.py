"""
product_mp 기반 포트폴리오 월간 수익률 + Rolling 경로 생성
==========================================================
bm_list (일별 가격지수) + MP_Position_20260317 (일별 MP 비중) ->
3개 펀드별 KRW 기준 일별 수익률 (일별 MP 비중 반영) -> 월간 수익률 -> Rolling 10년 경로 -> pkl

MP 비중 적용:
  - MP_Position 데이터 있는 날짜: 해당 일자 MP 비중 사용
  - MP_Position 이전 날짜: 가장 오래된 MP 비중으로 소급 적용 (backfill)

USD 자산: T-1 가격 x T 환율변동 적용 (KRW 환산)
KRW 자산: 그대로 사용

실행: python build_product_paths.py
출력: product_paths.pkl
"""

import pandas as pd
import numpy as np
import pickle
import sys, io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================================
# 자산 매핑
# ============================================================================

ASSET_MAP = {
    '미국 성장주':      ('M2US000G Index',   'USD'),
    '미국 가치주':      ('M2US000V Index',   'USD'),
    '한국 주식':        ('M2KR INDEX',       'KRW'),
    '호주 주식':        ('GDDUAS INDEX',     'USD'),
    '선진국 주식':      ('TAD09XU Index',    'USD'),
    '신흥국 주식':      ('M2EF Index',       'USD'),
    '금':              ('XAU Curncy',        'USD'),
    '글로벌 원자재':    ('SPGSCITR Index',    'USD'),
    '미국 부동산':      ('FNRETR Index',      'USD'),
    '미국외 부동산':    ('DWXRSN Index',      'USD'),
    '글로벌 인프라':    ('SPGTIND Index',     'USD'),
    '미국 물가채권':    ('LBUTTRUU Index',    'USD'),
    '미국 하이일드채권': ('LF98TRUU Index',   'USD'),
    '한국 10년국고채권': ('KOSEF_10yr',       'KRW'),
    '한국 3년국고채권':  ('KTBTR Index',      'KRW'),
    '한국 종합채권':    ('KBPMKTMB Index',    'KRW'),
    '한국 단기채권':    ('BMA03',             'KRW'),
    '미국 10년국고채권': ('IEF',              'USD'),
    '미국 종합국채':    ('IEF',              'USD'),
    '한국 중장기국공채권': ('BMA02',           'KRW'),
}

# MP 자산명 표준화 (변형 이름 → ASSET_MAP 키로 통일)
MP_NAME_ALIAS = {
    '한국 10년국고채': '한국 10년국고채권',
    '한국 3년국고채':  '한국 3년국고채권',
    '한국 단기채':     '한국 단기채권',
}

FX_COL = 'USDKRW Curncy'
TARGET_FUNDS = ['Golden Growth', 'MS GROWTH', 'MS STABLE']


# ============================================================================
# 데이터 로딩
# ============================================================================

print("1. 데이터 로딩...")
bm = pd.read_csv('../bm_list', sep='\t', index_col=0, parse_dates=True)

mp_pos = pd.read_csv('../MP_Position_20260317', sep='\t')
mp_pos['기준일자'] = pd.to_datetime(mp_pos['기준일자'])

print(f"   bm_list: {bm.index[0].date()} ~ {bm.index[-1].date()} ({len(bm)}일)")
print(f"   MP_Position: {len(mp_pos)}행, 기간: {mp_pos['기준일자'].min().date()} ~ {mp_pos['기준일자'].max().date()}")
print(f"   대상 펀드: {TARGET_FUNDS}")


# ============================================================================
# 일별 MP 비중 테이블 구축 (펀드별)
# ============================================================================

print("\n2. 일별 MP 비중 테이블 구축...")

fund_daily_weights = {}  # fund -> DataFrame (index=date, columns=자산군_소)

for fund in TARGET_FUNDS:
    sub = mp_pos[mp_pos['펀드설명'] == fund][['기준일자', '자산군_소', 'daily_weight_MP']].copy()

    # MP 자산명 표준화
    sub['자산군_소'] = sub['자산군_소'].replace(MP_NAME_ALIAS)

    # pivot: 날짜 x 자산 -> 비중
    pivot = sub.pivot_table(index='기준일자', columns='자산군_소', values='daily_weight_MP', aggfunc='first')
    pivot = pivot.sort_index()

    # bm_list의 전체 날짜 인덱스로 확장
    full_idx = bm.index
    pivot_full = pivot.reindex(full_idx)

    # MP_Position 이전 날짜는 가장 오래된 MP로 소급 (bfill 후 ffill)
    # 1) 먼저 forward-fill: MP 변경 이후 유지
    # 2) 그 다음 back-fill: MP 데이터 이전 날짜에 최초 MP 적용
    pivot_full = pivot_full.ffill().bfill()

    # MP에 없는 자산은 0으로 처리
    pivot_full = pivot_full.fillna(0)

    fund_daily_weights[fund] = pivot_full

    # 리밸런싱 시점 확인
    changes = pivot_full.diff().abs().sum(axis=1)
    rebal_dates = changes[changes > 0.001].index
    print(f"   {fund}: {pivot.index[0].date()} ~ {pivot.index[-1].date()}, "
          f"MP 리밸런싱 {len(rebal_dates)}회, 자산 {len(pivot.columns)}개")
    print(f"     소급 적용: {full_idx[0].date()} ~ {pivot.index[0].date()} (최초 MP 고정)")


# ============================================================================
# KRW 환산 일별 수익률 계산
# ============================================================================

print("\n3. KRW 환산 일별 수익률 계산...")

fx = bm[FX_COL].copy()
fx_ret = fx.pct_change()

asset_returns_krw = {}
for asset_name, (col, ccy) in ASSET_MAP.items():
    if col not in bm.columns:
        print(f"   [SKIP] {asset_name}: {col} 컬럼 없음")
        continue
    price = bm[col].copy()
    if ccy == 'USD':
        # T일 KRW 기준가 = USD가격(T-1) × 환율(T)
        # KRW 수익률(T) = USD수익률(T-1) × 환율변동(T)
        usd_ret = price.pct_change().shift(1)  # T-1일 USD 수익률을 T일에 적용
        krw_ret = (1 + usd_ret) * (1 + fx_ret) - 1
    else:
        krw_ret = price.pct_change()
    asset_returns_krw[asset_name] = krw_ret

returns_df = pd.DataFrame(asset_returns_krw)
print(f"   수익률 DataFrame: {returns_df.shape}")


# ============================================================================
# 포트폴리오 일별 수익률 (일별 MP 비중 반영) -> 월간 수익률
# ============================================================================

print("\n4. 포트폴리오 월간 수익률 계산 (일별 MP 비중 반영)...")

fund_monthly = {}

for fund in TARGET_FUNDS:
    wt_df = fund_daily_weights[fund]  # 일별 비중 (날짜 x 자산)

    # 공통 자산 & 날짜
    common_assets = [a for a in wt_df.columns if a in returns_df.columns]
    missing = [a for a in wt_df.columns if a not in returns_df.columns and (wt_df[a] > 0.001).any()]
    if missing:
        print(f"   [WARN] {fund}: 매핑 없는 자산 {missing}")

    # 일별 비중 (공통 자산만, 정규화)
    wt_common = wt_df[common_assets].copy()
    wt_sum = wt_common.sum(axis=1)
    wt_norm = wt_common.div(wt_sum, axis=0)  # 매일 합계=1로 정규화

    # 일별 수익률 (공통 자산만)
    ret_common = returns_df[common_assets]

    # 유효 행: 비중과 수익률 모두 존재하는 날짜
    valid_mask = ret_common.notna().all(axis=1) & wt_norm.notna().all(axis=1)
    ret_valid = ret_common.loc[valid_mask]
    wt_valid = wt_norm.loc[valid_mask]

    # 일별 포트폴리오 수익률 = 각 자산 수익률 x 해당일 비중 합산
    daily_ret = (ret_valid * wt_valid).sum(axis=1)

    # 월간 수익률
    monthly_ret = (1 + daily_ret).resample('MS').prod() - 1
    monthly_ret = monthly_ret.dropna()

    fund_monthly[fund] = monthly_ret

    ann_ret = (1 + monthly_ret.mean()) ** 12 - 1
    ann_vol = monthly_ret.std() * np.sqrt(12)
    print(f"   {fund}: {monthly_ret.index[0].date()} ~ {monthly_ret.index[-1].date()}, "
          f"{len(monthly_ret)}개월, 연수익률 {ann_ret*100:.2f}%, 연변동성 {ann_vol*100:.2f}%")


# ============================================================================
# Rolling 10년 경로 생성 (월초 기준)
# ============================================================================

print("\n5. Rolling + Bootstrap 경로 생성...")

from engine import generate_paths_bootstrap

T_MONTHS = 120
BOOTSTRAP_N_GENERATE = 5000
BOOTSTRAP_N_SAMPLE = 181
BOOTSTRAP_SEED = 42

fund_paths = {}

for fund in TARGET_FUNDS:
    mr = fund_monthly[fund]
    mr_values = mr.values
    n = len(mr)

    # Rolling 경로
    rolling_paths = []
    rolling_starts = []
    for i in range(n - T_MONTHS + 1):
        rolling_paths.append(mr.iloc[i:i + T_MONTHS].values)
        rolling_starts.append(mr.index[i])

    # Bootstrap 경로 (181개 샘플링)
    bootstrap_paths = generate_paths_bootstrap(
        mr_values, T_months=T_MONTHS,
        n_paths=BOOTSTRAP_N_GENERATE, seed=BOOTSTRAP_SEED
    )
    # 181개 서브샘플링
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    sample_idx = rng.choice(len(bootstrap_paths), size=BOOTSTRAP_N_SAMPLE, replace=False)
    bootstrap_sampled = [bootstrap_paths[i] for i in sample_idx]

    fund_paths[fund] = {
        'rolling_paths': rolling_paths,
        'rolling_start_dates': rolling_starts,
        'bootstrap_paths': bootstrap_sampled,
        'monthly_returns': mr,
    }
    print(f"   {fund}: Rolling {len(rolling_paths)}개 + Bootstrap {len(bootstrap_sampled)}개")


# ============================================================================
# pkl 저장
# ============================================================================

print("\n6. 저장...")

output = {
    'funds': TARGET_FUNDS,
    'fund_daily_weights': {f: fund_daily_weights[f].to_dict() for f in TARGET_FUNDS},
    'fund_paths': fund_paths,
    'T_months': T_MONTHS,
    'asset_map': ASSET_MAP,
}

with open('product_paths.pkl', 'wb') as f:
    pickle.dump(output, f)

print(f"   product_paths.pkl 저장 완료")
print(f"\n완료!")
for fund in TARGET_FUNDS:
    fp = fund_paths[fund]
    print(f"  {fund}: Rolling {len(fp['rolling_paths'])}개 + Bootstrap {len(fp['bootstrap_paths'])}개")
