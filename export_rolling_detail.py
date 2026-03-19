"""
2025.12.31 종료 Rolling 경로 — 일별 수식 기반 엑셀
===================================================
일별 데이터로 Python과 완전 동일한 로직.
BM 지수 = KRW 환산 (T-1 래그 반영). 인출 = 월초 영업일에만.
NAV 로직: 전일 NAV(인출후) × (1 + 전일 수익률) → 월초면 인출.

실행: python export_rolling_detail.py
"""

import pandas as pd
import numpy as np
import sys, io
import xlsxwriter

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

W0 = 1.2e8
INIT_WR = 0.12
BAND = 0.05
TARGET_RATIO = INIT_WR / 12

ASSET_MAP = {
    '미국 성장주':('M2US000G Index','USD'), '미국 가치주':('M2US000V Index','USD'),
    '한국 주식':('M2KR INDEX','KRW'), '호주 주식':('GDDUAS INDEX','USD'),
    '선진국 주식':('TAD09XU Index','USD'), '신흥국 주식':('M2EF Index','USD'),
    '금':('XAU Curncy','USD'), '글로벌 원자재':('SPGSCITR Index','USD'),
    '미국 부동산':('FNRETR Index','USD'), '미국외 부동산':('DWXRSN Index','USD'),
    '글로벌 인프라':('SPGTIND Index','USD'), '미국 물가채권':('LBUTTRUU Index','USD'),
    '미국 하이일드채권':('LF98TRUU Index','USD'),
    '한국 10년국고채권':('KOSEF_10yr','KRW'), '한국 3년국고채권':('KTBTR Index','KRW'),
    '한국 종합채권':('KBPMKTMB Index','KRW'), '한국 단기채권':('BMA03','KRW'),
    '미국 10년국고채권':('IEF','USD'), '미국 종합국채':('IEF','USD'),
    '한국 중장기국공채권':('BMA02','KRW'),
}
MP_NAME_ALIAS = {
    '한국 10년국고채': '한국 10년국고채권',
    '한국 3년국고채': '한국 3년국고채권',
    '한국 단기채': '한국 단기채권',
}
FX_COL = 'USDKRW Curncy'
TARGET_FUNDS = ['Golden Growth', 'MS GROWTH', 'MS STABLE']

# ============================================================================
print("1. 데이터 로딩...")
bm = pd.read_csv('../bm_list', sep='\t', index_col=0, parse_dates=True)
bm = bm[~bm.index.duplicated(keep='last')]  # 중복 날짜 제거 (마지막 값 유지)
mp_pos = pd.read_csv('../MP_Position_20260317', sep='\t')
mp_pos['기준일자'] = pd.to_datetime(mp_pos['기준일자'])
fx = bm[FX_COL].copy()
fx_ret = fx.pct_change()

# 일별 KRW 환산 수익률 (T-1 래그)
asset_returns = {}
for name, (col, ccy) in ASSET_MAP.items():
    if col not in bm.columns:
        continue
    price = bm[col].copy()
    if ccy == 'USD':
        usd_ret = price.pct_change().shift(1)
        krw_ret = (1 + usd_ret) * (1 + fx_ret) - 1
    else:
        krw_ret = price.pct_change()
    asset_returns[name] = krw_ret
returns_df = pd.DataFrame(asset_returns)

# KRW 환산 누적지수
krw_index_df = pd.DataFrame({
    name: (1 + returns_df[name].fillna(0)).cumprod() * 100
    for name in returns_df.columns
})

# MP 비중
fund_daily_weights = {}
for fund in TARGET_FUNDS:
    sub = mp_pos[mp_pos['펀드설명'] == fund][['기준일자', '자산군_소', 'daily_weight_MP']].copy()
    sub['자산군_소'] = sub['자산군_소'].replace(MP_NAME_ALIAS)
    pivot = sub.pivot_table(index='기준일자', columns='자산군_소', values='daily_weight_MP', aggfunc='first')
    pivot = pivot.sort_index().reindex(bm.index).ffill().bfill().fillna(0)
    fund_daily_weights[fund] = pivot

# ============================================================================
start_target = pd.Timestamp('2016-01-01')
end_target = pd.Timestamp('2025-12-31')

print("2. 엑셀 생성 (일별)...")

fname = 'rolling_detail_20160101_20251231.xlsx'
workbook = xlsxwriter.Workbook(fname)

fmt_hdr = workbook.add_format({'bold': True, 'bg_color': '#37474F', 'font_color': 'white',
                                'border': 1, 'text_wrap': True, 'valign': 'vcenter', 'font_size': 9})
fmt_n2 = workbook.add_format({'num_format': '#,##0.00', 'font_size': 9})
fmt_pct = workbook.add_format({'num_format': '0.00%', 'font_size': 9})
fmt_r6 = workbook.add_format({'num_format': '0.000000', 'font_size': 9})
fmt_txt = workbook.add_format({'font_size': 9})
fmt_int = workbook.add_format({'num_format': '#,##0', 'font_size': 9})
fmt_month_row = workbook.add_format({'num_format': '#,##0.00', 'font_size': 9, 'bg_color': '#E3F2FD'})
fmt_month_pct = workbook.add_format({'num_format': '0.00%', 'font_size': 9, 'bg_color': '#E3F2FD'})
fmt_month_txt = workbook.add_format({'font_size': 9, 'bg_color': '#E3F2FD'})

for fund in TARGET_FUNDS:
    print(f"  {fund}...")
    wt_df = fund_daily_weights[fund]
    common_assets = [a for a in wt_df.columns if a in returns_df.columns]
    n_ast = len(common_assets)

    # 기간 내 영업일 (전 자산 유효)
    mask = (returns_df.index >= start_target) & (returns_df.index <= end_target)
    valid = returns_df.loc[mask, common_assets].notna().all(axis=1)
    dates = returns_df.loc[mask].index[valid]

    # 월초 영업일 판별
    month_starts = set()
    seen_months = set()
    for dt in dates:
        ym = (dt.year, dt.month)
        if ym not in seen_months:
            seen_months.add(ym)
            month_starts.add(dt)

    ws = workbook.add_worksheet(fund[:31])
    ws.freeze_panes(2, 3)

    # 열 배치
    # A: #, B: 날짜, C: 월초여부
    # D ~ D+n-1: BM지수(KRW)
    # D+n ~ D+2n-1: 일별수익률 (수식: 현행/전행-1)
    # D+2n ~ D+3n-1: 비중
    # D+3n: 포트일별수익률 (SUMPRODUCT)
    # D+3n+1: G_NAV(인출전)
    # D+3n+2: G_인출시도
    # D+3n+3: W/NAV비율
    # D+3n+4: 밴드상한
    # D+3n+5: 밴드하한
    # D+3n+6: 밴드걸림
    # D+3n+7: G_인출실제
    # D+3n+8: G_NAV(인출후)
    # D+3n+9: G_누적인출
    # D+3n+10: F_인출금
    # D+3n+11: F_NAV(인출후)
    # D+3n+12: F_누적인출

    CB = 3  # BM 시작 열
    col_bm = CB
    col_ret = CB + n_ast
    col_wt = CB + 2 * n_ast
    col_pr = CB + 3 * n_ast       # 포트 일별수익률
    col_nw = col_pr + 1           # NAV(인출없음)
    col_nb = col_nw + 1           # G_NAV(인출전)
    col_wa = col_nb + 1           # G_인출시도
    col_ra = col_wa + 1           # W/NAV비율
    col_up = col_ra + 1           # 밴드상한
    col_lo = col_up + 1           # 밴드하한
    col_bh = col_lo + 1           # 밴드걸림
    col_wd = col_bh + 1           # G_인출실제
    col_gwr = col_wd + 1          # G_인출/NAV비율
    col_na = col_gwr + 1          # G_NAV(인출후)
    col_gr = col_na + 1           # G_수익률반영NAV
    col_cu = col_gr + 1           # G_누적인출
    col_gt = col_cu + 1           # G_총가치
    col_fb = col_gt + 1           # F_NAV(인출전)
    col_fw = col_fb + 1           # F_인출금
    col_fwr = col_fw + 1          # F_인출/NAV비율
    col_fn = col_fwr + 1          # F_NAV(인출후)
    col_fr = col_fn + 1           # F_수익률반영NAV
    col_fc = col_fr + 1           # F_누적인출
    col_ft = col_fc + 1           # F_총가치

    # 파라미터 (row 0)
    ws.write(0, 0, '초기투자금', fmt_txt); ws.write(0, 1, W0, fmt_n2)
    ws.write(0, 2, '연인출률', fmt_txt); ws.write(0, 3, INIT_WR, fmt_pct)
    ws.write(0, 4, 'Band', fmt_txt); ws.write(0, 5, BAND, fmt_pct)
    ws.write(0, 6, '월목표비율', fmt_txt); ws.write_formula(0, 7, '=D1/12', fmt_r6)

    # 헤더 (row 1)
    h = 1
    ws.write(h, 0, '#', fmt_hdr); ws.write(h, 1, '날짜', fmt_hdr); ws.write(h, 2, '월초', fmt_hdr)
    for i, a in enumerate(common_assets):
        ws.write(h, col_bm + i, f'BM_{a}', fmt_hdr)
        ws.write(h, col_ret + i, f'r_{a}', fmt_hdr)
        ws.write(h, col_wt + i, f'w_{a}', fmt_hdr)
    ws.write(h, col_pr, 'r_port', fmt_hdr)
    for c, lbl in [(col_nw,'NAV(인출없음)'),(col_nb,'G_NAV전'),(col_wa,'G_인출시도'),(col_ra,'W/NAV'),
                    (col_up,'상한'),(col_lo,'하한'),(col_bh,'밴드'),
                    (col_wd,'G_인출'),(col_gwr,'G_인출/NAV'),(col_na,'G_NAV후'),(col_gr,'G_수익률NAV'),
                    (col_cu,'G_누적인출'),(col_gt,'G_총가치'),
                    (col_fb,'F_NAV전'),(col_fw,'F_인출'),(col_fwr,'F_인출/NAV'),(col_fn,'F_NAV후'),
                    (col_fr,'F_수익률NAV'),(col_fc,'F_누적인출'),(col_ft,'F_총가치')]:
        ws.write(h, c, lbl, fmt_hdr)

    ws.set_column(0, 0, 5); ws.set_column(1, 1, 11); ws.set_column(2, 2, 4)
    ws.set_column(col_bm, col_bm + n_ast - 1, 12)
    ws.set_column(col_ret, col_ret + n_ast - 1, 10)
    ws.set_column(col_wt, col_wt + n_ast - 1, 8)
    ws.set_column(col_pr, col_ft, 14)

    def cl(r, c):
        return f'{xlsxwriter.utility.xl_col_to_name(c)}{r + 1}'

    # === Row 2: #0 행 (전일 BM 지수만, 인출/NAV 없음) ===
    r0 = 2
    ws.write(r0, 0, 0, fmt_int)
    # 전일 = start_target 직전 영업일의 KRW 환산 지수
    prev_mask = krw_index_df.index < start_target
    if prev_mask.any():
        prev_day = krw_index_df.loc[prev_mask].iloc[-1]
        prev_date = krw_index_df.loc[prev_mask].index[-1]
        ws.write(r0, 1, prev_date.strftime('%Y-%m-%d'), fmt_txt)
        for i, a in enumerate(common_assets):
            val = prev_day[a] if a in prev_day.index and pd.notna(prev_day[a]) else None
            if val is not None:
                ws.write(r0, col_bm + i, val, fmt_n2)

    # 월초 행 번호 추적 (인출시도 = 전월초 인출실제 참조용)
    prev_ms_row = None
    curr_ms_row = None

    # 데이터 행 (row 3부터)
    for di, dt in enumerate(dates):
        r = di + 3  # row 2=#0, row 3=첫 데이터
        is_ms = dt in month_starts
        f_n = fmt_month_row if is_ms else fmt_n2
        f_p = fmt_month_pct if is_ms else fmt_pct
        f_t = fmt_month_txt if is_ms else fmt_txt

        ws.write(r, 0, di + 1, fmt_int)
        ws.write(r, 1, dt.strftime('%Y-%m-%d'), f_t)
        ws.write(r, 2, 'Y' if is_ms else '', f_t)

        # BM 지수 (KRW 환산)
        for i, a in enumerate(common_assets):
            try:
                v = krw_index_df.loc[dt, a]
                val = float(v.iloc[-1]) if hasattr(v, 'iloc') else float(v)
            except (KeyError, IndexError, TypeError):
                val = None
            if val is not None and not np.isnan(val):
                ws.write(r, col_bm + i, val, f_n)
            else:
                ws.write(r, col_bm + i, '', f_t)

        # 일별 수익률 (수식: 전부 현행/전행-1, #0행 덕분에 첫 행도 수식)
        for i, a in enumerate(common_assets):
            bc = col_bm + i
            rc = col_ret + i
            ws.write_formula(r, rc,
                f'=IF(OR({cl(r, bc)}="",{cl(r-1, bc)}=""),0,{cl(r, bc)}/{cl(r-1, bc)}-1)', f_p)

        # 비중
        for i, a in enumerate(common_assets):
            try:
                v = wt_df.loc[dt, a]
                val = float(v.iloc[-1]) if hasattr(v, 'iloc') else float(v)
            except (KeyError, IndexError, TypeError):
                val = 0
            ws.write(r, col_wt + i, val if not np.isnan(val) else 0, f_p)

        # 포트 일별수익률 = SUMPRODUCT
        rr = f'{cl(r, col_ret)}:{cl(r, col_ret + n_ast - 1)}'
        wr = f'{cl(r, col_wt)}:{cl(r, col_wt + n_ast - 1)}'
        ws.write_formula(r, col_pr, f'=SUMPRODUCT({rr},{wr})', f_p)

        # ============================================================
        # Guardrail NAV (일별)
        # 월초: NAV(인출전) → 인출 → NAV(인출후) → 다음날 수익률 적용
        # 비월초: NAV(인출전) = NAV(인출후) 그대로 (인출 없음)
        # 다음행 NAV(인출전) = 전행 NAV(인출후) × (1 + 전행 포트수익률)
        # ============================================================
        pr_cell = cl(r, col_pr)

        # NAV(인출없음) = 전행 × (1 + 전행 수익률)
        if di == 0:
            ws.write_formula(r, col_nw, f'=$B$1*(1+{pr_cell})', f_n)
        else:
            ws.write_formula(r, col_nw, f'={cl(r-1, col_nw)}*(1+{cl(r-1, col_pr)})', f_n)

        if is_ms:
            prev_ms_row = curr_ms_row
            curr_ms_row = r

        # ============================================================
        # Guardrail
        # ============================================================
        if di == 0:
            ws.write_formula(r, col_nb, f'=$B$1', f_n)                    # NAV전 = W0
            ws.write_formula(r, col_wa, f'=$B$1*$D$1/12', f_n)           # 인출시도 = 초기값
        else:
            ws.write_formula(r, col_nb, f'={cl(r-1, col_gr)}', f_n)      # NAV전 = 전행 수익률반영NAV
            if is_ms:
                ws.write_formula(r, col_wa, f'={cl(prev_ms_row, col_wd)}', f_n)  # 인출시도 = 전월초 인출실제
            else:
                ws.write(r, col_wa, 0, f_n)

        # W/NAV, 밴드
        ws.write_formula(r, col_ra,
            f'=IF(OR({cl(r, col_nb)}=0,{cl(r, col_wa)}=0),0,{cl(r, col_wa)}/{cl(r, col_nb)})', fmt_r6)
        ws.write_formula(r, col_up, f'=$H$1*(1+$F$1)', fmt_r6)
        ws.write_formula(r, col_lo, f'=$H$1*(1-$F$1)', fmt_r6)

        if is_ms or di == 0:
            ws.write_formula(r, col_bh,
                f'=IF({cl(r, col_ra)}>{cl(r, col_up)},"상한",'
                f'IF({cl(r, col_ra)}<{cl(r, col_lo)},"하한","밴드내"))', f_t)
            ws.write_formula(r, col_wd,
                f'=IF({cl(r, col_bh)}="상한",{cl(r, col_up)}*{cl(r, col_nb)},'
                f'IF({cl(r, col_bh)}="하한",{cl(r, col_lo)}*{cl(r, col_nb)},'
                f'{cl(r, col_wa)}))', f_n)
        else:
            ws.write(r, col_bh, '', f_t)
            ws.write(r, col_wd, 0, f_n)

        # G_인출/NAV비율 = 인출실제 / NAV전
        ws.write_formula(r, col_gwr,
            f'=IF({cl(r, col_nb)}=0,0,{cl(r, col_wd)}/{cl(r, col_nb)})', fmt_pct)

        # NAV(인출후) = NAV전 - 인출실제
        ws.write_formula(r, col_na, f'={cl(r, col_nb)}-{cl(r, col_wd)}', f_n)

        # 수익률반영NAV = NAV후 × (1 + 당일 포트수익률) → 다음행의 NAV전
        ws.write_formula(r, col_gr, f'={cl(r, col_na)}*(1+{cl(r, col_pr)})', f_n)

        # 누적인출
        if di == 0:
            ws.write_formula(r, col_cu, f'={cl(r, col_wd)}', f_n)
        else:
            ws.write_formula(r, col_cu, f'={cl(r-1, col_cu)}+{cl(r, col_wd)}', f_n)

        # 총가치 = 수익률반영NAV + 누적인출
        ws.write_formula(r, col_gt, f'={cl(r, col_gr)}+{cl(r, col_cu)}', f_n)

        # ============================================================
        # Fixed
        # ============================================================
        if di == 0:
            ws.write_formula(r, col_fb, f'=$B$1', f_n)                   # NAV전 = W0
            ws.write_formula(r, col_fw, f'=$B$1*$D$1/12', f_n)          # 인출 = 고정
            ws.write_formula(r, col_fn, f'={cl(r, col_fb)}-{cl(r, col_fw)}', f_n)  # NAV후
        else:
            ws.write_formula(r, col_fb, f'={cl(r-1, col_fr)}', f_n)     # NAV전 = 전행 수익률반영NAV
            if is_ms:
                ws.write_formula(r, col_fw,
                    f'=MIN($B$1*$D$1/12,MAX({cl(r, col_fb)},0))', f_n)  # 인출 (NAV 있으면)
            else:
                ws.write(r, col_fw, 0, f_n)
            ws.write_formula(r, col_fn, f'=MAX({cl(r, col_fb)}-{cl(r, col_fw)},0)', f_n)

        # F_인출/NAV비율 = 인출 / NAV전
        ws.write_formula(r, col_fwr,
            f'=IF({cl(r, col_fb)}=0,0,{cl(r, col_fw)}/{cl(r, col_fb)})', fmt_pct)

        # 수익률반영NAV = NAV후 × (1 + 당일 포트수익률)
        ws.write_formula(r, col_fr, f'={cl(r, col_fn)}*(1+{cl(r, col_pr)})', f_n)

        # 누적인출
        if di == 0:
            ws.write_formula(r, col_fc, f'={cl(r, col_fw)}', f_n)
        else:
            ws.write_formula(r, col_fc, f'={cl(r-1, col_fc)}+{cl(r, col_fw)}', f_n)

        # 총가치 = 수익률반영NAV + 누적인출
        ws.write_formula(r, col_ft, f'={cl(r, col_fr)}+{cl(r, col_fc)}', f_n)

    print(f"    {fund}: {len(dates)}행 완료")

# 요약
ws_s = workbook.add_worksheet('요약')
for c, lbl in enumerate(['펀드','기간','초기투자금','인출률','Band','모델','데이터']):
    ws_s.write(0, c, lbl, fmt_hdr)
for i, fund in enumerate(TARGET_FUNDS):
    ws_s.write(i+1, 0, fund, fmt_txt)
    ws_s.write(i+1, 1, '2016-01 ~ 2025-12', fmt_txt)
    ws_s.write(i+1, 2, W0, fmt_n2)
    ws_s.write(i+1, 3, INIT_WR, fmt_pct)
    ws_s.write(i+1, 4, BAND, fmt_pct)
    ws_s.write(i+1, 5, '월초인출, 일별수익률 복리', fmt_txt)
    ws_s.write(i+1, 6, 'KRW환산 일별 (USD T-1래그)', fmt_txt)

workbook.close()
print(f"\n{fname} 저장 완료")
print("완료!")
