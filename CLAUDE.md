# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고하는 가이드입니다.

## 프로젝트 목적

과거 데이터 기반 퇴직 포트폴리오 인출률 백테스트 및 최적화 시스템.
한국 자산운용 실무에서 사용할 증거 기반 인출율 추천 도구.

## 실행 명령어

```bash
# Historical 그리드 서치 (grid_results_full.pkl 생성, 수 분 소요)
python grid_search.py

# GBM 그리드 서치 (gbm_results.pkl 생성, ~2분 소요)
python gbm_grid_search.py

# 엔진 검증 테스트 실행
python engine.py

# Streamlit 뷰어 실행 (grid_results_full.pkl + gbm_results.pkl 필요)
streamlit run viewer.py
```

## 아키텍처

계산과 시각화를 완전히 분리한다.

**계산 파이프라인** (오프라인, 스크립트):
- Historical: `withdrawal_backtest.py` -> `engine.py` -> `grid_search.py` -> `grid_results_full.pkl`
- GBM MC: `engine.py` -> `gbm_grid_search.py` -> `gbm_results.pkl`

**시각화** (Streamlit, 읽기 전용):
`viewer.py`가 `grid_results_full.pkl` + `gbm_results.pkl`을 로드 -> 차트, 필터링, drill-down

Streamlit에서 무거운 시뮬레이션을 돌리지 않는다. 단, Strategy Detail 탭의 NAV 경로 시뮬레이션은 on-demand로 실행.

### 데이터 흐름

1. `benchmark_data.pkl` (8개 자산 일별 가격지수, 2001~2025)
2. `withdrawal_backtest.py::DataPreprocessor`가 일별 수익률 + 월초 거래일 플래그 계산, `PortfolioCalculator`가 6개 포트폴리오 수익률 생성
3. `engine.py::daily_to_monthly_returns`로 월간 변환, `generate_paths_rolling`/`generate_paths_bootstrap`로 수익률 경로 생성
4. `engine.py::simulate_withdrawal_on_path`로 경로별 인출 시뮬레이션, `evaluate_strategy`로 집계
5. `grid_search.py`가 파라미터 그리드 순회 -> `pareto_frontier` / `select_optimal` 플래그와 함께 결과 저장
6. `viewer.py`가 pkl 로드 후 7개 탭 렌더링 (설득 흐름):
   1) Guardrail 효과 — Fixed vs Guardrail 비교 인포그래픽 (240개월 시뮬레이션)
   DV) 데이터 검증 — 실제 과거 데이터 기반 스파게티 차트 + 기말잔액 분포
   2) 이론 분석 (GBM) — 변동성별 Fixed vs Guardrail 체계 비교
   3) 실제 포트폴리오 검증 — 2컬럼 (좌: Historical 히트맵/곡선, 우: GBM 히트맵/곡선) + full-width 효과 분석/분석결과
   4) Band 최적화 — Band별 성공률 곡선 (간결화)
   5) Vol-Adjusted — 변동성 조정 Guardrail 전략 분석 (Fixed vs Guardrail vs Vol-Adjusted 3종 비교)
   6) Assumptions — 시뮬레이션 전제 조건 및 파라미터 정리

### 모듈별 역할

- **`withdrawal_backtest.py`**: `DataPreprocessor`, `PORTFOLIOS` 딕셔너리 (6개 포트폴리오: Port_4.0%~Port_9.0%), `BENCHMARK_MAPPING`, `PortfolioCalculator`. engine.py, grid_search.py, viewer.py가 공통으로 import하는 데이터 레이어.
- **`engine.py`**: 핵심 시뮬레이션 엔진. 경로 생성(rolling/bootstrap/GBM), `simulate_withdrawal_on_path` (scalar), `simulate_withdrawal_on_paths_vectorized` (numpy 벡터화), `evaluate_strategy` / `evaluate_gbm_strategy` (집계 지표), `pareto_frontier`, `select_optimal`. `__main__` 블록에 검증 테스트 포함 (벡터화 vs scalar 일치 확인).
- **`grid_search.py`**: Historical 파라미터 그리드 정의. 출력: `grid_results_full.pkl`, `grid_results_summary.xlsx`.
- **`gbm_grid_search.py`**: GBM MC 파라미터 그리드 (mu 2-12%, sigma 2-20%, init_wr 3-15%, band 5종, beta 5종). Fixed 케이스에 Closed-form 확률도 동시 계산 (`cf_success_rate`). 출력: `gbm_results.pkl` (130,625 레코드), `gbm_results_summary.xlsx`.
- **`viewer.py`**: Streamlit 앱 (7개 탭, 설득 흐름 구조). `@st.cache_data`로 데이터 캐싱. `BETA_LABELS` 5단계. 사이드바에 글로벌 컨트롤 3개 (원본 대비 기말잔액 비율 / 초기 인출률 / 데이터 기반). Tab 1(Fixed vs Guardrail 비교 인포그래픽), Tab DV(데이터 검증 스파게티), Tab 2(GBM 변동성별 분석), Tab 3(2컬럼: 좌 Historical + 우 GBM 비교 + full-width 효과 분석/분석결과), Tab 4(Band별 성공률 곡선만), Tab 5(Vol-Adjusted 3종 비교), Tab 6(Assumptions). 디폴트 데이터 기반: Bootstrap.
- **`guardrail_infographic.py`**: standalone matplotlib 인포그래픽 (NAV + 인출 화살표). 출력: `guardrail_infographic.png/svg`.
- **`guardrail_persuasion.py`**: standalone matplotlib 비교 차트 (Fixed vs Guardrail NAV + 인출액). 출력: `guardrail_persuasion.png/svg`.
- **`dynamic_simulator.py`**: 기존 시뮬레이터 (Fixed/Guardrails 전략). 참조용. 추후 통합 여부 판단.

### 신규 상품 펀드 파이프라인 (v4+)

- **`build_product_paths.py`**: bm_list + MP_Position → 3개 펀드(Golden Growth, MS GROWTH, MS STABLE)의 KRW 환산 일별 수익률 → 월간 수익률 → Rolling + Bootstrap 경로 생성. 출력: `product_paths.pkl`. USD 자산은 T-1 래그 반영 (`usd_ret.shift(1)`). MP 자산명 표준화 (`MP_NAME_ALIAS`).
- **`grid_search_product.py`**: product_paths.pkl 기반 grid search. 기존 grid_results_full.pkl에 합산. 3개 펀드 × rolling/bootstrap.
- **`run_product_sim.py`**: product_paths.pkl 기반 인출 시뮬레이션 (Fixed vs Guardrail). 출력: `product_sim_results.pkl`.
- **`export_rolling_detail.py`**: 2016-01~2025-12 rolling 경로의 **일별 수식 기반 엑셀** 출력. 전체 셀이 엑셀 수식. BM 지수는 KRW 환산 누적지수. 월초에만 인출 발생. NAV 로직: `NAV(인출후) = NAV(인출전) - 인출`, `수익률반영NAV = NAV(인출후) × (1 + r)`, 다음행 `NAV(인출전) = 전행 수익률반영NAV`.
- **`proposal_charts.py`**: 판매사 본부 설득용 HTML 제안서 (11장 + 부록). Port_9.0% / 12% 인출 / Band +-5%.

### Efficient Frontier 분석 도구

- **`_frontier.py`**: 비율밴드 efficient frontier + vol-adjusted 비대칭 밴드 alpha 시각화. 3펀드 × 3차트(NAV/인출, Tot/Std, Tot/Worst) = 9탭 HTML. 출력: `efficient_frontier_vol_ratio.html`. Vol-adj 모델: 1σ SNR 기준 bl/bh 스위칭.
- **`_frontier_xlsx.py`**: frontier 데이터를 엑셀로 출력. Frontier 곡선 + FA + FR + G5% + Vol 5/8%, 5/10%, 5/15%. 출력: `frontier_data.xlsx` (6시트).
- **`quadrant_ruin_worst.html`**: 2x2 사분면 차트 (X: Worst Cut, Y: Ruin Rate). FA vs FR vs Guardrail vs Vol-Adj 포지셔닝.

### 전략 비교 핵심 결론

- **FA vs FR vs Guardrail**: Guardrail = FR의 파산 0건 + FA의 인출 안정성 결합
- **Tot에서는 FR이 항상 우위** (Guardrail은 -1~2 양보). 알파는 인출 안정성(worst cut)에서만 존재
- **Vol-Adj 5/8~15%**: 고변동성 펀드(MS GROWTH)에서 worst cut 개선 -5%p 수준
- **셀링 포인트**: "FR처럼 파산 0건이면서, 하락장에서 인출이 급감하지 않음"

### 신규 상품 데이터 흐름

1. `../bm_list` (17개 자산 일별 가격지수 + USDKRW 환율)
2. `../MP_Position_20260317` (일별 MP 비중, 리밸런싱 시점에만 변경)
3. `build_product_paths.py`: 일별 KRW 환산 수익률 (USD T-1 래그) → 월간 복리 → Rolling 55개 + Bootstrap 181개
4. `grid_search_product.py`: 9개 포트폴리오 총 27,000건 grid results
5. `viewer.py`: 기존 6개 + 신규 3개 포트폴리오 통합 표시

### USD 자산 KRW 환산 규칙

- **T일 KRW 기준가 = USD가격(T-1) × 환율(T)**
- 일별: `krw_ret = (1 + usd_ret.shift(1)) * (1 + fx_ret) - 1`
- 월간 엑셀에서 이 래그를 직접 구현하면 **한 달 래그**가 되므로, 반드시 **일별로 래그 적용 후 월간 복리 합산**해야 함

### 인출 시뮬레이션 순서 — engine.py (매월 반복)

**현행 engine.py 로직 (Ordinary Annuity):**
1. 수익률 적용: `W_t = W_{t-1} * (1 + r_t)`
2. 인출 시도액: `prev_withdraw` (이전 달 인출액 유지, 초기값 = `W0 * init_wr / 12`)
3. 수익률 기반 보정 (옵션): lookback 기간 -> threshold 확인 -> +-adj%
3.5. **변동성 밴드 조정** (vol_adj=True일 때):
   - `σ_realized = std(최근 12개월 수익률, ddof=1) × √12` (연율화)
   - `sigma_ratio = clip(σ_realized / σ_target, 0.5, 2.0)`
   - `effective_band = base_band × sigma_ratio`
   - 고변동성 → 밴드 확대 (인출 조절 강화), 저변동성 → 밴드 축소 (인출 안정 유지)
   - `sigma_ratio = 1.0`이면 기존 guardrail과 동일 (backward compatible)
4. Guardrail 밴드 (W/NAV 비율 기준):
   - `target_ratio = init_wr / 12` (목표 월비율)
   - `upper_ratio = target_ratio * (1 + effective_band)`, `lower_ratio = target_ratio * (1 - effective_band)`
   - `ratio = withdraw / W_t` → 비율이 밴드 벗어나면 `upper_ratio * W_t` 또는 `lower_ratio * W_t`로 조정
   - 밴드 내이면 이전 인출액 그대로 유지 (고정 base로 리셋하지 않음)
5. 인출 실행: `W_t = W_t - withdraw_final`, `prev_withdraw = withdraw_final` (다음 달 기준 갱신)

### 그리드 서치 전략 유형

- **fixed_baseline**: `band=99.0` (사실상 guardrail 없음), `adj_on=False`. 순수 고정 인출.
- **dynamic**: 실제 guardrail 밴드 (0.05~0.20), 고정 밴드 폭.
- **vol_adjusted**: `vol_adj=True`. 실현 변동성으로 밴드 폭을 동적 조정. `sigma_target`은 포트폴리오의 `target_risk` (PORTFOLIOS 딕셔너리에 정의된 목표 변동성).

### 성공/실패 정의

- **Ruin (파산)**: 경로 중 `W_t <= 0` 발생
- **Terminal 실패**: `W_T < beta * W0` (원본 대비 기말잔액 비율: 0.1/0.25/0.5/0.75/1.0 중 선택)
- **성공**: ruin 없음 AND terminal 실패 없음

## 차후 개발 예정

아래 항목은 코딩 전 검토 필요.

1. ~~**Tab 1 메커니즘 시각화 강화**~~ — v6.0에서 Fixed vs Guardrail 비교 인포그래픽으로 교체 완료
2. ~~**Beta 위젯 위치 재배치**~~ — v5.3에서 사이드바 글로벌 컨트롤로 통합
3. ~~**Beta 값 확장**~~ — grid_search.py, gbm_grid_search.py 모두 beta=[0.1, 0.25, 0.5, 0.75, 1.0] 적용 + pkl 재생성 완료
4. ~~**모든 시각화에 핵심 코멘트**~~ — v5에서 완료
5. ~~**Band Trade-off 비교 그래프**~~ — v5.3에서 간결화 완료
6. ~~**전략 상세 + 나의 전략 조합 탭 통합**~~ — v5에서 해소
7. ~~**NAV 경로 시뮬레이션 탭 복원**~~ — 불필요로 삭제
8. ~~**engine.py Guardrail 로직 통일**~~ — v6.0에서 W/NAV 비율 밴드 방식으로 통일 완료
9. ~~**Tab 3 리팩토링**~~ — v7.0에서 2컬럼 레이아웃 + Section 4,5 삭제 + 분석결과 확장 완료
10. ~~**Volatility-Adjusted Guardrail**~~ — v7.2에서 engine.py vol_adj 파라미터 + grid_search/gbm_grid_search vol_adjusted 전략 추가 + viewer.py 3종 비교 완료
11. **engine.py 월초 인출(Annuity Due) 로직 전환** — 엑셀은 월초 인출로 구현 완료. engine.py는 아직 Ordinary Annuity. 수치 차이 0.6% 수준. 통일 검토 필요.
12. **엑셀 검증열과 Python 값 일치 확인** — 첫 행 수익률 0% 문제 수정 완료. 최종 총가치 일치 여부 검증 진행 중.

## 내부 의사결정 장표 (6장) — 제안서 구조

### 장표 1. 시장 배경

핵심 메시지: 퇴직연금 인출기 진입 가속 — 적립기 솔루션만으로는 부족

- DB→DC/IRP 전환 추세, 베이비붐 세대 퇴직 본격화
- 퇴직급여 일시금 vs 연금 선택 비율 변화
- 기존 상품 라인업: 적립기(MS GROWTH/STABLE, TDF, TIF, Golden Growth) → 인출기 공백
- 시장 기회: guardrail 기반 동적 인출 솔루션 수요

### 장표 2. 고정 인출의 위험

핵심 메시지: Fixed 인출은 시장 하락기에 원금 잠식 → 파산 리스크

- viewer.py Tab 1 그림: Fixed vs Guardrail 비교 인포그래픽 (240개월 시뮬레이션)
- Sequence-of-returns risk 시각화
- "같은 평균 수익률이라도 초반 하락 시 회복 불가" 메시지

### 장표 3. 솔루션: Guardrail 분배형 자펀드

핵심 메시지: 기존 모펀드에 분배 기능만 얹는 구조 — 신규 운용 불필요

- 구조도: 수급자 ← 분배형 자펀드(신규, 인출 규칙만) ← 모펀드(기존, 운용 그대로)
- 왜 자펀드인가: 모펀드 규약 변경 불필요, 기존 적립기 자펀드와 병렬
- Guardrail 메커니즘 도식: 밴드 내 유지 → 상한 초과 시 인출 삭감 / 하한 미달 시 인출 증액
- 대상 모펀드: MS GROWTH(성장형), MS STABLE(안정형), Golden Growth

### 장표 4. 백테스트 결과 — Historical

핵심 메시지: 과거 10년(2016~2025) 실제 MP 비중 기반 시뮬레이션

- rolling_detail Excel 기반: MS GROWTH / MS STABLE / Golden Growth
- Fixed vs Guardrail NAV 경로 비교 차트
- 누적 인출금 + 기말잔액 비교
- 연 12% 인출 시나리오, Band 5% 기준

### 장표 5. 백테스트 결과 — Monte Carlo & 최적화

핵심 메시지: 다양한 시장 환경에서도 Guardrail이 성공률 제고

- viewer.py Tab 3 그림: Historical(좌) + GBM(우) 히트맵
- 포트폴리오별 성공률 비교 (Fixed vs Guardrail vs Vol-Adjusted)
- Band 최적화 결과 (viewer Tab 4)
- 핵심 수치: 성공률 XX%p 개선, 기말잔액 XX% 보존

### 장표 6. 대상 모펀드 실적

핵심 메시지: 검증된 운용 트랙 레코드 위에 인출 솔루션 탑재

- DB_OCIO_Webview 데이터 활용: 모펀드 NAV 추이, AUM, 설정후 수익률
- 자산배분 현황 (8분류 도넛)
- Brinson PA 요약 (자산군별 기여수익률)
- "운용은 이미 증명됨 — 인출 규칙만 추가"

### 장표 4~5 데이터 소스

- viewer.py + rolling_detail Excel에서 캡처/추출
- 장표 6은 DB_OCIO_Webview에서 추출
- 포맷: 미정 (HTML / PPT 검토 중)

---

## SCIP DB에서 벤치마크 데이터 가져오기

### DB 접속

```python
import pymysql
conn = pymysql.connect(host='192.168.195.55', user='solution', password='Solution123!',
                       db='SCIP', charset='utf8mb4')
```

### 종목 검색 (back_dataset)

```sql
-- 이름/심볼로 검색
SELECT id, name, ISIN, symbol FROM back_dataset
WHERE name LIKE '%키워드%' OR symbol LIKE '%키워드%'

-- 예: S&P 500 → id=24, Gold → id=408, IEF → id=74
```

### 사용 가능한 dataseries 확인

종목마다 연결된 dataseries가 다르므로, 데이터 추출 전 반드시 확인:

```sql
SELECT dp.dataseries_id, ds.name, COUNT(*) as cnt,
       MIN(DATE(dp.timestamp_observation)) as first_date,
       MAX(DATE(dp.timestamp_observation)) as last_date
FROM back_datapoint dp
JOIN back_dataseries ds ON dp.dataseries_id = ds.id
WHERE dp.dataset_id = {찾은 id}
GROUP BY dp.dataseries_id, ds.name
```

### 주요 dataseries_id

| id | name | blob 형태 | 용도 | 대상 |
|----|------|-----------|------|------|
| 6 | FG Return | `{"USD": x, "KRW": y}` | 가격/수익률 지수 | ETF, 주식지수 |
| 9 | TOT RETURN INDEX NET DVDS | 단순 숫자 | 총수익지수 | Bloomberg 지수 |
| 15 | FG Price | `{"USD": x, "KRW": y}` | 가격 | ETF |
| 33 | KIS Bond Index | `{"totRtnIndex": "149.08", ...}` | 채권 총수익지수 | KIS 채권지수 |
| 39 | FG Total Return Index | 단순 숫자 | 무헷지 총수익 | MSCI 등 |

### 데이터 추출 + blob 파싱

```python
import pandas as pd, json

df = pd.read_sql('''
    SELECT DATE(timestamp_observation) as date, data
    FROM back_datapoint
    WHERE dataset_id = %s AND dataseries_id = %s
    ORDER BY timestamp_observation
''', conn, params=[dataset_id, dataseries_id])

def parse_blob(blob, key=None):
    if isinstance(blob, (bytes, bytearray)):
        blob = blob.decode('utf-8')
    blob = blob.strip()
    if blob.startswith('{'):
        obj = json.loads(blob)
        if key:
            return float(obj[key])
        if isinstance(obj, dict):
            return {k: float(v) for k, v in obj.items()}
    return float(blob.replace(',', ''))

# 사용 예:
# dataseries_id=6 (FG Return): parse_blob(blob, 'KRW') 또는 parse_blob(blob, 'USD')
# dataseries_id=33 (KIS Bond): parse_blob(blob, 'totRtnIndex')
# dataseries_id=9 (Bloomberg TR): parse_blob(blob) → 단순 숫자
```

### bm_list에 새 지수 추가하는 패턴

```python
# 1. SCIP에서 추출
prices = []
for _, r in df.iterrows():
    prices.append({'date': r['date'], 'NEW_COL': parse_blob(r['data'], 'USD')})
new_df = pd.DataFrame(prices).set_index('date')
new_df.index = pd.to_datetime(new_df.index)

# 2. 기존 bm_list에 join
bm = pd.read_csv('../bm_list', sep='\t', index_col=0, parse_dates=True)
bm = bm.join(new_df, how='left')
bm.to_csv('../bm_list', sep='\t')
```

### 주요 dataset_id 참조

| id | name | symbol | 비고 |
|----|------|--------|------|
| 24 | S&P 500 ETF Trust | SPY-US | dataseries=6, blob `{"USD","KRW"}` |
| 74 | iShares 7-10Y Treasury | IEF-US | dataseries=6, blob `{"USD","KRW"}` |
| 152 | KIS 종합채권국공채4~5Y | BMA02 | dataseries=33, blob totRtnIndex |
| 247 | Bloomberg EM Gov Bond | LHMN29341 | dataseries=6, blob `{"USD","KRW"}` |
| 273 | KAP 종합채권 AA- | KBPMKTMB Index | dataseries=9, 단순숫자 |
| 279 | KIS 종합채권 | BMM01 | dataseries=33, blob totRtnIndex |
| 408 | Gold Spot | XAU Curncy | dataseries=15, blob `{"USD","KRW"}` |

---

## 코딩 컨벤션

- 과도한 모듈화 금지. 재사용되지 않는 함수 만들지 않기.
- 선형적이고 읽기 쉬운 코드. 시간 흐름이 명확해야 함.
- 섹션별 `# ====` 주석 블록으로 구분.
- 함수는 순수 계산 또는 재사용 로직에만 사용.
- 분석 코드이지 프로덕션 라이브러리가 아님.
- 한국어 주석/UI, 금융 전문용어는 영문 병기.

## Git Workflow

- PR 기반 개발
- 브랜치명: `claude/...`
- 커밋 메시지: 영어
- 푸시 전 로컬 테스트 필수

## 의존성

Python: pandas, numpy, plotly, streamlit, scipy, openpyxl, tqdm (선택, 없어도 동작)
