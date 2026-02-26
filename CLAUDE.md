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
6. `viewer.py`가 pkl 로드 후 5개 탭 렌더링 (설득 흐름):
   1) Guardrail 효과 — Fixed vs Guardrail 비교 인포그래픽 (240개월 시뮬레이션)
   DV) 데이터 검증 — 실제 과거 데이터 기반 스파게티 차트 + 기말잔액 분포
   2) 이론 분석 (GBM) — 변동성별 Fixed vs Guardrail 체계 비교
   3) 실제 포트폴리오 검증 — 2컬럼 (좌: Historical 히트맵/곡선, 우: GBM 히트맵/곡선) + full-width 효과 분석/분석결과
   4) Band 최적화 — Band별 성공률 곡선 (간결화)

### 모듈별 역할

- **`withdrawal_backtest.py`**: `DataPreprocessor`, `PORTFOLIOS` 딕셔너리 (6개 포트폴리오: Port_4.0%~Port_9.0%), `BENCHMARK_MAPPING`, `PortfolioCalculator`. engine.py, grid_search.py, viewer.py가 공통으로 import하는 데이터 레이어.
- **`engine.py`**: 핵심 시뮬레이션 엔진. 경로 생성(rolling/bootstrap/GBM), `simulate_withdrawal_on_path` (scalar), `simulate_withdrawal_on_paths_vectorized` (numpy 벡터화), `evaluate_strategy` / `evaluate_gbm_strategy` (집계 지표), `pareto_frontier`, `select_optimal`. `__main__` 블록에 검증 테스트 포함 (벡터화 vs scalar 일치 확인).
- **`grid_search.py`**: Historical 파라미터 그리드 정의. 출력: `grid_results_full.pkl`, `grid_results_summary.xlsx`.
- **`gbm_grid_search.py`**: GBM MC 파라미터 그리드 (mu 2-12%, sigma 2-20%, init_wr 3-15%, band 5종, beta 5종). Fixed 케이스에 Closed-form 확률도 동시 계산 (`cf_success_rate`). 출력: `gbm_results.pkl` (130,625 레코드), `gbm_results_summary.xlsx`.
- **`viewer.py`**: Streamlit 앱 (5개 탭, 설득 흐름 구조). `@st.cache_data`로 데이터 캐싱. `BETA_LABELS` 5단계. 사이드바에 글로벌 컨트롤 3개 (원본 대비 기말잔액 비율 / 초기 인출률 / 데이터 기반). Tab 1(Fixed vs Guardrail 비교 인포그래픽), Tab DV(데이터 검증 스파게티), Tab 2(GBM 변동성별 분석), Tab 3(2컬럼: 좌 Historical + 우 GBM 비교 + full-width 효과 분석/분석결과), Tab 4(Band별 성공률 곡선만). 디폴트 데이터 기반: Bootstrap.
- **`guardrail_infographic.py`**: standalone matplotlib 인포그래픽 (NAV + 인출 화살표). 출력: `guardrail_infographic.png/svg`.
- **`guardrail_persuasion.py`**: standalone matplotlib 비교 차트 (Fixed vs Guardrail NAV + 인출액). 출력: `guardrail_persuasion.png/svg`.
- **`dynamic_simulator.py`**: 기존 시뮬레이터 (Fixed/Guardrails 전략). 참조용. 추후 통합 여부 판단.

### 인출 시뮬레이션 순서 (매월 반복, 순서 고정)

1. 수익률 적용: `W_t = W_{t-1} * (1 + r_t)`
2. 인출 시도액: `prev_withdraw` (이전 달 인출액 유지, 초기값 = `W0 * init_wr / 12`)
3. 수익률 기반 보정 (옵션): lookback 기간 -> threshold 확인 -> +-adj%
4. Guardrail 밴드 (W/NAV 비율 기준):
   - `target_ratio = init_wr / 12` (목표 월비율)
   - `upper_ratio = target_ratio * (1 + band)`, `lower_ratio = target_ratio * (1 - band)`
   - `ratio = withdraw / W_t` → 비율이 밴드 벗어나면 `upper_ratio * W_t` 또는 `lower_ratio * W_t`로 조정
   - 밴드 내이면 이전 인출액 그대로 유지 (고정 base로 리셋하지 않음)
5. 인출 실행: `W_t = W_t - withdraw_final`, `prev_withdraw = withdraw_final` (다음 달 기준 갱신)

### 그리드 서치 전략 유형

- **fixed_baseline**: `band=99.0` (사실상 guardrail 없음), `adj_on=False`. 순수 고정 인출.
- **dynamic**: 실제 guardrail 밴드 (0.05~0.20), 수익률 보정 옵션 포함.

### 성공/실패 정의

- **Ruin (파산)**: 경로 중 `W_t <= 0` 발생
- **Terminal 실패**: `W_T < beta * W0` (원본 대비 기말잔액 비율: 0.1/0.25/0.5/0.75/1.0 중 선택)
- **성공**: ruin 없음 AND terminal 실패 없음

## 차후 개발 예정

아래 항목은 코딩 전 검토 필요.

1. ~~**Tab 1 메커니즘 시각화 강화**~~ — v6.0에서 Fixed vs Guardrail 비교 인포그래픽으로 교체 완료
2. ~~**Beta 위젯 위치 재배치**~~ — v5.3에서 사이드바 글로벌 컨트롤로 통합
3. **Beta 값 확장** — grid_search.py에 beta=[0.1, 0.25, 0.5, 0.75, 1.0] 추가 후 pkl 재생성 필요 (현재 데이터는 0.1/0.5/1.0만 존재)
4. ~~**모든 시각화에 핵심 코멘트**~~ — v5에서 완료
5. ~~**Band Trade-off 비교 그래프**~~ — v5.3에서 간결화 완료
6. ~~**전략 상세 + 나의 전략 조합 탭 통합**~~ — v5에서 해소
7. **NAV 경로 시뮬레이션 탭 복원** — 전략 상세(NAV Fan Chart) 별도 탭 또는 drill-down
8. ~~**engine.py Guardrail 로직 통일**~~ — v6.0에서 W/NAV 비율 밴드 방식으로 통일 완료
9. ~~**Tab 3 리팩토링**~~ — v7.0에서 2컬럼 레이아웃 + Section 4,5 삭제 + 분석결과 확장 완료

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
