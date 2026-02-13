# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고하는 가이드입니다.

## 프로젝트 목적

과거 데이터 기반 퇴직 포트폴리오 인출률 백테스트 및 최적화 시스템.
한국 자산운용 실무에서 사용할 증거 기반 인출율 추천 도구.

## 실행 명령어

```bash
# 그리드 서치 실행 (grid_results_full.pkl 생성, 수 분 소요)
python grid_search.py

# 엔진 검증 테스트 실행
python engine.py

# Streamlit 뷰어 실행 (grid_results_full.pkl 필요)
streamlit run viewer.py

# 기존 Streamlit 앱 (viewer.py로 교체 예정)
streamlit run app.py
```

## 아키텍처

계산과 시각화를 완전히 분리한다.

**계산 파이프라인** (오프라인, 스크립트):
`withdrawal_backtest.py` -> `engine.py` -> `grid_search.py` -> `grid_results_full.pkl`

**시각화** (Streamlit, 읽기 전용):
`viewer.py`가 `grid_results_full.pkl`을 로드 -> 차트, 필터링, drill-down

Streamlit에서 무거운 시뮬레이션을 돌리지 않는다. 단, Strategy Detail 탭의 NAV 경로 시뮬레이션은 on-demand로 실행.

### 데이터 흐름

1. `benchmark_data.pkl` (8개 자산 일별 가격지수, 2001~2025)
2. `withdrawal_backtest.py::DataPreprocessor`가 일별 수익률 + 월초 거래일 플래그 계산, `PortfolioCalculator`가 6개 포트폴리오 수익률 생성
3. `engine.py::daily_to_monthly_returns`로 월간 변환, `generate_paths_rolling`/`generate_paths_bootstrap`로 수익률 경로 생성
4. `engine.py::simulate_withdrawal_on_path`로 경로별 인출 시뮬레이션, `evaluate_strategy`로 집계
5. `grid_search.py`가 파라미터 그리드 순회 -> `pareto_frontier` / `select_optimal` 플래그와 함께 결과 저장
6. `viewer.py`가 pkl 로드 후 6개 탭 렌더링 (스토리텔링 순서):
   1) Guardrail이란? — 개념 도입, 메커니즘 시각화, 이중 역할
   2) 언제 유리한가? — Fixed vs Guardrail 성공률/누적인출금 차이 히트맵
   3) 최적 Band는? — Band별 성공률 곡선, 최적 Band 분포, Trade-off 표
   4) 데이터 신뢰도 — Rolling/Bootstrap/GBM 3종 비교 검증
   5) 나의 전략 조합 — 사용자 선택형 탐색기 (Beta별/Path Method별 비교)
   6) 전략 상세 — NAV 경로 시뮬레이션 (on-demand)

### 모듈별 역할

- **`withdrawal_backtest.py`**: `DataPreprocessor`, `PORTFOLIOS` 딕셔너리 (6개 포트폴리오: Port_4.0%~Port_9.0%), `BENCHMARK_MAPPING`, `PortfolioCalculator`. engine.py, grid_search.py, viewer.py가 공통으로 import하는 데이터 레이어.
- **`engine.py`**: 핵심 시뮬레이션 엔진. 경로 생성(rolling/bootstrap), `simulate_withdrawal_on_path` (월별 인출 순서 고정: 수익률 적용 -> base -> 보정 -> guardrail 캡 -> 인출 실행), `evaluate_strategy` (집계 지표), `pareto_frontier`, `select_optimal`. `__main__` 블록에 검증 테스트 포함.
- **`grid_search.py`**: `CONFIG` 딕셔너리에 전체 파라미터 그리드 정의. `load_and_generate_paths()` -> `run_grid_search()` -> `save_results()`. 출력: `grid_results_full.pkl` (list of dicts), `grid_results_summary.xlsx`.
- **`viewer.py`**: Streamlit 앱 (6개 탭, 스토리텔링 구조). `@st.cache_data`로 데이터 캐싱. `BETA_LABELS` 5단계 (0.1/0.25/0.5/0.75/1.0). 전략 상세 탭에서 `simulate_paths_for_strategy`로 on-demand 시뮬레이션 실행.
- **`dynamic_simulator.py`**: 기존 시뮬레이터 (Fixed/Guardrails 전략). 참조용. 추후 통합 여부 판단.

### 인출 시뮬레이션 순서 (매월 반복, 순서 고정)

1. 수익률 적용: `W_t = W_{t-1} * (1 + r_t)`
2. Base 인출액 계산: `base = W0 * init_wr / 12`
3. 수익률 기반 보정 (옵션): lookback 기간 -> threshold 확인 -> +-adj%
4. Guardrail 캡: `current_wr`가 `[init_wr*(1-band), init_wr*(1+band)]` 밖이면 경계로 제한
5. 인출 실행: `W_t = W_t - withdraw_final`

### 그리드 서치 전략 유형

- **fixed_baseline**: `band=99.0` (사실상 guardrail 없음), `adj_on=False`. 순수 고정 인출.
- **dynamic**: 실제 guardrail 밴드 (0.05~0.20), 수익률 보정 옵션 포함.

### 성공/실패 정의

- **Ruin (파산)**: 경로 중 `W_t <= 0` 발생
- **Terminal 실패**: `W_T < beta * W0` (beta: 0.1/0.25/0.5/0.75/1.0 중 선택)
- **성공**: ruin 없음 AND terminal 실패 없음

## 차후 개발 예정 (v4.1)

아래 항목은 코딩 전 검토 필요. devlog_20260213.md에 상세 기술.

1. **Tab 1 메커니즘 시각화 강화** — 상승장/하락장에서 Fixed vs Guardrail 인출 차이를 확실히 보여주는 그래프 + 성공률/누적인출금 시각화 추가
2. **Beta 위젯 위치 재배치** — 사이드바에서 제거, 연결된 컨텐츠 섹션 상단에 인라인 배치
3. **Beta 값 확장** — grid_search.py에 beta=[0.1, 0.25, 0.5, 0.75, 1.0] 추가 후 pkl 재생성 필요 (현재 데이터는 0.1/0.5/1.0만 존재)
4. **모든 시각화에 핵심 코멘트** — 전체 차트에 해석 가이드 코멘트 일관 추가
5. **Band Trade-off 비교 그래프** — Tab 3에 테이블만 존재, scatter/bar 그래프 추가 필요
6. **전략 상세 + 나의 전략 조합 탭 통합** — 6탭 → 5탭, 탐색→지표→NAV 한 흐름으로
7. **분석 데이터 테이블 탭 추가** — Beta별 Fixed vs Guardrail 비교, Rolling vs GBM vs Bootstrap 비교, 핵심발견 요약을 데이터 테이블로 제공하는 새 탭

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
