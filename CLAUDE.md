# CLAUDE.md - Dynamic Withdrawal 프로젝트 현황

# 프로젝트 현황

## 목적
과거 데이터 기반 퇴직 포트폴리오 인출률 백테스트 및 최적화 시스템.
한국 자산운용 실무에서 사용할 증거 기반 인출율 추천 도구를 만든다.

## 아키텍처 방향 (확정)

계산과 시각화를 완전히 분리한다:
- **VSCode (노트북/스크립트)**: 엔진 구현, 그리드 서치, 시뮬레이션 실행 → 결과를 pkl로 저장
- **Streamlit**: pkl 파일 로드 → 프론티어 시각화, 제약 필터링, drill-down 조회 전용 뷰어

Streamlit에서 무거운 시뮬레이션을 돌리지 않는다.

## 파일 구조 및 상태

### 유지하는 파일
- `withdrawal_backtest.py`: DataPreprocessor, PORTFOLIOS, BENCHMARK_MAPPING 정의. 데이터 전처리 모듈로 계속 사용.
- `dynamic_simulator.py`: 기존 전략(Fixed/Guardrails) 시뮬레이터. 단일 경로 조회(get_single_path_detail) 참조/재활용 가능. 새 엔진 완성 후 통합 여부 판단.
- `app.py`: 기존 Streamlit 앱. 새 뷰어 완성 후 교체 예정.
- `test_dynamic_strategies.ipynb`: 기존 테스트 노트북. 참조용.
- `benchmark_data.pkl`: 벤치마크 가격 데이터 (2001-01-03 ~ 2025-12-31, 8개 자산)

### 삭제 대상 (더 이상 사용하지 않음)
- `dynamic_strategy_ui.py`: 기존 Streamlit UI 컴포넌트. 새 뷰어로 대체.
- `optimizer.py`: 기존 3전략 그리드 서치. 새 최적화 엔진으로 대체.
- `metrics_calculator.py`: 기존 메트릭 계산. 새 스펙의 지표로 대체.

### 새로 만들 파일 (아직 미구현)
- `engine.py` (가칭): 새 최적화 엔진. Path 생성(Rolling/Bootstrap), 인출 시뮬레이션(Base+보정+Guardrail캡), 집계, Pareto frontier 계산.
- `grid_search.py` (가칭): 그리드 서치 실행 스크립트. 결과를 pkl로 저장.
- `viewer.py` (가칭): 새 Streamlit 뷰어. pkl 로드 → 프론티어 시각화, 필터링, drill-down.

## 포트폴리오 정의 (withdrawal_backtest.py의 PORTFOLIOS)
6개 포트폴리오: Port_4.0% ~ Port_9.0%
자산: 한국주식, 미국성장주, 한국종합채권, 한국국고채10년, 신흥국달러채권, 미국채권, 미국외글로벌채권, 금
벤치마크 데이터: 8개 자산 일별 가격지수

## 새 최적화 엔진 스펙 요약

### Path 생성
A) Rolling Historical Window: 실제 연속 구간, 시작점 이동
B) Block Bootstrap: 월간 수익률 블록(12/24/36개월) 랜덤 샘플링

### 인출 전략 구조 (적용 순서 고정)
1. Base 인출: W0 × init_wr / 12
2. 전월 수익률 기반 보정 (옵션): lookback(1m/3m), threshold, ±adj%
3. Guardrail 캡 (필수): current_wr가 band 밖이면 경계로 제한

### 성공/실패 정의
- Ruin: 경로 중 W_t ≤ 0
- Terminal: W_T < β×W0 (β=0.5)
- 성공: ruin 없음 AND terminal 조건 충족

### 집계 지표 (θ별)
- x축: 성공확률
- y축: 누적 인출액 median
- 보조: CV(인출 변동성), worst_cut, p5 월인출
- 실패 분해: P_ruin, P_terminal_fail

### 파라미터 탐색 범위
- init_wr: 0.03 ~ 0.10
- band: 0.05 ~ 0.20
- 수익률 보정: lookback {1m, 3m}, threshold, adj ±2~10%

### 프론티어 & 최적 선택
- Pareto frontier: dominated 제거
- 제약: 성공확률 ≥ 0.90, P_ruin ≤ 0.01
- 최적: 제약 통과 후 누적 인출액 최대화

## 기술 스택
- Python, Pandas, NumPy
- Plotly (시각화)
- Streamlit (뷰어)
- pickle/parquet (결과 저장)

## 코딩 컨벤션
- 과도한 모듈화 금지. 재사용되지 않는 함수 만들지 않기.
- 선형적이고 읽기 쉬운 코드. 시간 흐름이 명확해야 함.
- 섹션별 주석으로 구분.
- 함수는 순수 계산 또는 재사용 로직에만 사용.
- 분석 코드이지 프로덕션 라이브러리가 아님.
- 한국어 주석/UI, 금융 전문용어는 영문 병기.

## 개발 순서
1단계: 엔진 구현 (path 생성, 인출 시뮬레이션, 집계, 프론티어)
2단계: 그리드 서치 스크립트 (결과 pkl 저장)
3단계: Streamlit 뷰어 (pkl 로드 → 시각화/필터링)

## Git Workflow
- PR-based development
- Feature branches named `claude/...`
- Commit messages in English
- Test changes locally before pushing
