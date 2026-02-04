# Dynamic Withdrawal Strategies - 사용 가이드

## 개요

이 가이드는 Dynamic Withdrawal Strategy 시스템의 사용 방법과 테스트 방법을 설명합니다.

## 시스템 구조

```
Dynamic Withdrawal 시스템
├── 전략 구현
│   ├── Fixed Rate (기존)
│   ├── Guardrails (신규)
│   └── Guyton-Klinger (신규)
├── 핵심 모듈
│   ├── dynamic_simulator.py: 시뮬레이션 엔진
│   ├── metrics_calculator.py: 메트릭 계산
│   └── optimizer.py: 최적화 엔진
└── UI/테스트
    ├── dynamic_strategy_ui.py: Streamlit UI
    ├── app.py: Streamlit 메인 앱
    ├── test_dynamic_strategies.ipynb: Jupyter 테스트
    └── test_streamlit_ui.py: 간단한 테스트 스크립트
```

## 테스트 방법

### 방법 1: Jupyter Notebook (권장)

```bash
jupyter notebook test_dynamic_strategies.ipynb
```

**장점:**
- 단계별 실행 가능
- 각 단계의 출력 확인 가능
- 오류 원인을 명확히 파악 가능
- 진행 상황 실시간 확인

**단계별 테스트:**
1. 라이브러리 및 데이터 로드
2. DataPreprocessor 검증
3. Dynamic Simulator 초기화
4. Guardrails 백테스트
5. 메트릭 계산
6. Guyton-Klinger 백테스트
7. Optimizer 테스트
8. 전체 최적화 실행
9. 결과 요약
10. UI 컴포넌트 테스트

### 방법 2: 간단한 Python 스크립트

```bash
# 환경이 설정된 후
python test_streamlit_ui.py
```

**장점:**
- 빠른 실행
- 주요 기능 검증
- 문제 진단용으로 좋음

### 방법 3: Streamlit UI

```bash
streamlit run app.py
```

**순서:**
1. 왼쪽 사이드바에서 포트폴리오, 기간, 인출률 설정
2. 하단의 "🎯 Dynamic 전략" 탭 선택
3. "🔍 최적화" 탭에서 설정 후 "🚀 최적화 실행" 클릭
4. "📊 비교 분석", "📈 상세 분석" 탭에서 결과 확인

## 세 가지 전략 설명

### 1. Fixed Rate (기준 전략)

```
Year 1: 100만원 × 5% = 5만원
Year 2: 100만원 × 5% × 1.02 = 5.1만원
Year 3: 100만원 × 5% × 1.02² = 5.2만원
...
```

**특징:**
- 매년 일정 금액 + 인플레이션 조정
- 시장 상황에 무관하게 일정한 인출
- 가장 간단한 전략

### 2. Guardrails (즉각 반응 방식)

```
NAV 기준 인출률 계산:
  현재 인출률 = 지난해 인출액 / 현재 NAV

상한 위반 (6% 초과):
  즉시 조정: 현재 NAV × 6% 로 재설정

하한 미달 (4% 이하):
  즉시 조정: 현재 NAV × 4% 로 재설정

정상 범위 (4%-6%):
  인플레이션 조정 계속 진행
```

**특징:**
- NAV 변화에 빠르게 대응
- 포트폴리오 고갈 위험 ↓
- 인출액이 크게 변할 수 있음

**예시:**
```
Year 1: NAV=1000, WR=5% → 50원
Year 2: 시장 하락, NAV=700
        현재 WR = 50/700 = 7.1% > 6% (위반!)
        → 즉시 조정: 700 × 6% = 42원 (-16%)
Year 3: 시장 회복, NAV=800
        현재 WR = 42/800 = 5.25% (정상)
        → 인플레이션 조정 계속
```

### 3. Guyton-Klinger (단계적 조정)

```
Rule 1: Guardrail 위반 시
  고정 비율 조정: 이전 인출액 × (1 ± 10%)
  → 여러 해에 걸쳐 단계적 조정

Rule 2: Portfolio Management Rule
  지난해 수익률 < -10% 시:
  → 동결 (인플레이션 조정 안함)

Rule 3: 정상 범위
  → 인플레이션 조정 계속
```

**특징:**
- 급격한 인출 변화 완화
- 큰 손실 후 보수적 대응
- Guardrails보다 안정적

**예시:**
```
Year 1: NAV=1000, WR=5% → 50원
Year 2: 시장 하락 30%, NAV=700
        현재 WR = 50/700 = 7.1% > 6% (위반!)
        Rule 1: 50 × 0.9 = 45원 (-10%)
        또한 Return = -30% < -10%
        Rule 2: 동결 (인플레이션 조정 안함)

Year 3: 시장 소폭 회복, NAV=750
        현재 WR = 45/750 = 6% (여전히 경계)
        Rule 1: 45 × 0.9 = 40.5원 (-10%)

Year 4: 시장 정상화, NAV=900
        현재 WR = 40.5/900 = 4.5% (정상)
        Rule 3: 40.5 × 1.02 = 41.3원 (인플레이션)
```

## 주요 메트릭

### Total Withdrawal (총 인출액)
- **의미**: 10년 동안 인출한 총액
- **목표**: 최대화
- **계산**: 120개월 인출액 합계

### YoY Volatility (연도별 변동성)
- **의미**: 연간 인출액의 변화 안정성
- **목표**: 최소화
- **계산**: 연도별 인출액의 YoY 변화율 표준편차
- **해석**: 낮을수록 생활비가 안정적

### Failure Rate (실패율)
- **의미**: 포트폴리오가 고갈될 확률
- **목표**: 최소화 (5% 이하)
- **계산**: 롤링 윈도우 중 실패한 경우의 비율

### Terminal NAV (최종 자산)
- **의미**: 10년 후 남은 자산 비율
- **목표**: 높을수록 좋음 (50% 이상)
- **계산**: 최종 NAV / 초기 NAV

## 일반적인 문제 해결

### Q1: 최적화가 느릴 때

**원인**: 많은 포트폴리오 × 많은 인출률 = 많은 시나리오

**해결:**
1. 포트폴리오 개수 줄이기
2. 인출률 범위 좁히기
3. 단계 크기 증가시키기

### Q2: "제약 조건을 만족하는 솔루션이 없습니다"

**원인**: 제약이 너무 엄격함

**해결:**
1. Max YoY Volatility 증가 (15% → 20%)
2. Max Failure Rate 증가 (5% → 10%)
3. Min Terminal NAV 감소 (50% → 40%)

### Q3: Streamlit 앱에서 Dynamic 탭이 안 보임

**해결:**
1. 앱 다시 시작: `Ctrl+C` 후 `streamlit run app.py`
2. 캐시 초기화: `streamlit cache clear`
3. 개발자 도구 확인: F12 → Console 탭에서 오류 확인

### Q4: 각 전략의 결과가 거의 같을 때

**원인**:
- 시장이 안정적인 기간
- 포트폴리오가 변동성이 낮음
- 제약 조건이 유사

**확인사항:**
- 데이터 기간 확인
- 포트폴리오 변동성 확인
- 제약 조건 비교

## 파일별 역할

### 동적 전략 핵심 파일

| 파일 | 역할 | 핵심 클래스/함수 |
|------|------|-----------------|
| `dynamic_simulator.py` | 백테스팅 엔진 | `GuardrailsWithdrawal`, `GuytonKlingerWithdrawal` |
| `metrics_calculator.py` | 메트릭 계산 | `MetricsCalculator` |
| `optimizer.py` | 최적화 엔진 | `WithdrawalOptimizer` |
| `dynamic_strategy_ui.py` | UI 컴포넌트 | `render_dynamic_strategies_tab()` |

### 테스트 파일

| 파일 | 용도 | 실행 방법 |
|------|------|----------|
| `test_dynamic_strategies.ipynb` | 단계별 테스트 | `jupyter notebook` |
| `test_streamlit_ui.py` | 빠른 검증 | `python test_streamlit_ui.py` |

## 다음 단계 (Phase 2)

- [ ] 파라미터 자동 최적화 (guardrail width, adjustment %)
- [ ] Regime-based 분석 (상승장 vs 하강장)
- [ ] Out-of-sample 테스트
- [ ] 몬테카를로 시뮬레이션
- [ ] 동적 포트폴리오 리밸런싱

## 참고 자료

### Guardrails 방식
- Vanguard's Dynamic Spending Approach
- 특징: NAV 기준 즉각 반응

### Guyton-Klinger 방식
- "Decision Rules for a Sustainable Withdrawal Rate"
- 저자: Jonathan T. Guyton & William J. Klinger (2006)
- 특징: 포트폴리오 수익률 기반 규칙

---

**문의사항이나 버그 리포트는 GitHub Issue에 등록해주세요.**
