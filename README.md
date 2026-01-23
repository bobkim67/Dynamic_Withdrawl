# Withdrawal Rate Backtest Framework

Historical backtest framework for evaluating withdrawal rates using rolling window analysis.

## Overview

이 프로젝트는 은퇴 포트폴리오의 인출 전략(Withdrawal Strategy)을 역사적 데이터를 기반으로 백테스트하는 프레임워크입니다.

### 주요 기능

- **롤링 윈도우 백테스트**: 일별 롤링으로 모든 가능한 투자 시점 평가
- **포트폴리오 구성**: 벤치마크 자산을 조합한 포트폴리오 수익률 자동 계산
- **실패율 분석**: 초기 자본 대비 최종 자본 비교 (P(VT < V0))
- **시각화**: 경로 분석, 시점별 성공/실패, 월별 누적 막대 그래프

## Features

### 1. 데이터 처리
- 일별 수익률 계산
- 월초 거래일 자동 식별
- 포트폴리오 가중평균 수익률 계산

### 2. 시뮬레이션
- **인출 방식**: 월초 첫 거래일에 고정 금액 인출
- **재투자**: 인출 후 잔액은 일별 수익률로 복리 운용
- **파산 방지**: 잔액이 0이 되면 더 이상 인출 불가

### 3. 백테스트 설정
- **Horizon**: 10~20년 (설정 가능)
- **WR (Withdrawal Rate)**: 0~10% (1% 단위 또는 자유 설정)
- **초기 자본**: V0 = 100 (상대적 단위)

### 4. 시각화
- 모든 시뮬레이션 경로 (성공/실패 색상 구분)
- 시작일별 최종 포트폴리오 가치
- 월별 성공/실패 누적 막대 그래프 + 실패율 트렌드
- 주요 금융 이벤트 표시 (리먼 브라더스, COVID-19)

## Installation

```bash
pip install pandas numpy matplotlib seaborn
```

## Usage

### 1. 기본 사용법

```python
import pandas as pd
from withdrawal_backtest import DataPreprocessor, WithdrawalSimulator, WithdrawalVisualizer

# 데이터 로드
data = pd.read_pickle('benchmark_data.pkl')

# 전처리 (포트폴리오 자동 추가)
preprocessor = DataPreprocessor(data, add_portfolios=True)
returns, month_starts = preprocessor.get_data()

# 시뮬레이터 생성
simulator = WithdrawalSimulator(returns, month_starts)

# 백테스트 실행
vt_array = simulator.run_rolling_backtest(
    benchmark='Port_6.0%',
    horizon_years=10,
    wr=0.04
)

# 결과 분석
failure_rate = (vt_array < 100).mean()
print(f"실패율: {failure_rate*100:.2f}%")
```

### 2. 시각화

```python
# 시각화 객체 생성
visualizer = WithdrawalVisualizer(simulator)

# 모든 경로 시각화 (성공/실패 색상 구분)
visualizer.plot_all_paths(
    benchmark='Port_6.0%',
    horizon_years=10,
    wr=0.04,
    save_path='paths.png'
)

# 월별 누적 막대 그래프
visualizer.plot_failure_rate_by_start_month(
    benchmark='Port_6.0%',
    horizon_years=10,
    wr=0.04,
    save_path='monthly_stacked.png'
)
```

### 3. 포트폴리오 커스터마이징

코드 상단의 `PORTFOLIOS` 딕셔너리를 수정:

```python
PORTFOLIOS = {
    'My_Portfolio': {
        'target_return': 7.0,
        'target_risk': 6.0,
        'weights': {
            '한국주식': 10.0,
            '미국성장주': 30.0,
            '한국종합채권': 40.0,
            '한국국고채10년': 10.0,
            '금': 10.0
        }
    }
}
```

## Data Format

### 입력 데이터 (benchmark_data.pkl)

```
date        미국성장주  국내주식  미국국채  ...  금
2001-01-03  100.0    100.0    100.0   ...  100.0
2001-01-04  107.8    100.5    98.6    ...  98.1
...
```

- **인덱스**: DatetimeIndex (거래일만, 주말/공휴일 제외)
- **컬럼**: 벤치마크 지수 레벨 (100 기준 정규화)

## Project Structure

```
Dynamic_withdrawl/
├── withdrawal_backtest.py    # 메인 코드
├── benchmark_data.pkl         # 벤치마크 지수 데이터
├── README.md                  # 프로젝트 설명
└── requirements.txt           # 패키지 의존성
```

## Key Concepts

### Failure Definition
- **실패**: VT < V0 (최종 자본이 초기 자본보다 작음)
- **성공**: VT ≥ V0
- **파산**: VT = 0

### Critical Withdrawal Rate (WR*)
- P(VT < V0) = 50%를 만족하는 인출율
- 이분 탐색 또는 그리드 서치로 계산

### Rolling Window
- 매 거래일을 시작점으로 설정
- 예: 10년 horizon, 2001-2015 데이터
  - 2001-01-04 ~ 2011-01-04
  - 2001-01-05 ~ 2011-01-05
  - ...
  - 2015-12-31 ~ 2025-12-31

## Limitations (Step 1)

현재 버전에서는 다음 기능이 제외되어 있습니다:
- ❌ 포트폴리오 리밸런싱
- ❌ 동적 인출 전략 (Dynamic Withdrawal)
- ❌ Monte Carlo 시뮬레이션
- ❌ Regime switching
- ❌ 확률적 수익률 모델

이러한 기능들은 향후 버전에서 추가될 예정입니다.

## Results Example

### 백테스트 결과 (10년, WR 4%)

| Portfolio | 연수익률 | 연변동성 | 실패율 |
|-----------|---------|---------|--------|
| Port_4.0% | 4.92%   | 2.65%   | 26.8%  |
| Port_6.0% | 7.10%   | 5.39%   | 0.0%   |
| Port_8.0% | 9.57%   | 8.95%   | 0.0%   |

### 주요 인사이트
- 보수적 포트(채권 88%)는 WR 4%를 감당하지 못함
- 중립/공격 포트는 WR 4%에서 안정적
- 금융위기 시작 시점(2001, 2008)의 실패율이 높음

## License

MIT

## Contact

bobkim67 @ GitHub
