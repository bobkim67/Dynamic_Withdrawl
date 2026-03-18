"""
Withdrawal Rate Backtest Framework - Step 1
============================================
Historical backtest only (no Monte Carlo)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# tqdm 선택적 import
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARN]  tqdm not available, progress bars will be disabled")


# ============================================================================
# 포트폴리오 정의 (자유롭게 수정 가능)
# ============================================================================

# 벤치마크 이름 매핑 (데이터 컬럼명 -> 포트폴리오에서 사용하는 이름)
BENCHMARK_MAPPING = {
    '한국주식': '국내주식',
    '미국성장주': '미국성장주',
    '한국종합채권': '국내중기채',
    '한국국고채10년': '국내장기채',
    '미국채권': '미국국채',
    '미국외글로벌채권': '미국외국채',
    '신흥국달러채권': '신흥국달러채권',
    '금': '금'
}

# 포트폴리오 정의 (가중치는 % 단위)
PORTFOLIOS = {
    'Port_4.0%': {
        'target_return': 4.0,
        'target_risk': 3.75,
        'weights': {
            '한국주식': 3.12,
            '미국성장주': 5.47,
            '한국종합채권': 88.37,
            '신흥국달러채권': 0.14,
            '금': 2.90
        }
    },
    'Port_5.0%': {
        'target_return': 5.0,
        'target_risk': 4.18,
        'weights': {
            '한국주식': 3.55,
            '미국성장주': 12.83,
            '한국종합채권': 75.87,
            '신흥국달러채권': 0.15,
            '금': 7.60
        }
    },
    'Port_6.0%': {
        'target_return': 6.0,
        'target_risk': 5.00,
        'weights': {
            '한국주식': 3.99,
            '미국성장주': 20.20,
            '한국종합채권': 63.37,
            '신흥국달러채권': 0.14,
            '금': 12.30
        }
    },
    'Port_7.0%': {
        'target_return': 7.0,
        'target_risk': 6.05,
        'weights': {
            '한국주식': 4.13,
            '미국성장주': 27.66,
            '한국종합채권': 46.36,
            '한국국고채10년': 4.75,
            '신흥국달러채권': 0.14,
            '금': 16.96
        }
    },
    'Port_8.0%': {
        'target_return': 8.0,
        'target_risk': 7.18,
        'weights': {
            '한국주식': 3.72,
            '미국성장주': 35.49,
            '한국종합채권': 22.49,
            '한국국고채10년': 16.90,
            '금': 21.40
        }
    },
    'Port_9.0%': {
        'target_return': 9.0,
        'target_risk': 8.36,
        'weights': {
            '한국주식': 3.88,
            '미국성장주': 42.67,
            '한국국고채10년': 27.02,
            '신흥국달러채권': 0.14,
            '금': 26.29
        }
    },
    # === 신규 상품 펀드 (product_mp 기반, MP 비중) ===
    'Golden Growth': {
        'target_return': 12.10,
        'target_risk': 7.40,
        'weights': {
            '미국성장주': 49.14, '금': 17.20, '한국3년국고채': 13.36,
            '한국주식': 10.70, '한국10년국고채': 7.64, '미국하이일드채권': 1.96,
        }
    },
    'MS GROWTH': {
        'target_return': 12.85,
        'target_risk': 8.73,
        'weights': {
            '미국성장주': 40.84, '금': 22.51, '한국10년국고채': 17.34,
            '한국주식': 8.96, '호주주식': 2.56, '미국하이일드채권': 1.60,
            '글로벌원자재': 1.28, '미국부동산': 1.28, '미국외부동산': 1.28,
            '글로벌인프라': 1.28, '미국물가채권': 1.07,
        }
    },
    'MS STABLE': {
        'target_return': 5.99,
        'target_risk': 3.97,
        'weights': {
            '한국10년국고채': 69.34, '미국성장주': 10.21, '미국하이일드채권': 6.40,
            '금': 5.63, '미국물가채권': 4.26, '한국주식': 2.24,
            '호주주식': 0.64, '글로벌원자재': 0.32, '미국부동산': 0.32,
            '미국외부동산': 0.32, '글로벌인프라': 0.32,
        }
    },
}

# 사용할 포트폴리오 선택 (또는 'all'로 전체 사용)
# 예: SELECTED_PORTFOLIOS = ['Port_4.0%', 'Port_6.0%', 'Port_8.0%']
SELECTED_PORTFOLIOS = 'all'  # 'all' 또는 리스트

# ============================================================================


# ============================================================================


class PortfolioCalculator:
    """
    포트폴리오 수익률 계산
    - 벤치마크 수익률을 가중평균하여 포트폴리오 수익률 계산
    - 리밸런싱은 없음 (Buy & Hold)
    """
    
    @staticmethod
    def calculate_portfolio_returns(returns_df: pd.DataFrame, 
                                    portfolio_weights: Dict[str, float],
                                    portfolio_name: str) -> pd.Series:
        """
        포트폴리오 수익률 계산
        
        Parameters
        ----------
        returns_df : pd.DataFrame
            벤치마크 일별 수익률
        portfolio_weights : Dict[str, float]
            포트폴리오 가중치 (% 단위)
        portfolio_name : str
            포트폴리오 이름
            
        Returns
        -------
        portfolio_returns : pd.Series
            포트폴리오 일별 수익률
        """
        # 가중치를 비율로 변환 (% -> 소수)
        weights_decimal = {k: v/100 for k, v in portfolio_weights.items()}
        
        # 가중치 합 검증
        total_weight = sum(weights_decimal.values())
        if not np.isclose(total_weight, 1.0, atol=0.01):
            print(f"[WARN]  {portfolio_name} 가중치 합: {total_weight*100:.2f}%")
        
        # 포트폴리오 수익률 = 가중평균
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        
        for asset_name, weight in weights_decimal.items():
            # 벤치마크 이름 매핑
            benchmark_name = BENCHMARK_MAPPING.get(asset_name, asset_name)
            
            if benchmark_name not in returns_df.columns:
                print(f"[WARN]  {portfolio_name}: '{benchmark_name}' 벤치마크를 찾을 수 없습니다")
                continue
            
            # 가중치 적용
            portfolio_returns += returns_df[benchmark_name] * weight
        
        portfolio_returns.name = portfolio_name
        return portfolio_returns
    
    @staticmethod
    def add_portfolios_to_returns(returns_df: pd.DataFrame,
                                  portfolios: Dict = None,
                                  selected: List[str] = None) -> pd.DataFrame:
        """
        수익률 DataFrame에 포트폴리오 컬럼 추가
        
        Parameters
        ----------
        returns_df : pd.DataFrame
            벤치마크 수익률
        portfolios : Dict
            포트폴리오 정의 (None이면 PORTFOLIOS 사용)
        selected : List[str]
            선택된 포트폴리오 리스트 (None이면 전체)
            
        Returns
        -------
        combined_df : pd.DataFrame
            벤치마크 + 포트폴리오 수익률
        """
        if portfolios is None:
            portfolios = PORTFOLIOS
        
        # 선택된 포트폴리오 결정
        if selected is None or selected == 'all':
            selected_portfolios = list(portfolios.keys())
        else:
            selected_portfolios = selected
        
        print(f"\n포트폴리오 수익률 계산:")
        print(f"  추가할 포트폴리오: {len(selected_portfolios)}개")
        
        # 결과 DataFrame
        result_df = returns_df.copy()
        
        # 각 포트폴리오 계산
        for port_name in selected_portfolios:
            if port_name not in portfolios:
                print(f"[WARN]  '{port_name}' 포트폴리오를 찾을 수 없습니다")
                continue
            
            port_config = portfolios[port_name]
            port_returns = PortfolioCalculator.calculate_portfolio_returns(
                returns_df, 
                port_config['weights'],
                port_name
            )
            
            # DataFrame에 추가
            result_df[port_name] = port_returns
            
            # 통계 출력
            annual_return = port_returns.mean() * 252 * 100
            annual_vol = port_returns.std() * np.sqrt(252) * 100
            print(f"  [OK] {port_name}: 연수익률 {annual_return:.2f}%, 연변동성 {annual_vol:.2f}%")
        
        return result_df


class DataPreprocessor:
    """
    데이터 로딩 및 전처리
    - 일별 수익률 계산
    - 월초 거래일 식별
    - 포트폴리오 수익률 추가
    """
    
    def __init__(self, data: pd.DataFrame, add_portfolios: bool = True):
        """
        Parameters
        ----------
        data : pd.DataFrame
            DatetimeIndex와 벤치마크 컬럼을 가진 DataFrame
            (이미 거래일만 포함, 주말 제외됨)
        add_portfolios : bool
            포트폴리오 수익률 추가 여부
        """
        self.data = data.copy()
        self.returns = None
        self.month_starts = None
        
        print("=== 데이터 전처리 시작 ===")
        self._validate_data()
        self._calculate_returns()
        
        # 포트폴리오 추가
        if add_portfolios:
            self._add_portfolios()
        
        self._identify_month_starts()
        print("[OK] 전처리 완료\n")
    
    def _validate_data(self):
        """데이터 검증"""
        # 인덱스가 DatetimeIndex인지 확인
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("데이터 인덱스는 DatetimeIndex여야 합니다")
        
        # 정렬 확인
        if not self.data.index.is_monotonic_increasing:
            print("[WARN]  데이터를 날짜순으로 정렬합니다")
            self.data.sort_index(inplace=True)
        
        # 결측치 확인
        null_counts = self.data.isnull().sum()
        if null_counts.sum() > 0:
            print(f"[WARN]  결측치 발견:\n{null_counts[null_counts > 0]}")
        
        print(f"데이터 기간: {self.data.index[0].date()} ~ {self.data.index[-1].date()}")
        print(f"총 거래일: {len(self.data):,}일")
        print(f"벤치마크: {len(self.data.columns)}개")
    
    def _calculate_returns(self):
        """일별 수익률 계산: r_t = I_t / I_{t-1} - 1"""
        print("\n일별 수익률 계산 중...", end=" ")
        
        # pct_change()는 (I_t - I_{t-1}) / I_{t-1} 계산
        self.returns = self.data.pct_change()
        
        # 첫 행은 NaN이므로 제거
        self.returns = self.returns.iloc[1:]
        
        print(f"[OK] ({len(self.returns):,}개 수익률)")
        
        # 수익률 통계
        print("\n수익률 통계 (연율화):")
        annual_stats = pd.DataFrame({
            '연평균수익률': self.returns.mean() * 252 * 100,
            '연변동성': self.returns.std() * np.sqrt(252) * 100
        }).round(2)
        print(annual_stats)
    
    def _add_portfolios(self):
        """포트폴리오 수익률 추가"""
        self.returns = PortfolioCalculator.add_portfolios_to_returns(
            self.returns,
            portfolios=PORTFOLIOS,
            selected=SELECTED_PORTFOLIOS
        )
    
    def _identify_month_starts(self):
        """월초 첫 거래일 식별"""
        print("\n월초 거래일 식별 중...", end=" ")
        
        # 월별로 그룹화해서 첫 거래일 찾기
        # resample을 사용해 각 월의 첫 날짜를 찾음
        monthly_first = self.returns.resample('MS').first()
        
        # 실제 첫 거래일은 resample 결과의 인덱스와는 다를 수 있음
        # 각 월별로 실제 데이터의 첫 거래일을 찾아야 함
        month_start_dates = []
        for year_month in self.returns.index.to_period('M').unique():
            # 해당 월의 모든 날짜
            mask = self.returns.index.to_period('M') == year_month
            # 첫 날짜
            first_date = self.returns.index[mask][0]
            month_start_dates.append(first_date)
        
        # boolean Series 생성
        self.month_starts = pd.Series(
            False, 
            index=self.returns.index,
            name='is_month_start'
        )
        self.month_starts.loc[month_start_dates] = True
        
        n_months = self.month_starts.sum()
        print(f"[OK] ({n_months}개월)")
        
        # 첫 몇 개 월초 날짜 출력
        month_start_dates_show = self.returns.index[self.month_starts][:10]
        print(f"첫 10개 월초: {[d.date() for d in month_start_dates_show]}")
    
    def get_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        전처리된 데이터 반환
        
        Returns
        -------
        returns : pd.DataFrame
            일별 수익률
        month_starts : pd.Series
            월초 여부 (boolean)
        """
        return self.returns, self.month_starts


class WithdrawalSimulator:
    """
    [Legacy] 단일 경로 시뮬레이션 및 롤링 백테스트.
    engine.py의 simulate_withdrawal_on_path() / generate_paths_rolling()으로 대체됨.
    __main__ 테스트 블록에서만 사용.
    """
    
    def __init__(self, returns: pd.DataFrame, month_starts: pd.Series):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            일별 수익률 (벤치마크 × 날짜)
        month_starts : pd.Series
            월초 거래일 여부 (boolean)
        """
        self.returns = returns
        self.month_starts = month_starts
        self.dates = returns.index
        
        # 월초 인덱스 미리 계산 (성능 최적화)
        self.month_start_indices = np.where(month_starts.values)[0]
        
        print("=== 시뮬레이터 초기화 ===")
        print(f"거래일 수: {len(self.dates):,}")
        print(f"월 수: {len(self.month_start_indices)}")
        print(f"벤치마크: {list(returns.columns)}")
    
    def _get_end_index(self, start_idx: int, horizon_years: int) -> int:
        """
        시작 인덱스로부터 정확히 horizon_years 후의 인덱스 찾기
        
        Parameters
        ----------
        start_idx : int
            시작 인덱스
        horizon_years : int
            투자 기간 (년)
            
        Returns
        -------
        end_idx : int or None
            종료 인덱스 (데이터 부족시 None)
        """
        start_date = self.dates[start_idx]
        target_date = start_date + pd.DateOffset(years=horizon_years)
        
        # target_date 이후의 첫 거래일 찾기
        valid_dates = self.dates[self.dates >= target_date]
        
        if len(valid_dates) == 0:
            return None
        
        end_date = valid_dates[0]
        end_idx = self.dates.get_loc(end_date)
        
        return end_idx
    
    def simulate_single_path(self, 
                            benchmark: str, 
                            start_idx: int, 
                            end_idx: int, 
                            wr: float, 
                            v0: float = 100.0) -> float:
        """
        단일 경로 시뮬레이션
        
        Parameters
        ----------
        benchmark : str
            벤치마크 이름
        start_idx : int
            시작 인덱스
        end_idx : int
            종료 인덱스
        wr : float
            연간 인출률 (예: 0.04 = 4%)
        v0 : float
            초기 포트폴리오 가치
            
        Returns
        -------
        vt : float
            최종 포트폴리오 가치
        """
        V = v0
        monthly_withdrawal = wr * v0 / 12
        
        # 해당 기간의 수익률과 월초 플래그
        period_returns = self.returns[benchmark].iloc[start_idx:end_idx+1].values
        period_month_starts = self.month_starts.iloc[start_idx:end_idx+1].values
        
        # 해당 기간의 월초 인덱스들 (상대 인덱스)
        month_indices = np.where(period_month_starts)[0]
        
        # 월별로 처리
        for i, month_start_idx in enumerate(month_indices):
            # 월초 인출
            V = max(V - monthly_withdrawal, 0)
            
            if V == 0:
                return 0.0
            
            # 이번 달의 마지막 인덱스
            if i < len(month_indices) - 1:
                month_end_idx = month_indices[i + 1]
            else:
                month_end_idx = len(period_returns)
            
            # 해당 월의 일별 수익률들
            month_returns = period_returns[month_start_idx:month_end_idx]
            
            # 누적 수익률 적용: V * (1+r1) * (1+r2) * ... * (1+rn)
            cumulative_return = np.prod(1 + month_returns)
            V = V * cumulative_return
            V = max(V, 0)
            
            if V == 0:
                return 0.0
        
        return V
    
    def run_rolling_backtest(self, 
                            benchmark: str, 
                            horizon_years: int, 
                            wr: float,
                            v0: float = 100.0,
                            verbose: bool = True) -> np.ndarray:
        """
        롤링 윈도우 백테스트
        
        Parameters
        ----------
        benchmark : str
            벤치마크 이름
        horizon_years : int
            투자 기간 (년)
        wr : float
            연간 인출률
        v0 : float
            초기 포트폴리오 가치
        verbose : bool
            진행 상황 출력 여부
            
        Returns
        -------
        vt_array : np.ndarray
            모든 롤링 윈도우의 최종 포트폴리오 가치
        """
        vt_list = []
        
        # 유효한 시작 인덱스 찾기
        valid_start_indices = []
        for start_idx in range(len(self.dates)):
            end_idx = self._get_end_index(start_idx, horizon_years)
            if end_idx is not None:
                valid_start_indices.append((start_idx, end_idx))
        
        n_paths = len(valid_start_indices)
        
        if verbose:
            print(f"\n롤링 백테스트: {benchmark}, {horizon_years}년, WR={wr*100:.1f}%")
            print(f"총 경로 수: {n_paths:,}")
        
        # 진행바 설정
        if TQDM_AVAILABLE:
            iterator = tqdm(valid_start_indices, desc=f"H{horizon_years}Y WR{wr*100:.0f}%", leave=False)
        else:
            iterator = valid_start_indices
        
        # 각 롤링 윈도우 시뮬레이션
        for start_idx, end_idx in iterator:
            vt = self.simulate_single_path(benchmark, start_idx, end_idx, wr, v0)
            vt_list.append(vt)
        
        vt_array = np.array(vt_list)
        
        if verbose:
            print(f"완료: {n_paths:,}개 경로")
        
        return vt_array
    
    def get_valid_horizons(self, max_horizon: int = 20) -> List[int]:
        """
        데이터에서 가능한 horizon 리스트 반환
        
        Parameters
        ----------
        max_horizon : int
            최대 horizon (년)
            
        Returns
        -------
        valid_horizons : List[int]
            가능한 horizon 리스트
        """
        valid_horizons = []
        
        for horizon in range(10, max_horizon + 1):
            # 첫 날짜에서 horizon년 후 데이터 존재 여부 확인
            end_idx = self._get_end_index(0, horizon)
            if end_idx is not None:
                valid_horizons.append(horizon)
        
        return valid_horizons


class WithdrawalVisualizer:
    """
    [Legacy] 백테스트 결과 시각화.
    viewer.py의 Plotly 기반 Streamlit 시각화로 대체됨.
    __main__ 테스트 블록에서만 사용.
    """
    
    def __init__(self, simulator: WithdrawalSimulator):
        """
        Parameters
        ----------
        simulator : WithdrawalSimulator
            시뮬레이터 인스턴스
        """
        self.simulator = simulator
    
    def plot_all_paths(self,
                      benchmark: str,
                      horizon_years: int,
                      wr: float,
                      v0: float = 100.0,
                      max_paths: int = 500,
                      figsize: tuple = (14, 8),
                      save_path: str = None):
        """
        모든 시뮬레이션 경로를 시각화
        
        Parameters
        ----------
        benchmark : str
            벤치마크/포트폴리오 이름
        horizon_years : int
            투자 기간
        wr : float
            연간 인출율
        v0 : float
            초기 자본
        max_paths : int
            그릴 최대 경로 수 (샘플링)
        figsize : tuple
            그래프 크기
        save_path : str
            저장 경로
        """
        print(f"\n경로 시각화: {benchmark}, {horizon_years}년, WR={wr*100:.1f}%")
        
        # 유효한 시작 인덱스 찾기
        valid_starts = []
        for start_idx in range(len(self.simulator.dates)):
            end_idx = self.simulator._get_end_index(start_idx, horizon_years)
            if end_idx is not None:
                valid_starts.append((start_idx, end_idx))
        
        n_total = len(valid_starts)
        
        # 샘플링 (너무 많으면)
        if n_total > max_paths:
            sample_indices = np.random.choice(n_total, max_paths, replace=False)
            sample_indices = np.sort(sample_indices)
            sampled_starts = [valid_starts[i] for i in sample_indices]
            print(f"총 {n_total}개 중 {max_paths}개 샘플링")
        else:
            sampled_starts = valid_starts
            print(f"총 {n_total}개 경로")
        
        # 각 경로 시뮬레이션
        success_paths = []
        failure_paths = []
        
        print("경로 계산 중...")
        for start_idx, end_idx in sampled_starts:
            path = self._simulate_path_with_history(
                benchmark, start_idx, end_idx, wr, v0
            )
            
            if path[-1] >= v0:
                success_paths.append(path)
            else:
                failure_paths.append(path)
        
        # 시각화
        fig, ax = plt.subplots(figsize=figsize)
        
        # 실패 경로 (빨간색)
        for path in failure_paths:
            months_path = np.arange(len(path))
            ax.plot(months_path, path, color='red', alpha=0.3, linewidth=0.5)
        
        # 성공 경로 (초록색)
        for path in success_paths:
            months_path = np.arange(len(path))
            ax.plot(months_path, path, color='green', alpha=0.2, linewidth=0.5)
        
        # 기준선 (초기 자본)
        ax.axhline(y=v0, color='blue', linestyle='--', linewidth=2, 
                   label=f'초기 자본 (${v0})', zorder=10)
        
        # 0선
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # 레전드 추가
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label=f'초기 자본 (${v0})'),
            Line2D([0], [0], color='green', alpha=0.6, linewidth=2, 
                   label=f'성공 ({len(success_paths)}개, {len(success_paths)/len(sampled_starts)*100:.1f}%)'),
            Line2D([0], [0], color='red', alpha=0.6, linewidth=2, 
                   label=f'실패 ({len(failure_paths)}개, {len(failure_paths)/len(sampled_starts)*100:.1f}%)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
        
        # 레이블
        ax.set_xlabel('월 (Months)', fontsize=12)
        ax.set_ylabel('포트폴리오 가치 ($)', fontsize=12)
        ax.set_title(f'{benchmark} - {horizon_years}년 백테스트 (WR={wr*100:.1f}%)\n'
                    f'롤링 시뮬레이션: {len(sampled_starts):,}개 경로',
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"저장: {save_path}")
        
        plt.show()
        
        return fig
    
    def _simulate_path_with_history(self, 
                                    benchmark: str,
                                    start_idx: int,
                                    end_idx: int,
                                    wr: float,
                                    v0: float) -> np.ndarray:
        """
        경로 전체 히스토리와 함께 시뮬레이션
        
        Returns
        -------
        path : np.ndarray
            월별 포트폴리오 가치
        """
        V = v0
        monthly_withdrawal = wr * v0 / 12
        
        period_returns = self.simulator.returns[benchmark].iloc[start_idx:end_idx+1].values
        period_month_starts = self.simulator.month_starts.iloc[start_idx:end_idx+1].values
        
        month_indices = np.where(period_month_starts)[0]
        
        # 월별 가치 저장
        path = [V]
        
        for i, month_start_idx in enumerate(month_indices):
            # 월초 인출
            V = max(V - monthly_withdrawal, 0)
            
            if i < len(month_indices) - 1:
                month_end_idx = month_indices[i + 1]
            else:
                month_end_idx = len(period_returns)
            
            # 해당 월 수익률
            month_returns = period_returns[month_start_idx:month_end_idx]
            cumulative_return = np.prod(1 + month_returns)
            V = V * cumulative_return
            V = max(V, 0)
            
            path.append(V)
        
        return np.array(path)
    
    def plot_terminal_values_by_start_date(self,
                                          benchmark: str,
                                          horizon_years: int,
                                          wr: float,
                                          v0: float = 100.0,
                                          figsize: tuple = (14, 6),
                                          save_path: str = None):
        """
        시작 시점별 최종 포트폴리오 가치
        
        Parameters
        ----------
        benchmark : str
            벤치마크/포트폴리오 이름
        horizon_years : int
            투자 기간
        wr : float
            연간 인출율
        v0 : float
            초기 자본
        figsize : tuple
            그래프 크기
        save_path : str
            저장 경로
        """
        print(f"\n시작일별 결과: {benchmark}, {horizon_years}년, WR={wr*100:.1f}%")
        
        # 롤링 백테스트
        valid_starts = []
        vt_list = []
        start_dates = []
        
        for start_idx in range(len(self.simulator.dates)):
            end_idx = self.simulator._get_end_index(start_idx, horizon_years)
            if end_idx is not None:
                vt = self.simulator.simulate_single_path(
                    benchmark, start_idx, end_idx, wr, v0
                )
                vt_list.append(vt)
                start_dates.append(self.simulator.dates[start_idx])
                valid_starts.append(start_idx)
        
        vt_array = np.array(vt_list)
        failure_mask = vt_array < v0
        
        # 시각화
        fig, ax = plt.subplots(figsize=figsize)
        
        # 실패 (빨간색)
        ax.scatter(np.array(start_dates)[failure_mask], 
                  vt_array[failure_mask],
                  c='red', alpha=0.5, s=10, label='실패 (VT < V0)')
        
        # 성공 (초록색)
        ax.scatter(np.array(start_dates)[~failure_mask], 
                  vt_array[~failure_mask],
                  c='green', alpha=0.3, s=10, label='성공 (VT ≥ V0)')
        
        # 기준선
        ax.axhline(y=v0, color='blue', linestyle='--', linewidth=2, 
                   label=f'초기 자본 (${v0})')
        
        # 통계선
        median = np.median(vt_array)
        ax.axhline(y=median, color='orange', linestyle=':', linewidth=2, 
                   label=f'중앙값 (${median:.2f})')
        
        # 레이블
        ax.set_xlabel('시작 날짜', fontsize=12)
        ax.set_ylabel('최종 포트폴리오 가치 ($)', fontsize=12)
        ax.set_title(f'{benchmark} - 시작일별 최종 결과 (Horizon={horizon_years}년, WR={wr*100:.1f}%)\n'
                    f'실패율: {failure_mask.mean()*100:.2f}% ({failure_mask.sum():,}/{len(vt_array):,})',
                    fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # x축 날짜 포맷
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"저장: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_failure_rate_by_start_month(self,
                                        benchmark: str,
                                        horizon_years: int,
                                        wr: float,
                                        v0: float = 100.0,
                                        figsize: tuple = (16, 6),
                                        save_path: str = None):
        """
        시작 월별 성공/실패 누적 막대 그래프
        
        Parameters
        ----------
        benchmark : str
            벤치마크/포트폴리오 이름
        horizon_years : int
            투자 기간
        wr : float
            연간 인출율
        v0 : float
            초기 자본
        figsize : tuple
            그래프 크기
        save_path : str
            저장 경로
        """
        print(f"\n월별 실패율 시각화: {benchmark}, {horizon_years}년, WR={wr*100:.1f}%")
        
        # 모든 롤링 시뮬레이션 실행
        vt_list = []
        start_dates = []
        
        for start_idx in range(len(self.simulator.dates)):
            end_idx = self.simulator._get_end_index(start_idx, horizon_years)
            if end_idx is not None:
                vt = self.simulator.simulate_single_path(
                    benchmark, start_idx, end_idx, wr, v0
                )
                vt_list.append(vt)
                start_dates.append(self.simulator.dates[start_idx])
        
        # DataFrame 생성
        results_df = pd.DataFrame({
            'start_date': start_dates,
            'vt': vt_list,
            'success': np.array(vt_list) >= v0
        })
        
        # 월별로 집계
        results_df['year_month'] = results_df['start_date'].dt.to_period('M')
        
        monthly_stats = results_df.groupby('year_month').agg({
            'success': ['sum', 'count']
        }).reset_index()
        monthly_stats.columns = ['year_month', 'success_count', 'total_count']
        monthly_stats['failure_count'] = monthly_stats['total_count'] - monthly_stats['success_count']
        monthly_stats['failure_rate'] = monthly_stats['failure_count'] / monthly_stats['total_count'] * 100
        
        # 날짜로 변환
        monthly_stats['date'] = monthly_stats['year_month'].dt.to_timestamp()
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # 상단: 누적 막대 그래프
        width = 20  # 막대 너비 (일 단위)
        
        ax1.bar(monthly_stats['date'], monthly_stats['failure_count'], 
               width=width, label='실패', color='red', alpha=0.7)
        ax1.bar(monthly_stats['date'], monthly_stats['success_count'], 
               width=width, bottom=monthly_stats['failure_count'],
               label='성공', color='green', alpha=0.7)
        
        ax1.set_ylabel('시뮬레이션 개수', fontsize=12)
        ax1.set_title(f'{benchmark} - 시작 월별 성공/실패 (Horizon={horizon_years}년, WR={wr*100:.1f}%)',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 하단: 실패율 선 그래프
        ax2.plot(monthly_stats['date'], monthly_stats['failure_rate'], 
                color='darkred', linewidth=2, marker='o', markersize=3)
        ax2.fill_between(monthly_stats['date'], 0, monthly_stats['failure_rate'],
                        color='red', alpha=0.3)
        
        ax2.set_xlabel('시작 날짜', fontsize=12)
        ax2.set_ylabel('실패율 (%)', fontsize=12)
        ax2.set_title('월별 실패율', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        # 주요 금융 이벤트 표시
        events = [
            ('2008-09-15', '리먼 브라더스'),
            ('2020-03-15', 'COVID-19'),
        ]
        
        for event_date, event_name in events:
            event_dt = pd.to_datetime(event_date)
            if monthly_stats['date'].min() <= event_dt <= monthly_stats['date'].max():
                ax1.axvline(x=event_dt, color='black', linestyle='--', 
                           linewidth=1.5, alpha=0.5)
                ax1.text(event_dt, ax1.get_ylim()[1] * 0.95, event_name,
                        rotation=90, va='top', ha='right', fontsize=9, alpha=0.7)
                ax2.axvline(x=event_dt, color='black', linestyle='--',
                           linewidth=1.5, alpha=0.5)
        
        # x축 날짜 포맷
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"저장: {save_path}")
        
        plt.show()
        
        # 통계 출력
        print(f"\n월별 통계:")
        print(f"  전체 기간: {monthly_stats['date'].min().date()} ~ {monthly_stats['date'].max().date()}")
        print(f"  총 개월 수: {len(monthly_stats)}")
        print(f"  평균 실패율: {monthly_stats['failure_rate'].mean():.2f}%")
        print(f"  최대 실패율: {monthly_stats['failure_rate'].max():.2f}% ({monthly_stats.loc[monthly_stats['failure_rate'].idxmax(), 'date'].date()})")
        print(f"  최소 실패율: {monthly_stats['failure_rate'].min():.2f}% ({monthly_stats.loc[monthly_stats['failure_rate'].idxmin(), 'date'].date()})")
        
        return fig, monthly_stats


# [Legacy] 클라우드 노트북 환경에서 작성된 테스트 코드. 로컬 실행 시 경로 수정 필요.
if __name__ == "__main__":
    import os
    print("=" * 60)
    print("Step 1: 시각화 테스트")
    print("=" * 60)

    # 데이터 로드 (로컬 경로 우선, 없으면 클라우드 경로 시도)
    local_pkl = 'benchmark_data.pkl'
    cloud_pkl = '/mnt/user-data/uploads/benchmark_data.pkl'
    pkl_path = local_pkl if os.path.exists(local_pkl) else cloud_pkl
    data = pd.read_pickle(pkl_path)

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # 전처리 (포트폴리오 자동 추가)
    preprocessor = DataPreprocessor(data, add_portfolios=True)
    returns, month_starts = preprocessor.get_data()
    
    # 시뮬레이터 생성
    simulator = WithdrawalSimulator(returns, month_starts)
    
    # 시각화 객체 생성
    visualizer = WithdrawalVisualizer(simulator)
    
    # ========================================
    # 시각화 1: Port_4.0% (실패율 높음)
    # ========================================
    print("\n" + "=" * 60)
    print("시각화 1: Port_4.0% - 실패 케이스 많은 포트")
    print("=" * 60)
    
    fig1 = visualizer.plot_all_paths(
        benchmark='Port_4.0%',
        horizon_years=10,
        wr=0.04,
        max_paths=500,
        save_path=os.path.join(output_dir, 'paths_port4_wr4.png')
    )
    plt.close()
    
    # ========================================
    # 시각화 2: Port_6.0% (성공률 높음)
    # ========================================
    print("\n" + "=" * 60)
    print("시각화 2: Port_6.0% - 성공 케이스 많은 포트")
    print("=" * 60)
    
    fig2 = visualizer.plot_all_paths(
        benchmark='Port_6.0%',
        horizon_years=10,
        wr=0.04,
        max_paths=500,
        save_path=os.path.join(output_dir, 'paths_port6_wr4.png')
    )
    plt.close()
    
    # ========================================
    # 시각화 3: Port_6.0% 더 높은 WR
    # ========================================
    print("\n" + "=" * 60)
    print("시각화 3: Port_6.0% - 더 높은 인출률 (6%)")
    print("=" * 60)
    
    fig3 = visualizer.plot_all_paths(
        benchmark='Port_6.0%',
        horizon_years=10,
        wr=0.06,
        max_paths=500,
        save_path=os.path.join(output_dir, 'paths_port6_wr6.png')
    )
    plt.close()
    
    # ========================================
    # 시각화 4: 시작일별 최종 결과
    # ========================================
    print("\n" + "=" * 60)
    print("시각화 4: 시작일별 최종 포트폴리오 가치")
    print("=" * 60)
    
    fig4 = visualizer.plot_terminal_values_by_start_date(
        benchmark='Port_4.0%',
        horizon_years=10,
        wr=0.04,
        save_path=os.path.join(output_dir, 'terminal_by_date_port4.png')
    )
    plt.close()
    
    fig5 = visualizer.plot_terminal_values_by_start_date(
        benchmark='Port_6.0%',
        horizon_years=10,
        wr=0.04,
        save_path=os.path.join(output_dir, 'terminal_by_date_port6.png')
    )
    plt.close()
    
    # ========================================
    # 시각화 5: 미국성장주 (벤치마크)
    # ========================================
    print("\n" + "=" * 60)
    print("시각화 5: 미국성장주 (벤치마크)")
    print("=" * 60)
    
    fig6 = visualizer.plot_all_paths(
        benchmark='미국성장주',
        horizon_years=10,
        wr=0.04,
        max_paths=500,
        save_path=os.path.join(output_dir, 'paths_us_growth_wr4.png')
    )
    plt.close()
    
    print("\n" + "=" * 60)
    print("[OK] 모든 시각화 완료!")
    print("=" * 60)
    
    # ========================================
    # 추가: 월별 누적 막대 그래프
    # ========================================
    print("\n" + "=" * 60)
    print("시각화 6: 월별 성공/실패 누적 막대 (Port_4.0%)")
    print("=" * 60)
    
    fig7, stats7 = visualizer.plot_failure_rate_by_start_month(
        benchmark='Port_4.0%',
        horizon_years=10,
        wr=0.04,
        save_path=os.path.join(output_dir, 'monthly_stacked_port4.png')
    )
    plt.close()
    
    print("\n" + "=" * 60)
    print("시각화 7: 월별 성공/실패 누적 막대 (Port_6.0%)")
    print("=" * 60)
    
    fig8, stats8 = visualizer.plot_failure_rate_by_start_month(
        benchmark='Port_6.0%',
        horizon_years=10,
        wr=0.04,
        save_path=os.path.join(output_dir, 'monthly_stacked_port6.png')
    )
    plt.close()
    
    print("\n" + "=" * 60)
    print("시각화 8: 월별 성공/실패 누적 막대 (미국성장주)")
    print("=" * 60)
    
    fig9, stats9 = visualizer.plot_failure_rate_by_start_month(
        benchmark='미국성장주',
        horizon_years=10,
        wr=0.04,
        save_path=os.path.join(output_dir, 'monthly_stacked_us_growth.png')
    )
    plt.close()
    
    print("\n" + "=" * 60)
    print("[OK] 모든 시각화 완료! (누적 막대 포함)")
    print("=" * 60)
