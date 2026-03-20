# 2026-03-20: Frontier Analysis v2 — bm_list 지수 교체 + 전략 비교

## bm_list 지수 교체
- 한국 채권 4종: BMA03/KTBTR/BMA02/KOSEF_10yr → KIS 만기별 채권지수 (BC001056~60, 2001년~)
- 선진국 주식: TAD09XU → MXWOU Index (MSCI World ex-USA, 2001년~)
- 미국 국채: IEF → LT10TRUU (10년, Bloomberg), LUATTRUU (종합국채)
- 결과: 전 자산 2001년부터 사용 가능, Rolling 경로 40→54~64개로 증가

## 시뮬레이션 기간 통일
- T_MONTHS: 120 → 240 (20년)
- 퇴직 후 20년 시나리오 (65세→85세)
- FA 파산률이 극적으로 나타남 (Port_9%/GG/MSG: 100%)

## 전략 비교 결과 (T=240, Rolling)

### FA vs Guard 5%
- 파산률: 100% → 0% (Port_9%, GG, MSG)
- 총가치: +55~65 개선
- Guard가 FA를 완전 지배 (dominant)

### FR vs Guard 5%
- MoM Std: 50~60% 개선 (전 펀드 일관)
- 인출 변동성 측정: MoM 변화율 Std로 확정
- 월별 인출 Std로는 Guard가 불리 (NAV 소진 효과 때문)

### Guard 5% vs Vol-Adj (고정 밴드 비대칭)
- 15/5: Tot 양보 큼 (-4~13), MoM 약간 개선
- 5/15: Tot 양보 없이 MoM 개선, Worst 동일
- 10/10: Tot 중간 양보, MoM/Worst 균형
- 차이 미미 → 5/5 대칭으로 충분

### Guard 5% vs Vol-Adj (연속 스케일링)
- Vol-Adj 5/15: 전 펀드에서 Tot/MoM/Worst 3개 모두 개선 (유일)
- Vol-Adj 10/10: MoM/Worst 최강, Tot 3~10 양보

### SNR 스위칭 테스트
- 정방향 (SNR높으면 넓게): Worst Cut 펀드별 비일관 → 폐기
- 역방향 (SNR높으면 좁게): 전 지표 악화 → 폐기
- 결론: SNR 스위칭 방식 부적합

## Vol-Adj 확정 방향
- 연속 스케일링 + 비대칭 밴드 (band_upper/band_lower)
- 유력 후보: Vol-Adj 5/15
- 미해결: σ_target 사전 설정 방법
