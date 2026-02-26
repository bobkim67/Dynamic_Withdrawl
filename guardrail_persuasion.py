"""
Guardrail Persuasion Infographic
- 동일 시장 경로에서 Fixed vs Guardrail 인출 → NAV 경로 분기 시각화
- matplotlib + numpy only
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.gridspec as gridspec

# ===== 한글 폰트 =====
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ===== 시뮬레이션 파라미터 =====
np.random.seed(7)
n_steps = 240

# 연 5% 인출 → 월 ~0.417%
target_ratio = 0.05 / 12   # ~0.00417
lower_band = target_ratio * 0.95   # -5%
upper_band = target_ratio * 1.05   # +5%

NAV0 = 100.0
W0 = target_ratio * NAV0

# ===== 수익률 경로 생성 (regime-stitched) =====
# 설득력 있는 시나리오: 하락기에 Fixed가 확실히 열위, Guardrail이 회복
# Phase 1 (0~59): 초기 안정기
r1 = np.random.normal(0.005, 0.020, 60)
# Phase 2 (60~119): 하락기 — GFC급 (깊은 하락)
r2 = np.random.normal(-0.018, 0.038, 60)
# Phase 3 (120~179): 회복기 — V자 반등
r3 = np.random.normal(0.018, 0.020, 60)
# Phase 4 (180~239): 후기 랠리
r4 = np.random.normal(0.013, 0.016, 60)

returns = np.concatenate([r1, r2, r3, r4])

# ===== Market NAV (인출 없음, 참조선) =====
mkt_nav = np.zeros(n_steps + 1)
mkt_nav[0] = NAV0
for t in range(n_steps):
    mkt_nav[t + 1] = mkt_nav[t] * (1 + returns[t])

# ===== Fixed Withdrawal NAV =====
nav_fixed = np.zeros(n_steps + 1)
nav_fixed[0] = NAV0
W_fixed_arr = np.full(n_steps + 1, W0)

for t in range(n_steps):
    nav_fixed[t + 1] = nav_fixed[t] * (1 + returns[t])
    nav_fixed[t + 1] = max(nav_fixed[t + 1] - W0, 0)

# ===== Guardrail Withdrawal NAV =====
nav_guard = np.zeros(n_steps + 1)
nav_guard[0] = NAV0
Wg = np.zeros(n_steps + 1)
Wg[0] = W0

for t in range(n_steps):
    nav_after_growth = nav_guard[t] * (1 + returns[t])
    if nav_after_growth <= 0:
        nav_guard[t + 1] = 0
        Wg[t + 1] = 0
        continue

    ratio = Wg[t] / nav_after_growth
    if ratio > upper_band:
        Wg[t + 1] = upper_band * nav_after_growth
    elif ratio < lower_band:
        Wg[t + 1] = lower_band * nav_after_growth
    else:
        Wg[t + 1] = Wg[t]

    nav_guard[t + 1] = max(nav_after_growth - Wg[t + 1], 0)

# 결과 미리보기
end_fixed = nav_fixed[-1]
end_guard = nav_guard[-1]
target_nav = NAV0 * 0.5

# ===== Figure 구성 (gridspec 3.2:1) =====
fig = plt.figure(figsize=(14, 8), facecolor='white')
gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1], hspace=0.12,
                       left=0.07, right=0.88, top=0.90, bottom=0.07)

ax_main = fig.add_subplot(gs[0])
ax_wd = fig.add_subplot(gs[1], sharex=ax_main)

time = np.arange(n_steps + 1)

# ===== 상단: NAV 경로 =====
ax_main.set_facecolor('white')

# Drawdown / Recovery 구간 음영
ax_main.axvspan(60, 120, alpha=0.05, color='#e74c3c', zorder=0)
ax_main.axvspan(120, 180, alpha=0.04, color='#16a34a', zorder=0)

# 구간 라벨 (상단 고정)
y_top = max(mkt_nav.max(), nav_guard.max(), nav_fixed.max()) * 1.04
ax_main.text(90, y_top, 'Drawdown', ha='center', va='top',
             fontsize=9.5, color='#cbd5e1', fontstyle='italic')
ax_main.text(150, y_top, 'Recovery', ha='center', va='top',
             fontsize=9.5, color='#cbd5e1', fontstyle='italic')

# Market NAV (참조)
ax_main.plot(time, mkt_nav, color='#d1d5db', linewidth=1.5, zorder=2,
             label='Market NAV (no withdrawal)')

# Fixed NAV
ax_main.plot(time, nav_fixed, color='#dc2626', linewidth=2.6, zorder=3,
             label='Fixed Withdrawal')

# Guardrail NAV
ax_main.plot(time, nav_guard, color='#16a34a', linewidth=2.6, zorder=4,
             label='Guardrail Withdrawal')

# Target 50% 수평선
ax_main.axhline(y=target_nav, color='#475569', linewidth=1.2, linestyle='--',
                zorder=1, alpha=0.7)
ax_main.text(n_steps + 3, target_nav, f'Target: {target_nav:.0f}\n(50% of initial)',
             va='center', ha='left', fontsize=8.5, color='#475569', fontweight='600')

# 종료점 마커
ax_main.plot(n_steps, end_fixed, 'o', color='#dc2626', markersize=9, zorder=5,
             markeredgecolor='white', markeredgewidth=1.5)
ax_main.plot(n_steps, end_guard, 'o', color='#16a34a', markersize=9, zorder=5,
             markeredgecolor='white', markeredgewidth=1.5)

# 종료 라벨 위치 (겹침 방지)
fixed_label_y = end_fixed
guard_label_y = end_guard
if abs(fixed_label_y - guard_label_y) < 8:
    if end_guard > end_fixed:
        guard_label_y = end_guard + 5
        fixed_label_y = end_fixed - 5
    else:
        fixed_label_y = end_fixed + 5
        guard_label_y = end_guard - 5

fixed_status = "BELOW" if end_fixed < target_nav else "ABOVE"
guard_status = "BELOW" if end_guard < target_nav else "ABOVE"

ax_main.annotate(
    f'Fixed ends: {end_fixed:.1f}  ({fixed_status} target)',
    xy=(n_steps, end_fixed), xytext=(n_steps - 50, fixed_label_y - 10),
    fontsize=9, fontweight='600', color='#dc2626',
    arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1, alpha=0.6),
    zorder=6,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#fecaca', alpha=0.9),
)

ax_main.annotate(
    f'Guardrail ends: {end_guard:.1f}  ({guard_status} target)',
    xy=(n_steps, end_guard), xytext=(n_steps - 50, guard_label_y + 10),
    fontsize=9, fontweight='600', color='#16a34a',
    arrowprops=dict(arrowstyle='->', color='#16a34a', lw=1, alpha=0.6),
    zorder=6,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#bbf7d0', alpha=0.9),
)

# 축 스타일 (상단)
ax_main.set_xlim(-3, n_steps + 40)
ax_main.set_ylim(0, y_top + 5)
ax_main.set_ylabel('NAV', fontsize=11, color='#64748b', labelpad=8)
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)
ax_main.spines['left'].set_color('#e2e8f0')
ax_main.spines['bottom'].set_visible(False)
ax_main.tick_params(colors='#94a3b8', labelsize=8.5, bottom=False, labelbottom=False)
ax_main.grid(False)

# 범례
legend = ax_main.legend(
    loc='upper left', frameon=True, fontsize=9,
    framealpha=0.95, edgecolor='#e2e8f0', fancybox=True, borderpad=0.8,
)
legend.get_frame().set_facecolor('#f8fafc')

# 밴드 정보 박스
info_text = (
    f'Annual W rate: {target_ratio*12*100:.1f}%\n'
    f'Band: ±5%\n'
    f'  ({lower_band*100:.2f}% – {upper_band*100:.2f}% monthly)\n'
    f'Initial NAV: {NAV0:.0f}'
)
props = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#e2e8f0', alpha=0.95)
ax_main.text(
    0.99, 0.98, info_text, transform=ax_main.transAxes,
    fontsize=8.5, verticalalignment='top', horizontalalignment='right',
    color='#475569', bbox=props, fontfamily='monospace',
)

# 타이틀
fig.text(
    0.475, 0.96, 'Guardrails Improve Portfolio Recovery Under Withdrawals',
    ha='center', va='bottom',
    fontsize=17, fontweight='700', color='#1e293b',
)
fig.text(
    0.475, 0.93, 'Same market path, different withdrawal rules → different ending NAV',
    ha='center', va='bottom',
    fontsize=10.5, color='#64748b',
)

# ===== 하단: 인출액 비교 =====
ax_wd.set_facecolor('white')

# Drawdown / Recovery 구간 음영 (동기화)
ax_wd.axvspan(60, 120, alpha=0.05, color='#e74c3c', zorder=0)
ax_wd.axvspan(120, 180, alpha=0.04, color='#16a34a', zorder=0)

ax_wd.fill_between(time, W0, alpha=0.15, color='#dc2626', step='mid',
                    label='Fixed W', zorder=2)
ax_wd.plot(time, W_fixed_arr, color='#dc2626', linewidth=1.2, alpha=0.5,
           linestyle='--', zorder=3)

ax_wd.fill_between(time, Wg, alpha=0.2, color='#16a34a', step='mid',
                    label='Guardrail W', zorder=2)
ax_wd.plot(time, Wg, color='#16a34a', linewidth=1.5, zorder=4)

# 축 스타일 (하단)
ax_wd.set_xlim(-3, n_steps + 40)
ax_wd.set_ylabel('Withdrawal\n(monthly)', fontsize=9, color='#64748b', labelpad=8)
ax_wd.set_xlabel('Time (months)', fontsize=10, color='#64748b', labelpad=8)
ax_wd.spines['top'].set_visible(False)
ax_wd.spines['right'].set_visible(False)
ax_wd.spines['left'].set_color('#e2e8f0')
ax_wd.spines['bottom'].set_color('#e2e8f0')
ax_wd.tick_params(colors='#94a3b8', labelsize=8)
ax_wd.grid(False)

# 하단 범례
wd_legend = ax_wd.legend(
    loc='upper right', frameon=True, fontsize=8,
    framealpha=0.9, edgecolor='#e2e8f0', fancybox=True, ncol=2,
)
wd_legend.get_frame().set_facecolor('#f8fafc')

# Drawdown 구간 인출 축소 annotation
crisis_t = 95
if Wg[crisis_t] > 0:
    ax_wd.annotate(
        '인출 축소\n(자본 보전)',
        xy=(crisis_t, Wg[crisis_t]), xytext=(crisis_t + 28, Wg[crisis_t] * 0.55),
        fontsize=8, color='#475569', fontweight='600',
        arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=0.8),
        zorder=5,
    )

# Recovery 구간 인출 확대 annotation
rally_t = 165
if Wg[rally_t] > 0:
    ax_wd.annotate(
        '인출 확대\n(수익 향유)',
        xy=(rally_t, Wg[rally_t]), xytext=(rally_t + 28, Wg[rally_t] * 1.5),
        fontsize=8, color='#475569', fontweight='600',
        arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=0.8),
        zorder=5,
    )

# ===== 저장 =====
fig.savefig('guardrail_persuasion.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig('guardrail_persuasion.svg', format='svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)

print(f"Market ending NAV: {mkt_nav[-1]:.1f}")
print(f"Fixed  ending NAV: {end_fixed:.1f}  ({'PASS' if end_fixed >= target_nav else 'FAIL'} vs target {target_nav:.0f})")
print(f"Guard  ending NAV: {end_guard:.1f}  ({'PASS' if end_guard >= target_nav else 'FAIL'} vs target {target_nav:.0f})")
print(f"Guardrail advantage: +{end_guard - end_fixed:.1f}")
print("Saved: guardrail_persuasion.png (300dpi)")
print("Saved: guardrail_persuasion.svg (vector)")
