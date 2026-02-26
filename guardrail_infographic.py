"""
Guardrail Dynamic Withdrawal Infographic
- matplotlib only, single figure
- NAV path + Guardrail band-based dynamic withdrawal visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ===== 한글 폰트 설정 (Windows Malgun Gothic) =====
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ===== 시뮬레이션 파라미터 =====
np.random.seed(42)
n_steps = 240
dt = 1 / 12
mu = 0.06
sigma = 0.14

target_ratio = 0.01        # 초기 W/NAV = 1.0%
lower_band = 0.007          # 하한 0.7%
upper_band = 0.013          # 상한 1.3%

# ===== NAV 경로 생성 (GBM — 스타일화) =====
z = np.random.randn(n_steps)
log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
# 위기 구간 삽입 (60~100): 하락 → 회복 드라마 강화
log_returns[60:80] = -0.03 + 0.005 * np.random.randn(20)
log_returns[80:100] = 0.005 + 0.005 * np.random.randn(20)
# 후반 랠리 (160~200)
log_returns[160:200] = 0.015 + 0.005 * np.random.randn(40)

nav = np.zeros(n_steps + 1)
nav[0] = 100.0
for i in range(n_steps):
    nav[i + 1] = nav[i] * np.exp(log_returns[i])

# ===== 인출액 계산 (Guardrail 로직) =====
W = np.zeros(n_steps + 1)
W[0] = target_ratio * nav[0]

for t in range(1, n_steps + 1):
    ratio = W[t - 1] / nav[t]
    if ratio > upper_band:
        W[t] = upper_band * nav[t]
    elif ratio < lower_band:
        W[t] = lower_band * nav[t]
    else:
        W[t] = W[t - 1]

W_fixed = W[0]  # 고정 인출 (비교용)

# ===== 화살표 그릴 시점 선택 =====
arrow_times = list(range(10, n_steps + 1, 18))
for extra in [65, 75, 90, 170, 190]:
    if extra not in arrow_times and extra <= n_steps:
        arrow_times.append(extra)
arrow_times = sorted(set(arrow_times))

# ===== 그리기 =====
fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
ax.set_facecolor('white')

time = np.arange(n_steps + 1)

# NAV 경로
ax.plot(time, nav, color='#1e293b', linewidth=2.8, zorder=5, label='NAV')

# 화살표 스케일: 인출액을 화살표 길이로 변환
arrow_scale = nav.max() * 0.22 / W.max()

# 고정 인출 화살표 (faint)
for t in arrow_times:
    arrow_len = W_fixed * arrow_scale
    ax.annotate(
        '', xy=(t, nav[t] - arrow_len), xytext=(t, nav[t]),
        arrowprops=dict(
            arrowstyle='->', color='#e74c3c', lw=1.3,
            alpha=0.3, shrinkA=0, shrinkB=0,
            connectionstyle='arc3,rad=0',
        ),
        zorder=3,
    )

# Guardrail 인출 화살표
for t in arrow_times:
    arrow_len = W[t] * arrow_scale
    ax.annotate(
        '', xy=(t, nav[t] - arrow_len), xytext=(t, nav[t]),
        arrowprops=dict(
            arrowstyle='->', color='#16a34a', lw=2.2,
            alpha=0.85, shrinkA=0, shrinkB=0,
            connectionstyle='arc3,rad=0',
        ),
        zorder=4,
    )

# 위기 구간 하이라이트
ax.axvspan(60, 100, alpha=0.04, color='#e74c3c', zorder=0)
ax.axvspan(160, 200, alpha=0.04, color='#16a34a', zorder=0)

# 구간 라벨
ax.text(80, nav[60:101].min() * 0.88, 'Drawdown', ha='center', va='top',
        fontsize=10, color='#94a3b8', fontstyle='italic', fontweight='500')
ax.text(180, nav[160:201].max() * 1.04, 'Rally', ha='center', va='bottom',
        fontsize=10, color='#94a3b8', fontstyle='italic', fontweight='500')

# 화살표 길이 비교 주석 (위기 구간)
t_crisis = 75
arrow_len_g = W[t_crisis] * arrow_scale
mid_y = nav[t_crisis] - arrow_len_g * 0.5
ax.annotate(
    '인출 축소\n(자본 보전)',
    xy=(t_crisis, mid_y), xytext=(t_crisis + 22, mid_y - 8),
    fontsize=9, color='#475569', fontweight='600',
    ha='left', va='center',
    arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1, connectionstyle='arc3,rad=-0.2'),
    zorder=6,
)

# 화살표 길이 비교 주석 (랠리 구간)
t_rally = 190
arrow_len_g = W[t_rally] * arrow_scale
mid_y = nav[t_rally] - arrow_len_g * 0.5
ax.annotate(
    '인출 확대\n(수익 향유)',
    xy=(t_rally, mid_y), xytext=(t_rally + 22, mid_y + 8),
    fontsize=9, color='#475569', fontweight='600',
    ha='left', va='center',
    arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1, connectionstyle='arc3,rad=0.2'),
    zorder=6,
)

# 축 정리
ax.set_xlim(-5, n_steps + 10)
y_min = nav.min() * 0.65
y_max = nav.max() * 1.12
ax.set_ylim(y_min, y_max)

ax.set_xlabel('Time (months)', fontsize=11, color='#64748b', labelpad=10)
ax.set_ylabel('NAV', fontsize=11, color='#64748b', labelpad=10)

# 축 스타일
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cbd5e1')
ax.spines['bottom'].set_color('#cbd5e1')
ax.tick_params(colors='#94a3b8', labelsize=9)
ax.grid(False)

# 타이틀
ax.text(
    0.5, 1.06, 'Dynamic Withdrawal with Guardrails',
    transform=ax.transAxes, ha='center', va='bottom',
    fontsize=18, fontweight='700', color='#1e293b',
)
ax.text(
    0.5, 1.02, 'Withdrawal adjusts to maintain W / NAV within band',
    transform=ax.transAxes, ha='center', va='bottom',
    fontsize=11, color='#64748b',
)

# 밴드 정보 박스
band_text = (
    f'Target W/NAV: {target_ratio*100:.1f}%\n'
    f'Lower band:   {lower_band*100:.1f}%\n'
    f'Upper band:   {upper_band*100:.1f}%'
)
props = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#e2e8f0', alpha=0.95)
ax.text(
    0.02, 0.97, band_text, transform=ax.transAxes,
    fontsize=9, verticalalignment='top', color='#475569',
    bbox=props, fontfamily='monospace',
)

# 범례
legend_elements = [
    plt.Line2D([0], [0], color='#1e293b', lw=2.8, label='NAV'),
    plt.Line2D([0], [0], color='#16a34a', lw=2.2, marker='v', markersize=6,
               label='Guardrail Withdrawal', markevery=[0]),
    plt.Line2D([0], [0], color='#e74c3c', lw=1.3, alpha=0.4, marker='v', markersize=5,
               label='Fixed Withdrawal (comparison)', markevery=[0]),
]
legend = ax.legend(
    handles=legend_elements, loc='upper right', frameon=True,
    fontsize=9, framealpha=0.95, edgecolor='#e2e8f0',
    fancybox=True, borderpad=0.8,
)
legend.get_frame().set_facecolor('#f8fafc')

plt.tight_layout(rect=[0, 0, 1, 0.96])

# ===== 저장 =====
fig.savefig('guardrail_infographic.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig('guardrail_infographic.svg', format='svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)

print("Saved: guardrail_infographic.png (300dpi)")
print("Saved: guardrail_infographic.svg (vector)")
