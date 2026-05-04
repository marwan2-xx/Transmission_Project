import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
from math import log2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TRANSMIT_POWER = 0.1      # 100 mW
NOISE_POWER    = 0.0001   # 0.1 mW

_CORRECT_N_INTERFERERS = {
    3:  [6, 2, 3, 4],
    4:  [6, 1, 2, 3],
    7:  [6, 1, 2, 3],
    9:  [6, 1, 2, 3],
    12: [6, 2, 3, 4],
    13: [6, 1, 2, 3],
    16: [6, 1, 2, 3],
    19: [6, 1, 2, 3],
}
_SECT_IDX = {
    "Omni (no sectoring)": 0,
    "60":  1,
    "120": 2,
    "180": 3,
}

# ─────────────────────────────────────────────
#  VALID N LIST
# ─────────────────────────────────────────────
def is_valid_N(N):
    for i in range(0, N + 1):
        for j in range(0, i + 1):
            val = i*i + i*j + j*j
            if val == N:  return True
            if val > N:   break
    return False

VALID_N_LIST = [N for N in range(3, 20) if is_valid_N(N)]

_HEX_DIRS = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]

def _get_ij(N):
    for i in range(0, N+1):
        for j in range(0, i+1):
            if i*i + i*j + j*j == N:
                return i, j
    return None

def _find_cochannel_cells(i, j):
    cells = []
    for d in range(6):
        q, r = 0, 0
        dq, dr = _HEX_DIRS[d]
        q += i*dq;  r += i*dr
        dq2, dr2 = _HEX_DIRS[(d+1) % 6]
        q += j*dq2; r += j*dr2
        cells.append((q, r))
    return cells


def get_n_for_sectoring(N, sectoring):
    row = _CORRECT_N_INTERFERERS.get(N)
    if row is None:
        return {"Omni (no sectoring)": 6, "180": 3, "120": 2, "60": 1}.get(sectoring, 6)
    return row[_SECT_IDX.get(sectoring, 0)]

def find_reuse_factor(c_i_linear):
    all_options = []
    for sectoring in ["Omni (no sectoring)", "180", "120", "60"]:
        for N in VALID_N_LIST:
            n  = get_n_for_sectoring(N, sectoring)
            ci = 3 * N / n
            if ci >= c_i_linear:
                all_options.append((sectoring, n, N, ci))
                break
    if not all_options:
        raise ValueError("Could not find a valid N for the given C/I requirement.")
    best = min(all_options, key=lambda x: x[2])
    return (best[2], best[0], best[1], best[3]), all_options

SECTORS_PER_CELL = {
    "Omni (no sectoring)": 1,
    "180": 2,
    "120": 3,
    "60":  6,
}

def hex_center(q, r, size):
    return size*math.sqrt(3)*(q+r/2), size*1.5*r

def hex_corners_pointy(cx, cy, size):
    return [(cx+size*math.cos(math.radians(60*i-30)),
             cy+size*math.sin(math.radians(60*i-30))) for i in range(6)]

def draw_hex_patch(ax, cx, cy, size, fc, ec='white', lw=2, alpha=1.0, zorder=1):
    from matplotlib.patches import Polygon as MPoly
    ax.add_patch(MPoly(hex_corners_pointy(cx,cy,size), closed=True,
                       facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder))

def draw_sector_wedges(ax, cx, cy, size, sectoring, colors, zorder=2):
    from matplotlib.patches import Wedge
    if sectoring == "Omni (no sectoring)": return
    ns = {"60":6,"120":3,"180":2}[sectoring]
    sa = 360.0/ns
    for k in range(ns):
        ax.add_patch(Wedge((cx,cy), size*0.80, k*sa-30, (k+1)*sa-30,
                           facecolor=colors[k%len(colors)], edgecolor='white',
                           linewidth=1.5, alpha=0.88, zorder=zorder))
    ax.plot(cx, cy, 'w.', markersize=5, zorder=zorder+1)

def draw_cochannel_map(N, sectoring, n_interferers, ci_db):
    ij = _get_ij(N)
    if ij is None:
        messagebox.showinfo("Info", f"N={N} has no valid (i,j)."); return
    i, j = ij
    cochannel_cells = _find_cochannel_cells(i, j)
    cochannel_set   = set(cochannel_cells)

    dq0, dr0 = _HEX_DIRS[0]
    wp = (i * dq0, i * dr0)

    size = 1.0
    grid_radius = int(math.sqrt(N)) + 2

    sec_pal = {
        "60":  ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6'],
        "120": ['#e74c3c','#3498db','#2ecc71'],
        "180": ['#e74c3c','#3498db'],
        "Omni (no sectoring)": ['#4e8ef7'],
    }
    sec_colors = sec_pal.get(sectoring, ['#4e8ef7'])
    ns = {"60":6, "120":3, "180":2}.get(sectoring, 0)

    BG = "#0f1117"
    fig, ax = plt.subplots(figsize=(8, 5))
    
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect('equal'); ax.axis('off')

    for q in range(-grid_radius, grid_radius + 1):
        for r in range(-grid_radius, grid_radius + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) > grid_radius:
                continue
            cx, cy = hex_center(q, r, size)
            if (q, r) == (0, 0):
                fc, ec, lw = '#ffe082', '#f9a825', 4.5
            elif (q, r) in cochannel_set:
                fc, ec, lw = '#ff8a65', '#e64a19', 3.5
            else:
                fc, ec, lw = '#1a1d2e', '#2e3250', 0.9
            draw_hex_patch(ax, cx, cy, size * 0.97, fc, ec, lw, zorder=1)
            if (q, r) in cochannel_set and (q, r) != (0, 0):
                from matplotlib.patches import Polygon as MPoly
                ax.add_patch(MPoly(hex_corners_pointy(cx, cy, size * 0.97), closed=True,
                    facecolor='none', edgecolor=ec, linewidth=lw, linestyle='dashed', zorder=2))

    def draw_sectors_on(q, r):
        cx, cy = hex_center(q, r, size)
        draw_sector_wedges(ax, cx, cy, size, sectoring, sec_colors, zorder=3)

    draw_sectors_on(0, 0)
    cx0, cy0 = hex_center(0, 0, size)
    ax.text(cx0, cy0, "Ref", ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1a1a1a', zorder=6)

    for (cq, cr) in cochannel_cells:
        draw_sectors_on(cq, cr)
        ccx, ccy = hex_center(cq, cr, size)
        ax.text(ccx, ccy - size * 0.30, "co-ch", ha='center', va='center',
                fontsize=8, fontweight='bold', color='#ffccbc', zorder=6)

    x0, y0 = hex_center(0, 0, size)
    xw, yw = hex_center(wp[0], wp[1], size)
    xc, yc = hex_center(cochannel_cells[0][0], cochannel_cells[0][1], size)
    arr = dict(arrowstyle="-|>", lw=2.0, mutation_scale=18)
    if i > 0:
        ax.annotate("", xy=(xw, yw), xytext=(x0, y0),
                    arrowprops=dict(**arr, color='#82b1ff'), zorder=7)
        ax.text((x0+xw)/2 + size*0.15, (y0+yw)/2 + size*0.15, f"i={i}",
                fontsize=11, fontweight='bold', color='#82b1ff',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1d2e', alpha=0.9), zorder=8)
    if j > 0:
        ax.annotate("", xy=(xc, yc), xytext=(xw, yw),
                    arrowprops=dict(**arr, color='#ce93d8'), zorder=7)
        ax.text((xw+xc)/2 + size*0.15, (yw+yc)/2 + size*0.15, f"j={j}",
                fontsize=11, fontweight='bold', color='#ce93d8',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1d2e', alpha=0.9), zorder=8)

    if ns > 0:
        handles = [plt.Rectangle((0,0),1,1, fc=sec_colors[k], alpha=0.88) for k in range(ns)]
        leg = ax.legend(handles, [f'Sector {k+1}' for k in range(ns)],
                  loc='upper right', fontsize=10, framealpha=0.95,
                  title='Sectors', title_fontsize=10)
        leg.get_frame().set_facecolor('#1a1d2e')
        leg.get_frame().set_edgecolor('#3a3d5a')
        for text in leg.get_texts(): text.set_color('#e0e4f0')
        leg.get_title().set_color('#e0e4f0')

    ci_str = f"{ci_db:.2f}" if ci_db is not None else "N/A"
    ax.set_title(
    f"Co-channel Map  ·  N={N}  ({sectoring})  ·  i={i}, j={j}  ·  n={n_interferers}  ·  C/I≈{ci_str} dB",
    fontsize=13, fontweight='bold', pad=24, color='white')

    lim = grid_radius * size * 1.8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    plt.tight_layout()
    plt.show()

sectoring_options = ["Omni (no sectoring)", "180", "120", "60"]

for N in VALID_N_LIST:
    for sectoring in sectoring_options:
        n = get_n_for_sectoring(N, sectoring)
        ci_linear = 3 * N / n
        ci_db = 10 * math.log10(ci_linear)
        print(f"N={N:2d} | sectoring={sectoring:20s} | n={n} | C/I={ci_db:.2f} dB")
        draw_cochannel_map(N, sectoring, n, ci_db)
        plt.savefig(f"N{N}_{sectoring.replace(' ', '_')}.png")
        plt.close()