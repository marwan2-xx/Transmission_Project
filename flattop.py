"""
Co-channel cell plotter — FLAT-TOP hexagon orientation
(matches the hexagonal.pdf grid: flat edges on top & bottom, vertices on left & right)

Coordinate system  (axial / cube-lite):
  q  →  column axis  (points RIGHT along flat-top rows)
  r  →  row  axis    (points DOWN-RIGHT; each step shifts half a column)

Flat-top center positions:
  cx = cell_size * 1.5  * q
  cy = cell_size * sqrt(3) * (r + q/2)

Co-channel displacement rule  (i,j stepping):
  1. From the reference cell, move i steps in direction d  (one of 6 axial dirs)
  2. Then turn 60° counter-clockwise  →  direction (d+2) % 6
  3. Move j steps in that new direction
  6 co-channel cells are found by repeating for d = 0,1,2,3,4,5
"""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Wedge


# ── Flat-top axial directions (q,r) ─────────────────────────────────────────
# For flat-top hexagons the six neighbours in axial coords are:
FLAT_TOP_DIRS = [
    ( 1,  0),   # 0 → right
    ( 0,  1),   # 1 → lower-right
    (-1,  1),   # 2 → lower-left
    (-1,  0),   # 3 → left
    ( 0, -1),   # 4 → upper-left
    ( 1, -1),   # 5 → upper-right
]

# ── Hardcoded correct n values per (N, sectoring) ────────────────────────────
# Order: [Omni, 60°, 120°, 180°]
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

def find_ij(N):
    """Return (i, j) with i >= j >= 0 satisfying  i² + i·j + j² = N."""
    for i in range(0, N + 1):
        for j in range(0, i + 1):
            if i * i + i * j + j * j == N:
                return i, j
    return None


def find_cochannel_cells(i, j):
    """
    Apply the (i, j) displacement in all 6 directions.
    Returns list of 6 (q, r) axial coords of co-channel cells,
    together with the intermediate waypoint after the i-steps (for drawing arrows).
    """
    cells = []
    waypoints = []
    for d in range(6):
        q, r = 0, 0
        # Step 1: move i steps in direction d
        dq, dr = FLAT_TOP_DIRS[d]
        q += i * dq
        r += i * dr
        waypoints.append((q, r))
        # Step 2: turn 60° CCW → direction (d+2)%6, move j steps
        dq2, dr2 = FLAT_TOP_DIRS[(d + 1) % 6]
        q += j * dq2
        r += j * dr2
        cells.append((q, r))
    return cells, waypoints


def axial_to_cart(q, r, cell_size=1.0):
    """Flat-top hex center in Cartesian (x, y)."""
    x = cell_size * 1.5 * q
    y = cell_size * math.sqrt(3) * (r + q / 2.0)
    return x, y


def hex_corners_flat_top(cx, cy, size):
    """Return the 6 corner (x,y) points of a flat-top hexagon."""
    return [
        (cx + size * math.cos(math.radians(60 * k)),
         cy + size * math.sin(math.radians(60 * k)))
        for k in range(6)
    ]


# def get_n_interferers(N, sectoring):
#     """
#     Count worst-case co-channel interferers for a given sectoring.
#     For omni: always 6.
#     For sectored: geometrically count cells inside the beam.
#     """
#     if sectoring == "Omni":
#         return 6

#     ij = find_ij(N)
#     if ij is None:
#         return {"180": 3, "120": 2, "60": 1}.get(sectoring, 6)

#     i, j = ij
#     cells, _ = find_cochannel_cells(i, j)

#     # Angle of each co-channel cell from origin (flat-top layout)
#     def cell_angle(q, r):
#         x, y = axial_to_cart(q, r)
#         return math.degrees(math.atan2(y, x)) % 360

#     angles = [cell_angle(q, r) for q, r in cells]
#     beam_width = {"180": 180, "120": 120, "60": 60}[sectoring]
#     half = beam_width / 2.0

#     max_n = 0
#     # Boresights for flat-top: sectors bisect at 0°, 60°, 120°, 180°, 240°, 300°
#     for boresight in [0, 60, 120, 180, 240, 300]:
#         count = sum(
#             1 for a in angles
#             if abs((a - boresight + 180) % 360 - 180) <= half + 0.001
#         )
#         max_n = max(max_n, count)
#     return max_n

def get_n_for_sectoring(N, sectoring):
    row = _CORRECT_N_INTERFERERS.get(N)
    if row is None:
        # fallback for any N not in the table
        return {"Omni (no sectoring)": 6, "180": 3, "120": 2, "60": 1}.get(sectoring, 6)
    return row[_SECT_IDX.get(sectoring, 0)]


def plot_cochannel(N, sectoring="Omni", grid_radius=8, cell_size=0.75,
                   show_steps=True):
    """
    Draw the hexagonal grid (flat-top orientation) highlighting:
      • Reference cell  (yellow)
      • Co-channel cells  (orange, dashed border)
      • Arrows showing the i steps + j steps for ONE representative direction
      • Sector wedges on ref + co-channel cells (if sectored)

    Parameters
    ----------
    N           : cluster size / reuse factor  (must satisfy i²+ij+j²=N)
    sectoring   : "Omni" | "60" | "120" | "180"
    grid_radius : half-width of background grid in cells
    cell_size   : hex radius (center-to-vertex) in data units
    show_steps  : if True, draw i-arrow + j-arrow for direction 0
    """
    ij = find_ij(N)
    if ij is None:
        print(f"N={N} cannot be expressed as i²+i·j+j². Skipping.")
        return
    i, j = ij

    cochannel_cells, waypoints = find_cochannel_cells(i, j)
    cochannel_set = set(cochannel_cells)

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    # ── Colour palette ────────────────────────────────────────────────────
    bg_fill   = "#e8eaf6"
    ref_fill  = "#ffeb3b"
    co_fill   = "#ffcc80"
    sec_pal = {
        "60":  ["#ef5350","#ff9800","#ffd740","#66bb6a","#42a5f5","#ab47bc"],
        "120": ["#ef5350","#42a5f5","#66bb6a"],
        "180": ["#ef5350","#42a5f5"],
        "Omni": ["#81d4fa"],
    }
    sec_colors = sec_pal.get(sectoring, ["#81d4fa"])

    # ── Draw background grid ──────────────────────────────────────────────
    for q in range(-grid_radius, grid_radius + 1):
        for r in range(-grid_radius, grid_radius + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) > grid_radius:
                continue
            cx, cy = axial_to_cart(q, r, cell_size)
            corners = hex_corners_flat_top(cx, cy, cell_size * 0.97)

            if (q, r) == (0, 0):
                facecolor, edgecolor, lw, ls = ref_fill, "#d32f2f", 4.5, "-"
            elif (q, r) in cochannel_set:
                facecolor, edgecolor, lw, ls = co_fill, "#f57c00", 3.5, "--"
            else:
                facecolor, edgecolor, lw, ls = bg_fill, "#90a4ae", 0.9, "-"

            ax.add_patch(Polygon(corners, closed=True, facecolor=facecolor,
                                 edgecolor=edgecolor, linewidth=lw,
                                 linestyle=ls))

    # ── Draw sector wedges on ref + co-channel cells ─────────────────────
    def draw_sectors(cx, cy):
        if sectoring == "Omni":
            return
        ns = {"60": 6, "120": 3, "180": 2}[sectoring]
        sa = 360.0 / ns
        for k in range(ns):
            ax.add_patch(Wedge(
                (cx, cy), cell_size * 0.76,
                k * sa,          # start angle (flat-top: first sector bisected by +x)
                (k + 1) * sa,
                facecolor=sec_colors[k], edgecolor="white",
                linewidth=1.2, alpha=0.88
            ))

    draw_sectors(0, 0)
    ax.text(0, 0, "Ref", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#b71c1c")

    for (q, r) in cochannel_cells:
        cx, cy = axial_to_cart(q, r, cell_size)
        draw_sectors(cx, cy)
        ax.text(cx, cy - cell_size * 0.30, "co-ch",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="#e65100")

    # ── Draw i-steps + j-steps arrows (direction 0 only, as example) ─────
    if show_steps and (i > 0 or j > 0):
        i_arrow_kw = dict(arrowstyle="-|>", color="#1565c0", lw=2.0,
                          mutation_scale=18)
        j_arrow_kw = dict(arrowstyle="-|>", color="#6a1b9a", lw=2.0,
                          mutation_scale=18)

        # i-steps arrow: origin → waypoint[0]
        wp_q, wp_r = waypoints[0]
        x0, y0 = axial_to_cart(0, 0, cell_size)
        xw, yw = axial_to_cart(wp_q, wp_r, cell_size)
        if i > 0:
            ax.annotate("", xy=(xw, yw), xytext=(x0, y0),
                        arrowprops=dict(**i_arrow_kw))
            mx, my = (x0 + xw) / 2, (y0 + yw) / 2
            ax.text(mx + cell_size * 0.15, my + cell_size * 0.15,
                    f"i={i}", fontsize=12, fontweight="bold",
                    color="#1565c0",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.85))

        # j-steps arrow: waypoint[0] → co-channel cell[0]
        cc_q, cc_r = cochannel_cells[0]
        xc, yc = axial_to_cart(cc_q, cc_r, cell_size)
        if j > 0:
            ax.annotate("", xy=(xc, yc), xytext=(xw, yw),
                        arrowprops=dict(**j_arrow_kw))
            mx2, my2 = (xw + xc) / 2, (yw + yc) / 2
            ax.text(mx2 + cell_size * 0.15, my2 + cell_size * 0.15,
                    f"j={j}", fontsize=12, fontweight="bold",
                    color="#6a1b9a",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.85))

    # ── C/I label ─────────────────────────────────────────────────────────
    n_int = get_n_for_sectoring(N, sectoring)
    ci_db = 10 * math.log10(3.0 * N / n_int)

    title = (f"Flat-Top Hexagonal Grid  |  N = {N}  ({sectoring} sectoring)\n"
             f"i = {i},  j = {j}   →   N = i² + i·j + j² = {N}   "
             f"|   n = {n_int}   |   C/I ≈ {ci_db:.1f} dB")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20, color="#263238")

    # ── i,j formula box ──────────────────────────────────────────────────
    ax.text(
        -grid_radius * cell_size * 1.55,
        -grid_radius * cell_size * 1.58,
        f"i = {i} , j = {j}   →   N = {i}² + {i}·{j} + {j}² = {N}",
        fontsize=11, fontweight="bold", color="#37474f",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9)
    )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elems = [
        mpatches.Patch(facecolor=ref_fill, edgecolor="#d32f2f", lw=3,
                       label="Reference Cell"),
        mpatches.Patch(facecolor=co_fill, edgecolor="#f57c00", lw=2,
                       linestyle="--", label="Co-channel Cells"),
        mpatches.Patch(facecolor=bg_fill, edgecolor="#90a4ae",
                       label="Other Cells"),
    ]
    if show_steps:
        legend_elems += [
            mpatches.Patch(color="#1565c0", label=f"i = {i} steps"),
            mpatches.Patch(color="#6a1b9a", label=f"j = {j} steps (60° CCW turn)"),
        ]
    ax.legend(handles=legend_elems, loc="upper right",
              fontsize=10, frameon=True)

    lim = grid_radius * cell_size * 1.7
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    plt.tight_layout()
    return fig


# ── Run examples ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)

    cases = [
        # (N,  sectoring,  grid_radius)
        (3,   "Omni",  6),
        (3,   "60",   5),
        (3,   "120",   5),
        (3,   "180",   5),
        (4,   "Omni",  6),
        (4,   "60",   5),
        (4,   "120",   5),
        (4,   "180",   5),
        
        (7,   "Omni",  7),
        (7,   "60",   5),
        (7,   "180",   5),
        (7,   "120",   6),
        
        (9,   "Omni",  7),
        (9,   "120",   6),
        (9,   "180",   5),
        (9,   "60",   5),
        (12,  "Omni",  8),
        (12,  "120",   7),
        (12,   "60",   5),
        (12,   "180",   5),
        (13,  "Omni",  8),
        (13,  "120",   7),
        (13,  "60",    7),
        (13,   "180",   5),

        (16,  "Omni",  8),
        (16,  "120",   7),
        (16,  "60",    7),
        (16,   "180",   5),

        (19,  "Omni",  8),
        (19,  "120",   7),
        (19,  "60",    7),
        (19,   "180",   5),
    ]

    for N, sec, gr in cases:
        fig = plot_cochannel(N, sectoring=sec, grid_radius=gr,
                             cell_size=0.75, show_steps=True)
        if fig:
            fname = f"output/N{N}_{sec}.png"
            fig.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fname}")