import numpy as np
import matplotlib.pyplot as plt
import math
from math import log2, ceil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TRANSMIT_POWER = 0.1      # 100 mW
NOISE_POWER    = 0.0001   # 0.1 mW

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

VALID_N_LIST = [N for N in range(3, 19) if is_valid_N(N)]   # N>=3 (N=1 impractical)

# ─────────────────────────────────────────────
#  CELLULAR PLANNING
# ─────────────────────────────────────────────
def get_area_km2(length, width):
    return (length / 1000) * (width / 1000)

def get_total_subscribers(area_km2, subscriber_density):
    return area_km2 * subscriber_density

def get_total_channels(total_BW, trunk_BW):
    return int(total_BW / trunk_BW)

# Hex directions (pointy-top axial)
_HEX_DIRS = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)] 

def _get_ij(N):
    """Return (i,j) pair satisfying i²+ij+j²=N (smallest i,j)."""
    for i in range(0, N+1):
        for j in range(0, i+1):
            if i*i + i*j + j*j == N:
                return i, j
    return None

def _axial_to_angle(q, r):
    """Angle in degrees from origin to hex cell (q,r) in pointy-top layout."""
    x = math.sqrt(3)*q + math.sqrt(3)/2*r
    y = 1.5 * r
    return math.degrees(math.atan2(y, x)) % 360

def _find_cochannel_cells(i, j):
    """
    Apply the (i,j) displacement rule in all 6 directions.
    Move i steps in direction d, turn 60 deg CCW (direction d+2), move j steps.
    Returns list of 6 (q,r) axial coords of co-channel cells.
    """
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
    """
    Geometrically compute number of co-channel interferers n:
    1. Find (i,j) for this N using i²+ij+j² = N
    2. Locate 6 co-channel cells via the (i,j) displacement rule
    3. For each hex-edge boresight (0,60,120,180,240,300 deg),
       count cells inside the beam (width = sectoring angle)
    4. Return worst-case (max) count across all orientations
    """
    if sectoring == "Omni (no sectoring)":
        return 6

    ij = _get_ij(N)
    if ij is None:
        return {"180":3, "120":2, "60":1}.get(sectoring, 6)
    i, j = ij

    angles     = [_axial_to_angle(q,r) for q,r in _find_cochannel_cells(i,j)]
    beam_width = {"180":180, "120":120, "60":60}[sectoring]
    half       = beam_width / 2.0

    # Boresights aligned to hex-edge directions (pointy-top)
    max_n = 0
    for boresight in [0, 60, 120, 180, 240, 300]:
        count = sum(
            1 for a in angles
            if abs((a - boresight + 180) % 360 - 180) <= half + 0.001
        )
        max_n = max(max_n, count)
    return max_n

def find_reuse_factor(c_i_linear):
    """
    For each sectoring type, find the smallest valid N (>=3) satisfying
    C/I = 3N/n. Returns all valid (sectoring, N) pairs.
    Best combo is chosen by find_best_reuse (minimum cells).
    """
    all_options = []
    for sectoring in ["Omni (no sectoring)", "180", "120", "60"]:
        for N in VALID_N_LIST:
            n  = get_n_for_sectoring(N, sectoring)
            ci = 3 * N / n
            if ci >= c_i_linear:
                all_options.append((sectoring, n, N, ci))
                break   # smallest valid N for this sectoring type
    if not all_options:
        raise ValueError("Could not find a valid N for the given C/I requirement.")
    # Placeholder best — real best determined by find_best_reuse
    best = min(all_options, key=lambda x: x[2])   # fallback: smallest N
    return (best[2], best[0], best[1], best[3]), all_options


# Number of physical sectors per cell per sectoring type
SECTORS_PER_CELL = {
    "Omni (no sectoring)": 1,
    "180": 2,
    "120": 3,
    "60":  6,
}

def erlang_b(A, C):
    if A == 0: return 0.0
    B = 1.0
    for k in range(1, int(C) + 1):
        B = (A * B) / (k + A * B)
    return B

def get_number_of_cells(total_subscribers, user_session_time,
                        requests_per_second, trunk_BW, total_BW,
                        blocking_probability, N,
                        sectoring="Omni (no sectoring)"):
    """
    Per-sector Erlang B analysis:
      channels_per_cell   = total_channels // N
      channels_per_sector = channels_per_cell // num_sectors
      A_per_sector        = A_total / (num_cells * num_sectors)
    Erlang B is evaluated per sector (each sector is its own trunk group).
    """
    traffic_per_user    = requests_per_second * user_session_time
    A_total             = total_subscribers * traffic_per_user
    total_channels      = get_total_channels(total_BW, trunk_BW)
    channels_per_cell   = max(1, int(total_channels // N))
    num_sectors         = SECTORS_PER_CELL.get(sectoring, 1)
    channels_per_sector = max(1, channels_per_cell // num_sectors)

    for num_cells in range(1, 10_000_001):
        A_per_sector = A_total / (num_cells * num_sectors)
        if erlang_b(A_per_sector, channels_per_sector) <= blocking_probability:
            return num_cells, channels_per_cell, A_total / num_cells
    return 10_000_000, channels_per_cell, A_total / 10_000_000


def find_best_reuse(all_options, total_subscribers, user_session_time,
                    requests_per_second, trunk_BW, total_BW, blocking_probability):
    """
    From all valid (sectoring, N) combos, pick the one giving minimum cells.
    Returns: best_result, num_cells, ch_per_cell, A_per_cell
    """
    best_cells, best_result, best_ch_pc, best_A_pc = None, None, None, None
    for sectoring, n, N, ci in all_options:
        cells, ch_pc, A_pc = get_number_of_cells(
            total_subscribers, user_session_time, requests_per_second,
            trunk_BW, total_BW, blocking_probability, N, sectoring)
        if best_cells is None or cells < best_cells:
            best_cells  = cells
            best_result = (N, sectoring, n, ci)
            best_ch_pc  = ch_pc
            best_A_pc   = A_pc
    return best_result, best_cells, best_ch_pc, best_A_pc


# def shannon_capacity(channel_bandwidth_hz, snr_linear):
#     return channel_bandwidth_hz * log2(1 + snr_linear)

# CORRECT shannon capacity according to the project note 
def shannon_capacity(trunk_bw, transmit_power, noise_power, c_i_linear):
    """
    C->Transmit power
    sigma ->noise power
    Shannon capacity treating interference as noise (per the project note).
    SINR = C / (sigma^2 + I),  where I = C / (C/I)_min
    """
    I = transmit_power / c_i_linear      # interference power
    sinr = transmit_power / (math.pow(noise_power , 2) + I)
    return trunk_bw * log2(1 + sinr)




# ─────────────────────────────────────────────
#  16-QAM
# ─────────────────────────────────────────────
def get_16qam_constellation():
    re = np.array([-3, -1, 1, 3])
    im = np.array([-3, -1, 1, 3])
    const = np.array([complex(r, i) for r in re for i in im])
    return const / np.sqrt(10)

def bits_to_symbols(bits):
    constellation = get_16qam_constellation()
    bits = np.asarray(bits, dtype=int)
    pad  = (4 - len(bits) % 4) % 4
    if pad: bits = np.append(bits, np.zeros(pad, dtype=int))
    symbols = []
    for i in range(0, len(bits), 4):
        idx = int(''.join(map(str, bits[i:i+4])), 2)
        symbols.append(constellation[idx])
    return np.array(symbols)

def transmit(symbols, tx_power, noise_power, c_i_linear):
    transmitted        = symbols * np.sqrt(tx_power)
    interference_power = tx_power / c_i_linear if c_i_linear > 0 else 0
    total_noise_var    = noise_power + interference_power
    noise = np.sqrt(total_noise_var / 2) * (
        np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
    )
    return transmitted + noise

def demodulate(received_symbols, tx_power):
    constellation = get_16qam_constellation() * np.sqrt(tx_power)
    received_bits = []
    for sym in received_symbols:
        idx  = np.argmin(np.abs(constellation - sym))
        bit4 = np.array(list(format(idx, '04b')), dtype=int)
        received_bits.extend(bit4)
    return np.array(received_bits, dtype=int)

def compute_BER(original_bits, received_bits):
    n      = min(len(original_bits), len(received_bits))
    errors = int(np.sum(original_bits[:n] != received_bits[:n]))
    return errors / n, errors

def generate_bitStream():
    return np.random.randint(0, 2, 10000)

# ─────────────────────────────────────────────
#  BER PLOTS
# ─────────────────────────────────────────────
def plot_ber_vs_tx_power(c_i_linear, noise_power):
    tx_powers = np.logspace(-3, 1, 25)
    bers = []
    for p in tx_powers:
        bits = generate_bitStream()
        rx   = transmit(bits_to_symbols(bits), p, noise_power, c_i_linear)
        ber, _ = compute_BER(bits, demodulate(rx, p))
        bers.append(max(ber, 1e-6))
    plt.figure(figsize=(8, 5))
    plt.semilogy(tx_powers * 1000, bers, marker='o', color='steelblue')
    plt.xlabel("Transmit Power (mW)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs Transmit Power (fixed noise power)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_ber_vs_noise_power(c_i_linear, tx_power):
    noise_powers = np.logspace(-5, 0, 25)
    bers = []
    for np_ in noise_powers:
        bits = generate_bitStream()
        rx   = transmit(bits_to_symbols(bits), tx_power, np_, c_i_linear)
        ber, _ = compute_BER(bits, demodulate(rx, tx_power))
        bers.append(max(ber, 1e-6))
    plt.figure(figsize=(8, 5))
    plt.semilogy(noise_powers * 1000, bers, marker='s', color='orange')
    plt.xlabel("Noise Power (mW)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"BER vs Noise Power (fixed TX = {tx_power*1000:.0f} mW)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
#  BONUS: IMAGE TRANSMISSION
# ─────────────────────────────────────────────
def transmit_image(image_path, tx_power, noise_power, c_i_linear):
    from PIL import Image
    img       = Image.open(image_path).convert("L").resize((256, 256))
    img_array = np.array(img, dtype=np.uint8)
    bits      = np.unpackbits(img_array.flatten())
    symbols   = bits_to_symbols(bits)
    received  = transmit(symbols, tx_power, noise_power, c_i_linear)
    rx_bits   = demodulate(received, tx_power)
    rx_bits   = rx_bits[:len(bits)].astype(np.uint8)
    rx_array  = np.packbits(rx_bits).reshape(256, 256)
    ber, errors = compute_BER(bits, rx_bits)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_array, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(rx_array,  cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Received Image (BER={ber:.4f})"); axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    return ber, errors


# ─────────────────────────────────────────────
#  CLUSTER FIGURE  (tkinter canvas)
# ─────────────────────────────────────────────
from math import cos, sin, radians, sqrt

CELL_COLORS = [
    "#4e8ef7","#e8734a","#50c97f","#f5c842","#a87de8",
    "#e85c8a","#4ec9c9","#f97f51","#7bb8f5","#c9e86a",
    "#f5a623","#8e44ad","#2ecc71","#e74c3c","#1abc9c",
]
SECTOR_COLORS = {
    "60":  ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db","#9b59b6"],
    "120": ["#e74c3c","#3498db","#2ecc71"],
    "180": ["#e74c3c","#3498db"],
    "Omni (no sectoring)": ["#4e8ef7"],
}


def _hex_pixel(q, r, size):
    """Axial → pixel (pointy-top, flat-side hexagons)."""
    x = size * sqrt(3) * (q + r / 2)
    y = size * 1.5 * r
    return x, y


def _hex_corners_canvas(cx, cy, size):
    """6 corner coords of a pointy-top hexagon centred at (cx,cy)."""
    pts = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        pts.append(cx + size * math.cos(angle))
        pts.append(cy + size * math.sin(angle))
    return pts


def _cluster_cells_axial(N):
    """Return N axial (q,r) coords in a compact spiral cluster."""
    dirs = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    cells = [(0, 0)]
    ring  = 1
    while len(cells) < N:
        q, r = ring, 0
        for d in range(6):
            for _ in range(ring):
                if len(cells) < N:
                    cells.append((q, r))
                dq, dr = dirs[(d + 4) % 6]
                q += dq; r += dr
        ring += 1
    return cells[:N]


def _wedge_coords(cx, cy, radius, start_deg, end_deg, steps=20):
    """Approximate a pie-wedge with a polygon (for tk canvas)."""
    pts = [cx, cy]
    for s in range(steps + 1):
        a = math.radians(start_deg + (end_deg - start_deg) * s / steps)
        pts.append(cx + radius * math.cos(a))
        pts.append(cy + radius * math.sin(a))
    return pts


class ClusterWindow(tk.Toplevel):
    """Standalone Toplevel window that draws the hex cluster."""

    def __init__(self, master, N, sectoring, n_interferers, ci_db):
        super().__init__(master)
        self.title(f"Cellular Cluster — N={N}, {sectoring} sectoring")
        self.resizable(True, True)

        CELL_SIZE   = max(28, min(52, int(320 / max(N, 1) ** 0.5)))
        cells       = _cluster_cells_axial(N)

        # compute pixel centers and canvas offset so cluster is centred
        raw = [_hex_pixel(q, r, CELL_SIZE) for q, r in cells]
        cx_mean = sum(p[0] for p in raw) / len(raw)
        cy_mean = sum(p[1] for p in raw) / len(raw)

        # neighbour ring
        seen = set(cells); nb_ring = []
        for q, r in cells:
            for dq, dr in [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]:
                nb = (q+dq, r+dr)
                if nb not in seen:
                    nb_ring.append(nb); seen.add(nb)

        all_pts  = cells + nb_ring
        all_px   = [_hex_pixel(q, r, CELL_SIZE)[0] - cx_mean for q,r in all_pts]
        all_py   = [_hex_pixel(q, r, CELL_SIZE)[1] - cy_mean for q,r in all_pts]
        pad      = CELL_SIZE * 2
        W        = int(max(all_px) - min(all_px) + pad * 2 + 200)
        H        = int(max(all_py) - min(all_py) + pad * 2 + 120)
        ox       = -min(all_px) + pad          # pixel offset X
        oy       = -min(all_py) + pad + 50     # pixel offset Y (leave room for title)

        cv = tk.Canvas(self, width=W, height=H, bg="#eef2ff")
        cv.pack(fill=tk.BOTH, expand=True)

        sec_colors = SECTOR_COLORS.get(sectoring, ["#4e8ef7"])
        ns = {"60":6, "120":3, "180":2}.get(sectoring, 0)

        # ── draw neighbour ring ──────────────────────────────────────
        for q, r in nb_ring:
            px = _hex_pixel(q, r, CELL_SIZE)[0] - cx_mean + ox
            py = _hex_pixel(q, r, CELL_SIZE)[1] - cy_mean + oy
            pts = _hex_corners_canvas(px, py, CELL_SIZE * 0.97)
            cv.create_polygon(pts, fill="#d4d9f5", outline="#9aa5d4",
                              width=1)

        # ── draw cluster cells ───────────────────────────────────────
        ij = _get_ij(N)
        ref_q, ref_r = 0, 0   # reference cell is always at origin

        for idx, (q, r) in enumerate(cells):
            px = _hex_pixel(q, r, CELL_SIZE)[0] - cx_mean + ox
            py = _hex_pixel(q, r, CELL_SIZE)[1] - cy_mean + oy
            fc = CELL_COLORS[idx % len(CELL_COLORS)]
            is_ref = (q == ref_q and r == ref_r)

            # hex background
            pts = _hex_corners_canvas(px, py, CELL_SIZE * 0.97)
            cv.create_polygon(pts, fill=fc, outline="white",
                              width=2, stipple="gray25" if not is_ref else "")

            # sector wedges
            if ns > 0:
                sa = 360.0 / ns
                for k in range(ns):
                    start_a = k * sa - 30
                    end_a   = start_a + sa
                    wpts = _wedge_coords(px, py, CELL_SIZE * 0.80,
                                         start_a, end_a)
                    cv.create_polygon(wpts,
                                      fill=sec_colors[k % len(sec_colors)],
                                      outline="white", width=1)

            # hex outline (drawn on top of wedges)
            cv.create_polygon(pts, fill="", outline=fc, width=3)
            if is_ref:
                cv.create_polygon(pts, fill="", outline="#c0392b", width=4)

            # cell number
            fs = max(8, 15 - N // 4)
            cv.create_text(px, py, text=str(idx + 1),
                           font=("Segoe UI", fs, "bold"), fill="#1a1a2e")

        # ── reference cell label ─────────────────────────────────────
        rpx = _hex_pixel(ref_q, ref_r, CELL_SIZE)[0] - cx_mean + ox
        rpy = _hex_pixel(ref_q, ref_r, CELL_SIZE)[1] - cy_mean + oy
        cv.create_text(rpx, rpy + CELL_SIZE * 1.3,
                       text="▲ Reference cell",
                       font=("Segoe UI", 9, "bold"), fill="#c0392b")

        # ── co-channel cells (highlight with dashed red border) ──────
        if ij:
            ii, jj = ij
            cochannel = _find_cochannel_cells(ii, jj)
            for cq, cr in cochannel:
                cpx = _hex_pixel(cq, cr, CELL_SIZE)[0] - cx_mean + ox
                cpy = _hex_pixel(cq, cr, CELL_SIZE)[1] - cy_mean + oy
                cpts = _hex_corners_canvas(cpx, cpy, CELL_SIZE * 0.97)
                cv.create_polygon(cpts, fill="", outline="#f39c12",
                                  width=3, dash=(6, 3))
                cv.create_text(cpx, cpy - CELL_SIZE * 0.35,
                               text="co-ch", font=("Segoe UI", 7),
                               fill="#f39c12")

        # ── sector legend ─────────────────────────────────────────────
        if ns > 0:
            lx, ly = W - 130, 10
            cv.create_text(lx + 55, ly + 8, text="Sectors",
                           font=("Segoe UI", 9, "bold"), fill="#333")
            for k in range(ns):
                cv.create_rectangle(lx, ly + 22 + k*20,
                                    lx + 16, ly + 36 + k*20,
                                    fill=sec_colors[k], outline="")
                cv.create_text(lx + 22, ly + 29 + k*20,
                               text=f"Sector {k+1}",
                               font=("Segoe UI", 9), anchor="w", fill="#333")

        # ── info box ──────────────────────────────────────────────────
        info = (f"N = {N}   |   Sectoring: {sectoring}   |"
                f"   n = {n_interferers}   |   C/I = {ci_db:.2f} dB")
        cv.create_rectangle(0, 0, W, 36, fill="#1a1a2e", outline="")
        cv.create_text(W // 2, 18, text=info,
                       font=("Segoe UI", 10, "bold"),
                       fill="white", anchor="center")

        # ── i,j label ─────────────────────────────────────────────────
        if ij:
            cv.create_text(10, H - 14,
                           text=f"i={ij[0]}, j={ij[1]}  →  N = i²+ij+j² = {N}",
                           font=("Segoe UI", 9), anchor="w", fill="#555")


def draw_cluster(N, sectoring, n_interferers, ci_db):
    """Open a new Toplevel window showing the hex cluster."""
    # need a root window — reuse if already exists
    try:
        root = tk._default_root
        if root is None:
            raise RuntimeError
    except Exception:
        root = tk.Tk(); root.withdraw()
    win = ClusterWindow(root, N, sectoring, n_interferers, ci_db)
    win.focus_force()


def draw_cochannel_map(N, sectoring, n_interferers, ci_db):
    """
    Draw the full hex grid highlighting the reference cell and its
    6 co-channel cells, with sector wedges on each — mirroring flattop.py style
    but using the pointy-top layout already used in this file.
    """
    ij = _get_ij(N)
    if ij is None:
        messagebox.showinfo("Info", f"N={N} has no valid (i,j)."); return
    i, j = ij
    cochannel_cells = _find_cochannel_cells(i, j)
    cochannel_set   = set(cochannel_cells)

    # Build waypoints for the i-step arrow (direction 0 only)
    dq0, dr0 = _HEX_DIRS[0]
    wp = (i * dq0, i * dr0)

    size = 1.0
    grid_radius = max(8, int(math.sqrt(N)) + 4)

    sec_pal = {
        "60":  ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6'],
        "120": ['#e74c3c','#3498db','#2ecc71'],
        "180": ['#e74c3c','#3498db'],
        "Omni (no sectoring)": ['#4e8ef7'],
    }
    sec_colors = sec_pal.get(sectoring, ['#4e8ef7'])
    ns = {"60":6, "120":3, "180":2}.get(sectoring, 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect('equal'); ax.axis('off')
    fig.patch.set_facecolor('#f8f9fa')

    # Draw background grid
    for q in range(-grid_radius, grid_radius + 1):
        for r in range(-grid_radius, grid_radius + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) > grid_radius:
                continue
            cx, cy = hex_center(q, r, size)

            if (q, r) == (0, 0):
                fc, ec, lw, ls = '#ffeb3b', '#d32f2f', 4.5, 'solid'
            elif (q, r) in cochannel_set:
                fc, ec, lw, ls = '#ffcc80', '#f57c00', 3.5, 'dashed'
            else:
                fc, ec, lw, ls = '#e8eaf6', '#90a4ae', 0.9, 'solid'

            draw_hex_patch(ax, cx, cy, size * 0.97, fc, ec, lw, zorder=1)
            # redraw with dashed linestyle for co-channel cells
            if ls == 'dashed':
                from matplotlib.patches import Polygon as MPoly
                ax.add_patch(MPoly(
                    hex_corners_pointy(cx, cy, size * 0.97), closed=True,
                    facecolor='none', edgecolor=ec,
                    linewidth=lw, linestyle='dashed', zorder=2))

    # Draw sector wedges on reference + co-channel cells
    def draw_sectors_on(q, r):
        cx, cy = hex_center(q, r, size)
        draw_sector_wedges(ax, cx, cy, size, sectoring, sec_colors, zorder=3)

    draw_sectors_on(0, 0)
    cx0, cy0 = hex_center(0, 0, size)
    ax.text(cx0, cy0, "Ref", ha='center', va='center',
            fontsize=12, fontweight='bold', color='#b71c1c', zorder=6)

    for (cq, cr) in cochannel_cells:
        draw_sectors_on(cq, cr)
        ccx, ccy = hex_center(cq, cr, size)
        ax.text(ccx, ccy - size * 0.30, "co-ch",
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='#e65100', zorder=6)

    # i-step arrow (blue) and j-step arrow (purple) for direction 0
    x0, y0   = hex_center(0, 0, size)
    xw, yw   = hex_center(wp[0], wp[1], size)
    xc, yc   = hex_center(cochannel_cells[0][0], cochannel_cells[0][1], size)

    arr = dict(arrowstyle="-|>", lw=2.0, mutation_scale=18)
    if i > 0:
        ax.annotate("", xy=(xw, yw), xytext=(x0, y0),
                    arrowprops=dict(**arr, color='#1565c0'), zorder=7)
        ax.text((x0+xw)/2 + size*0.15, (y0+yw)/2 + size*0.15,
                f"i={i}", fontsize=11, fontweight='bold', color='#1565c0',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85), zorder=8)
    if j > 0:
        ax.annotate("", xy=(xc, yc), xytext=(xw, yw),
                    arrowprops=dict(**arr, color='#6a1b9a'), zorder=7)
        ax.text((xw+xc)/2 + size*0.15, (yw+yc)/2 + size*0.15,
                f"j={j}", fontsize=11, fontweight='bold', color='#6a1b9a',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85), zorder=8)

    # Sector legend
    if ns > 0:
        handles = [plt.Rectangle((0,0),1,1, fc=sec_colors[k], alpha=0.88)
                   for k in range(ns)]
        ax.legend(handles, [f'Sector {k+1}' for k in range(ns)],
                  loc='upper right', fontsize=10, framealpha=0.92,
                  title='Sectors', title_fontsize=10)

    # Title / info
    ci_str = f"{ci_db:.2f}" if ci_db is not None else "N/A"
    ax.set_title(
        f"Co-channel Map  |  N={N}  ({sectoring})  |  "
        f"i={i}, j={j}  |  n={n_interferers}  |  C/I≈{ci_str} dB",
        fontsize=13, fontweight='bold', pad=18, color='#263238')

    lim = grid_radius * size * 1.8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  16-QAM
# ─────────────────────────────────────────────
def get_16qam_constellation():
    re = np.array([-3, -1, 1, 3])
    im = np.array([-3, -1, 1, 3])
    const = np.array([complex(r, i) for r in re for i in im])
    return const / np.sqrt(10)

def bits_to_symbols(bits):
    constellation = get_16qam_constellation()
    bits = np.asarray(bits, dtype=int)
    pad  = (4 - len(bits) % 4) % 4
    if pad: bits = np.append(bits, np.zeros(pad, dtype=int))
    symbols = []
    for i in range(0, len(bits), 4):
        idx = int(''.join(map(str, bits[i:i+4])), 2)
        symbols.append(constellation[idx])
    return np.array(symbols)

def transmit(symbols, tx_power, noise_power, c_i_linear):
    transmitted        = symbols * np.sqrt(tx_power)
    interference_power = tx_power / c_i_linear if c_i_linear > 0 else 0
    total_noise_var    = noise_power + interference_power
    noise = np.sqrt(total_noise_var / 2) * (
        np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
    )
    return transmitted + noise

def demodulate(received_symbols, tx_power):
    constellation = get_16qam_constellation() * np.sqrt(tx_power)
    received_bits = []
    for sym in received_symbols:
        idx  = np.argmin(np.abs(constellation - sym))
        bit4 = np.array(list(format(idx, '04b')), dtype=int)
        received_bits.extend(bit4)
    return np.array(received_bits, dtype=int)

def compute_BER(original_bits, received_bits):
    n      = min(len(original_bits), len(received_bits))
    errors = int(np.sum(original_bits[:n] != received_bits[:n]))
    return errors / n, errors

def generate_bitStream():
    return np.random.randint(0, 2, 10000)

# ─────────────────────────────────────────────
#  BER PLOTS
# ─────────────────────────────────────────────
def plot_ber_vs_tx_power(c_i_linear, noise_power):
    tx_powers = np.logspace(-3, 1, 25)
    bers = []
    for p in tx_powers:
        bits = generate_bitStream()
        rx   = transmit(bits_to_symbols(bits), p, noise_power, c_i_linear)
        ber, _ = compute_BER(bits, demodulate(rx, p))
        bers.append(max(ber, 1e-6))
    plt.figure(figsize=(8, 5))
    plt.semilogy(tx_powers * 1000, bers, marker='o', color='steelblue')
    plt.xlabel("Transmit Power (mW)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs Transmit Power (fixed noise power)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_ber_vs_noise_power(c_i_linear, tx_power):
    noise_powers = np.logspace(-5, 0, 25)
    bers = []
    for np_ in noise_powers:
        bits = generate_bitStream()
        rx   = transmit(bits_to_symbols(bits), tx_power, np_, c_i_linear)
        ber, _ = compute_BER(bits, demodulate(rx, tx_power))
        bers.append(max(ber, 1e-6))
    plt.figure(figsize=(8, 5))
    plt.semilogy(noise_powers * 1000, bers, marker='s', color='orange')
    plt.xlabel("Noise Power (mW)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"BER vs Noise Power (fixed TX = {tx_power*1000:.0f} mW)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
#  BONUS: IMAGE TRANSMISSION
# ─────────────────────────────────────────────
def transmit_image(image_path, tx_power, noise_power, c_i_linear):
    from PIL import Image
    img       = Image.open(image_path).convert("L").resize((256, 256))
    img_array = np.array(img, dtype=np.uint8)
    bits      = np.unpackbits(img_array.flatten())
    symbols   = bits_to_symbols(bits)
    received  = transmit(symbols, tx_power, noise_power, c_i_linear)
    rx_bits   = demodulate(received, tx_power)
    rx_bits   = rx_bits[:len(bits)].astype(np.uint8)
    rx_array  = np.packbits(rx_bits).reshape(256, 256)
    ber, errors = compute_BER(bits, rx_bits)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_array, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(rx_array,  cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Received Image (BER={ber:.4f})"); axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    return ber, errors


# ─────────────────────────────────────────────
#  CLUSTER FIGURE
# ─────────────────────────────────────────────
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

def cluster_cells(N):
    dirs = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    cells = [(0,0)]
    ring = 1
    while len(cells) < N:
        q,r = ring,0
        for d in range(6):
            for _ in range(ring):
                if len(cells) < N: cells.append((q,r))
                dq,dr = dirs[(d+4)%6]; q+=dq; r+=dr
        ring += 1
    return cells[:N]

def draw_cluster(N, sectoring, n_interferers, ci_db):
    cells = cluster_cells(N)
    size  = 1.0
    cell_colors = ['#4e8ef7','#e8734a','#50c97f','#f5c842','#a87de8',
                   '#e85c8a','#4ec9c9','#f97f51','#7bb8f5','#c9e86a',
                   '#f5a623','#8e44ad','#2ecc71','#e74c3c','#1abc9c']
    sec_pal = {"60":  ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6'],
               "120": ['#e74c3c','#3498db','#2ecc71'],
               "180": ['#e74c3c','#3498db'],
               "Omni (no sectoring)": ['#4e8ef7']}
    sec_colors = sec_pal.get(sectoring, ['#4e8ef7'])

    centers  = [hex_center(q,r,size) for q,r in cells]
    cx_mean  = np.mean([c[0] for c in centers])
    cy_mean  = np.mean([c[1] for c in centers])

    fig, ax = plt.subplots(figsize=(11,10))
    ax.set_aspect('equal'); ax.axis('off')
    fig.patch.set_facecolor('#f8f9ff'); ax.set_facecolor('#f8f9ff')

    # neighbor ring
    seen = set(cells); nb_ring = []
    for q,r in cells:
        for dq,dr in [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]:
            nb=(q+dq,r+dr)
            if nb not in seen: nb_ring.append(nb); seen.add(nb)
    for q,r in nb_ring:
        cx=hex_center(q,r,size)[0]-cx_mean; cy=hex_center(q,r,size)[1]-cy_mean
        draw_hex_patch(ax,cx,cy,size*0.97,'#dde3f5','#b0bce8',0.8,0.6,0)

    # cluster cells
    for idx,(q,r) in enumerate(cells):
        cx=hex_center(q,r,size)[0]-cx_mean; cy=hex_center(q,r,size)[1]-cy_mean
        fc=cell_colors[idx%len(cell_colors)]
        draw_hex_patch(ax,cx,cy,size*0.97,fc,'white',2.5,0.18,1)
        draw_sector_wedges(ax,cx,cy,size,sectoring,sec_colors,2)
        draw_hex_patch(ax,cx,cy,size*0.97,'none',fc,2.5,1.0,3)
        fs=max(8,15-N//3)
        ax.text(cx,cy,str(idx+1),ha='center',va='center',
                fontsize=fs,fontweight='bold',color='#1a1a2e',zorder=6)

    # reference cell
    cx0=hex_center(0,0,size)[0]-cx_mean; cy0=hex_center(0,0,size)[1]-cy_mean
    draw_hex_patch(ax,cx0,cy0,size*0.97,'none','#c0392b',4,1.0,7)
    ax.annotate('Reference\ncell', xy=(cx0,cy0-size*0.95),
                xytext=(cx0-size*2.0,cy0-size*2.4),
                fontsize=9,color='#c0392b',fontweight='bold',
                arrowprops=dict(arrowstyle='->',color='#c0392b',lw=1.5),zorder=8)

    # sector legend
    ns = {"60":6,"120":3,"180":2}.get(sectoring,0)
    if ns > 0:
        handles=[plt.Rectangle((0,0),1,1,fc=sec_colors[k],alpha=0.88) for k in range(ns)]
        ax.legend(handles,[f'Sector {k+1}' for k in range(ns)],
                  loc='lower right',fontsize=10,framealpha=0.92,
                  title='Sectors',title_fontsize=10,edgecolor='#ccc')

    # info box
    props=dict(boxstyle='round,pad=0.7',facecolor='#fffde7',edgecolor='#f0c040',alpha=0.97)
    ax.text(0.02,0.98,
            f"Cluster size  N = {N}\nSectoring      = {sectoring}\n"
            f"Interferers  n = {n_interferers}\nC/I achieved  = {ci_db:.2f} dB",
            transform=ax.transAxes,fontsize=10,va='top',ha='left',
            bbox=props,family='monospace',zorder=10)

    ax.set_title(f'Cellular Cluster — N={N} cells,  {sectoring} sectoring',
                 fontsize=14,fontweight='bold',pad=16,color='#1a1a2e')

    all_pts = cells+nb_ring
    all_x=[hex_center(q,r,size)[0]-cx_mean for q,r in all_pts]
    all_y=[hex_center(q,r,size)[1]-cy_mean for q,r in all_pts]
    pad=size*1.5
    ax.set_xlim(min(all_x)-pad,max(all_x)+pad)
    ax.set_ylim(min(all_y)-pad,max(all_y)+pad)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        root.title("NETW601 — Cellular Network Planner + 16-QAM Simulator")
        root.resizable(True, True)

        # ── Style ────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel",     font=("Segoe UI", 10))
        style.configure("TButton",    font=("Segoe UI", 10), padding=6)
        style.configure("TEntry",     font=("Segoe UI", 10), padding=4)
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Title.TLabel",  font=("Segoe UI", 14, "bold"))
        style.configure("Green.TLabel",  font=("Segoe UI", 10), foreground="#1a7a1a")
        style.configure("Red.TLabel",    font=("Segoe UI", 10), foreground="#cc0000")

        # ── Main layout ──────────────────────────
        main = ttk.Frame(root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="NETW601 — Cellular Network Planner + 16-QAM Simulator",
                  style="Title.TLabel").grid(row=0, column=0, columnspan=4,
                  pady=(0, 16), sticky="w")

        # ── Input panel ──────────────────────────
        inp = ttk.LabelFrame(main, text="Inputs", padding=12)
        inp.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        fields = [
            ("Length (m)",                  "length"),
            ("Width (m)",                   "width"),
            ("User density (users/km²)",    "density"),
            ("Min C/I (dB)",                "ci"),
            ("Session time (s)",            "session"),
            ("Requests per second",         "rps"),
            ("Trunk bandwidth (Hz)",        "trunk"),
            ("Total bandwidth (Hz)",        "total"),
            ("Blocking probability",        "blocking"),
        ]
        self.entries = {}
        self.err_labels = {}
        defaults = ["2000","2000","200","18","5","0.01","200000","5000000","0.02"]

        for row, ((label, key), default) in enumerate(zip(fields, defaults)):
            ttk.Label(inp, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0,8))
            e = ttk.Entry(inp, width=18)
            e.insert(0, default)
            e.grid(row=row, column=1, sticky="ew", pady=3)
            e.bind("<Return>", lambda ev: self.run())
            self.entries[key] = e
            lbl = ttk.Label(inp, text="", style="Red.TLabel")
            lbl.grid(row=row, column=2, sticky="w", padx=(6,0))
            self.err_labels[key] = lbl

        inp.columnconfigure(1, weight=1)

        # ── Buttons ──────────────────────────────
        btn_frame = ttk.Frame(inp)
        btn_frame.grid(row=len(fields), column=0, columnspan=3, pady=(12, 0), sticky="ew")

        ttk.Button(btn_frame, text="Run Simulation",
                   command=self.run).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(btn_frame, text="Plot BER vs TX Power",
                   command=self.plot_tx).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(btn_frame, text="Plot BER vs Noise",
                   command=self.plot_noise).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(btn_frame, text="Show Cluster",
                   command=self.show_cluster).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(inp, text="Transmit Image (Bonus)",
                   command=self.run_image).grid(row=len(fields)+1, column=0,
                   columnspan=3, pady=(6, 0), sticky="ew")
        ttk.Button(btn_frame, text="Show Co-channel Map",
           command=self.show_cochannel_map).pack(side=tk.LEFT, fill=tk.X, expand=True)
        # ── Results panel ────────────────────────
        res = ttk.LabelFrame(main, text="Results", padding=12)
        res.grid(row=1, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        # Metric cards row
        cards = ttk.Frame(res)
        cards.pack(fill=tk.X, pady=(0, 12))

        self.card_vars = {}
        card_defs = [
            ("Min Cells",    "cells"),
            ("Sectoring",    "sect"),
            ("Capacity",     "cap"),
            ("BER",          "ber"),
            ("Ch/Cell",      "chpc"),
            ("Traffic/Cell", "traf"),
        ]
        for col, (title, key) in enumerate(card_defs):
            f = ttk.Frame(cards, relief="groove", padding=8)
            f.grid(row=0, column=col, padx=4, sticky="nsew")
            cards.columnconfigure(col, weight=1)
            ttk.Label(f, text=title, font=("Segoe UI", 9),
                      foreground="#666").pack()
            var = tk.StringVar(value="—")
            ttk.Label(f, textvariable=var,
                      font=("Segoe UI", 13, "bold")).pack()
            self.card_vars[key] = var

        # Sectoring table
        ttk.Label(res, text="Sectoring Analysis  (C/I = 3N/n)",
                  style="Header.TLabel").pack(anchor="w", pady=(0, 4))

        tbl_frame = ttk.Frame(res)
        tbl_frame.pack(fill=tk.X, pady=(0, 10))

        cols = ("Type", "n", "N", "C/I achieved")
        self.tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=4)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        self.tree.tag_configure("chosen", background="#d4edda", foreground="#155724")
        self.tree.pack(fill=tk.X)

        # Bit streams
        ttk.Label(res, text="Transmitted bit stream (first 40 bits)",
                  style="Header.TLabel").pack(anchor="w", pady=(4,2))
        self.tx_text = scrolledtext.ScrolledText(res, height=2, font=("Courier", 10),
                                                  state="disabled", wrap=tk.WORD)
        self.tx_text.pack(fill=tk.X)

        ttk.Label(res, text="Received bit stream (first 40 bits)",
                  style="Header.TLabel").pack(anchor="w", pady=(8,2))
        self.rx_text = scrolledtext.ScrolledText(res, height=2, font=("Courier", 10),
                                                  state="disabled", wrap=tk.WORD)
        self.rx_text.pack(fill=tk.X)

        # Additional info
        ttk.Label(res, text="Additional Info",
                  style="Header.TLabel").pack(anchor="w", pady=(10,2))
        self.info_text = scrolledtext.ScrolledText(res, height=7, font=("Courier", 10),
                                                    state="disabled", wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(main, textvariable=self.status,
                  font=("Segoe UI", 9), foreground="#555").grid(
                  row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Store last computed c_i_linear for plot buttons
        self.last_ci = None

    # ── Helpers ──────────────────────────────────
    def get_val(self, key):
        return float(self.entries[key].get().strip())

    def clear_errors(self):
        for lbl in self.err_labels.values():
            lbl.config(text="")

    def set_err(self, key, msg):
        self.err_labels[key].config(text=msg)

    def set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state="disabled")

    def validate(self):
        self.clear_errors()
        ok = True
        def err(k, m): self.set_err(k, m); nonlocal ok; ok = False

        try: l = self.get_val("length");    (err("length","Must be > 0") if l<=0 else None)
        except: err("length", "Invalid")
        try: w = self.get_val("width");     (err("width","Must be > 0") if w<=0 else None)
        except: err("width", "Invalid")
        try: d = self.get_val("density");   (err("density","Must be > 0") if d<=0 else None)
        except: err("density", "Invalid")
        try: ci = self.get_val("ci");       (err("ci","Must be > 0") if ci<=0 else None)
        except: err("ci", "Invalid")
        try: s = self.get_val("session");   (err("session","Must be > 0") if s<=0 else None)
        except: err("session", "Invalid")
        try: r = self.get_val("rps");       (err("rps","Must be > 0") if r<=0 else None)
        except: err("rps", "Invalid")
        try: tr = self.get_val("trunk");    (err("trunk","Must be > 0") if tr<=0 else None)
        except: err("trunk", "Invalid"); tr = None
        try:
            tot = self.get_val("total")
            if tr and tot <= tr: err("total", f"Must be > trunk ({tr:.0f})")
            elif tot <= 0:       err("total", "Must be > 0")
        except: err("total", "Invalid")
        try:
            bp = self.get_val("blocking")
            if not (0 < bp < 1): err("blocking", "Must be 0 < p < 1")
        except: err("blocking", "Invalid")
        return ok

    # ── Run ──────────────────────────────────────
    def run(self):
        if not self.validate(): return
        self.status.set("Running simulation...")
        self.root.update()
        try:
            length   = self.get_val("length")
            width    = self.get_val("width")
            density  = self.get_val("density")
            ci_dB    = self.get_val("ci")
            session  = self.get_val("session")
            rps      = self.get_val("rps")
            trunk    = self.get_val("trunk")
            total    = self.get_val("total")
            blocking = self.get_val("blocking")

            ci_lin     = 10 ** (ci_dB / 10)
            self.last_ci = ci_lin
            area_km2   = get_area_km2(length, width)
            total_subs = get_total_subscribers(area_km2, density)
            total_ch   = get_total_channels(total, trunk)

            _, all_opts = find_reuse_factor(ci_lin)
            (N, sect, n_int, ci_ach), num_cells, ch_pc, A_pc = find_best_reuse(
                all_opts, total_subs, session, rps, trunk, total, blocking)
            # New - Correct way
            # Old (wrong)
            # cap = shannon_capacity(trunk, ci_lin)        # ← Remove this
            cap = shannon_capacity(trunk, TRANSMIT_POWER, NOISE_POWER, ci_lin)

            bits    = generate_bitStream()
            syms    = bits_to_symbols(bits)
            rx_syms = transmit(syms, TRANSMIT_POWER, NOISE_POWER, ci_lin)
            rx_bits = demodulate(rx_syms, TRANSMIT_POWER)
            ber, errors = compute_BER(bits, rx_bits)

            # Cards
            self.card_vars["cells"].set(f"{num_cells:,}")
            self.card_vars["sect"].set(sect)
            self.card_vars["cap"].set(f"{cap/1e3:.1f} kbps")
            self.card_vars["ber"].set(f"{ber*100:.3f}%")
            self.card_vars["chpc"].set(str(ch_pc))
            self.card_vars["traf"].set(f"{A_pc:.4f} E")

            # Sectoring table
            for row in self.tree.get_children():
                self.tree.delete(row)
            for s_label, s_n, s_N, s_ci in all_opts:
                chosen = (s_N == N and s_label == sect)
                tag    = ("chosen",) if chosen else ()
                mark   = " ✓" if chosen else ""
                self.tree.insert("", tk.END, values=(
                    s_label + mark, s_n, s_N,
                    f"{10*math.log10(s_ci):.2f} dB"), tags=tag)

            # Bit streams
            self.set_text(self.tx_text, str(bits[:40].tolist()))
            self.set_text(self.rx_text, str(rx_bits[:40].tolist()))

            # Additional info
            info = (
                f"  Area                : {area_km2:.4f} km²\n"
                f"  Total subscribers   : {total_subs:.0f}\n"
                f"  Total channels      : {total_ch}\n"
                f"  Reuse factor N      : {N}\n"
                f"  Co-channel interf n : {n_int}\n"
                f"  Achieved C/I        : {10*math.log10(ci_ach):.2f} dB\n"
                f"  Bit errors          : {errors} / {len(bits)}\n"
            )
            self.set_text(self.info_text, info)
            self.status.set(f"Done. BER = {ber*100:.3f}%,  cells = {num_cells:,}")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.set("Error — see message box.")

    def show_cluster(self):
        if self.last_ci is None:
            messagebox.showinfo("Info", "Run the simulation first.")
            return
        try:
            (N, sect, n_int, ci_ach), _ = find_reuse_factor(self.last_ci)
            ci_db = 10 * math.log10(ci_ach)
            self.status.set("Drawing cluster figure...")
            self.root.update()
            draw_cluster(N, sect, n_int, ci_db)
            self.status.set("Cluster figure shown.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_cochannel_map(self):
        if self.last_ci is None:
            messagebox.showinfo("Info", "Run the simulation first.")
            return
        try:
            (N, sect, n_int, ci_ach), _ = find_reuse_factor(self.last_ci)
            ci_db = 10 * math.log10(ci_ach)
            self.status.set("Drawing co-channel map...")
            self.root.update()
            draw_cochannel_map(N, sect, n_int, ci_db)
            self.status.set("Co-channel map shown.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    def plot_tx(self):
        if self.last_ci is None:
            messagebox.showinfo("Info", "Run the simulation first.")
            return
        self.status.set("Generating BER vs TX power plot...")
        self.root.update()
        plot_ber_vs_tx_power(self.last_ci, NOISE_POWER)
        self.status.set("Plot done.")

    def plot_noise(self):
        if self.last_ci is None:
            messagebox.showinfo("Info", "Run the simulation first.")
            return
        self.status.set("Generating BER vs noise power plot...")
        self.root.update()
        plot_ber_vs_noise_power(self.last_ci, TRANSMIT_POWER)
        self.status.set("Plot done.")

    def run_image(self):
        if self.last_ci is None:
            messagebox.showinfo("Info", "Run the simulation first.")
            return
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All", "*.*")]
        )
        if not path: return
        try:
            self.status.set("Transmitting image...")
            self.root.update()
            ber, errors = transmit_image(path, TRANSMIT_POWER, NOISE_POWER, self.last_ci)
            self.status.set(f"Image done. BER = {ber*100:.3f}%,  errors = {errors}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.set("Image error.")

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = App(root)
    root.mainloop()
