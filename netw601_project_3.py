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

# ─────────────────────────────────────────────
#  CELLULAR PLANNING
# ─────────────────────────────────────────────
def get_area_km2(length, width):
    return (length / 1000) * (width / 1000)

def get_total_subscribers(area_km2, subscriber_density):
    return area_km2 * subscriber_density

def get_total_channels(total_BW, trunk_BW):
    return int(total_BW / trunk_BW)

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

def shannon_capacity(trunk_bw, transmit_power, noise_power, c_i_linear):
    I = transmit_power / c_i_linear if c_i_linear > 0 else 0.0
    sinr = transmit_power / (noise_power + I)
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
def _apply_plot_style(fig, ax):
    """Apply dark, modern style to matplotlib figures."""
    BG   = "#0f1117"
    GRID = "#2a2d3a"
    TEXT = "#e0e4f0"
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.spines[:].set_color(GRID)
    ax.grid(True, which="both", color=GRID, linestyle="--", alpha=0.7)
    fig.tight_layout()

def plot_ber_vs_tx_power(c_i_linear, noise_power):
    tx_powers = np.logspace(-3, 1, 25)
    bers = []
    for p in tx_powers:
        bits = generate_bitStream()
        rx   = transmit(bits_to_symbols(bits), p, noise_power, c_i_linear)
        ber, _ = compute_BER(bits, demodulate(rx, p))
        bers.append(max(ber, 1e-6))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(tx_powers * 1000, bers, marker='o', color='#5b9cf6',
                linewidth=2, markersize=6, markerfacecolor='#a8c8ff')
    ax.set_xlabel("Transmit Power (mW)", fontsize=11)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=11)
    ax.set_title("BER vs Transmit Power  ·  16-QAM", fontsize=13, fontweight='bold')
    _apply_plot_style(fig, ax)
    plt.show()

def plot_ber_vs_noise_power(c_i_linear, tx_power):
    noise_powers = np.logspace(-5, 0, 25)
    bers = []
    for np_ in noise_powers:
        bits = generate_bitStream()
        rx   = transmit(bits_to_symbols(bits), tx_power, np_, c_i_linear)
        ber, _ = compute_BER(bits, demodulate(rx, tx_power))
        bers.append(max(ber, 1e-6))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(noise_powers * 1000, bers, marker='s', color='#f0a050',
                linewidth=2, markersize=6, markerfacecolor='#ffd580')
    ax.set_xlabel("Noise Power (mW)", fontsize=11)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=11)
    ax.set_title(f"BER vs Noise Power  ·  TX = {tx_power*1000:.0f} mW", fontsize=13, fontweight='bold')
    _apply_plot_style(fig, ax)
    plt.show()

# ─────────────────────────────────────────────
#  CO-CHANNEL MAP
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
    grid_radius = max(8, int(math.sqrt(N)) + 4)

    sec_pal = {
        "60":  ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6'],
        "120": ['#e74c3c','#3498db','#2ecc71'],
        "180": ['#e74c3c','#3498db'],
        "Omni (no sectoring)": ['#4e8ef7'],
    }
    sec_colors = sec_pal.get(sectoring, ['#4e8ef7'])
    ns = {"60":6, "120":3, "180":2}.get(sectoring, 0)

    BG = "#0f1117"
    fig, ax = plt.subplots(figsize=(14, 12))
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
        fontsize=13, fontweight='bold', pad=18, color='#e0e4f0')

    lim = grid_radius * size * 1.8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
#  IMAGE TRANSMISSION
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

    BG = "#0f1117"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(BG)
    for a in axes: a.set_facecolor(BG)
    axes[0].imshow(img_array, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Image", color='#e0e4f0', fontsize=12); axes[0].axis("off")
    axes[1].imshow(rx_array,  cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Received  (BER={ber:.4f})", color='#e0e4f0', fontsize=12); axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    return ber, errors


# ─────────────────────────────────────────────
#  PALETTE & THEME CONSTANTS
# ─────────────────────────────────────────────
C_BG        = "#0f1117"   # main background
C_SURFACE   = "#16192a"   # panel background
C_SURFACE2  = "#1e2235"   # card / entry background
C_BORDER    = "#2e3250"   # subtle border
C_ACCENT    = "#5b9cf6"   # primary blue accent
C_ACCENT2   = "#7c6af7"   # purple accent
C_SUCCESS   = "#4caf82"   # green
C_WARN      = "#f0a050"   # amber
C_TEXT      = "#e0e4f0"   # primary text
C_MUTED     = "#8890b0"   # secondary text
C_ERROR     = "#f06090"   # error red


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        root.title("NETW601 — Cellular Network Planner + 16-QAM")
        root.configure(bg=C_BG)
        root.resizable(True, True)
        root.minsize(960, 640)

        self._configure_styles()
        self._build_ui()
        self.last_ci = None

    # ── ttk style configuration ──────────────────────────────────────
    def _configure_styles(self):
        s = ttk.Style()
        s.theme_use("clam")

        s.configure(".", background=C_BG, foreground=C_TEXT,
                    font=("Consolas", 10), borderwidth=0)

        s.configure("TFrame",      background=C_BG)
        s.configure("Panel.TFrame", background=C_SURFACE)

        s.configure("TLabel",      background=C_BG, foreground=C_TEXT,
                    font=("Consolas", 10))
        s.configure("Muted.TLabel", background=C_BG, foreground=C_MUTED,
                    font=("Consolas", 9))
        s.configure("Error.TLabel", background=C_BG, foreground=C_ERROR,
                    font=("Consolas", 9))
        s.configure("Title.TLabel", background=C_BG, foreground=C_TEXT,
                    font=("Consolas", 15, "bold"))
        s.configure("Sub.TLabel",  background=C_BG, foreground=C_MUTED,
                    font=("Consolas", 9))

        # Buttons
        for name, bg, fg, abg in [
            ("Accent.TButton",  C_ACCENT,  C_BG,    "#3a7ce0"),
            ("Ghost.TButton",   C_SURFACE2, C_TEXT,  "#252840"),
            ("Warn.TButton",    C_WARN,    C_BG,    "#d08030"),
            ("Danger.TButton",  "#e05070", C_BG,    "#c03050"),
        ]:
            s.configure(name, background=bg, foreground=fg,
                        font=("Consolas", 10, "bold"), relief="flat",
                        padding=(12, 7), borderwidth=0)
            s.map(name, background=[("active", abg)])

        # Entry
        s.configure("TEntry", fieldbackground=C_SURFACE2, foreground=C_TEXT,
                    insertcolor=C_TEXT, borderwidth=1,
                    font=("Consolas", 10))
        s.map("TEntry", bordercolor=[("focus", C_ACCENT), ("!focus", C_BORDER)])

        # LabelFrame
        s.configure("TLabelframe", background=C_SURFACE, bordercolor=C_BORDER,
                    relief="flat", padding=16)
        s.configure("TLabelframe.Label", background=C_SURFACE, foreground=C_ACCENT,
                    font=("Consolas", 10, "bold"))

        # Treeview
        s.configure("Treeview", background=C_SURFACE2, foreground=C_TEXT,
                    fieldbackground=C_SURFACE2, font=("Consolas", 10),
                    rowheight=26, borderwidth=0)
        s.configure("Treeview.Heading", background=C_SURFACE, foreground=C_ACCENT,
                    font=("Consolas", 10, "bold"), relief="flat")
        s.map("Treeview", background=[("selected", C_ACCENT2)])
        s.configure("Scrollbar.Vertical.TScrollbar", background=C_SURFACE2,
                    troughcolor=C_SURFACE)

    # ── UI builder ────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # ── Header bar ───────────────────────────────────────────────
        hdr = tk.Frame(root, bg=C_SURFACE, height=56)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="◈  NETW601", bg=C_SURFACE, fg=C_ACCENT,
                 font=("Consolas", 15, "bold")).pack(side=tk.LEFT, padx=20, pady=12)
        tk.Label(hdr, text="Cellular Network Planner + 16-QAM Simulator",
                 bg=C_SURFACE, fg=C_MUTED, font=("Consolas", 10)).pack(side=tk.LEFT)

        # status pill (right side)
        self.status_var = tk.StringVar(value="● Ready")
        tk.Label(hdr, textvariable=self.status_var, bg=C_SURFACE, fg=C_SUCCESS,
                 font=("Consolas", 9)).pack(side=tk.RIGHT, padx=20)

        # ── Body (two columns) ────────────────────────────────────────
        body = tk.Frame(root, bg=C_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ── LEFT: Input Panel ─────────────────────────────────────────
        left = ttk.LabelFrame(body, text="  Parameters", style="TLabelframe")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(1, weight=1)

        fields = [
            ("Length (m)",              "length",   "5000"),
            ("Width (m)",               "width",    "5000"),
            ("User density (users/km²)","density",  "1000"),
            ("Min C/I (dB)",            "ci",       "10"),
            ("Session time (s)",        "session",  "0.0069"),
            ("Requests / second",       "rps",      "1"),
            ("Trunk bandwidth (Hz)",    "trunk",    "200"),
            ("Total bandwidth (Hz)",    "total",    "20000"),
            ("Blocking probability",    "blocking", "0.02"),
        ]
        self.entries    = {}
        self.err_labels = {}

        for row, (label, key, default) in enumerate(fields):
            # divider line every 3 rows
            if row in (3, 6):
                sep = tk.Frame(left, bg=C_BORDER, height=1)
                sep.grid(row=row*2-1, column=0, columnspan=3,
                         sticky="ew", pady=(4, 0))

            tk.Label(left, text=label, bg=C_SURFACE, fg=C_MUTED,
                     font=("Consolas", 9)).grid(
                     row=row*2, column=0, sticky="w", padx=(4,10), pady=(6,0))

            e = ttk.Entry(left, width=16)
            e.insert(0, default)
            e.grid(row=row*2, column=1, sticky="ew", padx=(0,4), pady=(6,0))
            e.bind("<Return>", lambda ev: self.run())
            self.entries[key] = e

            err = tk.Label(left, text="", bg=C_SURFACE, fg=C_ERROR,
                           font=("Consolas", 8))
            err.grid(row=row*2, column=2, sticky="w")
            self.err_labels[key] = err

        # ── Action buttons ────────────────────────────────────────────
        btn_area = tk.Frame(left, bg=C_SURFACE)
        btn_area.grid(row=99, column=0, columnspan=3, sticky="ew", pady=(18, 4))
        btn_area.columnconfigure((0,1), weight=1)

        self._btn(btn_area, "▶  Run Simulation",  "Accent.TButton", self.run,  0, 0, cspan=2)
        self._btn(btn_area, "≋  BER vs TX Power",  "Ghost.TButton", self.plot_tx,    1, 0)
        self._btn(btn_area, "≋  BER vs Noise",     "Ghost.TButton", self.plot_noise, 1, 1)
        self._btn(btn_area, "◎  Co-channel Map",   "Ghost.TButton", self.show_cochannel_map, 2, 0)
        self._btn(btn_area, "⊡  Image TX (Bonus)", "Warn.TButton",  self.run_image,  2, 1)

        # ── RIGHT: Results Panel ──────────────────────────────────────
        right = ttk.LabelFrame(body, text="  Results", style="TLabelframe")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        # ── Metric cards ──────────────────────────────────────────────
        cards_wrap = tk.Frame(right, bg=C_SURFACE)
        cards_wrap.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        card_defs = [
            ("Min Cells",    "cells",  C_ACCENT),
            ("Sectoring",    "sect",   C_ACCENT2),
            ("Capacity",     "cap",    C_SUCCESS),
            ("BER",          "ber",    C_WARN),
            ("Ch / Cell",    "chpc",   C_ACCENT),
            ("Traffic/Cell", "traf",   C_MUTED),
        ]
        self.card_vars = {}
        for col, (title, key, color) in enumerate(card_defs):
            cards_wrap.columnconfigure(col, weight=1)
            card = tk.Frame(cards_wrap, bg=C_SURFACE2,
                            highlightbackground=C_BORDER, highlightthickness=1)
            card.grid(row=0, column=col, padx=(0 if col==0 else 6, 0), sticky="nsew")
            tk.Label(card, text=title, bg=C_SURFACE2, fg=C_MUTED,
                     font=("Consolas", 8)).pack(pady=(8, 2))
            var = tk.StringVar(value="—")
            tk.Label(card, textvariable=var, bg=C_SURFACE2, fg=color,
                     font=("Consolas", 13, "bold")).pack(pady=(0, 8))
            self.card_vars[key] = var

        # ── Sectoring table ───────────────────────────────────────────
        tk.Label(right, text="Sectoring Analysis  —  C/I = 3N/n",
                 bg=C_SURFACE, fg=C_TEXT, font=("Consolas", 10, "bold")).grid(
                 row=1, column=0, sticky="w", pady=(0, 4))

        cols = ("Type", "n", "N", "C/I achieved")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=4)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        self.tree.tag_configure("chosen",
                                background="#1e3a2e", foreground=C_SUCCESS)
        self.tree.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        # ── Bit streams + Info ────────────────────────────────────────
        detail = tk.Frame(right, bg=C_SURFACE)
        detail.grid(row=3, column=0, sticky="nsew")
        detail.columnconfigure(0, weight=1)
        detail.rowconfigure(2, weight=1)

        tk.Label(detail, text="TX bit stream  (first 40 bits)",
                 bg=C_SURFACE, fg=C_MUTED, font=("Consolas", 9, "bold")).grid(
                 row=0, column=0, sticky="w")
        self.tx_text = self._make_text(detail, 2)
        self.tx_text.grid(row=1, column=0, sticky="ew", pady=(2, 8))

        tk.Label(detail, text="RX bit stream  (first 40 bits)",
                 bg=C_SURFACE, fg=C_MUTED, font=("Consolas", 9, "bold")).grid(
                 row=2, column=0, sticky="w")
        self.rx_text = self._make_text(detail, 2)
        self.rx_text.grid(row=3, column=0, sticky="ew", pady=(2, 8))

        tk.Label(detail, text="Simulation Summary",
                 bg=C_SURFACE, fg=C_MUTED, font=("Consolas", 9, "bold")).grid(
                 row=4, column=0, sticky="w")
        self.info_text = self._make_text(detail, 7)
        self.info_text.grid(row=5, column=0, sticky="nsew", pady=(2, 0))
        detail.rowconfigure(5, weight=1)

    # ── Widget helpers ────────────────────────────────────────────────
    def _btn(self, parent, text, style, cmd, row, col, cspan=1):
        b = ttk.Button(parent, text=text, style=style, command=cmd)
        b.grid(row=row, column=col, columnspan=cspan,
               sticky="ew", padx=(0 if col==0 else 6, 0), pady=(6, 0))
        return b

    def _make_text(self, parent, height):
        t = tk.Text(parent, height=height, bg=C_SURFACE2, fg=C_TEXT,
                    insertbackground=C_TEXT, relief="flat",
                    font=("Consolas", 9), wrap=tk.WORD,
                    highlightbackground=C_BORDER, highlightthickness=1,
                    padx=8, pady=6, state="disabled")
        return t

    def _set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state="disabled")

    def _set_status(self, msg, color=C_MUTED):
        self.status_var.set(f"● {msg}")

    # ── Validation ────────────────────────────────────────────────────
    def _clear_errors(self):
        for lbl in self.err_labels.values():
            lbl.config(text="")

    def _err(self, key, msg):
        self.err_labels[key].config(text=f"⚠ {msg}")

    def _val(self, key):
        return float(self.entries[key].get().strip())

    def validate(self):
        self._clear_errors()
        ok = True
        def flag(k, m):
            nonlocal ok; self._err(k, m); ok = False

        checks = [
            ("length",   lambda v: v > 0, "Must be > 0"),
            ("width",    lambda v: v > 0, "Must be > 0"),
            ("density",  lambda v: v > 0, "Must be > 0"),
            ("ci",       lambda v: v > 0, "Must be > 0"),
            ("session",  lambda v: v > 0, "Must be > 0"),
            ("rps",      lambda v: v > 0, "Must be > 0"),
            ("trunk",    lambda v: v > 0, "Must be > 0"),
            ("blocking", lambda v: 0 < v < 1, "0 < p < 1"),
        ]
        vals = {}
        for key, test, msg in checks:
            try:
                v = self._val(key)
                if not test(v): flag(key, msg)
                else: vals[key] = v
            except: flag(key, "Invalid number")

        try:
            tot = self._val("total")
            tr  = vals.get("trunk", 1)
            if tot <= tr: flag("total", f"Must exceed trunk ({tr:.0f} Hz)")
            elif tot <= 0: flag("total", "Must be > 0")
        except: flag("total", "Invalid number")

        return ok

    # ── Main simulation ───────────────────────────────────────────────
    def run(self):
        if not self.validate(): return
        self._set_status("Running…")
        self.root.update()
        try:
            length   = self._val("length")
            width    = self._val("width")
            density  = self._val("density")
            ci_dB    = self._val("ci")
            session  = self._val("session")
            rps      = self._val("rps")
            trunk    = self._val("trunk")
            total    = self._val("total")
            blocking = self._val("blocking")

            ci_lin      = 10 ** (ci_dB / 10)
            self.last_ci = ci_lin
            area_km2    = get_area_km2(length, width)
            total_subs  = get_total_subscribers(area_km2, density)
            total_ch    = get_total_channels(total, trunk)

            _, all_opts = find_reuse_factor(ci_lin)
            (N, sect, n_int, ci_ach), num_cells, ch_pc, A_pc = find_best_reuse(
                all_opts, total_subs, session, rps, trunk, total, blocking)

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

            # Table
            for row in self.tree.get_children():
                self.tree.delete(row)
            for s_label, s_n, s_N, s_ci in all_opts:
                chosen = (s_N == N and s_label == sect)
                tag    = ("chosen",) if chosen else ()
                mark   = "  ✓" if chosen else ""
                self.tree.insert("", tk.END, values=(
                    s_label + mark, s_n, s_N,
                    f"{10*math.log10(s_ci):.2f} dB"), tags=tag)

            # Bit streams
            self._set_text(self.tx_text, " ".join(map(str, bits[:40].tolist())))
            self._set_text(self.rx_text, " ".join(map(str, rx_bits[:40].tolist())))

            # Info
            info = (
                f"  Area                :  {area_km2:.4f} km²\n"
                f"  Total subscribers   :  {total_subs:.0f}\n"
                f"  Total channels      :  {total_ch}\n"
                f"  Reuse factor  N     :  {N}\n"
                f"  Co-channel int. n   :  {n_int}\n"
                f"  Achieved C/I        :  {10*math.log10(ci_ach):.2f} dB\n"
                f"  Bit errors          :  {errors} / {len(bits)}\n"
            )
            self._set_text(self.info_text, info)
            self._set_status(f"Done  ·  BER {ber*100:.3f}%  ·  {num_cells:,} cells", C_SUCCESS)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._set_status("Error — see dialog", C_ERROR)

    # ── Handlers ─────────────────────────────────────────────────────
    def show_cochannel_map(self):
        if not self._need_run(): return
        try:
            (N, sect, n_int, ci_ach), _ = find_reuse_factor(self.last_ci)
            ci_db = 10 * math.log10(ci_ach)
            self._set_status("Drawing co-channel map…")
            self.root.update()
            draw_cochannel_map(N, sect, n_int, ci_db)
            self._set_status("Co-channel map shown.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_tx(self):
        if not self._need_run(): return
        self._set_status("Generating BER vs TX power…")
        self.root.update()
        plot_ber_vs_tx_power(self.last_ci, NOISE_POWER)
        self._set_status("Plot done.")

    def plot_noise(self):
        if not self._need_run(): return
        self._set_status("Generating BER vs noise…")
        self.root.update()
        plot_ber_vs_noise_power(self.last_ci, TRANSMIT_POWER)
        self._set_status("Plot done.")

    def run_image(self):
        if not self._need_run(): return
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All", "*.*")])
        if not path: return
        try:
            self._set_status("Transmitting image…")
            self.root.update()
            ber, errors = transmit_image(path, TRANSMIT_POWER, NOISE_POWER, self.last_ci)
            self._set_status(f"Image done  ·  BER {ber*100:.3f}%  ·  {errors} errors", C_SUCCESS)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._set_status("Image error.", C_ERROR)

    def _need_run(self):
        if self.last_ci is None:
            messagebox.showinfo("Run first", "Please run the simulation first.")
            return False
        return True


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = App(root)
    root.mainloop()