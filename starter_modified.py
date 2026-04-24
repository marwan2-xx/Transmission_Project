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
#  VALID N LIST & REUSE PARAMETERS
# ─────────────────────────────────────────────
def get_reuse_parameters(n_array):
    """
    Takes an array of reuse factors (N) and returns a dictionary
    mapping N to its valid (i, j) tuple based on N = i^2 + ij + j^2.
    """
    results = {}
    for N in n_array:
        found = False
        limit = int(N**0.5) + 1
        for i in range(limit + 1):
            for j in range(i + 1):   # ensures i >= j
                if (i**2 + i*j + j**2) == N:
                    results[N] = (i, j)
                    found = True
                    break
            if found:
                break
        if not found:
            results[N] = None
    return results

def is_valid_N(N):
    for i in range(0, N + 1):
        for j in range(0, i + 1):
            val = i*i + i*j + j*j
            if val == N:  return True
            if val > N:   break
    return False

valid_Ns = [N for N in range(3, 20) if is_valid_N(N)]

# ─────────────────────────────────────────────
#  REUSE TABLE  (manually mapped per tutorial)
# ─────────────────────────────────────────────
# Order of sectoring types matches sectoring_labels index-for-index
sectoring_labels = ["no sectoring", "60-degrees", "120-degrees", "180-degrees"]

valid_ns = {
    3:  [6, 2, 3, 2],
    4:  [6, 1, 2, 2],
    7:  [6, 3, 2, 2],
    9:  [6, 3, 2, 2],
    12: [6, 2, 2, 2],
    13: [6, 3, 2, 2],
    16: [6, 3, 2, 2],
    19: [6, 2, 2, 2],
}

def find_min_reuse(ns_table: dict, required_CIR: float) -> dict:
    """
    For each N in ns_table, iterates over its n values (one per sectoring type)
    and picks the first n where 3N/n >= required_CIR.

    Returns
    -------
    { N: (n, sectoring_label) }  for every N that has at least one match.
    """
    results = {}
    for N, ns in ns_table.items():
        for n, label in zip(ns, sectoring_labels):
            if (3 * N) / n >= required_CIR:
                results[N] = (n, label)
                break   # first (least-restrictive) match wins
    return results

def find_reuse_factor(c_i_linear):
    """
    Adapter so the GUI (which expects the old return signature) keeps working.

    Calls find_min_reuse, then picks the smallest accepted N as 'best',
    and returns:
        best_result : (N, sectoring_label, n, achieved_CIR)
        all_opts    : [(sectoring_label, n, N, achieved_CIR), ...]  – one row per accepted N
    """
    matches = find_min_reuse(valid_ns, c_i_linear)
    if not matches:
        raise ValueError("Could not find a valid N for the given C/I requirement.")

    all_opts = [
        (label, n, N, 3 * N / n)
        for N, (n, label) in sorted(matches.items())
    ]

    # Best = smallest N that satisfies the requirement
    best_N, (best_n, best_label) = min(matches.items(), key=lambda x: x[0])
    best_result = (best_N, best_label, best_n, 3 * best_N / best_n)

    return best_result, all_opts

# ─────────────────────────────────────────────
#  CELLULAR PLANNING
# ─────────────────────────────────────────────
def get_total_channels(total_BW, trunk_BW):
    return int(total_BW / trunk_BW)
def get_area_km2(length, width):
    return (length / 1000) * (width / 1000)

def get_total_subscribers(area_km2, subscriber_density):
    return area_km2 * subscriber_density
def erlang_b(A, C):
    if A == 0: return 0.0
    B = 1.0
    for k in range(1, int(C) + 1):
        B = (A * B) / (k + A * B)
    return B

def get_number_of_cells(total_subscribers, user_session_time,
                        requests_per_second, trunk_BW, total_BW,
                        blocking_probability, N):
    traffic_per_user  = requests_per_second * user_session_time
    A_total           = total_subscribers * traffic_per_user
    total_channels    = get_total_channels(total_BW, trunk_BW)
    channels_per_cell = max(1, int(total_channels // N))
    for num_cells in range(1, 10_000_001):
        A_per_cell = A_total / num_cells
        if erlang_b(A_per_cell, channels_per_cell) <= blocking_probability:
            return num_cells, channels_per_cell, A_per_cell
    return 10_000_000, channels_per_cell, A_total / 10_000_000

def shannon_capacity(channel_bandwidth_hz, snr_linear):
    return channel_bandwidth_hz * log2(1 + snr_linear)

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
        defaults = ["2000","2000","200","10","5","0.01","200000","5000000","0.02"]

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
                   command=self.plot_noise).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(inp, text="Transmit Image (Bonus)",
                   command=self.run_image).grid(row=len(fields)+1, column=0,
                   columnspan=3, pady=(6, 0), sticky="ew")

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

            (N, sect, n_int, ci_ach), all_opts = find_reuse_factor(ci_lin)
            num_cells, ch_pc, A_pc = get_number_of_cells(
                total_subs, session, rps, trunk, total, blocking, N)
            cap = shannon_capacity(trunk, ci_lin)

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

