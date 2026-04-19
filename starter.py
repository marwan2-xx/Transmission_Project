import numpy as np
import matplotlib.pyplot as plt
import math
from math import log2, ceil

# ─────────────────────────────────────────────
#  CONSTANTS (fixed by project spec)
# ─────────────────────────────────────────────
TRANSMIT_POWER = 0.1      # 100 mW
NOISE_POWER    = 0.0001   # 0.1 mW

# ─────────────────────────────────────────────
#  VALID N LIST
# ─────────────────────────────────────────────
def is_valid_N(N):
    """Check if N can be written as i² + i*j + j²  (i >= j >= 0)."""
    for i in range(0, N + 1):
        for j in range(0, i + 1):
            val = i*i + i*j + j*j
            if val == N:
                return True
            if val > N:
                break
    return False

VALID_N_LIST = [N for N in range(1, 500) if is_valid_N(N)]

# ─────────────────────────────────────────────
#  INPUT VALIDATION
# ─────────────────────────────────────────────
def validate_inputs(length, width, subscriber_density, c_i_dB,
                    user_session_time, requests_per_second,
                    trunk_BW, total_BW, blocking_prob):
    errors = []

    if length <= 0:
        errors.append("Area length must be positive.")
    if width <= 0:
        errors.append("Area width must be positive.")
    if subscriber_density <= 0:
        errors.append("User density must be positive.")
    if c_i_dB <= 0:
        errors.append("Minimum C/I must be a positive dB value.")
    if user_session_time <= 0:
        errors.append("Session time must be positive.")
    if requests_per_second <= 0:
        errors.append("Requests per second must be positive.")
    if trunk_BW <= 0:
        errors.append("Trunk bandwidth must be positive.")
    if total_BW <= 0:
        errors.append("Total bandwidth must be positive.")
    if trunk_BW >= total_BW:
        errors.append("Trunk bandwidth must be smaller than total bandwidth.")
    if not (0 < blocking_prob < 1):
        errors.append("Blocking probability must be between 0 and 1 (e.g. 0.02 for 2%).")

    area_km2   = (length / 1000) * (width / 1000)
    total_subs = area_km2 * subscriber_density
    A_total    = total_subs * requests_per_second * user_session_time
    total_ch   = int(total_BW / trunk_BW)
    if A_total > total_ch * 10000:
        errors.append(
            f"Offered traffic ({A_total:.1f} Erlangs) is extremely high for "
            f"{total_ch} total channels. Reduce session time or request rate."
        )

    if errors:
        print("\n[INPUT ERRORS] Please fix the following:")
        for idx, e in enumerate(errors, 1):
            print(f"  {idx}. {e}")
        raise ValueError(f"{len(errors)} invalid input(s). See details above.")

# ─────────────────────────────────────────────
#  CELLULAR PLANNING
# ─────────────────────────────────────────────
def get_area_km2(length, width):
    return (length / 1000) * (width / 1000)

def get_total_subscribers(area_km2, subscriber_density):
    return area_km2 * subscriber_density

def get_total_channels(total_BW, trunk_BW):
    return int(total_BW / trunk_BW)

def find_reuse_factor(c_i_linear):
    """
    Select sectoring type and reuse factor N:

    Decision logic (industry-standard approach):
      1. Try Omni (n=6, no sectoring). If it satisfies C/I with N <= 7, use it.
      2. Otherwise try 120-degree sectoring (n=2). If N <= 9, use it.
      3. Otherwise use 60-degree sectoring (n=1).

    This reflects real practice: use the simplest antenna scheme that works,
    and only add sectoring when omni requires an impractically large N.
    """
    # Step 1: try omni
    for N in VALID_N_LIST:
        if (3 * N / 6) >= c_i_linear:
            if N <= 7:
                return N, 180, 6, 3 * N / 6   # omni is sufficient
            break

    # Step 2: try 120-degree sectoring
    for N in VALID_N_LIST:
        if (3 * N / 2) >= c_i_linear:
            if N <= 9:
                return N, 120, 2, 3 * N / 2
            break

    # Step 3: fall back to 60-degree sectoring
    for N in VALID_N_LIST:
        if (3 * N / 1) >= c_i_linear:
            return N, 60, 1, 3 * N / 1

    raise ValueError("Could not find a valid N for the given C/I requirement.")

def erlang_b(A, C):
    """Erlang B formula using Jagerman iterative recursion (overflow-safe)."""
    if A == 0:
        return 0.0
    B = 1.0
    for k in range(1, int(C) + 1):
        B = (A * B) / (k + A * B)
    return B

def get_number_of_cells(total_subscribers, user_session_time,
                        requests_per_second, trunk_BW, total_BW,
                        blocking_probability, N):
    """
    Find minimum number of cells so that Erlang-B blocking on
    (channels_per_cell) channels <= blocking_probability.
    """
    traffic_per_user  = requests_per_second * user_session_time
    A_total           = total_subscribers * traffic_per_user
    total_channels    = get_total_channels(total_BW, trunk_BW)
    channels_per_cell = max(1, int(total_channels // N))

    MAX_CELLS = 10_000_000
    for num_cells in range(1, MAX_CELLS + 1):
        A_per_cell = A_total / num_cells
        if erlang_b(A_per_cell, channels_per_cell) <= blocking_probability:
            return num_cells, channels_per_cell, A_per_cell

    print(f"\n[WARNING] Could not meet blocking probability with {MAX_CELLS:,} cells.")
    print(f"  Total traffic = {A_total:.1f} Erlangs, channels/cell = {channels_per_cell}")
    print(f"  Consider increasing total_BW or reducing session time / request rate.")
    return MAX_CELLS, channels_per_cell, A_total / MAX_CELLS

def shannon_capacity(channel_bandwidth_hz, snr_linear):
    """Shannon capacity in bits/s."""
    return channel_bandwidth_hz * log2(1 + snr_linear)

# ─────────────────────────────────────────────
#  16-QAM MODULATION / DEMODULATION
# ─────────────────────────────────────────────
def get_16qam_constellation():
    """Standard 16-QAM constellation normalised to unit average power."""
    re = np.array([-3, -1, 1, 3])
    im = np.array([-3, -1, 1, 3])
    const = np.array([complex(r, i) for r in re for i in im])
    return const / np.sqrt(10)

def bits_to_symbols(bits):
    """Map groups of 4 bits to 16-QAM symbols."""
    constellation = get_16qam_constellation()
    bits = np.asarray(bits, dtype=int)
    pad  = (4 - len(bits) % 4) % 4
    if pad:
        bits = np.append(bits, np.zeros(pad, dtype=int))
    symbols = []
    for i in range(0, len(bits), 4):
        idx = int(''.join(map(str, bits[i:i+4])), 2)
        symbols.append(constellation[idx])
    return np.array(symbols)

def transmit(symbols, tx_power, noise_power, c_i_linear):
    """Transmit through AWGN + interference channel."""
    transmitted        = symbols * np.sqrt(tx_power)
    interference_power = tx_power / c_i_linear if c_i_linear > 0 else 0
    total_noise_var    = noise_power + interference_power
    noise = np.sqrt(total_noise_var / 2) * (
        np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
    )
    return transmitted + noise

def demodulate(received_symbols, tx_power):
    """Nearest-neighbour hard decision demodulation."""
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
def plot_ber_vs_tx_power(c_i_linear, noise_power, num_bits=10000):
    """BER vs Transmit Power (fixed noise power)."""
    tx_powers = np.logspace(-3, 1, 25)
    bers      = []
    for p in tx_powers:
        bits     = generate_bitStream()
        symbols  = bits_to_symbols(bits)
        received = transmit(symbols, p, noise_power, c_i_linear)
        rx_bits  = demodulate(received, p)
        ber, _   = compute_BER(bits, rx_bits)
        bers.append(max(ber, 1e-6))

    plt.figure(figsize=(8, 5))
    plt.semilogy(tx_powers * 1000, bers, marker='o', color='steelblue')
    plt.xlabel("Transmit Power (mW)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs Transmit Power (fixed noise power)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_ber_vs_noise_power(c_i_linear, tx_power, num_bits=10000):
    """BER vs Noise Power (fixed transmit power)."""
    noise_powers = np.logspace(-5, 0, 25)
    bers         = []
    for np_ in noise_powers:
        bits     = generate_bitStream()
        symbols  = bits_to_symbols(bits)
        received = transmit(symbols, tx_power, np_, c_i_linear)
        rx_bits  = demodulate(received, tx_power)
        ber, _   = compute_BER(bits, rx_bits)
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
    """Transmit a 256x256 image through the 16-QAM channel."""
    from PIL import Image

    img       = Image.open(image_path).convert("L").resize((256, 256))
    img_array = np.array(img, dtype=np.uint8)
    bits      = np.unpackbits(img_array.flatten())

    symbols  = bits_to_symbols(bits)
    received = transmit(symbols, tx_power, noise_power, c_i_linear)
    rx_bits  = demodulate(received, tx_power)

    rx_bits  = rx_bits[:len(bits)].astype(np.uint8)
    rx_array = np.packbits(rx_bits).reshape(256, 256)

    ber, errors = compute_BER(bits, rx_bits)
    print(f"\n[Image BER] {ber:.4f}  ({errors} bit errors out of {len(bits)})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_array, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(rx_array, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Received Image (BER={ber:.4f})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def get_float_input(prompt, condition=lambda x: x > 0, error_msg="Value must be positive."):
    """Keep asking until the user enters a valid non-empty float that passes condition."""
    while True:
        try:
            raw = input(prompt)
            if raw.strip() == "":
                print("  [ERROR] Input cannot be empty. Please enter a number.")
                continue
            value = float(raw)
            if not condition(value):
                print(f"  [ERROR] {error_msg}")
            else:
                return value
        except ValueError:
            print("  [ERROR] Invalid input. Please enter a number.")


def main():
    print("=" * 60)
    print("  NETW 601 - Cellular Network Planner + 16-QAM Simulator")
    print("=" * 60)

    length             = get_float_input("\nEnter the length (meters)                        : ")
    width              = get_float_input("Enter the width (meters)                          : ")
    subscriber_density = get_float_input("Enter users density (users/km2)                   : ")
    c_i_dB             = get_float_input("Enter the min signal to interference ratio (dB)   : ")
    user_session_time  = get_float_input("Enter average session time per user (seconds)     : ")
    requests_per_second= get_float_input("Enter the average session requests per second     : ")
    trunk_BW           = get_float_input("Enter the trunk bandwidth (Hz)                    : ")
    total_BW           = get_float_input(
        "Enter the total bandwidth (Hz)                    : ",
        condition=lambda x: x > trunk_BW,
        error_msg=f"Total bandwidth must be greater than trunk bandwidth ({trunk_BW} Hz)."
    )
    blocking_prob      = get_float_input(
        "Enter the blocking probability (e.g. 0.02)        : ",
        condition=lambda x: 0 < x < 1,
        error_msg="Blocking probability must be between 0 and 1 (e.g. 0.02 for 2%)."
    )

    # Traffic sanity check
    area_km2   = (length / 1000) * (width / 1000)
    total_subs = area_km2 * subscriber_density
    A_total    = total_subs * requests_per_second * user_session_time
    total_ch   = int(total_BW / trunk_BW)
    if A_total > total_ch * 10000:
        print(f"\n  [WARNING] Offered traffic ({A_total:.1f} Erlangs) is extremely high "
              f"for {total_ch} total channels.")
        print("  Consider reducing session time or request rate.\n")

    c_i_linear = 10 ** (c_i_dB / 10)

    area_km2   = get_area_km2(length, width)
    total_subs = get_total_subscribers(area_km2, subscriber_density)
    total_ch   = get_total_channels(total_BW, trunk_BW)

    N, sectoring, n_interferers, ci_achieved = find_reuse_factor(c_i_linear)

    num_cells, ch_per_cell, A_per_cell = get_number_of_cells(
        total_subs, user_session_time, requests_per_second,
        trunk_BW, total_BW, blocking_prob, N
    )

    capacity_bps = shannon_capacity(trunk_BW, c_i_linear)

    bit_stream  = generate_bitStream()
    symbols     = bits_to_symbols(bit_stream)
    received    = transmit(symbols, TRANSMIT_POWER, NOISE_POWER, c_i_linear)
    rx_bits     = demodulate(received, TRANSMIT_POWER)
    ber, errors = compute_BER(bit_stream, rx_bits)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  1. Minimum number of cells    : {num_cells}")
    sect_label = "Omni (no sectoring)" if sectoring == 180 else f"{sectoring} degrees"
    print(f"  2. Sectoring type             : {sect_label}")
    print(f"  3. Shannon capacity (per ch)  : {capacity_bps/1e3:.2f} kbps")
    print()
    print(f"  4. Received bit stream (first 40 bits):")
    print(f"     {rx_bits[:40].tolist()}")
    print()
    print(f"  5. BER                        : {ber:.4f}  ({ber*100:.2f}%)")
    print(f"     Bit errors                 : {errors} / {len(bit_stream)}")
    print("=" * 60)
    print()
    print("  [ Additional Info ]")
    print(f"  Area                          : {area_km2:.4f} km2")
    print(f"  Total subscribers             : {total_subs:.0f}")
    print(f"  Total channels available      : {total_ch}")
    print(f"  Frequency reuse factor  N     : {N}")
    print(f"  Co-channel interferers  n     : {n_interferers}")
    print(f"  Achieved C/I                  : {10*math.log10(ci_achieved):.2f} dB")
    print(f"  Channels per cell             : {ch_per_cell}")
    print(f"  Offered traffic per cell      : {A_per_cell:.4f} Erlangs")
    print(f"  Transmitted bit stream (first 40 bits):")
    print(f"     {bit_stream[:40].tolist()}")
    print("=" * 60)

    print("\nGenerating BER plots...")
    plot_ber_vs_tx_power(c_i_linear, NOISE_POWER)
    plot_ber_vs_noise_power(c_i_linear, TRANSMIT_POWER)

    do_image = input("\n[Bonus] Transmit an image? (y/n): ").strip().lower()
    if do_image == 'y':
        img_path = input("  Enter path to image: ").strip()
        transmit_image(img_path, TRANSMIT_POWER, NOISE_POWER, c_i_linear)


if __name__ == "__main__":
    main()
