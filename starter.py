import numpy as np
import matplotlib.pyplot as plt
from math import log2

#inputs 
transmit_power = 0.1
noise_power = 0.001
length = eval(input("Enter the length : "))
width = eval(input("Enter the width : "))
subscriber_density = eval(input("Enter users density in a given area(users/square kilometers) : "))
c_i = eval(input("Enter the min signal to interference ratio : "))
user_session_time = eval(input("Enter avearge session time per user : "))
requests_per_second = eval(input("Enter the average session requests per second : "))
trunk_BW = eval(input("Enter the trunk bandwidth : "))
total_BW = eval(input("Enter the total bandwidth : "))
Blocking_probability = eval(input("Enter the blocking probability : "))




def get_Area(length , width):
    return length*width

def get_total_subscribers(area , subscriber_density):
    return area*subscriber_density

def get_total_channels(trunk_BW , total_BW):
    return total_BW/trunk_BW

def get_n():
    return
def get_N(c_i , n):
    return ((c_i)*n)/3

def get_number_of_cells():
    return

def get_channels_per_cell(total_channels , N):
    return total_channels/N


def shannon_capacity(B , c_i):
    return B * log2(1 + c_i)

def generate_bitStream():
    return np.random.randint(0, 2, 10000)

def get_16qam_constellation(): #not sure of that function
    """Standard 16-QAM constellation (normalized average power = 1)"""
    re = np.array([-3, -1, 1, 3])
    im = np.array([-3, -1, 1, 3])
    const = np.array([complex(r, i) for r in re for i in im])
    return const / np.sqrt(10)   # normalization

def bits_to_symbols(bits):#not sure of that function
    
    constellation = get_16qam_constellation()
    """Map groups of 4 bits to 16-QAM symbols (binary mapping)"""
    symbols = []
    for i in range(0, len(bits), 4):
        idx = int(''.join(map(str, bits[i:i+4])), 2)
        symbols.append(constellation[idx])
    return np.array(symbols)

def transmit(symbols, tx_power, noise_power, c_i):#not sure of that function
    """Transmit through interference channel (interference treated as noise)"""
    # Scale to transmit power
    transmitted = symbols * np.sqrt(tx_power)
    
    # Interference power = Tx power / (C/I)
    interference_power = tx_power / c_i if c_i > 0 else 0
    
    # Total noise variance = noise + interference
    total_noise_var = noise_power + interference_power
    
    # Complex AWGN
    noise = np.sqrt(total_noise_var / 2) * (
        np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
    )
    
    return transmitted + noise


def demodulate(received_symbols, constellation):#not sure of that function
    """Nearest-neighbor demodulation"""
    received_bits = []
    for sym in received_symbols:
        distances = np.abs(constellation - sym)
        idx = np.argmin(distances)
        bit4 = np.array(list(format(idx, '04b')), dtype=int)
        received_bits.extend(bit4)
    return np.array(received_bits)


def compute_BER(original_bits, received_bits):#not sure of that function
    errors = np.sum(original_bits != received_bits)
    ber = errors / len(original_bits)
    return ber, errors

def plot_ber_vs_tx_power(c_i, noise_power, num_bits=10000):
    constellation = get_16qam_constellation()
    """Plot BER vs Transmit Power (required output)"""
    tx_powers = np.logspace(-3, 0, 20)   # 0.001 W to 1 W
    bers = []
    
    for p in tx_powers:
        bits = generate_bitStream()
        symbols = bits_to_symbols(bits)
        received = transmit(symbols, p, noise_power, c_i)
        received_bits = demodulate(received, constellation)
        ber, _ = compute_BER(bits, received_bits)
        bers.append(ber)
    
    plt.figure(figsize=(8, 5))
    plt.semilogy(tx_powers * 1000, bers, marker='o')   # plot in mW
    plt.xlabel("Transmit Power (mW)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs Transmit Power")
    plt.grid(True, which="both")
    plt.show()


bit_stream = generate_bitStream()
area= get_Area(length , width)
print(area)
print(bit_stream)

total_channels = get_total_channels(trunk_BW , total_BW)
total_subscirbers = get_total_subscribers(area , subscriber_density)





