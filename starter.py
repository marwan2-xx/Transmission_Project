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

def modulate():
    return

def transmit():
    return
def demodulate():
    return

def compute_BER():
    return


bit_stream = generate_bitStream()
area= get_Area(length , width)

total_channels = get_total_channels(trunk_BW , total_BW)
total_subscirbers = get_total_subscribers(area , subscriber_density)








