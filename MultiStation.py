
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict

# Simulation parameters
AIFS = 43
slot_time = 9
mean_backoff = 67.5
phy_header = 44
SIFS = 16
block_ack = 44
bitrate = 34.4e6  # 34.4 Mbps
frame_size_bytes = 1000
frame_size_bits = frame_size_bytes * 8
OV = 214.5  # µs
lambda_i = 15e-5  # arrivals per µs
D_i = 240  # µs
SIM_TIME = 1e6  # 1 second

random.seed(42)

def transmission_time(n_frames):
    data_time = (frame_size_bits * n_frames) / bitrate * 1e6
    return OV + data_time

class Packet:
    def __init__(self, dest_id, arrival_time):
        self.dest_id = dest_id
        self.arrival_time = arrival_time

class Aggregator:
    def __init__(self, env, strategy):
        self.env = env
        self.strategy = strategy
        self.buffer = deque()
        self.stats = []
        self.pool_sizes = []
        self.proc = env.process(self.run())

    def run(self):
        while True:
            if not self.buffer:
                yield self.env.timeout(1)
                continue

            jumbo_frame = []

            if self.strategy == 'FIFO':
                jumbo_frame.append(self.buffer.popleft())

            elif self.strategy == 'FIFO Pooling':
                first = self.buffer[0]
                same_dest = [pkt for pkt in self.buffer if pkt.dest_id == first.dest_id]
                if len(same_dest) > 1:
                    jumbo_frame = [pkt for pkt in list(self.buffer) if pkt.dest_id == first.dest_id]
                    for pkt in jumbo_frame:
                        self.buffer.remove(pkt)
                else:
                    dests = set()
                    for pkt in list(self.buffer):
                        if pkt.dest_id not in dests:
                            jumbo_frame.append(pkt)
                            dests.add(pkt.dest_id)
                    for pkt in jumbo_frame:
                        self.buffer.remove(pkt)

            elif self.strategy == 'MAX FIFO Pooling':
                first = self.buffer[0]
                same_dest = [pkt for pkt in self.buffer if pkt.dest_id == first.dest_id]
                dests = set()
                ofdma = []
                for pkt in self.buffer:
                    if pkt.dest_id not in dests:
                        ofdma.append(pkt)
                        dests.add(pkt.dest_id)
                agg_size = len(same_dest)
                ofdma_size = len(ofdma)
                if agg_size >= ofdma_size:
                    jumbo_frame = same_dest
                else:
                    jumbo_frame = ofdma
                for pkt in jumbo_frame:
                    self.buffer.remove(pkt)

            elif self.strategy == 'MAX Pooling':
                dest_groups = defaultdict(list)
                for pkt in list(self.buffer):
                    dest_groups[pkt.dest_id].append(pkt)
                largest_group = max(dest_groups.values(), key=lambda x: len(x))
                jumbo_frame = largest_group
                for pkt in jumbo_frame:
                    self.buffer.remove(pkt)

            n_frames = len(jumbo_frame)
            tx_time = transmission_time(n_frames)
            yield self.env.timeout(tx_time)
            self.pool_sizes.append(n_frames)
            for pkt in jumbo_frame:
                self.stats.append(self.env.now - pkt.arrival_time)

def packet_generator(env, aggregator, num_dest):
    while True:
        yield env.timeout(random.expovariate(lambda_i * num_dest))
        dest_id = random.randint(0, num_dest - 1)
        pkt = Packet(dest_id, env.now)
        aggregator.buffer.append(pkt)

def simulate(strategy, num_dest):
    env = simpy.Environment()
    agg = Aggregator(env, strategy)
    env.process(packet_generator(env, agg, num_dest))
    env.run(until=SIM_TIME)
    mean_sojourn = np.mean(agg.stats) if agg.stats else 0
    mean_pool = np.mean(agg.pool_sizes) if agg.pool_sizes else 0
    load = (len(agg.stats) * frame_size_bits) / (bitrate * (SIM_TIME * 1e-6))
    return mean_sojourn, mean_pool, load

# Run simulation for each strategy
strategies = ['FIFO', 'FIFO Pooling', 'MAX FIFO Pooling', 'MAX Pooling']
dest_range = range(2, 41, 2)
results = {s: {'sojourn': [], 'pool': [], 'load': []} for s in strategies}

for s in strategies:
    for N in dest_range:
        mean_s, mean_p, load = simulate(s, N)
        results[s]['sojourn'].append(mean_s)
        results[s]['pool'].append(mean_p)
        results[s]['load'].append(load)

# Plotting


fig, axs = plt.subplots(3, 1, figsize=(10, 15))
for s in strategies:
    axs[0].plot(dest_range, results[s]['load'], label=s)
for s in strategies:
    if s != 'FIFO':
        axs[1].plot(dest_range, results[s]['sojourn'], label=s)
        axs[2].plot(dest_range, results[s]['pool'], label=s)

axs[0].set_title("System Load vs Number of Destinations")
axs[0].set_xlabel("Number of Destinations")
axs[0].set_ylabel("System Load")
axs[0].legend()
axs[0].grid()

axs[1].set_title("Mean Sojourn Time vs Number of Destinations")
axs[1].set_xlabel("Number of Destinations")
axs[1].set_ylabel("Mean Sojourn Time (µs)")
axs[1].legend()
axs[1].grid()

axs[2].set_title("Mean Pooling Size vs Number of Destinations")
axs[2].set_xlabel("Number of Destinations")
axs[2].set_ylabel("Mean Pooling Size (frames)")
axs[2].legend()
axs[2].grid()

# plt.tight_layout()
plt.tight_layout(pad=4.0)  # add padding between subplots

plt.show()
