import os.path
import sys

def inp(): return int(sys.stdin.readline())
def st(): return list(sys.stdin.readline().strip())
def li(): return list(map(int, sys.stdin.readline().split()))
def mp(): return map(int, sys.stdin.readline().split())

def pr(n): return sys.stdout.write(str(n) + "\n")
def prl(n): return sys.stdout.write(str(n) + "")

if os.path.exists('input.txt'):
    sys.stdin = open('input.txt', 'r')
    sys.stdout = open('output.txt', 'w')

import numpy as np
import pandas as pd
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 12
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

class Packet:
    """Class representing a packet in the system."""
    def __init__(self, arrival_time, destination, service_time):
        self.arrival_time = arrival_time
        self.destination = destination  # 0 for first destination, 1 for second
        self.service_time = service_time
        self.departure_time = None
        self.sojourn_time = None

    def complete_service(self, time):
        """Mark packet as served and calculate sojourn time."""
        self.departure_time = time
        self.sojourn_time = self.departure_time - self.arrival_time

class QueueingSystem:
    """Base class for different queueing disciplines."""
    def __init__(self, arrival_rate_1, arrival_rate_2, service_time=1174.5e-6):
        self.count=0
        self.arrival_rate_1 = arrival_rate_1
        self.arrival_rate_2 = arrival_rate_2
        self.service_time = service_time
        self.aifs = 43
        self.slot_time = 9
        self.mean_backoff = 67.5
        self.physical_header = 44
        self.sifs = 16
        self.block_ack = 44
        self.mcs = 8.6
        self.frame_size = 1000
        self.ov = 214.5e-6
        self.current_time = 0
        self.server_busy_until = 0
        self.served_packets = []
        self.server_busy_time = 0
        self.jumbo_frame_sizes = []
        self.queue_length_history = []
        self.queue_length_time = []

    def generate_arrivals(self, num_packets):
        """Generate Poisson arrivals for both destinations."""
        arrivals_1 = []
        current_time = 0
        for _ in range(int(num_packets * self.arrival_rate_1 / (self.arrival_rate_1 + self.arrival_rate_2))):
            inter_arrival = np.random.exponential(1/self.arrival_rate_1)
            current_time += inter_arrival
            arrivals_1.append((current_time, 0))

        arrivals_2 = []
        current_time = 0
        for _ in range(int(num_packets * self.arrival_rate_2 / (self.arrival_rate_1 + self.arrival_rate_2))):
            inter_arrival = np.random.exponential(1/self.arrival_rate_2)
            current_time += inter_arrival
            arrivals_2.append((current_time, 1))

        all_arrivals = arrivals_1 + arrivals_2
        all_arrivals.sort()
        packets = [Packet(time, dest, self.service_time) for time, dest in all_arrivals[:num_packets]]

        return packets

    def run_simulation(self, num_packets):
        """Run the simulation for a specified number of packets."""
        packets = self.generate_arrivals(num_packets)
        max_time = packets[-1].arrival_time if packets else 0
        for packet in tqdm(packets, desc="Processing packets"):
            self.process_arrival(packet)
            queue_length = len(self.get_queue())
            self.queue_length_history.append(queue_length)
            self.queue_length_time.append(packet.arrival_time)
            self.current_time = self.server_busy_until

        self.add_time()
        total_time = self.server_busy_until
        server_load = self.server_busy_time / total_time if total_time > 0 else 0
        sojourn_times = [p.sojourn_time for p in self.served_packets]
        mean_sojourn_time = np.mean(sojourn_times) if sojourn_times else 0
        mean_pooling_size = np.mean(self.jumbo_frame_sizes) if self.jumbo_frame_sizes else 0
        return {
            'server_load': server_load,
            'mean_sojourn_time': mean_sojourn_time,
            'mean_pooling_size': mean_pooling_size,
            'sojourn_times': sojourn_times
        }


class FIFOQueueingSystem(QueueingSystem):
    """First-In-First-Out queueing discipline with a single queue."""
    def __init__(self, arrival_rate_1, arrival_rate_2, service_time=1174.5e-6):
        super().__init__(arrival_rate_1, arrival_rate_2, service_time)
        self.queue = deque()  # Single queue for all packets

    def process_arrival(self, packet):
        self.current_time = packet.arrival_time
        self._serve_packets()
        self.queue.append(packet)
        self._serve_packets()

    def _serve_packets(self):
        while self.queue and self.server_busy_until <= self.current_time:
            self.count+=1;
            next_packet = self.queue.popleft()
            pooled_packets = [next_packet]  # Serve one packet at a time
            self.jumbo_frame_sizes.append(len(pooled_packets))
            service_start = max(self.current_time, self.server_busy_until)
            self.server_busy_until = service_start + next_packet.service_time
            self.server_busy_time += next_packet.service_time
            next_packet.complete_service(self.server_busy_until)
            self.served_packets.append(next_packet)

    def get_queue(self):
        return list(self.queue)

    def add_time(self):
        while self.queue:
            next_packet = self.queue.popleft()
            pooled_packets = [next_packet]
            self.jumbo_frame_sizes.append(len(pooled_packets))
            service_start = max(self.server_busy_until, next_packet.arrival_time)
            self.server_busy_until = service_start + next_packet.service_time
            self.server_busy_time += next_packet.service_time
            next_packet.complete_service(self.server_busy_until)
            self.served_packets.append(next_packet)









class FIFOPoolingSystem(QueueingSystem):
    """FIFO Pooling - First packet selected, then pool based on destination."""
    def __init__(self, arrival_rate_1, arrival_rate_2, service_time=1174.5e-6):
        super().__init__(arrival_rate_1, arrival_rate_2, service_time)
        self.queue = deque()

    def process_arrival(self, packet):
        self.current_time = packet.arrival_time
        self._serve_packets()
        self.queue.append(packet)
        self._serve_packets()

    def _serve_packets(self):
        while self.queue and self.server_busy_until <= self.current_time:
            next_packet = self.queue.popleft()
            pooled_packets = [next_packet]
            if self.queue:  # Check if there's a second packet
                second_packet = self.queue[0]
                if second_packet.destination != next_packet.destination:
                    # Different destinations: send both packets if arrived
                    if second_packet.arrival_time <= self.current_time:
                        pooled_packets.append(self.queue.popleft())
                else:
                   # Same destination: aggregate until different destination or arrival time exceeded
                    while self.queue and self.queue[0].destination == next_packet.destination and self.queue[0].arrival_time <= self.current_time:
                        pooled_packets.append(self.queue.popleft())
            self.jumbo_frame_sizes.append(len(pooled_packets))
            # print(len(pooled_packets))
            # print(self.service_time)
            jumbo_service_time = (self.service_time-self.ov) * len(pooled_packets)+self.ov
            service_start = max(self.current_time, self.server_busy_until)
            self.server_busy_until = service_start + jumbo_service_time
            self.server_busy_time += jumbo_service_time
            for packet in pooled_packets:
                packet.complete_service(self.server_busy_until)
                self.served_packets.append(packet)

    def get_queue(self):
        return list(self.queue)
    def add_time(self):
        while self.queue:
            next_packet = self.queue.popleft()
            pooled_packets = [next_packet]
            if self.queue:
                second_packet = self.queue[0]
                if second_packet.destination != next_packet.destination:
                    if second_packet.arrival_time <= self.server_busy_until:
                        pooled_packets.append(self.queue.popleft())
                else:
                    while self.queue and self.queue[0].destination == next_packet.destination and self.queue[0].arrival_time <= self.server_busy_until:
                        pooled_packets.append(self.queue.popleft())
            self.jumbo_frame_sizes.append(len(pooled_packets))
            jumbo_service_time = (self.service_time - self.ov) * len(pooled_packets) + self.ov
            service_start = max(self.server_busy_until, next_packet.arrival_time)
            self.server_busy_until = service_start + jumbo_service_time
            self.server_busy_time += jumbo_service_time
            for packet in pooled_packets:
                packet.complete_service(self.server_busy_until)
                self.served_packets.append(packet)



class MaxFIFOPoolingSystem(QueueingSystem):
    """MAX FIFO POOLING takes first frame in FIFO order and maximizes jumbo frame size."""
    def __init__(self, arrival_rate_1, arrival_rate_2, service_time=1174.5e-6):
        super().__init__(arrival_rate_1, arrival_rate_2, service_time)
        self.queue = deque()

    def process_arrival(self, packet):
        self.current_time = packet.arrival_time
        self._serve_packets()
        self.queue.append(packet)
        self._serve_packets()

    def _serve_packets(self):
        while self.queue and self.server_busy_until <= self.current_time:
            next_packet = self.queue.popleft()
            destination = next_packet.destination
            pooled_packets = [next_packet]
            same_dest_packets = [p for p in self.queue if p.destination == destination and p.arrival_time <= self.current_time]
            different_dest_count = len(set(p.destination for p in self.queue if p.arrival_time <= self.current_time))
            if len(same_dest_packets) > different_dest_count:
                for _ in range(min(len(same_dest_packets), len(self.queue))):
                    i = 0
                    while i < len(self.queue):
                        if self.queue[i].destination == destination and self.queue[i].arrival_time <= self.current_time:
                            pooled_packets.append(self.queue[i])
                            self.queue.remove(self.queue[i])
                            break
                        i += 1
            else:
                seen_destinations = {destination}
                i = 0
                while i < len(self.queue):
                    if self.queue[i].destination not in seen_destinations and self.queue[i].arrival_time <= self.current_time:
                        pooled_packets.append(self.queue[i])
                        seen_destinations.add(self.queue[i].destination)
                        self.queue.remove(self.queue[i])
                    else:
                        i += 1
            self.jumbo_frame_sizes.append(len(pooled_packets))
            jumbo_service_time = (self.service_time-self.ov) * len(pooled_packets)+self.ov
            service_start = max(self.current_time, self.server_busy_until)
            self.server_busy_until = service_start + jumbo_service_time
            self.server_busy_time += jumbo_service_time
            for packet in pooled_packets:
                packet.complete_service(self.server_busy_until)
                self.served_packets.append(packet)


    def get_queue(self):
        return list(self.queue)
    def add_time(self):
        while self.queue:
            next_packet = self.queue.popleft()
            destination = next_packet.destination
            pooled_packets = [next_packet]

            same_dest_packets = [p for p in self.queue if p.destination == destination and p.arrival_time <= self.server_busy_until]
            different_dest_count = len(set(p.destination for p in self.queue if p.arrival_time <= self.server_busy_until))

            if len(same_dest_packets) > different_dest_count:
                for _ in range(min(len(same_dest_packets), len(self.queue))):
                    i = 0
                    while i < len(self.queue):
                        if self.queue[i].destination == destination and self.queue[i].arrival_time <= self.server_busy_until:
                            pooled_packets.append(self.queue[i])
                            self.queue.remove(self.queue[i])
                            break
                        i += 1
            else:
                seen_destinations = {destination}
                i = 0
                while i < len(self.queue):
                    if self.queue[i].destination not in seen_destinations and self.queue[i].arrival_time <= self.server_busy_until:
                        pooled_packets.append(self.queue[i])
                        seen_destinations.add(self.queue[i].destination)
                        self.queue.remove(self.queue[i])
                    else:
                        i += 1

            self.jumbo_frame_sizes.append(len(pooled_packets))
            jumbo_service_time = (self.service_time - self.ov) * len(pooled_packets) + self.ov
            service_start = max(self.server_busy_until, next_packet.arrival_time)
            self.server_busy_until = service_start + jumbo_service_time
            self.server_busy_time += jumbo_service_time
            for packet in pooled_packets:
                packet.complete_service(self.server_busy_until)
                self.served_packets.append(packet)


class MaxPoolingSystem(QueueingSystem):
    """MAX POOLING aims to maximize the jumbo frame size without considering arrival order."""
    def __init__(self, arrival_rate_1, arrival_rate_2, service_time=1174.5e-6):
        super().__init__(arrival_rate_1, arrival_rate_2, service_time)
        self.queue_1 = deque()
        self.queue_2 = deque()

    def process_arrival(self, packet):
        self.current_time = packet.arrival_time
        self._serve_packets()
        if packet.destination == 0:
            self.queue_1.append(packet)
        else:
            self.queue_2.append(packet)
        self._serve_packets()

    def _serve_packets(self):
        while (self.queue_1 or self.queue_2) and self.server_busy_until <= self.current_time:
            # Filter queues based on current time
            queue_1_arrived = [p for p in self.queue_1 if p.arrival_time <= self.current_time]
            queue_2_arrived = [p for p in self.queue_2 if p.arrival_time <= self.current_time]
            if len(queue_1_arrived) > len(queue_2_arrived):
                pooled_packets = queue_1_arrived
                # Remove served packets from queue_1
                self.queue_1 = deque([p for p in self.queue_1 if p not in pooled_packets])
            elif len(queue_2_arrived) > len(queue_1_arrived):
                pooled_packets = queue_2_arrived
                # Remove served packets from queue_2
                self.queue_2 = deque([p for p in self.queue_2 if p not in pooled_packets])
            else:
                if np.random.random() < 0.5 and queue_1_arrived:
                    pooled_packets = queue_1_arrived
                    # Remove served packets from queue_1
                    self.queue_1 = deque([p for p in self.queue_1 if p not in pooled_packets])
                elif queue_2_arrived:
                    pooled_packets = queue_2_arrived
                    # Remove served packets from queue_2
                    self.queue_2 = deque([p for p in self.queue_2 if p not in pooled_packets])
                else:
                    break
            self.jumbo_frame_sizes.append(len(pooled_packets))
            jumbo_service_time = (self.service_time-self.ov) * len(pooled_packets)+self.ov
            service_start = max(self.current_time, self.server_busy_until)
            self.server_busy_until = service_start + jumbo_service_time
            self.server_busy_time += jumbo_service_time
            for packet in pooled_packets:
                packet.complete_service(self.server_busy_until)
                self.served_packets.append(packet)


    def get_queue(self):
        return list(self.queue_1) + list(self.queue_2)
    def add_time(self):
        while self.queue_1 or self.queue_2:
            pooled_packets = []
            dest_seen = set()
            for queue in [self.queue_1, self.queue_2]:
                i = 0
                while i < len(queue):
                    if queue[i].destination not in dest_seen and queue[i].arrival_time <= self.server_busy_until:
                        pooled_packets.append(queue[i])
                        dest_seen.add(queue[i].destination)
                        queue.remove(queue[i])
                    else:
                        i += 1
            if not pooled_packets:
                break  # no eligible packets

            self.jumbo_frame_sizes.append(len(pooled_packets))
            jumbo_service_time = (self.service_time - self.ov) * len(pooled_packets) + self.ov
            service_start = max(self.server_busy_until, min(p.arrival_time for p in pooled_packets))
            self.server_busy_until = service_start + jumbo_service_time
            self.server_busy_time += jumbo_service_time
            for packet in pooled_packets:
                packet.complete_service(self.server_busy_until)
                self.served_packets.append(packet)


def run_comparative_simulation():
    """Run simulations for different arrival rates and queueing disciplines."""
    fixed_arrival_rate_1 = 3e-5
    service_time = 1174.5e-6
    num_packets = 10000
    arrival_rates_2 = np.linspace(3e-5, 1000, 20)
    disciplines = {
        'FIFO': FIFOQueueingSystem,
        'FIFO Pooling': FIFOPoolingSystem,
        'Max FIFO Pooling': MaxFIFOPoolingSystem,
        'Max Pooling': MaxPoolingSystem
    }
    results = {
        discipline: {
            'server_load': [],
            'mean_sojourn_time': [],
            'mean_pooling_size': []
        } for discipline in disciplines
    }
    for arrival_rate_2 in tqdm(arrival_rates_2, desc="Simulating arrival rates"):
        for discipline_name, discipline_class in disciplines.items():
            system = discipline_class(fixed_arrival_rate_1, arrival_rate_2, service_time)
            stats = system.run_simulation(num_packets)
            results[discipline_name]['server_load'].append(stats['server_load'])
            results[discipline_name]['mean_sojourn_time'].append(stats['mean_sojourn_time'])
            results[discipline_name]['mean_pooling_size'].append(stats['mean_pooling_size'])
    plot_results(arrival_rates_2, results)
    return results, arrival_rates_2

def plot_results(arrival_rates_2, results):
    """Plot the combined simulation results for all disciplines."""
    disciplines = list(results.keys())
    metrics = ['server_load', 'mean_sojourn_time', 'mean_pooling_size']
    titles = ['Server Load', 'Mean Sojourn Time(sec)', 'Mean Pooling Size']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if metric in ['mean_sojourn_time', 'mean_pooling_size']:
            # Exclude FIFO for Mean Sojourn Time and Mean Pooling Size
            for discipline in [d for d in disciplines if d != 'FIFO']:
                axes[i].plot(arrival_rates_2, results[discipline][metric], marker='o', label=discipline)
        else:
            # Include all disciplines for Server Load
            for discipline in disciplines:
                axes[i].plot(arrival_rates_2, results[discipline][metric], marker='o', label=discipline)
        axes[i].set_xlabel('Arrival Rate to Destination 2 (λ₂)')
        axes[i].set_ylabel(title)
        axes[i].set_title(title)
        axes[i].grid(True)
        axes[i].legend()
    plt.tight_layout()
    plt.savefig('queueing_simulation_combined_results.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    print("\nRunning comparative simulation for different queueing disciplines...")
    results, arrival_rates_2 = run_comparative_simulation()
    print("\nSummary of Results at Highest Load:")
    disciplines = list(results.keys())
    for discipline in disciplines:
        print(f"\n{discipline}:")
        print(f"  Server Load: {results[discipline]['server_load'][-1]:.4f}")
        print(f"  Mean Sojourn Time: {results[discipline]['mean_sojourn_time'][-1]:.4f}")
        print(f"  Mean Pooling Size: {results[discipline]['mean_pooling_size'][-1]:.4f}")
