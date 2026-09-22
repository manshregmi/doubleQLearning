import random
from profiling.initialize_profiling import get_profiling_data
import matplotlib.pyplot as plt
from reference_schedulers.random_scheduler import run_random_scheduler
from simulator.a2c_simulator import run_a2c_simulation
from simulator.doubleQ_simulator import run_simulation
from a2c.coarse_grained_dq import run_oneshot_doubleQ_simulation
from a2c.coarse_grained_a2c import run_oneshot_a2c_simulation
import numpy as np


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 10



    a2c_energy, a2c_time, a2c_deadline_misses = [], [], []
    deadline = 500

    devices = list(range(1,100,1))


    for device in devices:
        for _ in range(10000):
            times = []

            profiling_data = get_profiling_data(deadline=deadline, edge_devices=device)

            total_cloud_waiting_time = 0

            for level in range(len(profiling_data.layers)):
                congestion = profiling_data.get_max_layer_cloud_time(level)+ abs(profiling_data.get_max_layer_cloud_time(level) * (profiling_data.numberOfEdgeDevice - 1) * np.random.uniform(0,1))
                total_cloud_waiting_time+=congestion
            times.append(total_cloud_waiting_time)
        
        time = np.array(times)
        avg_time = np.mean(time)
        a2c_time.append(avg_time)




        # a2c_e, a2c_t, a2c_dm = run_a2c_simulation(profiling_data, 100000, 20, False)
        # a2c_energy.append(a2c_e)
        # a2c_time.append(a2c_t)
        # a2c_deadline_misses.append(a2c_dm/100000)
        # print("average energy is ", a2c_e, a2c_t)


    # Plot 1: Energy vs Deadline (Comparison)
    plt.figure(figsize=(14, 8))
    plt.plot(devices, a2c_time, label="Congestion", marker='*', linewidth=3)
    plt.xlabel("UAV swarm size", fontsize=28, fontfamily='Times New Roman')
    plt.ylabel("Contension time (ms)", fontsize=28, fontfamily='Times New Roman')
    plt.title("Contension various UAV swarm size", fontsize=28, fontfamily='Times New Roman')
    plt.legend(fontsize=24, loc='best')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=24, fontfamily='Times New Roman')
    plt.yticks(fontsize=24, fontfamily='Times New Roman')
    plt.tight_layout()
    plt.savefig("energy_vs_uav_swam_size_comparison.png", dpi=600)
    plt.show()


    