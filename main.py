import random
from profiling.initialize_profiling import get_profiling_data
import matplotlib.pyplot as plt
from reference_schedulers.random_scheduler import run_random_scheduler
from simulator.a2c_simulator import run_a2c_simulation
from simulator.doubleQ_simulator import run_simulation, run_simulation_all
from simulator.sac_simulator import run_sac_simulation
import numpy as np


if __name__ == "__main__":
    episodes = 1
    max_steps = 10
    deadlines = list(range(500, 505, 3))  # 1ms to 700ms

    dq_energy, dq_time, dq_r = [], [], []
    a2c_energy, a2c_time = [], []
    sac_energy, sac_time = [], []
    random_energy, random_time = [], []   
    edge_energy, edge_time = [], []
    cloud_energy, cloud_time = [], []

    for d in deadlines:
        print("Running simulations for deadline: {} ms".format(d))
        profiling_data = get_profiling_data(d)

        e, t, r = run_simulation_all(profiling_data, 1)
        dq_energy.append(e)
        dq_time.append(t)

        # a2c_e, a2c_t = run_a2c_simulation(profiling_data, episodes, max_steps)
        # a2c_energy.append(a2c_e)
        # a2c_time.append(a2c_t)

        # sac_e, sac_t = run_sac_simulation(profiling_data, episodes, max_steps)
        # sac_energy.append(sac_e)
        # sac_time.append(sac_t)

        # re, rt = run_random_scheduler(profiling_data, 100, max_steps, is_random=True, is_all_cloud=False)
        # random_energy.append(re)
        # random_time.append(rt)

        # ee, et = run_random_scheduler(profiling_data, 100, max_steps, is_random=False, is_all_cloud=False)
        # edge_energy.append(ee)
        # edge_time.append(et)

        # ce, ct = run_random_scheduler(profiling_data, 100, max_steps, is_random=False, is_all_cloud=True)
        # cloud_energy.append(ce)
        # cloud_time.append(ct)

    # # Plot Energy vs Deadline
    # plt.figure(figsize=(8, 6))
    # plt.plot(deadlines, dq_energy, label="Double Q", marker='o')
    # # plt.plot(deadlines, a2c_energy, label="A2C", marker='*')
    # # plt.plot(deadlines, sac_energy, label="SAC", marker='s')
    # plt.plot(deadlines, random_energy, label="Random", marker='+')
    # plt.plot(deadlines, edge_energy, label="All Edge", marker='x')
    # plt.plot(deadlines, cloud_energy, label="All Cloud", marker='^')
    # plt.xlabel("Deadline (ms)")
    # plt.ylabel("Average Energy (Joules)")
    # plt.title("Average Energy vs Deadline")
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.tight_layout()
    # plt.show()

    # # Plot Completion Time vs Deadline
    # plt.figure(figsize=(8, 6))
    # plt.plot(deadlines, dq_time, label="Double Q", marker='o')
    # # plt.plot(deadlines, a2c_time, label="A2C", marker='*')
    # # plt.plot(deadlines, sac_time, label="SAC", marker='s')
    # plt.plot(deadlines, random_time, label="Random", marker='+')
    # plt.plot(deadlines, edge_time, label="All Edge", marker='x')
    # plt.plot(deadlines, cloud_time, label="All Cloud", marker='^')
    # plt.xlabel("Deadline (ms)")
    # plt.ylabel("Average Completion Time (ms)")
    # plt.title("Average Completion Time vs Deadline")
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.tight_layout()
    # plt.show()


    print("Simulations completed.")
    # print("all cloud energy", np.mean(cloud_energy))
    # print("all edge energy", np.mean(edge_energy))
    print("dq energy", np.mean(dq_energy),"time", np.mean(dq_time), "reward", np.mean(dq_r))
    # print("a2c energy", np.mean(a2c_energy))
    # print("sac energy", np.mean(sac_energy))
    # print("random energy", np.mean(random_energy))