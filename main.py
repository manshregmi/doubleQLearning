import random
from a2c.coarse_grained_a2c import run_a2c_episode_level
from profiling.initialize_profiling import get_profiling_data
import matplotlib.pyplot as plt
from reference_schedulers.random_scheduler import run_random_scheduler
from simulator.a2c_simulator import run_a2c_simulation
from simulator.doubleQ_simulator import run_simulation
import numpy as np


if __name__ == "__main__":
    is_test = False
    episodes = 100000
    # episodes = 1
    max_steps = 10
    deadlines = list(range(400,505,50 ))  # 400ms to 500ms
    
    dq_energy, dq_time, dq_deadline_misses = [], [], []
    a2c_energy, a2c_time, a2c_deadline_misses = [], [], []
    cg_a2c_energy, cg_a2c_time, cg_a2c_deadline_misses = [], [], []
    sac_energy, sac_time = [], []
    random_energy, random_time = [], []   
    edge_energy, edge_time = [], []
    cloud_energy, cloud_time = [], []

    for d in deadlines:
        print("Running simulations for deadline: {} ms".format(d))
        profiling_data = get_profiling_data(d)

        e, t, dm = run_simulation(profiling_data, episodes, max_steps, is_test)
        dq_energy.append(e)
        dq_time.append(t)
        dq_deadline_misses.append(dm)

        a2c_e, a2c_t, a2c_dm = run_a2c_simulation(profiling_data, episodes, max_steps, is_test)
        a2c_energy.append(a2c_e)
        a2c_time.append(a2c_t)
        a2c_deadline_misses.append(a2c_dm)

        cg_a2c_e, cg_a2c_t, cg_a2c_dm = run_a2c_episode_level(profiling_data, episodes, max_steps, is_test)
        cg_a2c_energy.append(cg_a2c_e)
        cg_a2c_time.append(cg_a2c_t)
        cg_a2c_deadline_misses.append(cg_a2c_dm)

        # sac_e, sac_t = run_sac_simulation(profiling_data, episodes, max_steps, is_test)
        # sac_energy.append(sac_e)
        # sac_time.append(sac_t)

        re, rt = run_random_scheduler(profiling_data, episodes, max_steps, is_random=True, is_all_cloud=False)
        random_energy.append(re)
        random_time.append(rt)

        ee, et = run_random_scheduler(profiling_data, 1000, max_steps, is_random=False, is_all_cloud=False)
        edge_energy.append(ee)
        edge_time.append(et)

        ce, ct = run_random_scheduler(profiling_data, 1000, max_steps, is_random=False, is_all_cloud=True)
        cloud_energy.append(ce)
        cloud_time.append(ct)

    # Plot Energy vs Deadline
    plt.figure(figsize=(8, 6))
    plt.plot(deadlines, dq_energy, label="Double Q", marker='o')
    plt.plot(deadlines, a2c_energy, label="A2C", marker='*')
    # plt.plot(deadlines, sac_energy, label="SAC", marker='s')
    # plt.plot(deadlines, random_energy, label="Random", marker='+')
    # plt.plot(deadlines, edge_energy, label="All Edge", marker='x')
    # plt.plot(deadlines, cloud_energy, label="All Cloud", marker='^')
    plt.xlabel("Deadline (ms)")
    plt.ylabel("Average Energy (Joules)")
    plt.title("Average Energy vs Deadline")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Plot Energy vs Deadline
    plt.figure(figsize=(8, 6))
    plt.plot(deadlines, dq_deadline_misses, label="Double Q", marker='o')
    plt.plot(deadlines, a2c_deadline_misses, label="A2C", marker='*')
    # plt.plot(deadlines, sac_energy, label="SAC", marker='s')
    # plt.plot(deadlines, random_energy, label="Random", marker='+')
    # plt.plot(deadlines, edge_energy, label="All Edge", marker='x')
    # plt.plot(deadlines, cloud_energy, label="All Cloud", marker='^')
    plt.xlabel("Deadline (ms)")
    plt.ylabel("Deadline Misses")
    plt.title("Deadline Misses vs Deadline")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("deadline_misses.png", dpi=600)  # 600 dpi for publication quality
    plt.show()


    print("Simulations completed.")
    print("Double Q Energy:", dq_energy)
    print("A2C Energy:", a2c_energy)
    print("Coarse-Grained A2C Energy:", cg_a2c_energy)  


# ======================================================
    energies = np.array([dq_energy, a2c_energy, edge_energy, cloud_energy, cg_a2c_energy])

    labels = ['DQ', 'A2C', 'AllU', 'AllC', 'Coarse A2C']

    x = np.arange(len(deadlines))  # the label locations
    width = 0.15  # width of the bars

    plt.figure(figsize=(10, 6))

    # Plot each scheduler's bar, offsetting them by width
    for i in range(len(labels)):
        plt.bar(x + i*width, energies[i], width, label=labels[i])

    # Set x-axis ticks in the middle of each group
    plt.xticks(x + 2*width, [f'Deadline {d} ms' for d in deadlines])

    plt.ylabel("Average Energy (Joules)")
    plt.title("Average Energy for Different Schedulers by Deadline")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    # Save high-resolution figure
    plt.savefig("energy_bargraph.png", dpi=600)  # 600 dpi for publication quality

    plt.show()