from profiling.initialize_profiling import get_profiling_data
import matplotlib.pyplot as plt
from reference_schedulers.random_scheduler import run_random_scheduler
from simulator.doubleQ_simulator import run_simulation


if __name__ == "__main__":
    episodes = 10000   
    max_steps = 20
    deadlines = list(range(100, 300, 2))

    dq_energy, dq_time = [], []
    random_energy, random_time = [], []
    edge_energy, edge_time = [], []
    cloud_energy, cloud_time = [], []

    for d in deadlines:
        profiling_data = get_profiling_data(d)

        e, t = run_simulation(profiling_data, episodes, max_steps)
        dq_energy.append(e)
        dq_time.append(t)

        re, rt = run_random_scheduler(profiling_data, episodes, max_steps, is_random=True, is_all_cloud=False)
        random_energy.append(re)
        random_time.append(rt)

        ee, et = run_random_scheduler(profiling_data, episodes, max_steps, is_random=False, is_all_cloud=False)
        edge_energy.append(ee)
        edge_time.append(et)

        ce, ct = run_random_scheduler(profiling_data, episodes, max_steps, is_random=False, is_all_cloud=True)
        cloud_energy.append(ce)
        cloud_time.append(ct)

    # Plot Energy vs Deadline
    plt.figure(figsize=(8, 6))
    plt.plot(deadlines, dq_energy, label="Double Q")
    plt.plot(deadlines, random_energy, label="Random")
    plt.plot(deadlines, edge_energy, label="All Edge")
    plt.plot(deadlines, cloud_energy, label="All Cloud")
    plt.xlabel("Deadline (ms)")
    plt.ylabel("Average Energy (Joules)")
    plt.title("Average Energy vs Deadline")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Completion Time vs Deadline
    plt.figure(figsize=(8, 6))
    plt.plot(deadlines, dq_time, label="Double Q")
    plt.plot(deadlines, random_time, label="Random")
    plt.plot(deadlines, edge_time, label="All Edge")
    plt.plot(deadlines, cloud_time, label="All Cloud")
    plt.xlabel("Deadline (ms)")
    plt.ylabel("Average Completion Time (ms)")
    plt.title("Average Completion Time vs Deadline")
    plt.legend()
    plt.grid(True)
    plt.show()
