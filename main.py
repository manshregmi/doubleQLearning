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
    episodes = 1000
    # episodes = 1
    max_steps = 10
    deadlines = list(range(400,405,50 ))  # 400ms to 500ms
    
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

        e, t, dm, dq_episode_computation_time = run_simulation(profiling_data, episodes, max_steps, is_test)
        dq_energy.append(e)
        dq_time.append(t)
        dq_deadline_misses.append(dm)

        a2c_e, a2c_t, a2c_dm, a2c_episode_computation_time = run_a2c_simulation(profiling_data, episodes, max_steps, is_test)
        a2c_energy.append(a2c_e)
        a2c_time.append(a2c_t)
        a2c_deadline_misses.append(a2c_dm)

        cg_a2c_e, cg_a2c_t, cg_a2c_dm, cg_a2c_episode_computation_time = run_a2c_episode_level(profiling_data, episodes, max_steps, is_test)
        cg_a2c_energy.append(cg_a2c_e)
        cg_a2c_time.append(cg_a2c_t)
        cg_a2c_deadline_misses.append(cg_a2c_dm)


print("Double Q-Learning Results:")
print("computation times per episode:", np.mean(dq_episode_computation_time), "standard deviation: ", np.std(dq_episode_computation_time), "min: ", np.min(dq_episode_computation_time), "max: ", np.max(dq_episode_computation_time))


print("A2C Results:")
print("computation times per episode:", np.mean(a2c_episode_computation_time), "standard deviation: ", np.std(a2c_episode_computation_time), "min: ", np.min(a2c_episode_computation_time), "max: ", np.max(a2c_episode_computation_time))

print("Coarse-Grained A2C Results:")
print("computation times per episode:", np.mean(cg_a2c_episode_computation_time), "standard deviation: ", np.std(cg_a2c_episode_computation_time), "min: ", np.min(cg_a2c_episode_computation_time), "max: ", np.max(cg_a2c_episode_computation_time))


