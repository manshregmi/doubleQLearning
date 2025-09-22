def random_offload_scheduler(layers, nodesInEachLayer):
    schedule = {}
    for layer in range(layers):
        schedule[layer] = []
        for node in range(nodesInEachLayer[layer]):
            decision = random.choice([0, 1])  # 0=edge, 1=cloud
            schedule[layer].append(decision)
    return schedule
