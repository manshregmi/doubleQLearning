import matplotlib.pyplot as plt

# Threshold values
thresholds = [25, 50, 75, 100, 150, 200, 300]

# Energy values (J)
ternary = [4.653, 3.945, 3.539, 3.718, 3.881, 3.986, 4.272]
ternary_no_threshold = [4.786, 3.866, 3.854, 3.883, 3.940, 3.997, 4.112]
binary_optimistic = [4.752, 3.709, 3.725, 3.750, 3.801, 3.852, 3.954]
binary_conservative = [4.764, 4.357, 3.999, 4.021, 4.066, 4.110, 4.199]

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds, ternary,
    marker='o', linewidth=2,
    label='Ternary A2C'
)

plt.plot(
    thresholds, ternary_no_threshold,
    marker='s', linewidth=2,
    label='Ternary A2C (No Threshold)'
)

plt.plot(
    thresholds, binary_optimistic,
    marker='^', linewidth=2,
    label='Binary A2C (Optimistic)'
)

plt.plot(
    thresholds, binary_conservative,
    marker='D', linewidth=2,
    label='Binary A2C (Conservative)'
)

# Highlight the 75 ms operating point
plt.axvline(
    x=75,
    linestyle='--',
    linewidth=1.5,
    label='75 ms threshold'
)

plt.xlabel('Threshold (ms)', fontsize=12)
plt.ylabel('Energy per Episode (J)', fontsize=12)
plt.title('Energy vs. Timeout Threshold', fontsize=13)

plt.xticks(thresholds)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig(
    'energy_vs_threshold.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()