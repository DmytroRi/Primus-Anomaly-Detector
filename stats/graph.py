import numpy as np
import matplotlib.pyplot as plt

# Data
seg_lengths = ['15 s', '5 s', '1 s']
methods = ['Rough', 'Z-scoring', 'Deltas']
means = np.array([
    [23.31, 33.66, 33.59],
    [22.14, 33.46, 33.07],
    [15.75, 33.47, 33.72]
])  # rows: segment lengths, cols: methods

# Replicates for standard deviation
reps = np.array([
    [[23.63, 23.44, 22.85], [32.03, 33.40, 35.55], [31.64, 34.96, 34.18]],
    [[21.48, 22.66, 22.27], [34.57, 32.03, 33.79], [33.40, 32.62, 33.20]],
    [[15.62, 15.62, 16.02], [34.38, 30.47, 35.55], [31.45, 34.57, 35.16]]
])  # (3 segment lengths, 3 methods, 3 replicates)
errors = reps.std(axis=2)

# Plot grouped by method
x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots()
for i, sl in enumerate(seg_lengths):
    ax.bar(x + (i - 1) * width, means[i], width, yerr=errors[i], label=sl, capsize=5)

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel('Purity (%)')
ax.set_xlabel('Method')
ax.set_title('Clustering Purity by Method and Segmentation Length')
ax.legend(title='Segmentation Length')
plt.tight_layout()
plt.show()
