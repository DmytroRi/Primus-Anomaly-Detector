import numpy as np
import matplotlib.pyplot as plt

def KMeans():
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

def KNN():
    # Data
    k_values = list(range(1, 21))
    purity_1s = [1.0000, 0.7660, 0.7241, 0.7241, 0.6872, 0.6773, 0.6798, 0.6626, 0.6478, 0.6305, 0.6108, 0.5936, 0.5788, 0.5739, 0.5394, 0.5271, 0.5099, 0.5049, 0.4975, 0.4975]
    purity_5s = [1.0000, 0.7857, 0.7241, 0.7143, 0.7094, 0.6946, 0.6970, 0.6601, 0.6355, 0.6158, 0.6034, 0.5690, 0.5714, 0.5640, 0.5468, 0.5493, 0.5296, 0.5345, 0.5246, 0.5123]
    purity_15s = [1.0000, 0.8054, 0.7660, 0.7340, 0.6847, 0.6576, 0.6700, 0.6502, 0.6281, 0.6207, 0.6207, 0.6158, 0.5887, 0.5714, 0.5591, 0.5296, 0.5148, 0.5123, 0.5222, 0.5049]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, purity_15s, marker='o', label='15 s')
    plt.plot(k_values, purity_5s, marker='o', label='5 s')
    plt.plot(k_values, purity_1s, marker='o', label='1 s')
    plt.title('k-Nearest Neighbors Purity vs. k')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Purity')
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend(title='Segmentation Length')
    plt.tight_layout()
    plt.show()

def KNN_v1():
    # Data
    k_values = list(range(1, 71))
    precision = [0.3829, 0.3926, 0.4158, 0.4313, 0.4413, 0.4486, 0.4541, 0.4586, 0.4624, 0.4654,
                 0.4679, 0.4702, 0.4720, 0.4739, 0.4753, 0.4766, 0.4777, 0.4787, 0.4795, 0.4804,
                 0.4811, 0.4815, 0.4821, 0.4824, 0.4827, 0.4830, 0.4835, 0.4836, 0.4838, 0.4841,
                 0.4843, 0.4846, 0.4847, 0.4849, 0.4850, 0.4851, 0.4852, 0.4853, 0.4854, 0.4853,
                 0.4854, 0.4856, 0.4855, 0.4855, 0.4856, 0.4856, 0.4857, 0.4857, 0.4855, 0.4855,
                 0.4856, 0.4855, 0.4855, 0.4855, 0.4855, 0.4854, 0.4855, 0.4855, 0.4855, 0.4854,
                 0.4854, 0.4852, 0.4852, 0.4851, 0.4850, 0.4849, 0.4848, 0.4847, 0.4846, 0.4846]
    
     # Find max
    max_prec = max(precision)
    max_k = k_values[precision.index(max_prec)]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, precision, label='Precision')
    plt.plot(max_k, max_prec, 'ro', label=f'Max Precision: {max_prec:.4f} at k={max_k}')
    plt.axhline(y=max_prec, color='r', linestyle='--', label='Max Precision Line')
    plt.title('k-Nearest Neighbors Precision vs. k')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Precision')
    plt.xticks(k_values, rotation=45)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    pass


KNN_v1()