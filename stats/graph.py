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

def MFCC_vs_Deltas():
    k_values = list(range(1, 71))
    precision_nodelta = [0.5649, 0.5836, 0.6026, 0.6153, 0.6211, 0.6263, 0.6281, 0.6304, 0.6306, 0.6312,
                 0.6313, 0.6315, 0.6309, 0.6311, 0.6310, 0.6304, 0.6298, 0.6295, 0.6287, 0.6278,
                 0.6271, 0.6264, 0.6255, 0.6246, 0.6239, 0.6230, 0.6222, 0.6215, 0.6210, 0.6205,
                 0.6195, 0.6189, 0.6182, 0.6176, 0.6167, 0.6160, 0.6151, 0.6148, 0.6143, 0.6134,
                 0.6130, 0.6124, 0.6115, 0.6111, 0.6103, 0.6099, 0.6092, 0.6088, 0.6081, 0.6077,
                 0.6073, 0.6065, 0.6060, 0.6058, 0.6053, 0.6044, 0.6040, 0.6036, 0.6030, 0.6025,
                 0.6020, 0.6015, 0.6007, 0.6003, 0.5999, 0.5994, 0.5988, 0.5986, 0.5981, 0.5977]

    precision_delta = [0.4725, 0.5352, 0.5390, 0.5469, 0.5479, 0.5526, 0.5550, 0.5572, 0.5568, 0.5568,
                 0.5563, 0.5567, 0.5562, 0.5547, 0.5542, 0.5529, 0.5521, 0.5515, 0.5504, 0.5496,
                 0.5484, 0.5474, 0.5465, 0.5455, 0.5444, 0.5437, 0.5424, 0.5416, 0.5408, 0.5397,
                 0.5390, 0.5384, 0.5373, 0.5365, 0.5358, 0.5350, 0.5345, 0.5337, 0.5328, 0.5319,
                 0.5315, 0.5306, 0.5298, 0.5292, 0.5285, 0.5279, 0.5270, 0.5266, 0.5260, 0.5252,
                 0.5248, 0.5243, 0.5236, 0.5230, 0.5226, 0.5220, 0.5214, 0.5208, 0.5206, 0.5201,
                 0.5196, 0.5189, 0.5185, 0.5180, 0.5177, 0.5171, 0.5166, 0.5161, 0.5156, 0.5150]

    # Find max
    max_prec = max(precision_nodelta)
    max_k = k_values[precision_nodelta.index(max_prec)]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, precision_nodelta, label='Precision (no delta)')
    plt.plot(k_values, precision_delta, label='Precision (with delta)')
    plt.plot(max_k, max_prec, 'ro', label=f'Max Precision: {max_prec:.4f} at k={max_k}')
    plt.axhline(y=max_prec, color='r', linestyle='--', label='Max Precision Line')
    plt.title('Precision using MFCCs vs. MFCCs with Deltas')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Precision')
    plt.xticks(k_values, rotation=45)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    pass

def binary_classification():
    # Data
    k_values = list(range(1, 71))
    precision = [0.8885, 0.8911, 0.9017, 0.9074, 0.9103, 0.9130, 0.9120, 0.9141, 0.9128, 0.9146,
                 0.9129, 0.9144, 0.9128, 0.9141, 0.9126, 0.9137, 0.9123, 0.9134, 0.9120, 0.9130,
                 0.9117, 0.9127, 0.9115, 0.9123, 0.9112, 0.9120, 0.9110, 0.9117, 0.9107, 0.9114,
                 0.9104, 0.9111, 0.9102, 0.9108, 0.9099, 0.9106, 0.9097, 0.9103, 0.9095, 0.9101,
                 0.9093, 0.9099, 0.9091, 0.9096, 0.9089, 0.9094, 0.9087, 0.9092, 0.9085, 0.9090,
                 0.9084, 0.9088, 0.9082, 0.9087, 0.9081, 0.9085, 0.9079, 0.9083, 0.9078, 0.9081,
                 0.9076, 0.9080, 0.9075, 0.9079, 0.9073, 0.9077, 0.9072, 0.9076, 0.9071, 0.9074]
    
     # Find max
    max_prec = max(precision)
    max_k = k_values[precision.index(max_prec)]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, precision, label='Precision')
    plt.plot(max_k, max_prec, 'ro', label=f'Max Precision: {max_prec:.4f} at k={max_k}')
    plt.axhline(y=max_prec, color='r', linestyle='--', label='Max Precision Line')
    plt.title('k-Nearest Neighbors Precision vs. k (Binary Classification)')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Precision')
    plt.xticks(k_values, rotation=45)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    pass

def Multiclass_vs_Binary_v2():
    k_values = list(range(1, 31))
    precision_multiclass= [0.5556, 0.5556, 0.5926, 0.5679, 0.5432, 0.5802, 0.5556, 0.5309, 0.4938, 0.5185,
                         0.5062, 0.4938, 0.5309, 0.5185, 0.5062, 0.4938, 0.5062, 0.5185, 0.5185, 0.4815,
                         0.5062, 0.5185, 0.5556, 0.5185, 0.5062, 0.5185, 0.4938, 0.5185, 0.4938, 0.5185]

    precision_binary = [0.8642, 0.8765, 0.9630, 0.9506, 0.9012, 0.9136, 0.8889, 0.9136, 0.9259, 0.9136,
                        0.9012, 0.9136, 0.9259, 0.9012, 0.9136, 0.9259, 0.9259, 0.9136, 0.9259, 0.9136,
                        0.9136, 0.9259, 0.9136, 0.9012, 0.9136, 0.9136, 0.9136, 0.9136, 0.9136, 0.9136]

    # Find max
    max_prec = max(precision_binary)
    max_k = k_values[precision_binary.index(max_prec)]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, precision_multiclass, label='Precision (multiclass)')
    plt.plot(k_values, precision_binary, label='Precision (binary)')
    plt.plot(max_k, max_prec, 'ro', label=f'Max Precision: {max_prec:.4f} at k={max_k}')
    plt.axhline(y=max_prec, color='r', linestyle='--', label='Max Precision Line')
    plt.title('Precision multiclass vs. binary classification')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Precision')
    plt.xticks(k_values, rotation=45)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    pass

def external_v2():
    # Data
    k_values = list(range(1, 31))
    precision = [0.6650, 0.6750, 0.6750, 0.6350, 0.6650, 0.6450, 0.6400, 0.6050, 0.6050, 0.5850,
                 0.5600, 0.5750, 0.5800, 0.5850, 0.5900, 0.5850, 0.5750, 0.5750, 0.5900, 0.5850,
                 0.5750, 0.5650, 0.5650, 0.5600, 0.5700, 0.5650, 0.5600, 0.5650, 0.5700, 0.5700]
    
     # Find max
    max_prec = max(precision)
    max_k = k_values[precision.index(max_prec)]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, precision, label='Precision')
    plt.plot(max_k, max_prec, 'ro', label=f'Max Precision: {max_prec:.4f} at k={max_k}')
    plt.axhline(y=max_prec, color='r', linestyle='--', label='Max Precision Line')
    plt.title('k-Nearest Neighbors Precision vs. k (external dataset)')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Precision')
    plt.xticks(k_values, rotation=45)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    pass


def Multiclass_vs_Binary_v2_adjusted():
    k_values = list(range(1, 31))
    precision_multiclass= [0.5952, 0.5833, 0.6190, 0.6310, 0.6190, 0.5833, 0.5714, 0.5476, 0.5357, 0.5000,
                           0.5238, 0.5119, 0.5476, 0.5476, 0.5119, 0.5476, 0.5476, 0.5476, 0.5238, 0.5119,
                           0.5119, 0.5000, 0.5238, 0.5119, 0.5119, 0.5238, 0.5238, 0.5357, 0.5119, 0.5119]

    precision_binary = [0.9286, 0.9286, 0.9524, 0.9524, 0.9405, 0.9405, 0.9286, 0.9405, 0.9286, 0.9405,
                        0.9286, 0.9405, 0.9405, 0.9405, 0.9405, 0.9405, 0.9048, 0.9286, 0.9286, 0.9286,
                        0.9167, 0.9167, 0.9048, 0.9167, 0.9048, 0.9048, 0.8929, 0.9048, 0.9048, 0.9048]

    # Find max
    max_prec = max(precision_binary)
    max_k = k_values[precision_binary.index(max_prec)]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, precision_multiclass, label='Precision (multiclass)')
    plt.plot(k_values, precision_binary, label='Precision (binary)')
    plt.plot(max_k, max_prec, 'ro', label=f'Max Precision: {max_prec:.4f} at k={max_k}')
    plt.axhline(y=max_prec, color='r', linestyle='--', label='Max Precision Line')
    plt.title('Precision multiclass vs. binary classification')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Precision')
    plt.xticks(k_values, rotation=45)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    pass

Multiclass_vs_Binary_v2_adjusted()