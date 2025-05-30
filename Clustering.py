
import time
import numpy as np
import ConnectionDB as DB
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
from annoy import AnnoyIndex
from collections import defaultdict

K_MAX = 70
N_TREES = 20            # Number of trees for Annoy index
METRIC = 'euclidean'    # 'angular', 'euclidean', 'manhattan', 'hamming', 'dot'
TESTING_RATIO = 0.2
NEIGHBOURS = 5

def combine_frames(rows, duration_ms, hop_ms):
    """Groups tiny frames into larger segments and computes mean+std features."""
    frames_per_seg = duration_ms // hop_ms
    by_song = defaultdict(list)
    genres_map = {}
    for row in rows:
        song, genre = row[0], row[1]
        mfcc = row[3:]
        by_song[song].append(mfcc)
        genres_map[song] = genre
    feats_list, genre_list = [], []
    for song, feats in by_song.items():
        arr = np.array(feats, dtype=float)
        n_frames = arr.shape[0]
        n_segs = int(np.ceil(n_frames / frames_per_seg))
        for i in range(n_segs):
            seg = arr[i*frames_per_seg:(i+1)*frames_per_seg]
            if seg.size == 0: continue
            mean = seg.mean(axis=0)
            std  = seg.std(axis=0)
            feats_list.append(np.hstack([mean, std]))
            genre_list.append(genres_map[song])
    features = np.vstack(feats_list)
    genres  = genre_list

    return features, genres

def visualize_embedding(rows, method='pca'):
    """Visulalizes 2D embedding of MFCC features using PCA or t-SNE."""
    
    features, genres = combine_frames(rows, duration_ms=30000, hop_ms=30000)

    le = LabelEncoder().fit(genres)
    y  = le.transform(genres)

    if method == 'pca':
        emb = PCA(n_components=2).fit_transform(features)
    elif method == 'tsne':
        emb = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(features)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    pfig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
    handles, _ = scatter.legend_elements()
    genre_labels = list(le.classes_)
    ax.legend(
        handles=list(handles),
        labels=genre_labels,
        title="Genre",
        bbox_to_anchor=(1, 1),
        loc="upper left"
    )
    plt.title(f"{method.upper()} projection of MFCC features")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()

def split_data():
    """Loads rows from DB, extracts features & labels, splits, and scales."""
    rows = DB.upload_data_from_db()
    # rows: (song_name, song_genre, dummy_class, mfcc0…mfcc12)

    visualize_embedding(rows)

    raw_genres = [r[1] for r in rows]
    features   = np.array([r[3:] for r in rows], dtype=np.float32)

    le = LabelEncoder()
    y  = le.fit_transform(raw_genres)

    X_train, X_test, y_train, y_test, \
    names_train, names_test, genres_train, genres_test = train_test_split(
        features, y, [r[0] for r in rows], raw_genres,
        test_size=TESTING_RATIO,
        stratify=y,
        random_state=42
    )

    # scale features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"Split data into {len(X_train)} training and {len(X_test)} testing samples.")
    return (X_train, X_test, y_train, y_test,
            names_test, genres_test, le)

def build_annoy_index(X_train, n_trees=N_TREES, metric=METRIC):
    """Builds and returns an AnnoyIndex on X_train."""
    d   = X_train.shape[1]
    idx = AnnoyIndex(d, metric)
    print(f"Building Annoy index with {n_trees} trees…")
    t0 = time.time()
    for i, vec in enumerate(X_train):
        idx.add_item(i, vec.tolist())
    idx.build(n_trees)
    print(f"Index built in {time.time() - t0:.1f}s\n")
    return idx

def knn_with_annoy(idx, X_test, y_train, k):
    """
    Queries idx for each vector in X_test with k neighbors,
    returns predicted labels and query time.
    """
    N = len(X_test)
    y_pred = np.empty(N, dtype=y_train.dtype)
    t0 = time.time()
    for j, q in enumerate(X_test):
        nbrs = idx.get_nns_by_vector(q.tolist(), k)
        y_pred[j] = Counter(y_train[i] for i in nbrs).most_common(1)[0][0]
    return y_pred, time.time() - t0

def evaluate_best_k(idx, X_test, y_train, y_test, k_max=K_MAX):
    """
    Loops k = 1..k_max, runs knn_with_annoy, prints accuracy & time,
    and returns (best_k, best_acc, y_best).
    """
    best_k, best_acc, y_best = 1, 0.0, None
    for k in range(1, k_max + 1):
        y_pred, dt = knn_with_annoy(idx, X_test, y_train, k)
        acc = accuracy_score(y_test, y_pred)
        print(f"k={k:2d} -> acc={acc:.4f}  time={dt:.2f}s")
        if acc > best_acc:
            best_acc, best_k, y_best = acc, k, y_pred
    print(f"\nBest k = {best_k} with accuracy {best_acc:.4f}\n")
    return best_k, best_acc, y_best

def output_detailed_report(y_best, y_test, le, names_test, genres_test):
    """Prints classification report, confusion matrix and one sample mismatch."""
    print("Classification Report:")
    print(classification_report(y_test, y_best, target_names=le.classes_))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_best)
    print(cm)
    plot_confusion_matrix(cm, classes=le.classes_)

    # show first misclassified example
    for name, true_lab, pred_lab in zip(names_test, y_test, y_best):
        if true_lab != pred_lab:
            print(f"First misclassified: {name} -> true={le.classes_[true_lab]}, pred={le.classes_[pred_lab]}")
            break

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """Plots a confusion matrix."""

    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, where=row_sums!=0) * 100

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Percentage of true class (%)')

    # Set up axes
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Annotate each cell with count and percent
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_percent[i, j]
            label = f"{pct:.2f}%"
            # Use a threshold based on the max percentage in the row for text color contrast
            text_color = "white" if pct > (cm_percent[i].max() / 2) else "black"
            ax.text(j, i, label,
                    ha="center", va="center",
                    color=text_color)

    plt.tight_layout()
    plt.show()

def main():
    print("\n--- Split & Scale Data ---")
    X_train, X_test, y_train, y_test, names_test, genres_test, le = split_data()

    print("\n--- Build Index ---")
    annoy_idx = build_annoy_index(X_train)

    print("\n--- Evaluate k = 1…{K_MAX} ---".format(K_MAX=K_MAX))
    best_k, best_acc, y_best = evaluate_best_k(
        annoy_idx, X_test, y_train, y_test, k_max=K_MAX
    )

    print("\n--- Detailed Report for Best k ---")
    output_detailed_report(y_best, y_test, le, names_test, genres_test)

if __name__ == "__main__":
    main()