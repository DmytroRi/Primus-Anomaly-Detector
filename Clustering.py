
import numpy as np
import ConnectionDB as DB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

K_MAX = 20
TESTING_RATIO = 0.2
NEIGHBOURS = 5

def compute_knn(
    X_train, X_test,
    y_train, y_test,
    names_test=None,
    genres_test=None,
    n_neighbors: int = NEIGHBOURS,
    algorithm: str = 'auto'
):
    """
    Trains a K-Nearest Neighbors classifier on the training set,
    predicts on the test set, and returns evaluation metrics.

    Args:
        X_train (np.ndarray): shape (n_train, n_features)
        X_test  (np.ndarray): shape (n_test,  n_features)
        y_train (array-like): shape (n_train,)
        y_test  (array-like): shape (n_test,)
        names_test (list[str], optional): song names for each test sample
        genres_test(list[str], optional): genres for each test sample
        n_neighbors (int): number of neighbors to use
        algorithm (str): 'auto', 'ball_tree', 'kd_tree' or 'brute'

    Returns:
        dict: {
            'accuracy': float,
            'classification_report': str,
            'confusion_matrix': np.ndarray,
            'y_pred': np.ndarray,
            'names_test': list[str] (if provided),
            'genres_test': list[str] (if provided)
        }
    """
    print("Computing KNN...")
    # 1) Fit
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    knn.fit(X_train, y_train)

    # 2) Predict
    y_pred = knn.predict(X_test)
    
    # 2.1) Decode labels
    le = LabelEncoder()
    y_pred = le.inverse_transform(y_pred)  # Decode predictions back to original labels

    # 3) Evaluate
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    # 4) Package results
    results = {
        'accuracy': acc,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

    if names_test is not None:
        results['names_test'] = names_test
    if genres_test is not None:
        results['genres_test'] = genres_test

    print("KNN computation completed.")
    return results

def output_results(results):
    """
    Outputs the results of the KNN evaluation.

    Args:
        results (dict): Results from compute_knn function.
    """
    print(f"Accuracy: {results['accuracy']}")
    print("Classification Report:")
    print(results['classification_report'])
    print("Confusion Matrix:")
    print(results['confusion_matrix'])

    if 'names_test' in results:
        print("Test Song Names:")
        print(results['names_test'][:10])  # Print first 10 names
    if 'genres_test' in results:
        print("Test Genres:")
        print(results['genres_test'][:10])  # Print first 10 genres


def split_data():
    """
    Splits the dataset into training, validation, and test sets.
    """
    print("Splitting data into training, validation, and test sets...")
    
    rows = DB.upload_data_from_db()

    song_names  = [r[0] for r in rows]                              # shape (N,)
    song_genres = [r[1] for r in rows]                              # shape (N,)           
    labels = np.array([r[2] for r in rows], dtype=np.int32)         # shape (N,)
    features = np.array([r[3:] for r in rows], dtype=np.float32)    # shape (N, 13)

    le = LabelEncoder()
    y = le.fit_transform(song_genres)                               # Encode labels to integers

    X_train, X_test, y_train, y_test, \
    names_train, names_test, \
    genres_train, genres_test = train_test_split(
    features,
    y,
    song_names,
    song_genres,
    test_size=TESTING_RATIO,  
    stratify=y,            # keep distribution of classes
    random_state=42
    )

    print("Data split completed.")
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples.")

    results = compute_knn(
        X_train, X_test,
        y_train, y_test,
        names_test=names_test,
        genres_test=genres_test
    )
    
    output_results(results)

    pass

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