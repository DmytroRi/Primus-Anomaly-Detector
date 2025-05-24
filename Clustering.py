
import numpy as np
import ConnectionDB as DB
from sklearn.model_selection import train_test_split

TESTING_RATIO = 0.2

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

    X_train, X_test, y_train, y_test, \
    names_train, names_test, \
    genres_train, genres_test = train_test_split(
    features,
    labels,
    song_names,
    song_genres,
    test_size=TESTING_RATIO,  
    stratify=labels,            # keep distribution of classes
    random_state=42
    )


    print("Data split completed.")
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    pass