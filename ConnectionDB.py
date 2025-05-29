import sqlite3
import numpy as np
from typing import List, Tuple, Sequence

DB_PATH = "computed_data/features.db"      # Path to the SQLite database
WORKING_TABLE = "FeaturesExtended"         # Working table name for MFCC features

"""
TABLES NAMING CONVENTION:
[FRAME_LENGTH][HOP_LENGTH]_[DELTA_TYPE]_[ADDITIONAL_INFO]
"""
WORKING_TABLE_EXTERN    = "fr20h10_nodelta_extern"                   # 20ms frame, 10ms hop, no delta features, no CMVN, external dataset
WORKING_TABLE0          = "fr20h10_nodelta"                          # 20ms frame, 10ms hop, no delta features, no CMVN
WORKING_TABLE1          = "fr20h10_nodelta_noprimus"                 # 20ms frame, 10ms hop, no delta features, no CMVN, no primus

def insert_in_DB(
    records: Sequence[Tuple[str, str, int, List[float]]]
):
    """
    Inserts multiple MFCC frames into WORKING_TABLE0 in one batch.

    Args:
      records: an iterable of tuples
        (song_name, song_genre, classification, mfcc_values)
        where mfcc_values is a list of exactly 13 floats
    """
    # Build the parameter tuples
    params = []
    for song_name, song_genre, classification, mfcc_values in records:
        if len(mfcc_values) != 13:
            raise ValueError("Each mfcc_values must have 13 floats.")
        # flatten into one tuple of length 16
        params.append((song_name, song_genre, classification, *mfcc_values))

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # turn off autocommit so all INSERTs live in a single transaction
        cur.execute("BEGIN")
        cur.executemany(f"""
            INSERT INTO {WORKING_TABLE0} (
                SONG_NAME,
                SONG_GENRE,
                CLASSIFICATION,
                MFCC0, MFCC1, MFCC2, MFCC3, MFCC4,
                MFCC5, MFCC6, MFCC7, MFCC8,
                MFCC9, MFCC10, MFCC11, MFCC12
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?
            );
        """, params)
        conn.commit()
        print(f"Inserted {len(params)} rows in one transaction")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"SQLite error during bulk insert: {e}")
    finally:
        conn.close()


def reset_DB():
    """
    Resets the database by removing all rows from WORKING_TABLE
    and resetting the sqlite_sequence entry for this table.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # Remove all rows
        cur.execute(f"DELETE FROM {WORKING_TABLE1};")
        # Reset the sqlite_sequence entry for this table
        cur.execute(f"DELETE FROM sqlite_sequence WHERE name = '{WORKING_TABLE1}';")
        conn.commit()
        print("Database reset successfully.")
    except sqlite3.Error as e:
        print(f"SQLite error during database reset: {e}")
    finally:
        conn.close()

def create_table_features_extended():
    """
    Creates the WORKING_TABLE0 table in the database.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {WORKING_TABLE0} (
        SONG_NAME      TEXT,
        SONG_GENRE     TEXT,
        CLASSIFICATION TEXT,
        SEGMENT_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
        MFCC0          REAL, MFCC1  REAL, MFCC2  REAL,
        MFCC3          REAL, MFCC4  REAL, MFCC5  REAL,
        MFCC6          REAL, MFCC7  REAL, MFCC8  REAL,
        MFCC9          REAL, MFCC10 REAL, MFCC11 REAL,
        MFCC12         REAL
        );
        """)
        conn.commit()
        print("Table created successfully.")
    except sqlite3.Error as e:
        print(f"SQLite error during table creation: {e}")
    finally:
        conn.close()

def create_table_features_extended_zscoring():
    """
    Creates the FeaturesExtendedZScroing table in the database.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {WORKING_TABLE1} (
        SONG_NAME      TEXT,
        SONG_GENRE     TEXT,
        CLASSIFICATION TEXT,
        SEGMENT_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
        MFCC0          REAL, MFCC1  REAL, MFCC2  REAL,
        MFCC3          REAL, MFCC4  REAL, MFCC5  REAL,
        MFCC6          REAL, MFCC7  REAL, MFCC8  REAL,
        MFCC9          REAL, MFCC10 REAL, MFCC11 REAL,
        MFCC12         REAL
        );
        """)
        conn.commit()
        print("Table created successfully.")
    except sqlite3.Error as e:
        print(f"SQLite error during table creation: {e}")
    finally:
        conn.close()

def compute_and_save_zscored_mfcc(
    source_table: str = "FeaturesExtended",
    target_table: str = "FeaturesExtendedZScoring"
):
    """
    Reads all MFCC rows from `source_table`, applies Z-Scoring
    to MFCC0…MFCC12, and writes the normalized rows into `target_table`.
    """
    print(f"Computing Z-Scored MFCCs from {source_table} to {target_table}...")
    
    # Load existing data
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"""
      SELECT SONG_NAME, SONG_GENRE, CLASSIFICATION,
             MFCC0, MFCC1, MFCC2, MFCC3, MFCC4,
             MFCC5, MFCC6, MFCC7, MFCC8, MFCC9,
             MFCC10, MFCC11, MFCC12
      FROM {source_table};
    """)
    rows = cur.fetchall()
    if not rows:
        print("No data found in source table.")
        conn.close()
        return
    
    # Separate metadata vs MFCC matrix
    metadata = [(r[0], r[1], r[2]) for r in rows]
    mfcc_matrix = np.array([r[3:] for r in rows], dtype=float)

    # Compute per-column mean & std
    means = mfcc_matrix.mean(axis=0)
    stds  = mfcc_matrix.std(axis=0, ddof=0)
    # avoid divide-by-zero
    stds[stds == 0] = 1.0

    # Apply Z-Score
    zscored = (mfcc_matrix - means) / stds

    # Prepare insert parameters
    params = []
    for (song, genre, cls), mfcc_vals in zip(metadata, zscored):
        params.append((song, genre, cls, *mfcc_vals.tolist()))

    # Bulk-insert
    col_list = ",".join([
        "SONG_NAME", "SONG_GENRE", "CLASSIFICATION",
        "MFCC0","MFCC1","MFCC2","MFCC3","MFCC4",
        "MFCC5","MFCC6","MFCC7","MFCC8",
        "MFCC9","MFCC10","MFCC11","MFCC12"
    ])
    placeholders = ",".join(["?"] * 16)
    sql = f"INSERT INTO {target_table} ({col_list}) VALUES ({placeholders});"

    try:
        cur.execute("BEGIN")
        cur.executemany(sql, params)
        conn.commit()
        print(f"Inserted {len(params)} z-scored rows into {target_table}.")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Error during bulk insert: {e}")
    finally:
        conn.close()

def upload_data_from_db() -> List[Tuple[str, str, int, float, float, float, float, float, float, float, float, float, float, float]]:
    """
    Uploads data from the specified table in the database.
    """
    print(f"Uploading data from table {WORKING_TABLE0}...")
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(f"""
                SELECT 
                "SONG_NAME", "SONG_GENRE", "CLASSIFICATION",
                "MFCC0",    "MFCC1",    "MFCC2",    "MFCC3",    "MFCC4",
                "MFCC5",    "MFCC6",    "MFCC7",    "MFCC8",
                "MFCC9",    "MFCC10",   "MFCC11",   "MFCC12"
                FROM {WORKING_TABLE0};
                """)
    rows = cur.fetchall()
    conn.close()
    print(f"Uploaded {len(rows)} rows from {WORKING_TABLE0}.")
    return rows

def check_table_empty(table_name: str):
    """
    Checks if the specified table is empty an resets it if needed.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cur.fetchone()[0]
    conn.close()
    if count != 0:
        print(f"Table {table_name} is not empty. Do you want to reset it? (y/n)")
        answer = input().strip().lower()
        if answer == 'y':
            print(f"Resetting table {table_name}...")
            reset_DB()
        else:
            print("Table not reset. Exiting.")
    pass