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
WORKING_TABLE_EXTERN_EXT= "fr20h10_delta_deltadelta_extern"          # 20ms frame, 10ms hop, delta features, external dataset
WORKING_TABLE0          = "fr20h10_nodelta"                          # 20ms frame, 10ms hop, no delta features, no CMVN
WORKING_TABLE1          = "fr20h10_nodelta_noprimus"                 # 20ms frame, 10ms hop, no delta features, no CMVN, no primus
###########################################################
## Table versions 2
FT_TABLE_F4096_H1024    = "f4096h1024"                              # 4096 samples frame, 1024 samples hop

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
    for song_name, song_genre, classification, features in records:
        if len(features) != 52:
            raise ValueError("Each mfcc_values must have 52 floats.")
        # flatten into one tuple of length 16
        params.append((song_name, song_genre, classification, *features))

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

def insert_extended_in_DB(
    records: Sequence[Tuple[str, str, int, List[float]]]
):
    """
    Inserts multiple MFCC frames into WORKING_TABLE_EXTERN_EXT in one batch.

    Args:
      records: an iterable of tuples
        (song_name, song_genre, classification, mfcc_values)
        where mfcc_values is a list of exactly 13 floats
    """
    # Build the parameter tuples
    params = []
    for song_name, song_genre, classification, features_values in records:
        if len(features_values) != 39:
            raise ValueError("Each features_values must have 39 floats.")
        # flatten into one tuple of length 42
        params.append((song_name, song_genre, classification, *features_values))

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # turn off autocommit so all INSERTs live in a single transaction
        cur.execute("BEGIN")
        cur.executemany(f"""
            INSERT INTO {WORKING_TABLE_EXTERN_EXT} (
                SONG_NAME,
                SONG_GENRE,
                CLASSIFICATION,
                MFCC0, MFCC1, MFCC2, MFCC3, MFCC4,
                MFCC5, MFCC6, MFCC7, MFCC8,
                MFCC9, MFCC10, MFCC11, MFCC12,
                DELTA0, DELTA1, DELTA2, DELTA3, DELTA4,
                DELTA5, DELTA6, DELTA7, DELTA8,
                DELTA9, DELTA10, DELTA11, DELTA12,
                DELTA_DELTA0, DELTA_DELTA1, DELTA_DELTA2,
                DELTA_DELTA3, DELTA_DELTA4, DELTA_DELTA5,
                DELTA_DELTA6, DELTA_DELTA7, DELTA_DELTA8,
                DELTA_DELTA9, DELTA_DELTA10, DELTA_DELTA11,
                DELTA_DELTA12
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
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

def create_table_features_extended_deltas():
    """
    Creates the WORKING_TABLE_EXTERN_EXT table in the database.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {WORKING_TABLE_EXTERN_EXT} (
        SONG_NAME      TEXT,
        SONG_GENRE     TEXT,
        CLASSIFICATION TEXT,
        SEGMENT_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
        MFCC0          REAL, MFCC1  REAL, MFCC2  REAL,
        MFCC3          REAL, MFCC4  REAL, MFCC5  REAL,
        MFCC6          REAL, MFCC7  REAL, MFCC8  REAL,
        MFCC9          REAL, MFCC10 REAL, MFCC11 REAL,
        MFCC12         REAL,
        DELTA0         REAL, DELTA1 REAL, DELTA2 REAL,
        DELTA3         REAL, DELTA4 REAL, DELTA5 REAL,
        DELTA6         REAL, DELTA7 REAL, DELTA8 REAL,
        DELTA9         REAL, DELTA10 REAL, DELTA11 REAL,
        DELTA12        REAL,
        DELTA_DELTA0    REAL, DELTA_DELTA1 REAL, DELTA_DELTA2 REAL,
        DELTA_DELTA3    REAL, DELTA_DELTA4 REAL, DELTA_DELTA5 REAL,
        DELTA_DELTA6    REAL, DELTA_DELTA7 REAL, DELTA_DELTA8 REAL,
        DELTA_DELTA9    REAL, DELTA_DELTA10 REAL, DELTA_DELTA11 REAL,
        DELTA_DELTA12   REAL
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

def upload_data_from_db() -> List[Tuple[str, str, int, float, float, float, float, float, float, float, float, float, float, float, float, float]]:
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


def upload_data_from_ext_db() -> List[Tuple[str, str, int, float, float, float, float, float, float, float, float, float, float, float, float, float,
                                                           float, float, float, float, float, float, float, float, float, float, float, float, float,
                                                           float, float, float, float, float, float, float, float, float, float, float, float, float]]:    
    """
    Uploads data from the specified table in the database.
    """
    print(f"Uploading data from table {WORKING_TABLE_EXTERN_EXT}...")
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(f"""
                SELECT 
                "SONG_NAME", "SONG_GENRE", "CLASSIFICATION",
                "MFCC0",    "MFCC1",    "MFCC2",    "MFCC3",    "MFCC4",
                "MFCC5",    "MFCC6",    "MFCC7",    "MFCC8",
                "MFCC9",    "MFCC10",   "MFCC11",   "MFCC12",
                "DELTA0",   "DELTA1",   "DELTA2",   "DELTA3",   "DELTA4",
                "DELTA5",   "DELTA6",   "DELTA7",   "DELTA8",
                "DELTA9",   "DELTA10",  "DELTA11",  "DELTA12",
                "DELTA_DELTA0", "DELTA_DELTA1", "DELTA_DELTA2",
                "DELTA_DELTA3", "DELTA_DELTA4", "DELTA_DELTA5",
                "DELTA_DELTA6", "DELTA_DELTA7", "DELTA_DELTA8",
                "DELTA_DELTA9", "DELTA_DELTA10", "DELTA_DELTA11",
                "DELTA_DELTA12"
                FROM {WORKING_TABLE_EXTERN_EXT};
                """)
    rows = cur.fetchall()
    conn.close()
    print(f"Uploaded {len(rows)} rows from {WORKING_TABLE_EXTERN_EXT}.")
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

def create_table_features_v2():
    """
    Creates the FeaturesExtendedV2 table in the database.
    This table is used for storing MFCC features with additional metadata.
    """
    columns = [
        "SONG_NAME",
        "SONG_GENRE",
        "CLASSIFICATION",
    ]

    for i in range(20):
        columns.append(f"MFCC{i}_mean REAL")
        columns.append(f"MFCC{i}_var  REAL")

    extras = [
        "SPECTRAL_CENTROID",
        "SPECTRAL_BANDWIDTH",
        "SPECTRAL_ROLLOFF",
        "RMSE",
        "ZCR",
        "TEMPO"
    ]
    for feat in extras:
        columns.append(f"{feat}_mean")
        columns.append(f"{feat}_var")
    
    cols_sql = ",\n    ".join(columns)
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {FT_TABLE_F4096_H1024} (
        {cols_sql}
    );
    """

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error creating table '{FT_TABLE_F4096_H1024}': {e}")
        conn.close()
        return

    

def insert_features_V2(records: Sequence[Tuple[str, str, int, List[float]]]):
    """Inserts multiple MFCC frames into FT_TABLE_F4096_H1024 in one batch."""
    # Build the parameter tuples
    params = []
    for song_name, song_genre, classification, features in records:
        if len(features) != 52:
            raise ValueError("Each features size must have 52 floats.")
        # flatten into one tuple of length 55
        params.append((song_name, song_genre, classification, *features))
    
    columns = [
        "SONG_NAME",
        "SONG_GENRE",
        "CLASSIFICATION",
    ]

    for i in range(20):
        columns.append(f"MFCC{i}_mean REAL")
        columns.append(f"MFCC{i}_var  REAL")

    extras = [
        "SPECTRAL_CENTROID",
        "SPECTRAL_BANDWIDTH",
        "SPECTRAL_ROLLOFF",
        "RMSE",
        "ZCR",
        "TEMPO"
    ]
    for feat in extras:
        columns.append(f"{feat}_mean")
        columns.append(f"{feat}_var")

    placeholders = ",".join(["?"] * len(columns))
    columns = ",".join(columns)

    sql = f"""INSERT INTO {FT_TABLE_F4096_H1024} ({columns})VALUES ({placeholders});"""

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("BEGIN")
        cur.executemany(columns, params)
        conn.commit()
        print(f"Inserted {len(params)} rows in one transaction")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"SQLite error during bulk insert: {e}")
    finally:
        conn.close()