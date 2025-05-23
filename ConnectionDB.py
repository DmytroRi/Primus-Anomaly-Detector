import sqlite3
from typing import List, Tuple, Sequence

DB_PATH = "clustering/Database/features.db"     # Path to the SQLite database

def insert_in_DB(
    records: Sequence[Tuple[str, str, int, List[float]]]
):
    """
    Inserts multiple MFCC frames into FeaturesExtended in one batch.

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
        cur.executemany("""
            INSERT INTO FeaturesExtended (
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
    Resets the database by removing all rows from FeaturesExtended
    and resetting the sqlite_sequence entry for this table.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # Remove all rows
        cur.execute("DELETE FROM FeaturesExtended;")
        # Reset the sqlite_sequence entry for this table
        cur.execute("DELETE FROM sqlite_sequence WHERE name = 'FeaturesExtended';")
        conn.commit()
        print("Database reset successfully.")
    except sqlite3.Error as e:
        print(f"SQLite error during database reset: {e}")
    finally:
        conn.close()