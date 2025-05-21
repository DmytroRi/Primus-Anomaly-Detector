import sqlite3

DB_PATH = "clustering/Database/features.db"     # Path to the SQLite database

def insert_in_DB(
    song_name: str,
    song_genre: str,
    classification: int,
    mfcc_values: list
):
    """
    Inserts a new feature row into the FeaturesExtended table.

    Args:
        song_name (str): Name of the song.
        song_genre (str): Genre of the song.
        classification (int): Classification result (e.g., 1 or 0).
        mfcc_values (list): A list of 13 MFCC float values [MFCC0, ..., MFCC12].
    """

    if len(mfcc_values) != 13:
        raise ValueError("Exactly 13 MFCC values are required.")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO FeaturesExtended (
                SONG_NAME, SONG_GENRE, CLASSIFICATION,
                MFCC0, MFCC1, MFCC2, MFCC3, MFCC4,
                MFCC5, MFCC6, MFCC7, MFCC8,
                MFCC9, MFCC10, MFCC11, MFCC12
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            song_name, song_genre, classification,
            *mfcc_values
        ))

        conn.commit()
    
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    
    finally:
        if conn:
            conn.close()


#-- Remove all rows
# DELETE FROM FeaturesExtended;
# 
# -- Reset the sqlite_sequence entry for this table
# DELETE FROM sqlite_sequence
#  WHERE name = 'FeaturesExtended';