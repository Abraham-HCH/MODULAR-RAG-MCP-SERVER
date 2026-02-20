"""Initialize ingestion_history SQLite database.

Creates `data/db/ingestion_history.db` and the `ingestion_history` table if not present.
Run: python scripts/init_ingestion_db.py
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "db", "ingestion_history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_history (
    file_hash TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_msg TEXT,
    chunk_count INTEGER
);
CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status);
CREATE INDEX IF NOT EXISTS idx_processed_at ON ingestion_history(processed_at);
"""


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.executescript(SCHEMA)
        conn.commit()
        print(f"Initialized ingestion DB at: {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
