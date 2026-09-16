import sqlite3
from datetime import datetime


DATABASE_NAME = "compliance.db"


def get_connection():
    """Create a connection to the SQLite database."""
    return sqlite3.connect(DATABASE_NAME)


def create_tables():
    """Create the scan history table if it doesn't exist."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT,
            mrp TEXT,
            net_quantity TEXT,
            manufacturer TEXT,
            address TEXT,
            overall_status TEXT,
            scanned_at TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_scan(product_data, compliance_result):
    """Save a completed scan to the database."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO scan_history (
            product_name,
            mrp,
            net_quantity,
            manufacturer,
            address,
            overall_status,
            scanned_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        product_data.get("product_name"),
        product_data.get("mrp"),
        product_data.get("net_quantity"),
        product_data.get("manufacturer"),
        product_data.get("address"),
        compliance_result.get("overall_status"),
        datetime.now().isoformat()
    ))

    connection.commit()
    connection.close()


def get_scan_history():
    """Return all previous scans."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM scan_history
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    connection.close()

    return records