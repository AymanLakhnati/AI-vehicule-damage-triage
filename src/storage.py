import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Storage:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    corrected_labels TEXT NOT NULL,
                    approved_labels TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS bookings (
                    id TEXT PRIMARY KEY,
                    partner_id TEXT NOT NULL,
                    city TEXT NOT NULL,
                    contact_name TEXT NOT NULL,
                    contact_phone TEXT NOT NULL,
                    preferred_time TEXT NOT NULL,
                    assessment_id TEXT,
                    location_consent INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS assessments (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def add_feedback(self, record):
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO feedback (id, sha256, corrected_labels, status) VALUES (?, ?, ?, ?)",
                (record["id"], record["sha256"], record["corrected_labels"], record["status"]),
            )

    def list_feedback(self, status=None):
        query = "SELECT * FROM feedback"
        parameters = ()
        if status:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY created_at, id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def review_feedback(self, feedback_id, status, approved_labels=None):
        with self._connection() as connection:
            connection.execute(
                "UPDATE feedback SET status = ?, approved_labels = COALESCE(?, approved_labels) WHERE id = ?",
                (status, approved_labels, feedback_id),
            )

    def add_booking(self, record):
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO bookings
                (id, partner_id, city, contact_name, contact_phone, preferred_time, assessment_id, location_consent, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["id"], record["partner_id"], record["city"], record["contact_name"],
                    record["contact_phone"], record["preferred_time"], record["assessment_id"],
                    int(record["location_consent"]), record["status"],
                ),
            )

    def list_bookings(self):
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM bookings ORDER BY created_at, id").fetchall()
        return [dict(row) for row in rows]

    def add_assessment(self, assessment_id, payload):
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO assessments (id, payload) VALUES (?, ?)",
                (assessment_id, json.dumps(payload)),
            )

    def get_assessment(self, assessment_id):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, payload, created_at FROM assessments WHERE id = ?",
                (assessment_id,),
            ).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "created_at": row["created_at"], **json.loads(row["payload"])}
