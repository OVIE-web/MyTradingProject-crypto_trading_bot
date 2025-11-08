import psycopg2
from datetime import datetime
from psycopg2.extras import RealDictCursor

class ModelRegistry:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            dbname="trading",
            user="postgres",
            password="postgres",
        )
        self.create_table()

    def create_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    accuracy FLOAT,
                    params JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            self.conn.commit()

    def register_model(self, model_name, model_path, accuracy, params):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_registry (model_name, model_path, accuracy, params)
                VALUES (%s, %s, %s, %s)
            """, (model_name, model_path, accuracy, json.dumps(params)))
            self.conn.commit()

    def latest(self):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM model_registry ORDER BY id DESC LIMIT 1;")
            record = cur.fetchone()
            if record:
                return record["id"], record["model_path"], record["accuracy"]
            return None

    def get(self, version_id):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM model_registry WHERE id = %s;", (version_id,))
            record = cur.fetchone()
            if record:
                return record["id"], record["model_path"], record["accuracy"]
            return None
