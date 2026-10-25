import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = BASE_DIR / "data" / "bodyfit_submissions.sqlite3"


def save_submission(
    *,
    database_path: Path | str | None = None,
    sex: str,
    age_years: int,
    height_cm: float,
    weight_kg: float,
    waist_cm: float,
    sleep_hours: float,
    physical_activity: str,
    bmi: float,
    body_fat_pct: float,
    model_name: str,
) -> str:
    path = Path(
        database_path
        or os.environ.get("BODYFIT_DB_PATH")
        or DEFAULT_DATABASE_PATH
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    submission_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                sex TEXT NOT NULL,
                age_years INTEGER NOT NULL,
                height_cm REAL NOT NULL,
                weight_kg REAL NOT NULL,
                waist_cm REAL NOT NULL,
                sleep_hours REAL NOT NULL,
                physical_activity TEXT NOT NULL,
                bmi REAL NOT NULL,
                body_fat_pct REAL NOT NULL,
                model_name TEXT NOT NULL,
                pose_validated INTEGER NOT NULL CHECK (pose_validated = 1)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO submissions (
                id, created_at, sex, age_years, height_cm, weight_kg,
                waist_cm, sleep_hours, physical_activity, bmi, body_fat_pct,
                model_name, pose_validated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                submission_id,
                created_at,
                sex,
                age_years,
                height_cm,
                weight_kg,
                waist_cm,
                sleep_hours,
                physical_activity,
                bmi,
                body_fat_pct,
                model_name,
            ),
        )

    os.chmod(path, 0o600)
    return submission_id
