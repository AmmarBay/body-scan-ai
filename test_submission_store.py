import sqlite3
import tempfile
import unittest
from pathlib import Path

from submission_store import save_submission


class SubmissionStoreTests(unittest.TestCase):
    def test_submission_is_persisted_without_a_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "nested" / "submissions.sqlite3"

            submission_id = save_submission(
                database_path=database_path,
                sex="Pria",
                age_years=30,
                height_cm=175,
                weight_kg=75,
                waist_cm=82,
                sleep_hours=7,
                physical_activity="Sedang",
                bmi=75 / (1.75**2),
                body_fat_pct=18.5,
                model_name="test model",
            )

            with sqlite3.connect(database_path) as connection:
                row = connection.execute(
                    """
                    SELECT id, sex, age_years, body_fat_pct, pose_validated
                    FROM submissions
                    """
                ).fetchone()
                columns = {
                    result[1]
                    for result in connection.execute(
                        "PRAGMA table_info(submissions)"
                    )
                }

            self.assertEqual(row, (submission_id, "Pria", 30, 18.5, 1))
            self.assertNotIn("photo", columns)
            self.assertEqual(database_path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
