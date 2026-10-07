import tempfile
from pathlib import Path
import shutil

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app

video_path = Path(tempfile.gettempdir()) / "hudle-smoke.mp4"
writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (320, 180))
for _ in range(8):
    writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
writer.release()

client = TestClient(app.app)
upload = client.post("/api/games", files={"video": ("smoke.mp4", video_path.open("rb"), "video/mp4")})
game = upload.json()
assert upload.status_code == 200, upload.text
assert client.get(f"/api/games/{game['id']}/frame?i=0").status_code == 200
partial = client.get(f"/api/games/{game['id']}/video", headers={"Range": "bytes=0-99"})
assert partial.status_code == 206 and len(partial.content) == 100
points = [{"px": 0, "py": 0, "fx": 30, "fy": 0}, {"px": 319, "py": 0, "fx": 30, "fy": 53.33}, {"px": 0, "py": 179, "fx": 45, "fy": 0}, {"px": 319, "py": 179, "fx": 45, "fy": 53.33}]
assert client.put(f"/api/games/{game['id']}/calibration", json={"frame": 0, "points": points}).status_code == 200
analysis = client.post("/api/analyze-play", json={"gameId": game["id"], "startTime": 0, "endTime": 0.75})
assert analysis.status_code == 200, analysis.text
assert "error" not in analysis.json()["cv"], analysis.json()["cv"]
assert len(client.get(f"/api/games/{game['id']}/plays").json()) == 1
assert b"Play,Start Time,End Time" in client.get(f"/api/games/{game['id']}/export.csv").content
connection = app.db()
play_id = analysis.json()["id"]
connection.execute("DELETE FROM player_detections WHERE playId=?", (play_id,))
connection.execute("DELETE FROM plays WHERE id=?", (play_id,))
connection.execute("DELETE FROM games WHERE id=?", (game["id"],))
connection.commit()
connection.close()
shutil.rmtree(app.PLAYS / play_id, ignore_errors=True)
for path in (app.UPLOADS / f"{game['id']}.mp4", app.UPLOADS / f"{game['id']}.calibration.json"):
    path.unlink(missing_ok=True)
video_path.unlink(missing_ok=True)
print("API and CV smoke check passed: upload, playback, calibration, full football-cv stage flow, persistence, CSV")
