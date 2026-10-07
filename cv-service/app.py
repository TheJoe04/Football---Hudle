from __future__ import annotations

import base64
import csv
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

ROOT = Path(__file__).resolve().parents[1]
CV_REPO = ROOT / "vendor" / "football-cv"
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
PLAYS = DATA / "plays"
DB = DATA / "coachfilm.sqlite3"
sys.path.insert(0, str(CV_REPO))
LOG_DIR = DATA / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("coachfilm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

app = FastAPI(title="Coach Film API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def request_log(request: Request, call_next):
    from time import perf_counter
    started = perf_counter()
    try:
        response = await call_next(request)
        logger.info("request method=%s path=%s status=%s elapsed_ms=%d", request.method, request.url.path, response.status_code, (perf_counter()-started)*1000)
        return response
    except Exception:
        logger.exception("request_failed method=%s path=%s", request.method, request.url.path)
        raise

class Point(BaseModel):
    px: float
    py: float
    fx: float
    fy: float

class Calibration(BaseModel):
    frame: int = 0
    points: list[Point] = Field(min_length=4)

class AnalyzeRequest(BaseModel):
    gameId: str
    startTime: float = Field(ge=0)
    endTime: float = Field(gt=0)
    calibration: Calibration | None = None
    down: int | None = Field(default=None, ge=1, le=4)
    distance: float | None = Field(default=None, ge=0)

class FieldValue(BaseModel):
    value: str | int | float | None
    confidence: float = Field(ge=0, le=1)

class Interpretation(BaseModel):
    unit: FieldValue
    down: FieldValue
    distance: FieldValue
    formation: FieldValue
    play: FieldValue
    gap: FieldValue
    result: FieldValue

PLAY_TYPES = ["Inside Zone", "Outside Zone", "Power", "Counter", "Duo", "Trap", "Sweep", "Draw", "QB Run", "Screen", "Pass", "RPO", "Other", "UNKNOWN"]
GAPS = ["A Left", "A Right", "B Left", "B Right", "C Left", "C Right", "D Left", "D Right", "Outside Left", "Outside Right", "N/A", "UNKNOWN"]
UNITS = ["OFFENSE", "DEFENSE", "SPECIAL_TEAMS", "UNKNOWN"]
SPECIAL = ["Kickoff", "Kick Return", "Punt", "Punt Return", "Field Goal", "PAT", "Field Goal Block", "Punt Block", "UNKNOWN"]
FORMATIONS = ["Trips Right", "Trips Left", "Doubles", "Empty", "I Formation", "Singleback", "Shotgun", "Pistol", "Unknown"]

def db():
    DATA.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE IF NOT EXISTS games(id TEXT PRIMARY KEY, filename TEXT, path TEXT, calibration TEXT, createdAt TEXT DEFAULT CURRENT_TIMESTAMP)")
    c.execute("CREATE TABLE IF NOT EXISTS plays(id TEXT PRIMARY KEY, gameId TEXT, playNumber INTEGER, startTime REAL, endTime REAL, snapTime REAL, unit TEXT, down INTEGER, distance REAL, formation TEXT, play TEXT, gap TEXT, result TEXT, overallConfidence REAL, rawCvOutput TEXT, rawAiOutput TEXT, confirmed INTEGER DEFAULT 0, createdAt TEXT DEFAULT CURRENT_TIMESTAMP, updatedAt TEXT DEFAULT CURRENT_TIMESTAMP)")
    c.execute("CREATE TABLE IF NOT EXISTS player_detections(playId TEXT, trackId INTEGER, team TEXT, positionGroup TEXT, preSnapX REAL, preSnapY REAL)")
    c.commit()
    return c

def read_json(path: Path, default):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return default

@app.get("/api/health")
def health():
    return {"ok": True, "cvRepo": str(CV_REPO), "openaiConfigured": bool(os.getenv("OPENAI_API_KEY"))}

@app.post("/api/games")
async def upload_game(video: UploadFile = File(...)):
    suffix = Path(video.filename or "film.mp4").suffix.lower()
    logger.info("upload_started filename=%r content_type=%r reported_size=%r", video.filename, video.content_type, video.size)
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}:
        logger.warning("upload_rejected filename=%r reason=unsupported_extension", video.filename)
        raise HTTPException(400, "Unsupported video format. Upload MP4, MOV, M4V, AVI, or MKV.")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    game_id = uuid.uuid4().hex[:10]
    path = UPLOADS / f"{game_id}{suffix}"
    try:
        with path.open("wb") as f:
            shutil.copyfileobj(video.file, f, length=4 * 1024 * 1024)
    except Exception:
        path.unlink(missing_ok=True)
        logger.exception("upload_save_failed game_id=%s filename=%r", game_id, video.filename)
        raise HTTPException(500, "The server could not save the uploaded video. Check data/logs/app.log.")
    c = db(); c.execute("INSERT INTO games(id,filename,path) VALUES(?,?,?)", (game_id, video.filename, str(path))); c.commit(); c.close()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release(); path.unlink(missing_ok=True)
        c = db(); c.execute("DELETE FROM games WHERE id=?", (game_id,)); c.commit(); c.close()
        logger.warning("upload_rejected game_id=%s filename=%r reason=opencv_cannot_open", game_id, video.filename)
        raise HTTPException(400, "OpenCV could not read this file. Try an H.264 MP4 or a different video export.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30; frames = cap.get(cv2.CAP_PROP_FRAME_COUNT); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    logger.info("upload_saved game_id=%s filename=%r bytes=%d path=%s duration=%.2f fps=%.3f dimensions=%dx%d", game_id, video.filename, path.stat().st_size, path, frames/fps if fps else 0, fps, width, height)
    return {"id": game_id, "filename": video.filename, "videoUrl": f"/api/games/{game_id}/video", "duration": round(frames / fps, 3) if fps else 0, "fps": fps, "width": width, "height": height}

@app.get("/api/games/{game_id}/video")
def video(game_id: str, request: Request):
    row = db().execute("SELECT path FROM games WHERE id=?", (game_id,)).fetchone()
    if not row: raise HTTPException(404, "Game not found")
    path = Path(row["path"])
    size = path.stat().st_size
    byte_range = request.headers.get("range")
    start, end, status = 0, size - 1, 200
    headers = {"Accept-Ranges": "bytes"}
    if byte_range:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", byte_range.strip())
        if not match:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        if match.group(1):
            start = int(match.group(1))
            end = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
        else:
            suffix = int(match.group(2) or 0)
            start = max(0, size - suffix)
        if start >= size or end < start:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    def stream_file():
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk: break
                remaining -= len(chunk)
                yield chunk
    return StreamingResponse(stream_file(), status_code=status, media_type=media_type, headers=headers)

@app.put("/api/games/{game_id}/calibration")
def save_calibration(game_id: str, calibration: Calibration):
    c = db()
    game = c.execute("SELECT path FROM games WHERE id=?", (game_id,)).fetchone()
    c.close()
    if not game: raise HTTPException(404, "Game not found")
    path = Path(game["path"]).parent / f"{game_id}.calibration.json"
    path.write_text(calibration.model_dump_json(indent=2), encoding="utf-8")
    connection = db()
    connection.execute("UPDATE games SET calibration=? WHERE id=?", (calibration.model_dump_json(), game_id))
    connection.commit()
    connection.close()
    return {"ok": True, "points": len(calibration.points)}

@app.get("/api/games/{game_id}/frame")
def frame(game_id: str, i: int = 0):
    row = db().execute("SELECT path FROM games WHERE id=?", (game_id,)).fetchone()
    if not row: raise HTTPException(404, "Game not found")
    cap = cv2.VideoCapture(row["path"]); cap.set(cv2.CAP_PROP_POS_FRAMES, max(i, 0)); ok, img = cap.read(); cap.release()
    if not ok: raise HTTPException(400, "Could not read video frame")
    ok, encoded = cv2.imencode(".jpg", img)
    return Response(encoded.tobytes(), media_type="image/jpeg")

def extract_clip(source: Path, destination: Path, start: float, end: float):
    cap = cv2.VideoCapture(str(source)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    ok, image = cap.read()
    if not ok: cap.release(); raise ValueError("Start time is beyond the video duration")
    h, w = image.shape[:2]; writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    current = start
    while ok and current <= end:
        writer.write(image); ok, image = cap.read(); current += 1 / fps
    cap.release(); writer.release()

def normalize_cv(play_id: str, pdir: Path, fps: float):
    labels = read_json(pdir / "labels.json", {})
    teams = read_json(pdir / "team_colors.json", {}).get("assignments", {})
    groups = read_json(pdir / "positions.json", {})
    coords = read_json(pdir / "field_coords.json", {}).get("tracks", [])
    tracks = read_json(pdir / "tracks.json", {}).get("tracks", [])
    byid = {str(t.get("track_id")): t for t in coords}
    boxbyid = {str(t.get("track_id")): t for t in tracks}
    snap = int(labels.get("snap_frame", 0)); pre = max(0, snap - 3)
    players = []
    for tid, track in byid.items():
        seq = track.get("frames", [])
        nearest = min(seq, key=lambda x: abs(int(x.get("i", 0)) - pre), default={})
        assignment = teams.get(tid, {})
        token = groups.get(tid, groups.get(str(tid), ""))
        players.append({"trackId": int(tid), "team": assignment.get("team", labels.get("teams", {}).get(tid, "unknown")), "positionGroup": token, "x": nearest.get("x"), "y": nearest.get("y"), "confidence": assignment.get("confidence", 0), "color": assignment.get("color"), "trajectory": [{"frame": f.get("i"), "x": f.get("x"), "y": f.get("y")} for f in seq[::max(1, len(seq)//20)]] , "boxes": boxbyid.get(tid, {}).get("frames", [])})
    return {"playId": play_id, "snapFrame": snap, "snapTime": snap / fps, "lineOfScrimmage": labels.get("los_x"), "offenseTeam": labels.get("offense_team"), "players": players, "labels": labels, "teamColors": read_json(pdir / "team_colors.json", {}).get("clusters", {}), "artifacts": {p.name: str(p) for p in pdir.glob("*.json")}}

def frames_for_ai(clip: Path, snap_frame: int):
    cap = cv2.VideoCapture(str(clip)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); picks = sorted(set([max(0, min(n-1, snap_frame-18)), max(0, min(n-1, snap_frame-4)), max(0, min(n-1, snap_frame+12)), max(0, n-1)])); result=[]
    for i in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, img=cap.read()
        if ok:
            ok, buf=cv2.imencode(".jpg", img)
            if ok: result.append({"frame": i, "data": base64.b64encode(buf).decode()})
    cap.release(); return result

def unknown_interpretation(down=None, distance=None):
    return {"unit":{"value":"UNKNOWN","confidence":0},"down":{"value":down,"confidence":0 if down is None else .5},"distance":{"value":distance,"confidence":0 if distance is None else .5},"formation":{"value":"Unknown","confidence":0},"play":{"value":"UNKNOWN","confidence":0},"gap":{"value":"UNKNOWN","confidence":0},"result":{"value":"UNKNOWN","confidence":0}}

def interpret(context, images, down, distance):
    fallback = unknown_interpretation(down, distance)
    if not os.getenv("OPENAI_API_KEY"): return fallback, {"status":"skipped", "reason":"OPENAI_API_KEY is not configured"}
    try:
        from openai import OpenAI
        schema={"type":"object","additionalProperties":False,"required":["unit","down","distance","formation","play","gap","result"],"properties":{k:{"type":"object","additionalProperties":False,"required":["value","confidence"],"properties":{"value":{"type":["string","integer","number","null"]},"confidence":{"type":"number","minimum":0,"maximum":1}}} for k in ["unit","down","distance","formation","play","gap","result"]}}
        schema["properties"]["unit"]["properties"]["value"]={"type":"string","enum":UNITS}
        schema["properties"]["down"]["properties"]["value"]={"type":["integer","null"],"enum":[1,2,3,4,None]}
        schema["properties"]["formation"]["properties"]["value"]={"type":"string","enum":FORMATIONS}
        schema["properties"]["play"]["properties"]["value"]={"type":"string","enum":PLAY_TYPES+SPECIAL}
        schema["properties"]["gap"]["properties"]["value"]={"type":"string","enum":GAPS}
        prompt=("Analyze this American football play using the frames and supplied CV data. Return only the requested structured fields. Never infer down/distance from play shape; use the known value if supplied, otherwise UNKNOWN/null unless readable on the scoreboard. Use UNKNOWN when uncertain. Formation requires visible alignment. Identify result and approximate yards only when visible. For pass plays gap is N/A. Context: "+json.dumps({"cv":context,"knownDown":down,"knownDistance":distance,"allowedPlays":PLAY_TYPES,"specialTeams":SPECIAL,"allowedGaps":GAPS}))
        content=[{"type":"input_text","text":prompt}]+[{"type":"input_image","image_url":f"data:image/jpeg;base64,{x['data']}"} for x in images]
        response=OpenAI().responses.create(model=os.getenv("OPENAI_VISION_MODEL","gpt-4.1-mini"),input=[{"role":"user","content":content}],text={"format":{"type":"json_schema","name":"play_interpretation","strict":True,"schema":schema}})
        raw=response.output_text; obj=json.loads(raw); validated=Interpretation.model_validate(obj).model_dump()
        # Explicit coach-provided metadata outranks visual inference.
        if down is not None: validated["down"]={"value":down,"confidence":1.0}
        if distance is not None: validated["distance"]={"value":distance,"confidence":1.0}
        if validated["play"]["value"] not in PLAY_TYPES+SPECIAL: validated["play"]={"value":"UNKNOWN","confidence":0}
        if validated["gap"]["value"] not in GAPS: validated["gap"]={"value":"UNKNOWN","confidence":0}
        return validated, {"status":"ok","raw":raw,"provider":"openai","model":os.getenv("OPENAI_VISION_MODEL","gpt-4.1-mini")}
    except Exception as e:
        return fallback, {"status":"error","message":str(e)[:500]}

@app.post("/api/analyze-play")
def analyze_play(req: AnalyzeRequest):
    if req.endTime <= req.startTime: raise HTTPException(400, "End time must be after start time")
    c=db(); game=c.execute("SELECT * FROM games WHERE id=?",(req.gameId,)).fetchone()
    if not game: c.close(); raise HTTPException(404,"Game not found")
    game_calib=json.loads(game["calibration"]) if game["calibration"] else read_json(Path(game["path"]).parent / f"{req.gameId}.calibration.json", None)
    calibration=req.calibration.model_dump() if req.calibration else game_calib
    if not calibration or len(calibration.get("points",[]))<4: c.close(); raise HTTPException(400,"Add at least four field calibration points before analyzing this play")
    play_id="p"+uuid.uuid4().hex[:12]; pdir=PLAYS/play_id; pdir.mkdir(parents=True,exist_ok=True); clip=pdir/"clip.mp4"
    try: extract_clip(Path(game["path"]),clip,req.startTime,req.endTime)
    except Exception as e: c.close(); raise HTTPException(400,str(e))
    (pdir/"calibration.json").write_text(json.dumps(calibration),encoding="utf-8")
    cap=cv2.VideoCapture(str(clip)); fps=cap.get(cv2.CAP_PROP_FPS) or 30; width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    # Upstream stages read optional meta.json for native FPS and frame bounds,
    # but ingest does not currently write it. Supplying the clip facts keeps
    # snap timing and sideline filtering calibrated to the actual source.
    (pdir/"meta.json").write_text(json.dumps({"fps":fps,"width":width,"height":height}),encoding="utf-8")
    logs={}; failed=None
    try:
        from footballcv import cli
        # The requested segment is already at football-cv's canonical
        # plays/<id>/clip.mp4 path, so run its shipped stages in order.
        for stage in cli.ORDER:
            try: cli.STAGES[stage](play_id,DATA)
            except Exception as e:
                if stage == "autocal": # user has already supplied manual calibration
                    continue
                failed=stage; logs[stage]=str(e); break
        if failed:
            cv={"playId":play_id,"players":[],"snapFrame":0,"snapTime":0,"lineOfScrimmage":None,"error":logs[failed]}
        else: cv=normalize_cv(play_id,pdir,fps)
    except Exception as e:
        cv={"playId":play_id,"players":[],"snapFrame":0,"snapTime":0,"lineOfScrimmage":None,"error":str(e)}; logs["pipeline"]=str(e)
    images=frames_for_ai(clip,int(cv.get("snapFrame",0)))
    ai,ai_meta=interpret(cv,images,req.down,req.distance)
    confidences=[v["confidence"] for v in ai.values() if isinstance(v,dict)]
    overall=round(sum(confidences)/len(confidences),2) if confidences else 0
    num=c.execute("SELECT COALESCE(MAX(playNumber),0)+1 FROM plays WHERE gameId=?",(req.gameId,)).fetchone()[0]
    vals={k:ai[k]["value"] for k in ai}
    c.execute("INSERT INTO plays(id,gameId,playNumber,startTime,endTime,snapTime,unit,down,distance,formation,play,gap,result,overallConfidence,rawCvOutput,rawAiOutput) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(play_id,req.gameId,num,req.startTime,req.endTime,req.startTime+float(cv.get("snapTime",0)),vals["unit"],vals["down"],vals["distance"],vals["formation"],vals["play"],vals["gap"],vals["result"],overall,json.dumps(cv),json.dumps({"interpretation":ai,"debug":ai_meta,"logs":logs})))
    for p in cv.get("players",[]):
        c.execute("INSERT INTO player_detections VALUES(?,?,?,?,?,?)",(play_id,p["trackId"],p["team"],p["positionGroup"],p.get("x"),p.get("y")))
    c.commit(); c.close()
    return {"id":play_id,"gameId":req.gameId,"playNumber":num,"startTime":req.startTime,"endTime":req.endTime,"snapTime":req.startTime+float(cv.get("snapTime",0)),"fields":ai,"overallConfidence":overall,"cv":cv,"debug":{"clipUrl":f"/api/plays/{play_id}/clip","frames":images,"ai":ai_meta,"logs":logs,"cvError":cv.get("error"),"artifactFiles":cv.get("artifacts",{})}}

@app.get("/api/plays/{play_id}/clip")
def play_clip(play_id: str):
    p=PLAYS/play_id/"clip.mp4"
    if not p.is_file(): raise HTTPException(404,"Clip not found")
    return Response(p.read_bytes(),media_type="video/mp4")

@app.get("/api/games/{game_id}/plays")
def list_plays(game_id: str):
    rows=db().execute("SELECT * FROM plays WHERE gameId=? ORDER BY playNumber",(game_id,)).fetchall()
    return [{"id":r["id"],"gameId":r["gameId"],"playNumber":r["playNumber"],"startTime":r["startTime"],"endTime":r["endTime"],"snapTime":r["snapTime"],"fields":{k:{"value":r[col],"confidence":0} for k,col in [("unit","unit"),("down","down"),("distance","distance"),("formation","formation"),("play","play"),("gap","gap"),("result","result")]},"overallConfidence":r["overallConfidence"],"confirmed":bool(r["confirmed"]),"cv":json.loads(r["rawCvOutput"]),"debug":json.loads(r["rawAiOutput"]).get("debug",{})} for r in rows]

class SavePlay(BaseModel):
    fields: Interpretation
    confirmed: bool = False

@app.put("/api/plays/{play_id}")
def save_play(play_id: str, req: SavePlay):
    c=db(); f=req.fields.model_dump(); row=c.execute("SELECT id FROM plays WHERE id=?",(play_id,)).fetchone()
    if not row: c.close(); raise HTTPException(404,"Play not found")
    allowed={"unit":UNITS,"play":PLAY_TYPES+SPECIAL,"gap":GAPS,"formation":FORMATIONS}
    for key, values in allowed.items():
        value=f[key]["value"]
        if value not in values: c.close(); raise HTTPException(422,f"Invalid {key} value")
    if f["down"]["value"] not in (None,1,2,3,4): c.close(); raise HTTPException(422,"Down must be 1, 2, 3, or 4")
    if f["distance"]["value"] is not None and (not isinstance(f["distance"]["value"],(int,float)) or f["distance"]["value"]<0): c.close(); raise HTTPException(422,"Distance must be a nonnegative number")
    c.execute("UPDATE plays SET unit=?,down=?,distance=?,formation=?,play=?,gap=?,result=?,confirmed=?,updatedAt=CURRENT_TIMESTAMP WHERE id=?",(*(f[k]["value"] for k in ["unit","down","distance","formation","play","gap","result"]),int(req.confirmed),play_id)); c.commit(); c.close(); return {"ok":True}

@app.get("/api/games/{game_id}/export.csv")
def export_csv(game_id: str):
    rows=db().execute("SELECT * FROM plays WHERE gameId=? ORDER BY playNumber",(game_id,)).fetchall(); s=io.StringIO(); w=csv.writer(s); w.writerow(["Play","Start Time","End Time","Unit","Down","Distance","Formation","Play","Gap","Result"])
    for r in rows: w.writerow([r["playNumber"],r["startTime"],r["endTime"],r["unit"],r["down"],r["distance"],r["formation"],r["play"],r["gap"],r["result"]])
    return Response(s.getvalue(),media_type="text/csv",headers={"Content-Disposition":f'attachment; filename="{game_id}-plays.csv"'})

class Correction(BaseModel):
    text: str

@app.post("/api/plays/{play_id}/correction")
def correction(play_id: str, req: Correction):
    match=re.fullmatch(r"\s*(?:change|set)\s+(unit|down|distance|formation|play|gap|result)\s+to\s+(.+?)\s*",req.text,re.I)
    if not match: return {"ok":False,"message":"Try: change gap to A Right"}
    key,val=match.group(1).lower(),match.group(2).strip(); c=db(); row=c.execute("SELECT rawAiOutput FROM plays WHERE id=?",(play_id,)).fetchone()
    if not row: c.close(); raise HTTPException(404,"Play not found")
    vals=json.loads(row["rawAiOutput"]); parsed=vals.get("interpretation",unknown_interpretation())
    allowed={"unit":UNITS,"play":PLAY_TYPES,"gap":GAPS,"formation":FORMATIONS}
    if key in allowed and val.upper() not in [x.upper() for x in allowed[key]]: c.close(); return {"ok":False,"message":f"Choose a valid {key} value"}
    if key == "play" and val.upper() not in [x.upper() for x in PLAY_TYPES+SPECIAL]: c.close(); return {"ok":False,"message":"Choose a valid play or special teams value"}
    if key=="unit": val=next(x for x in UNITS if x.upper()==val.upper())
    if key in {"play","gap","formation"}: val=next(x for x in allowed[key] if x.upper()==val.upper())
    if key == "play": val=next(x for x in PLAY_TYPES+SPECIAL if x.upper()==val.upper())
    if key in {"down","distance"}:
        try: val=int(val) if key=="down" else float(val)
        except ValueError: c.close(); return {"ok":False,"message":f"{key.capitalize()} must be numeric"}
    if key=="down" and val not in [1,2,3,4]: c.close(); return {"ok":False,"message":"Down must be 1, 2, 3, or 4"}
    parsed[key]={"value":val,"confidence":1.0}; vals["interpretation"]=parsed
    cols={"unit":"unit","down":"down","distance":"distance","formation":"formation","play":"play","gap":"gap","result":"result"}
    c.execute(f"UPDATE plays SET {cols[key]}=?,rawAiOutput=?,updatedAt=CURRENT_TIMESTAMP WHERE id=?",(val,json.dumps(vals),play_id)); c.commit(); c.close()
    return {"ok":True,"field":key,"value":val}
