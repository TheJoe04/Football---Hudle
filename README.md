# Huddle · Football Film Assistant

Local MVP for tagging American football plays with an AI review chat beside the film. The app uses [football-cv](vendor/football-cv/README.md) directly for player detection, tracking, jersey-color teams, calibration, field coordinates, snap/LOS and position groups. Its MIT license and copyright notice are preserved in [the vendored license](vendor/football-cv/LICENSE). No new training pipeline is included.

The repo is split into a Next.js/React/TypeScript interface at `web/` and a FastAPI adapter at `cv-service/`. The adapter owns clipping, CV artifact normalization, optional OpenAI structured vision output, SQLite storage, coach corrections and CSV export. See [the CV service notes](cv-service/README.md) for the upstream artifact contract.

## Run locally (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r cv-service/requirements.txt
python -m pip install -e .\vendor\football-cv[detect,ml]
Copy-Item cv-service/.env.example cv-service/.env
# Set OPENAI_API_KEY in your environment if you want automatic tagging.

Start-Process powershell -ArgumentList '-NoExit','-Command','Set-Location cv-service; uvicorn app:app --reload --port 8000'
Set-Location web
npm install
npm run dev
```

Open http://localhost:3000, upload a video, set a play start and end, then analyze. The first analysis prompts for four manual field calibration points on two yard lines and both sidelines; that calibration is reused for plays from the same game. football-cv's optional detector uses pretrained weights and may download them on first use. `OPENAI_API_KEY` enables the multimodal provider; without it, CV still runs and AI fields default to UNKNOWN. For a non-OpenAI provider, implement the same `interpret(context, images, down, distance)` contract in `cv-service/app.py`.

Upload progress and errors appear above the workspace. Service request and file-save details go to `data/logs/app.log`; Uvicorn output is in `data/logs/uvicorn.stderr.log`, and Next.js output is in `data/logs/next.stdout.log` / `next.stderr.log`.

## Current MVP limits

- No whole-game automatic segmentation, authentication, billing, or training.
- Down/distance, play call, gap, and result are model suggestions and remain coach editable. The model is instructed to leave unknown values unknown.
- Video processing depends on OpenCV; FFmpeg is useful for production codecs but is not required by the current adapter.
- The development view exposes the normalized CV output, raw AI response, representative frames, players and clip for debugging.
