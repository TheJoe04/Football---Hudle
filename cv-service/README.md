# Coach Film CV service

FastAPI application adapter around the vendored `footballcv` package. It owns upload, clipping, normalized outputs, AI provider calls, SQLite storage, and export. The web app only consumes this service's stable API.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r cv-service/requirements.txt
pip install -e .\vendor\football-cv[detect,ml]
Set `OPENAI_API_KEY` in your shell to enable tagging, then run:

```powershell
Set-Location cv-service
uvicorn app:app --reload --port 8000
```
```

The detect extra installs a pretrained person detector and may require a compatible PyTorch/TorchVision build. `OPENAI_API_KEY` enables multimodal tagging; without it, CV analysis still runs and AI fields stay UNKNOWN. The adapter uses OpenAI Responses API behind `interpret()` so another provider can implement the same interface.

football-cv already emits these per-play artifacts under `data/plays/<play_id>/`:

| Artifact | Existing fields used by the adapter |
| --- | --- |
| `detections.json` | `model`, `conf_threshold`, `frames[]` with native-pixel person boxes `[x1,y1,x2,y2,confidence]` |
| `tracks.json` | `tracks[]` with `track_id` and frame-indexed pixel `box` values |
| `team_colors.json` | `clusters` RGB values; `assignments[track_id]` with `team`, `confidence`, and player RGB color; flip events |
| `field_coords.json` | per-frame homographies and `tracks[]` with field-yard `x,y` coordinates |
| `labels.json` | `snap_frame`, `los_x`, `teams[track_id]` (`O`/`D`), `position_confidence`, snap and offense-defense confidence |
| `positions.json` | `{track_id: position_group}`; defense `DL/LB/CB/S`, offense `OL/QB/RB/WR` |
| `calibration.json`, `snap.json`, `meta.json` | manual pixel-to-yard points, optional marked snap, and clip metadata. Upstream reads `meta.json` but its ingest stage does not create one; this adapter writes actual FPS and frame dimensions for positions/snap logic. |

`footballcv.analyze(video, calibration, snap, data_dir, play_id)` runs the pipeline and returns the position map. The CLI exposes the same ordered stages: `detect → track → teamcolor → stitch → autocal → field → positions`. The zero-build browser viewer contains the original manual calibration interaction and overlay renderer; this app follows the same calibration/artifact contracts in React so the coach experience stays TypeScript-based. See [the vendored README](../vendor/football-cv/README.md) and [MIT license](../vendor/football-cv/LICENSE). Upstream files are preserved without application-specific edits.
