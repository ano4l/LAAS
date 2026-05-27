# Football AI Analytics

End-to-end football analytics prototype built from the product brief in `Football analytics  (1).pdf`.

## What Runs Today

- FastAPI backend for uploads, demo jobs, job status, and media serving.
- Python video worker that generates:
  - annotated video
  - tracking JSON
  - possession metrics
  - tactical phase segments
  - AI-style tactical report
- React/Vite match workspace with:
  - upload/demo controls
  - live processing progress
  - video playback
  - animated tactical pitch
  - metrics, phase list, and report panel

The current worker uses a deterministic fallback analytics engine so the product can run immediately. The next CV milestone is swapping that fallback for YOLO + tracker + Eagle-style homography using the cloned references in `references/`.

## Local Development

```bash
npm install
npm --prefix apps/web install
python -m pip install numpy opencv-python python-multipart
python -m pip install -e . --no-deps
npm run dev
```

Open:

- Web app: http://127.0.0.1:5173
- API health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs

## Useful Commands

```bash
npm run build
python -m compileall apps/api workers
```

## Production Backend

The API uses SQLite for job/run metadata and filesystem manifests for heavy analysis artifacts. This keeps job records small while preserving full tactical outputs.

Key endpoints:

- `GET /health` liveness check
- `GET /ready` storage/database readiness
- `GET /capabilities` engine, upload, and AI capability discovery
- `POST /videos/upload` video upload and analysis job creation
- `POST /demo-job` local demo analysis job
- `GET /analysis-jobs/{job_id}` polling
- `GET /runs` completed analysis library
- `GET /runs/{run_id}` full run manifest
- `GET /ai/status` AI provider readiness
- `POST /runs/{run_id}/ai-summary` report enrichment boundary

Environment:

```bash
cp .env.example .env
```

Important variables:

- `STORAGE_ROOT` controls uploads, outputs, and SQLite database location.
- `MAX_UPLOAD_MB` sets the upload size limit.
- `FOOTBALL_ANALYTICS_ENGINE=auto|fallback|yolo` selects the analysis engine.
- `OPENAI_API_KEY` enables future AI report enrichment.
- `VITE_API_URL` can point the web app at a separate API service; leave empty for same-origin production serving.
- `CORS_ORIGINS` should include the deployed frontend origin if the API is not serving the built web app.

The FastAPI service serves `apps/web/dist` at `/` after `npm run build`, so Railway can run the product as a single web service.

## Railway

The repo includes `Dockerfile` and `railway.json`.

For Railway:

1. Set environment variables from `.env.example`.
2. Add a persistent volume mounted at the same path as `STORAGE_ROOT` if match outputs should survive redeploys.
3. Deploy with the Dockerfile builder.
4. Check `/health`, then `/ready`.

## Vercel Frontend

The repo also includes `vercel.json` for deploying the React/Vite web app from `apps/web`.

For a frontend-only demo, do not set `VITE_API_URL`. The app will load bundled static match analytics and all dashboard/modules will work without a backend.

When the Railway/FastAPI backend is deployed, set this Vercel environment variable:

```bash
VITE_API_URL=https://your-backend-domain.example
```

The video analysis API should stay on Railway or another long-running backend host. Vercel is used here for the web UI/demo, while FastAPI handles uploads, OpenCV processing, media files, and SQLite/object-storage-backed metadata elsewhere.

## Optional Real CV Engine

The app includes a YOLO adapter that can use:

`references/FootBall-Analytics-MVP/yolo11l.pt`

Install the optional CV dependencies when the network is stable:

```bash
python -m pip install ultralytics supervision pandas scikit-learn
```

Check readiness:

```bash
curl http://127.0.0.1:8000/capabilities
```

Engine behavior:

- `FOOTBALL_ANALYTICS_ENGINE=auto` tries YOLO when available and falls back automatically.
- `FOOTBALL_ANALYTICS_ENGINE=fallback` forces the deterministic fallback engine.
- `FOOTBALL_ANALYTICS_ENGINE=yolo` requires the YOLO path and raises errors if it fails.

## Reference Repos

The required repositories were cloned into `references/` for local inspection. Keep them quarantined from app code unless license and model-weight usage are explicitly cleared.

Primary references:

- `references/FootBall-Analytics-MVP`
- `references/Eagle`
- `references/football_analysis`
- `references/Soccer-Analytics`
