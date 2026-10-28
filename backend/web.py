"""Serve the bundled React SPA and FastAPI on one origin for cloud demos.

API routes are registered before the catch-all SPA route. This preserves API
404s and avoids exposing any files outside the frontend build directory.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .app import create_app


def create_web_app(public_dir: str | Path | None = None, api: FastAPI | None = None) -> FastAPI:
    application = api if api is not None else create_app()
    root = Path(public_dir or os.getenv("FRONTEND_DIST_DIR", "/app/public")).resolve()

    if root.is_dir() and (root / "index.html").is_file():
        assets = root / "assets"
        if assets.is_dir():
            application.mount("/assets", StaticFiles(directory=str(assets)), name="frontend-assets")

        @application.get("/", include_in_schema=False)
        def home():
            return FileResponse(root / "index.html")

        @application.get("/{requested_path:path}", include_in_schema=False)
        def frontend_fallback(requested_path: str):
            if requested_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API resource not found")
            candidate = (root / requested_path).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return FileResponse(candidate)
            # Do not return HTML for nonexistent JS, CSS, or other asset paths.
            if "." in requested_path.rsplit("/", 1)[-1]:
                raise HTTPException(status_code=404, detail="File not found")
            return FileResponse(root / "index.html")

    return application


app = create_web_app()
