from __future__ import annotations
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from backend import app

api = FastAPI(title="Blog Writing Agent API")

class GenerateRequest(BaseModel):
    topic: str
    as_of: str = date.today().isoformat()

def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"

def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()

def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()

def list_past_blogs() -> List[Path]:
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files

def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback

@api.post("/generate")
def generate_blog(req: GenerateRequest):
    inputs: Dict[str, Any] = {
        "topic": req.topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": req.as_of,
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }
    out = app.invoke(inputs)
    return JSONResponse(content=out)

@api.get("/blogs")
def get_blogs():
    past_files = list_past_blogs()
    results = []
    for p in past_files:
        md_text = read_md_file(p)
        title = extract_title_from_md(md_text, p.stem)
        results.append({"filename": p.name, "title": title})
    return {"blogs": results}

@api.get("/blogs/{filename}")
def get_blog(filename: str):
    p = Path(filename)
    md_text = read_md_file(p)
    return {"filename": filename, "content": md_text}

@api.get("/download/bundle/{filename}")
def download_bundle(filename: str):
    p = Path(filename)
    md_text = read_md_file(p)
    title = extract_title_from_md(md_text, p.stem)
    bundle_bytes = bundle_zip(md_text, filename, Path("images"))
    return StreamingResponse(
        BytesIO(bundle_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={safe_slug(title)}_bundle.zip"}
    )

@api.get("/download/images")
def download_images():
    images_dir = Path("images")
    zip_bytes = images_zip(images_dir)
    return StreamingResponse(
        BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=images.zip"}
    )