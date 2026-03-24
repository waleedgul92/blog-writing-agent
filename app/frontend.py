import json
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st
import requests

API_URL = "http://localhost:8000"

def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"

_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")

def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()

def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue
        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)
        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")
        i += 1

def list_past_blogs() -> List[Dict[str, str]]:
    r = requests.get(f"{API_URL}/blogs")
    return r.json().get("blogs", [])

def read_md_file(filename: str) -> str:
    r = requests.get(f"{API_URL}/blogs/{filename}")
    return r.json().get("content", "")

st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")
st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area("Topic", height=120)
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found.")
        selected_filename = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, str] = {}
        for pf in past_files[:50]:
            label = f"{pf['title']}  ·  {pf['filename']}"
            options.append(label)
            file_by_label[label] = pf['filename']

        selected_label = st.radio("Select a blog to load", options=options, index=0, label_visibility="collapsed")
        selected_filename = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_filename:
                md_text = read_md_file(selected_filename)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.session_state["topic_prefill"] = pf['title']

if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    pass

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

tab_plan, tab_evidence, tab_preview, tab_images = st.tabs(["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images"])

if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs = {"topic": topic.strip(), "as_of": as_of.isoformat()}
    status = st.status("Running graph via API...", expanded=True)

    resp = requests.post(f"{API_URL}/generate", json=inputs)
    out = resp.json()
    st.session_state["last_out"] = out
    status.update(label="✅ Done", state="complete", expanded=False)

out = st.session_state.get("last_out")
if out:
    with tab_plan:
        st.subheader("Plan")
        plan_dict = out.get("plan")
        if not plan_dict:
            st.info("No plan found in output.")
        else:
            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame([
                    {
                        "id": t.get("id"),
                        "title": t.get("title"),
                        "target_words": t.get("target_words"),
                        "requires_research": t.get("requires_research"),
                        "requires_citations": t.get("requires_citations"),
                        "requires_code": t.get("requires_code"),
                        "tags": ", ".join(t.get("tags") or []),
                    }
                    for t in tasks
                ]).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned.")
        else:
            rows = []
            for e in evidence:
                rows.append({
                    "title": e.get("title"),
                    "published_at": e.get("published_at"),
                    "source": e.get("source"),
                    "url": e.get("url"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            plan_dict = out.get("plan")
            if isinstance(plan_dict, dict):
                blog_title = plan_dict.get("blog_title", "blog")
            else:
                blog_title = "blog"

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button("⬇️ Download Markdown", data=final_md.encode("utf-8"), file_name=md_filename, mime="text/markdown")

            bundle = requests.get(f"{API_URL}/download/bundle/{md_filename}").content
            st.download_button("📦 Download Bundle (MD + images)", data=bundle, file_name=f"{safe_slug(blog_title)}_bundle.zip", mime="application/zip")

    with tab_images:
        st.subheader("Images")
        specs = out.get("image_specs") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("images/ exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, use_container_width=True)

            z = requests.get(f"{API_URL}/download/images").content
            if z:
                st.download_button("⬇️ Download Images (zip)", data=z, file_name="images.zip", mime="application/zip")