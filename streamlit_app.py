from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".mpeg", ".mpg")

st.set_page_config(page_title="BALLISTIC TRACKING // RESEARCH CONSOLE", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{--bg:#05070b;--panel:#0b1018;--panel2:#0f1621;--line:#26384f;--text:#e8f1f7;--muted:#78909f;--accent:#5aa9ff;--violet:#9b8cff}
.stApp{background:radial-gradient(circle at 50% -10%,rgba(52,108,180,.18),transparent 35%),linear-gradient(180deg,#05070b 0%,#070b11 55%,#05070b 100%);color:var(--text)}
[data-testid="stHeader"]{background:transparent}.block-container{max-width:1500px;padding-top:2rem}h1,h2,h3,h4{color:var(--text)!important;letter-spacing:.04em}
.hero{border:1px solid var(--line);background:linear-gradient(135deg,rgba(12,22,31,.96),rgba(7,11,17,.96));border-radius:18px;padding:28px 30px 24px;margin-bottom:18px;box-shadow:0 0 45px rgba(50,110,190,.10),inset 0 1px 0 rgba(255,255,255,.025)}
.eyebrow,.media-title,.section-kicker{color:var(--accent);font-size:.72rem;letter-spacing:.2em;font-weight:700}.hero-title{font-size:clamp(1.65rem,4vw,3rem);font-weight:800;margin:5px 0;letter-spacing:.08em}.hero-sub{color:var(--muted);font-size:.9rem}
.status{display:inline-block;border:1px solid #315b86;color:var(--accent);background:rgba(64,130,210,.09);border-radius:999px;padding:5px 11px;font-size:.72rem;letter-spacing:.08em}
.upload-box{border:1px dashed #31577d;background:rgba(8,18,26,.8);border-radius:16px;padding:10px;margin:12px 0 22px}.metric-card,.lab-card{background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:16px;min-height:92px;margin-bottom:12px}.metric-label{color:var(--muted);font-size:.68rem;letter-spacing:.14em}.metric-value{color:var(--text);font-size:1.25rem;font-weight:700;margin-top:5px}.frame-caption{font-size:.68rem;color:#78909f;text-align:center;margin:3px 0 10px}
.lab-card{min-height:0}.lab-card h3{margin-top:0}.muted{color:var(--muted)}.state-pending{color:var(--violet);font-weight:700;letter-spacing:.12em}.not-available{border:1px dashed #42546a;border-radius:10px;padding:14px;color:var(--muted);letter-spacing:.12em;text-align:center}
.architecture{white-space:pre-wrap;font-family:monospace;color:#b9d8f3;border:1px solid var(--line);background:#080d14;border-radius:12px;padding:18px;line-height:1.55}
div[data-testid="stFileUploader"]{background:rgba(9,16,24,.72);border:1px solid #26384f;border-radius:12px;padding:8px}div[data-testid="stFileUploaderDropzone"]{background:transparent}button[kind="secondary"]{border-color:#31516d}
@media(max-width:700px){.block-container{padding:1rem .75rem 2rem}.hero{padding:20px 18px;border-radius:14px}.hero-sub{font-size:.78rem}}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">COMPUTER VISION × STATE ESTIMATION</div><div class="hero-title">BALLISTIC TRACKING</div><div class="hero-sub">RESEARCH CONSOLE // MEDIA · DATASET · MODEL · AUDITS</div><br><span class="status">● READ-ONLY EXPERIMENT LAB</span></div>', unsafe_allow_html=True)


def _basename(name: str) -> str:
    return name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _is_skippable_member(name: str) -> bool:
    norm = name.replace("\\", "/")
    base = _basename(norm)
    return not norm or norm.endswith("/") or norm.startswith("__MACOSX/") or "/__MACOSX/" in norm or not base or base.startswith(".") or base.startswith("._")


def _is_image_member(name: str) -> bool:
    return _basename(name).lower().endswith(IMAGE_EXTENSIONS)


def natural_frame_key(name: str) -> Tuple[int, str]:
    stem = name.rsplit(".", 1)[0] if "." in name else name
    match = re.search(r"(\d+)", stem)
    return (int(match.group(1)), name.lower()) if match else (10**9, name.lower())


def _validate_image_bytes(data: bytes) -> bool:
    if not data or len(data) < 24:
        return False
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
        return True
    except Exception:
        return False


def _looks_like_zip(data: bytes) -> bool:
    return len(data) >= 4 and data[:2] == b"PK"


@st.cache_data(show_spinner="Reading sequence…", ttl=3600)
def load_frames_from_zip(zip_bytes: bytes) -> Tuple[List[Tuple[str, bytes]], Dict[str, int]]:
    stats = {"members_total": 0, "skipped_meta": 0, "candidates": 0, "unreadable": 0, "loaded": 0, "bytes": len(zip_bytes)}
    if not zip_bytes:
        raise ValueError("ZIP_EMPTY: o arquivo enviado está vazio.")
    if not _looks_like_zip(zip_bytes):
        raise ValueError("ZIP_OPEN_FAILED: o conteúdo recebido não parece um ZIP.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"ZIP_OPEN_FAILED: arquivo ZIP inválido ou incompleto ({exc}).") from exc
    try:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"ZIP_CORRUPT: entrada corrompida: {damaged}")
    except ValueError:
        raise
    except Exception:
        pass
    frames = []
    seen = set()
    for info in archive.infolist():
        stats["members_total"] += 1
        member = info.filename
        if info.is_dir() or _is_skippable_member(member):
            stats["skipped_meta"] += 1
            continue
        if not _is_image_member(member):
            continue
        stats["candidates"] += 1
        base = _basename(member)
        try:
            data = archive.read(info)
        except Exception:
            stats["unreadable"] += 1
            continue
        if not _validate_image_bytes(data):
            stats["unreadable"] += 1
            continue
        key = (member.lower(), len(data))
        if key in seen:
            continue
        seen.add(key)
        frames.append((base, data))
        stats["loaded"] += 1
    archive.close()
    if not frames:
        if stats["candidates"] == 0:
            raise ValueError("ZIP_NO_IMAGES: nenhuma imagem PNG/JPG/JPEG/WEBP/BMP foi encontrada no ZIP.")
        raise ValueError("ZIP_UNREADABLE_IMAGES: as imagens encontradas não puderam ser lidas.")
    frames.sort(key=lambda item: natural_frame_key(item[0]))
    return frames, stats


def decode_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def make_thumbnail(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    ratio = max_width / float(image.width)
    return image.resize((max_width, max(1, int(image.height * ratio))), Image.Resampling.BILINEAR)


def _validate_video_bytes(data: bytes) -> bool:
    if not data:
        return False
    try:
        import cv2
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".video") as handle:
            handle.write(data)
            handle.flush()
            capture = cv2.VideoCapture(handle.name)
            valid = capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0
            capture.release()
            return bool(valid)
    except Exception:
        return False


def detect_media_kind(name: str, data: bytes) -> str:
    lower = (name or "").lower()
    if lower.endswith(".zip") or _looks_like_zip(data):
        return "zip"
    if data.startswith((b"\x89PNG", b"\xff\xd8\xff", b"RIFF")) and _validate_image_bytes(data):
        return "image"
    if (data.startswith(b"\x00\x00\x00") and b"ftyp" in data[:64]) or data.startswith(b"\x1aE\xdf\xa3"):
        return "video" if _validate_video_bytes(data) else "invalid_video"
    if lower.endswith(IMAGE_EXTENSIONS):
        return "image"
    if lower.endswith(VIDEO_EXTENSIONS):
        return "video" if _validate_video_bytes(data) else "invalid_video"
    if _validate_image_bytes(data):
        return "image"
    return "unknown"


def _read_artifact(relative: str) -> str:
    path = ROOT / relative
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return ""


@st.cache_data(show_spinner=False)
def _artifact_texts() -> Dict[str, str]:
    paths = ["README.md", "evaluation.md", "docs/dataset.md", "docs/evaluation.md", "docs/experiment_notes.md", "docs/reproducibility.md", "docs/trajectory.md", "docs/yolo_training.md", "docs/kalman_filter.md", "docs/data_pipeline.md", "docs/pseudo_labeling.md"]
    return {path: _read_artifact(path) for path in paths}


def _first_match(text: str, patterns: List[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "NOT AVAILABLE"


def _available(relative: str) -> bool:
    return (ROOT / relative).is_file()


def not_available() -> None:
    st.markdown('<div class="not-available">NOT AVAILABLE</div>', unsafe_allow_html=True)


def render_overview(texts: Dict[str, str]) -> None:
    dataset = texts.get("docs/dataset.md", "")
    model = texts.get("docs/yolo_training.md", "")
    combined = "\n".join(texts.values())
    values = [
        ("DATASET", _first_match(dataset, [r"Dataset name:\*\*\s*([^\n]+)", r"## Source\s*\n\s*-\s*([^\n]+)"])),
        ("FRAMES", _first_match(dataset, [r"\| Number of frames \|\s*([^|]+)"])),
        ("RESOLUTION", _first_match(dataset, [r"\| Resolution \|\s*([^|]+)"])),
        ("PHOTRON FPS", _first_match(combined, [r"Photron FPS\s*\|?\s*([^\n|]+)", r"frame rate[^\n:]*:\s*([^\n]+)"])),
        ("MODEL", _first_match(model, [r"Architecture:\s*\*\*([^*]+)", r"Architecture:\s*([^\n]+)"])),
        ("STATE", _first_match(texts.get("docs/kalman_filter.md", ""), [r"state[^\n:]*:\s*([^\n]+)", r"4D"])),
    ]
    st.markdown('<div class="section-kicker">EXPERIMENT OVERVIEW</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for col, (label, value) in zip(cols, values):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)


def render_dataset(texts: Dict[str, str]) -> None:
    st.markdown('<div class="section-kicker">DATASET</div>', unsafe_allow_html=True)
    text = texts.get("docs/dataset.md", "")
    if not text:
        not_available()
        return
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="lab-card"><h3>DATASET RECORD</h3><p class="muted">Official source and acquisition details read from repository documentation.</p></div>', unsafe_allow_html=True)
        st.markdown(text, unsafe_allow_html=False)
    with right:
        st.markdown('<div class="lab-card"><h3>HDF5 STRUCTURE</h3></div>', unsafe_allow_html=True)
        match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
        st.code(match.group(1).strip() if match else "NOT AVAILABLE", language="text")


def render_model(texts: Dict[str, str]) -> None:
    st.markdown('<div class="section-kicker">MODEL</div>', unsafe_allow_html=True)
    text = texts.get("docs/yolo_training.md", "")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="lab-card"><h3>MODEL ARTIFACT</h3></div>', unsafe_allow_html=True)
        if _available("models/best.pt"):
            st.write({"path": "models/best.pt", "size_bytes": (ROOT / "models/best.pt").stat().st_size})
        else:
            not_available()
        st.caption("No inference or training is executed by this viewer.")
    with right:
        if text:
            st.markdown(text, unsafe_allow_html=False)
        else:
            not_available()


def _audit_card(title: str, body: str, available: bool = True) -> None:
    st.markdown(f'<div class="lab-card"><h3>{title}</h3></div>', unsafe_allow_html=True)
    if available and body:
        st.markdown(body, unsafe_allow_html=False)
    else:
        st.markdown('<div class="state-pending">PENDING</div>', unsafe_allow_html=True)


def render_audits(texts: Dict[str, str]) -> None:
    st.markdown('<div class="section-kicker">AUDITS</div>', unsafe_allow_html=True)
    text = texts.get("docs/evaluation.md", "") or texts.get("evaluation.md", "")
    tabs = st.tabs(["AUDIT 01 · CV × YOLO", "AUDIT 02 · LOW-CONFIDENCE RECOVERY", "AUDIT 03 · CV × YOLO × KALMAN", "AUDIT 04 · GATED KALMAN", "AUDIT 05 · ONE-STEP-AHEAD", "AUDIT 06 · TRAJECTORY DYNAMICS"])
    headings = ["Audit 1", "Audit 2", "Audit 3", "Audit 4", "Audit 5", "Next experiment"]
    for tab, heading in zip(tabs, headings):
        with tab:
            match = re.search(rf"(?:#+\s*)?{re.escape(heading)}[^\n]*\n(.*?)(?=\n(?:#+\s*)?(?:Audit \d|Interpretation|Current limitations|Next experiment)|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
            _audit_card(heading.upper(), match.group(1).strip() if match else "", bool(match))


def render_plots() -> None:
    st.markdown('<div class="section-kicker">SCIENTIFIC PLOTS</div>', unsafe_allow_html=True)
    candidates = ["docs/plots/results.png", "docs/plots/confusion_matrix.png", "docs/plots/BoxPR_curve.png", "docs/plots/BoxF1_curve.png", "plots/real_trajectory.png"]
    found = [path for path in candidates if _available(path)]
    if not found:
        not_available()
        return
    cols = st.columns(min(3, len(found)))
    for col, path in zip(cols, found):
        with col:
            st.caption(path)
            st.image(str(ROOT / path), use_container_width=True)


def render_trajectory(texts: Dict[str, str]) -> None:
    st.markdown('<div class="section-kicker">TRAJECTORY ANALYSIS</div>', unsafe_allow_html=True)
    trajectory_image = next((path for path in ["plots/real_trajectory.png", "docs/plots/real_trajectory.png"] if _available(path)), None)
    if trajectory_image:
        st.image(str(ROOT / trajectory_image), use_container_width=True)
    else:
        text = texts.get("docs/trajectory.md", "")
        if text:
            st.markdown(text, unsafe_allow_html=False)
        else:
            not_available()


def render_error_analysis(texts: Dict[str, str]) -> None:
    st.markdown('<div class="section-kicker">ERROR ANALYSIS</div>', unsafe_allow_html=True)
    text = texts.get("docs/evaluation.md", "") or texts.get("evaluation.md", "")
    matches = re.findall(r"(?:YOLO × CV|Kalman × CV|Kalman × YOLO|Prediction × CV)[^\n]*", text, flags=re.IGNORECASE)
    if matches:
        st.code("\n".join(matches), language="text")
    else:
        not_available()


def render_findings(texts: Dict[str, str]) -> None:
    st.markdown('<div class="section-kicker">RESEARCH FINDINGS</div>', unsafe_allow_html=True)
    text = texts.get("docs/evaluation.md", "") or texts.get("evaluation.md", "")
    match = re.search(r"(?:##\s*)?Interpretation\s*(.*?)(?=\n(?:##\s*)?Current limitations|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    limitations = re.search(r"(?:##\s*)?Current limitations\s*(.*?)(?=\n(?:##\s*)?Next experiment|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match and not limitations:
        not_available()
        return
    if match:
        st.markdown("**OBSERVATION**\n\n" + match.group(1).strip(), unsafe_allow_html=False)
    if limitations:
        st.markdown("**LIMITATION**\n\n" + limitations.group(1).strip(), unsafe_allow_html=False)


def render_reproducibility(texts: Dict[str, str]) -> None:
    st.markdown('<div class="section-kicker">REPRODUCIBILITY</div>', unsafe_allow_html=True)
    text = texts.get("docs/reproducibility.md", "")
    if text:
        st.markdown(text, unsafe_allow_html=False)
    else:
        not_available()


def render_architecture() -> None:
    st.markdown('<div class="section-kicker">EXPERIMENT ARCHITECTURE</div>', unsafe_allow_html=True)
    st.markdown('<div class="architecture">PHOTRON FRAMES\n      │\n      ▼\nCOMPUTER VISION\n      │\n      ▼\nYOLO DETECTION\n      │\n      ▼\nMEASUREMENT\n      │\n      ▼\nKALMAN FILTER\n      │\n      ▼\nSTATE ESTIMATION\n      │\n      ▼\nTRAJECTORY\n      │\n      ▼\nEVALUATION</div>', unsafe_allow_html=True)


def render_media_lab() -> None:
    st.markdown('<div class="media-title">MEDIA LAB // INPUT</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploads = st.file_uploader("Arraste arquivos para cá ou toque em Procurar arquivos", type=None, accept_multiple_files=True, help="Aceita imagens soltas, ZIPs com frames e vídeos. O conteúdo é validado pelo app.")
    st.markdown('</div>', unsafe_allow_html=True)
    if not uploads:
        st.info("Nenhuma mídia carregada. Envie PNG/JPG, um ZIP de frames ou um vídeo para iniciar a visualização.")
        return
    all_frames: List[Tuple[str, bytes]] = []
    videos = []
    unknown, invalid_videos, empty_files, zip_sources = [], [], [], []
    for uploaded in uploads:
        data = uploaded.getvalue()
        if not data:
            empty_files.append(uploaded.name)
            continue
        kind = detect_media_kind(uploaded.name, data)
        if kind == "zip":
            try:
                frames, _ = load_frames_from_zip(data)
                all_frames.extend(frames)
                zip_sources.append(f"{uploaded.name} · {len(frames)} frames")
            except ValueError as exc:
                st.error(f"{uploaded.name}: {exc}")
        elif kind == "image":
            if _validate_image_bytes(data):
                all_frames.append((_basename(uploaded.name), data))
            else:
                unknown.append(uploaded.name)
        elif kind == "video":
            videos.append((uploaded.name, data))
        elif kind == "invalid_video":
            invalid_videos.append(uploaded.name)
        else:
            unknown.append(uploaded.name)
    all_frames.sort(key=lambda item: natural_frame_key(item[0]))
    if empty_files: st.warning("Arquivos vazios ignorados: " + ", ".join(empty_files))
    if invalid_videos: st.error("Vídeos inválidos ou ilegíveis: " + ", ".join(invalid_videos))
    if unknown: st.warning("Formato não suportado ou arquivo inválido: " + ", ".join(unknown))
    cols = st.columns(4)
    for col, (label, value) in zip(cols, [("FRAMES", str(len(all_frames))), ("VIDEOS", str(len(videos))), ("SOURCES", str(len(uploads))), ("MODE", "VISUAL")]):
        with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    if zip_sources: st.caption("ZIP sources: " + " · ".join(zip_sources))
    for video_name, video_bytes in videos:
        st.markdown(f'<div class="media-title">VIDEO // {video_name}</div>', unsafe_allow_html=True)
        st.video(video_bytes)
    if all_frames:
        st.markdown('<div class="media-title">FRAME VIEWER</div>', unsafe_allow_html=True)
        selected_index = 0 if len(all_frames) == 1 else st.slider("Frame", 1, len(all_frames), min(max(1, (len(all_frames) + 1) // 2), len(all_frames)), label_visibility="collapsed") - 1
        name, raw = all_frames[selected_index]
        image = decode_image(raw)
        col_img, col_meta = st.columns([3, 1])
        with col_img:
            st.markdown(f"### FRAME {selected_index + 1:03d} / {len(all_frames)}")
            st.image(image, caption=name, use_container_width=True)
        with col_meta:
            st.markdown(f'<div class="metric-card"><div class="metric-label">SELECTED FRAME</div><div class="metric-value">{selected_index + 1:03d}</div></div><div class="metric-card"><div class="metric-label">FILE</div><div class="metric-value" style="font-size:.85rem;word-break:break-all">{name}</div></div><div class="metric-card"><div class="metric-label">RESOLUTION</div><div class="metric-value">{image.width} × {image.height}</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="media-title">SEQUENCE // ALL FRAMES</div>', unsafe_allow_html=True)
        grid_columns = min(8, max(3, len(all_frames)))
        for start in range(0, len(all_frames), grid_columns):
            row = all_frames[start:start + grid_columns]
            grid = st.columns(grid_columns)
            for col, (fname, fbytes) in zip(grid, row):
                with col:
                    st.image(make_thumbnail(decode_image(fbytes), 150), use_container_width=True)
                    match = re.search(r"(\d+)", fname.rsplit(".", 1)[0])
                    label = f"{int(match.group(1)):03d}" if match else fname
                    st.markdown(f'<p class="frame-caption">[{label}]</p>', unsafe_allow_html=True)
    st.divider()
    st.caption("VISUALIZATION ONLY · YOLO, Kalman, training code, weights and experimental results are not modified or executed here.")


media_tab, experiment_tab, model_tab, dataset_tab = st.tabs(["MEDIA LAB", "EXPERIMENT LAB", "MODEL", "DATASET"])
with media_tab:
    render_media_lab()
texts = _artifact_texts()
with experiment_tab:
    render_overview(texts)
    render_audits(texts)
    render_plots()
    render_trajectory(texts)
    render_error_analysis(texts)
    render_findings(texts)
    render_reproducibility(texts)
    render_architecture()
with model_tab:
    render_overview(texts)
    render_model(texts)
with dataset_tab:
    render_overview(texts)
    render_dataset(texts)
