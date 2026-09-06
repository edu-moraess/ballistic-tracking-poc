"""
Ballistic Tracking — Sequence Explorer (visualization only).

Primary source: upload of photron_frames.zip containing frame_000.png … frame_129.png.
Optional fallback: local directory of PNGs if already present on disk.

This app does not train YOLO, does not run the Kalman filter, and does not
modify experimental results.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page / theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ballistic Tracking — Sequence Explorer",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    h1, h2, h3, h4 { color: #f0f6fc !important; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .frame-caption {
        font-size: 0.72rem;
        color: #8b949e;
        text-align: center;
        margin-top: 0.15rem;
    }
    div[data-testid="stAlert"] {
        background-color: #21262d;
        border: 1px solid #30363d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("BALLISTIC TRACKING PoC")
st.subheader("Sequence Explorer")
st.caption("Computer Vision × State Estimation · visualization only")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
LOCAL_CANDIDATES = [
    Path("/content/drive/MyDrive/ballistic_tracking/extracted/photron"),
    Path("data/extracted/photron"),
    Path("data/photron"),
    Path("extracted/photron"),
]


def natural_frame_key(name: str) -> Tuple[int, str]:
    """frame_012.png → (12, name); unknown names sort last."""
    m = re.search(r"(\d+)", Path(name).stem)
    if m:
        return (int(m.group(1)), name.lower())
    return (10**9, name.lower())


def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_SUFFIXES


@st.cache_data(show_spinner="Extracting ZIP…")
def load_frames_from_zip(zip_bytes: bytes) -> List[Tuple[str, bytes]]:
    """
    Extract valid PNG/JPG/JPEG entries from a ZIP archive.
    Returns list of (filename, raw_bytes) sorted by frame number.
    Raises ValueError on corrupt ZIP or empty result.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Arquivo ZIP inválido ou corrompido: {exc}") from exc

    frames: List[Tuple[str, bytes]] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        # Use basename so nested folders still work
        name = Path(info.filename).name
        if not name or name.startswith("."):
            continue
        if not is_image_name(name):
            continue
        try:
            data = zf.read(info)
        except Exception:
            continue
        if not data:
            continue
        frames.append((name, data))

    if not frames:
        raise ValueError(
            "Nenhuma imagem PNG/JPG/JPEG válida encontrada dentro do ZIP."
        )

    frames.sort(key=lambda item: natural_frame_key(item[0]))
    return frames


@st.cache_data(show_spinner="Loading local frames…")
def load_frames_from_dir(dir_str: str) -> List[Tuple[str, bytes]]:
    directory = Path(dir_str)
    paths = [
        p
        for p in directory.iterdir()
        if p.is_file() and is_image_name(p.name)
    ]
    paths.sort(key=lambda p: natural_frame_key(p.name))
    return [(p.name, p.read_bytes()) for p in paths]


def resolve_local_dir() -> Optional[Path]:
    for candidate in LOCAL_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return None


def decode_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def make_thumbnail(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    ratio = max_width / float(image.width)
    size = (max_width, max(1, int(image.height * ratio)))
    return image.resize(size, Image.Resampling.BILINEAR)


# ---------------------------------------------------------------------------
# Source: ZIP upload (primary) / local directory (fallback)
# ---------------------------------------------------------------------------
st.sidebar.header("Source")

uploaded_zip = st.sidebar.file_uploader(
    "Upload photron_frames.zip",
    type=["zip"],
    help="ZIP contendo frame_000.png … frame_129.png",
)

frames: List[Tuple[str, bytes]] = []
source_label = "none"
load_error: Optional[str] = None

if uploaded_zip is not None:
    try:
        zip_bytes = uploaded_zip.getvalue()
        frames = load_frames_from_zip(zip_bytes)
        source_label = f"ZIP · {uploaded_zip.name}"
    except ValueError as exc:
        load_error = str(exc)
    except Exception as exc:
        load_error = f"Falha ao processar o ZIP: {type(exc).__name__}: {exc}"
else:
    local_dir = resolve_local_dir()
    if local_dir is not None:
        try:
            frames = load_frames_from_dir(str(local_dir))
            if frames:
                source_label = f"local · {local_dir}"
        except Exception as exc:
            load_error = f"Falha ao ler diretório local: {exc}"

if load_error:
    st.error(load_error)

if not frames:
    st.info(
        "Faça upload de **photron_frames.zip** na barra lateral.\n\n"
        "O arquivo deve conter as imagens Photron "
        "(`frame_000.png` … `frame_129.png`).\n\n"
        "As frames **não** estão versionadas no GitHub; "
        "o ZIP é a fonte principal deste viewer."
    )
    st.stop()

st.success(f"✓ {len(frames)} frames loaded")
st.sidebar.caption(f"Source: {source_label}")

# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------
st.sidebar.header("Frame selector")
selected_index = (
    st.sidebar.slider(
        "Frame",
        min_value=1,
        max_value=len(frames),
        value=min(max(1, (len(frames) + 1) // 2), len(frames)),
    )
    - 1
)

name, raw = frames[selected_index]
image = decode_image(raw)

# ---------------------------------------------------------------------------
# Selected frame (large)
# ---------------------------------------------------------------------------
col_img, col_meta = st.columns([3, 1])

with col_img:
    st.markdown(f"### FRAME {selected_index + 1:03d} / {len(frames)}")
    st.image(image, caption=name, use_container_width=True)

with col_meta:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**Metadata**")
    st.write(f"**Frame:** `{selected_index + 1} / {len(frames)}`")
    st.write(f"**File:** `{name}`")
    st.write(f"**Resolution:** `{image.width} × {image.height}`")
    st.write(f"**Total frames:** `{len(frames)}`")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sequence grid
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### SEQUENCE")
st.caption("Thumbnails ordered by frame number (`frame_000` → `frame_129`).")

columns = st.sidebar.slider("Grid columns", 4, 12, 8)
thumb_w = st.sidebar.slider("Thumbnail width", 60, 200, 100)

for start in range(0, len(frames), columns):
    row = frames[start : start + columns]
    cols = st.columns(columns)
    for col, (fname, fbytes) in zip(cols, row):
        with col:
            try:
                thumb = make_thumbnail(decode_image(fbytes), thumb_w)
                st.image(thumb, width=thumb_w)
            except Exception:
                st.write("—")
            # Show short index label: 001, 002, …
            m = re.search(r"(\d+)", Path(fname).stem)
            label = f"{int(m.group(1)):03d}" if m else fname
            st.markdown(
                f'<p class="frame-caption">[{label}]</p>',
                unsafe_allow_html=True,
            )

st.divider()
st.caption(
    "Visualization only · scientific code (`src/`), weights (`models/best.pt`) "
    "and experimental results remain unchanged."
)
