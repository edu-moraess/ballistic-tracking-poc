"""
Ballistic Tracking — Sequence Explorer (visualization only).

Primary source: upload of any .zip archive containing Photron PNG/JPG frames.
Images are discovered recursively inside the archive (no local filesystem scan).

This app does not train YOLO, does not run the Kalman filter, and does not
modify experimental results.
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import Dict, List, Optional, Tuple

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
# Helpers (string-based; no filesystem dependency for ZIP contents)
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _basename(member_name: str) -> str:
    """Return file basename from a zip member path (handles / and \\)."""
    name = member_name.replace("\\", "/")
    if name.endswith("/"):
        name = name[:-1]
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name


def _is_image_member(member_name: str) -> bool:
    base = _basename(member_name).lower()
    return base.endswith(IMAGE_EXTENSIONS)


def _is_skippable_member(member_name: str) -> bool:
    """Directories, macOS metadata, hidden files."""
    norm = member_name.replace("\\", "/")
    if not norm or norm.endswith("/"):
        return True
    if norm.startswith("__MACOSX/") or "/__MACOSX/" in norm:
        return True
    base = _basename(norm)
    if not base or base.startswith(".") or base.startswith("._"):
        return True
    return False


def natural_frame_key(name: str) -> Tuple[int, str]:
    """frame_012.png → (12, name); names without digits sort last."""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    m = re.search(r"(\d+)", stem)
    if m:
        return (int(m.group(1)), name.lower())
    return (10**9, name.lower())


def _validate_image_bytes(data: bytes) -> bool:
    """
    Return True if data is a readable raster image.
    Uses verify() then relies on original bytes for later display reopen.
    """
    if not data or len(data) < 24:
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        # verify() leaves the image in an unusable state; reopen + load
        with Image.open(io.BytesIO(data)) as im:
            im.load()
        return True
    except Exception:
        return False


@st.cache_data(show_spinner="Extracting ZIP…", ttl=3600)
def load_frames_from_zip(zip_bytes: bytes) -> Tuple[List[Tuple[str, bytes]], Dict[str, int]]:
    """
    Read any ZIP from bytes and collect valid PNG/JPG/JPEG members recursively.

    Returns
    -------
    frames : list of (basename, raw_bytes), sorted frame_000 → frame_129
    stats  : diagnostic counters
    """
    stats: Dict[str, int] = {
        "members_total": 0,
        "skipped_meta": 0,
        "candidates": 0,
        "unreadable": 0,
        "duplicates": 0,
        "loaded": 0,
    }

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"ZIP_OPEN_FAILED: não foi possível abrir o arquivo como ZIP ({exc})."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"ZIP_OPEN_FAILED: erro ao abrir o ZIP ({type(exc).__name__}: {exc})."
        ) from exc

    # Force full central-directory read / basic integrity check
    try:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(
                f"ZIP_CORRUPT: entrada corrompida detectada no ZIP: {bad}."
            )
    except ValueError:
        raise
    except Exception:
        # testzip can fail on some valid archives; continue and rely on per-file reads
        pass

    frames: List[Tuple[str, bytes]] = []
    seen: set = set()

    for info in zf.infolist():
        stats["members_total"] += 1
        member = info.filename

        if info.is_dir() or _is_skippable_member(member):
            stats["skipped_meta"] += 1
            continue

        if not _is_image_member(member):
            continue

        stats["candidates"] += 1
        base = _basename(member)

        if base in seen:
            stats["duplicates"] += 1
            continue

        try:
            data = zf.read(info)
        except Exception:
            stats["unreadable"] += 1
            continue

        if not _validate_image_bytes(data):
            stats["unreadable"] += 1
            continue

        seen.add(base)
        frames.append((base, data))
        stats["loaded"] += 1

    if not frames:
        if stats["candidates"] == 0:
            raise ValueError(
                "ZIP_NO_IMAGES: o ZIP abriu, mas nenhuma entrada "
                ".png/.jpg/.jpeg foi encontrada (incluindo subpastas). "
                f"Membros totais={stats['members_total']}, "
                f"metadados ignorados={stats['skipped_meta']}."
            )
        raise ValueError(
            "ZIP_UNREADABLE_IMAGES: existem candidatas a imagem no ZIP, "
            "mas nenhuma pôde ser lida pelo PIL. "
            f"candidatas={stats['candidates']}, "
            f"ilegíveis={stats['unreadable']}."
        )

    frames.sort(key=lambda item: natural_frame_key(item[0]))
    return frames, stats


def decode_image(data: bytes) -> Image.Image:
    """Reopen image bytes for display (safe after verify())."""
    im = Image.open(io.BytesIO(data))
    im.load()
    return im.convert("RGB")


def make_thumbnail(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    ratio = max_width / float(image.width)
    size = (max_width, max(1, int(image.height * ratio)))
    return image.resize(size, Image.Resampling.BILINEAR)


# ---------------------------------------------------------------------------
# Source: any ZIP upload
# ---------------------------------------------------------------------------
st.sidebar.header("Source")

uploaded_zip = st.sidebar.file_uploader(
    "Upload ZIP with frames",
    type=["zip"],
    help="Any .zip file; PNG/JPG/JPEG are collected recursively from the archive.",
)

frames: List[Tuple[str, bytes]] = []
source_label = "none"
load_error: Optional[str] = None
stats: Dict[str, int] = {}

if uploaded_zip is not None:
    try:
        zip_bytes = uploaded_zip.getvalue()
        if not zip_bytes:
            raise ValueError("ZIP_EMPTY: o arquivo enviado está vazio (0 bytes).")
        frames, stats = load_frames_from_zip(zip_bytes)
        source_label = f"ZIP · {uploaded_zip.name}"
    except ValueError as exc:
        load_error = str(exc)
    except Exception as exc:
        load_error = f"ZIP_ERROR: {type(exc).__name__}: {exc}"

if load_error:
    st.error(load_error)

if not frames:
    st.info(
        "Faça upload de um arquivo **.zip** na barra lateral.\n\n"
        "O ZIP pode ter qualquer nome. O app procura **recursivamente** "
        "por imagens PNG/JPG/JPEG (ex.: `frame_000.png` … `frame_129.png`).\n\n"
        "As frames **não** estão versionadas no GitHub; "
        "o ZIP é a fonte principal deste viewer."
    )
    st.stop()

st.success(f"✓ {len(frames)} frames loaded")
if uploaded_zip is not None:
    st.caption(
        f"Archive: `{uploaded_zip.name}` · "
        f"images: {len(frames)} · "
        f"members scanned: {stats.get('members_total', '?')}"
    )
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
            m = re.search(r"(\d+)", fname.rsplit(".", 1)[0])
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
