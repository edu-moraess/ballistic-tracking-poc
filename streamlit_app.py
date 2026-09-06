"""
Ballistic Tracking — Sequence Explorer (visualization only).

Primary source: upload of any .zip archive containing Photron PNG/JPG frames.
Images are discovered recursively inside the archive (no local filesystem scan).

Mobile-friendly upload: validates ZIP by magic bytes (not only file extension).

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
    """Return True if data is a readable raster image."""
    if not data or len(data) < 24:
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
        return True
    except Exception:
        return False


def _looks_like_zip(data: bytes) -> bool:
    """ZIP local-file or empty-archive magic numbers."""
    if len(data) < 4:
        return False
    # PK\x03\x04 (file), PK\x05\x06 (empty), PK\x07\x08 (spanned)
    return data[0:2] == b"PK"


@st.cache_data(show_spinner="Extracting ZIP…", ttl=3600)
def load_frames_from_zip(zip_bytes: bytes) -> Tuple[List[Tuple[str, bytes]], Dict[str, int]]:
    """
    Read any ZIP from bytes and collect valid PNG/JPG/JPEG members recursively.

    Returns (frames, stats) with frames sorted frame_000 → frame_129.
    """
    stats: Dict[str, int] = {
        "members_total": 0,
        "skipped_meta": 0,
        "candidates": 0,
        "unreadable": 0,
        "duplicates": 0,
        "loaded": 0,
        "bytes": len(zip_bytes),
    }

    if not zip_bytes:
        raise ValueError("ZIP_EMPTY: o arquivo enviado está vazio (0 bytes).")

    if not _looks_like_zip(zip_bytes):
        raise ValueError(
            "ZIP_OPEN_FAILED: o conteúdo recebido não parece um arquivo ZIP "
            f"(magic={zip_bytes[:4]!r}, tamanho={len(zip_bytes)} bytes). "
            "No celular, apague o item vermelho, confirme o Wi‑Fi e envie "
            "de novo um único photron_frames.zip."
        )

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"ZIP_OPEN_FAILED: não foi possível abrir como ZIP ({exc}). "
            f"Bytes recebidos={len(zip_bytes)}. "
            "Upload incompleto é comum em redes móveis — tente novamente no Wi‑Fi."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"ZIP_OPEN_FAILED: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(
                f"ZIP_CORRUPT: entrada corrompida no ZIP: {bad}."
            )
    except ValueError:
        raise
    except Exception:
        pass

    frames: List[Tuple[str, bytes]] = []
    seen: set = set()
    sample_members: List[str] = []

    for info in zf.infolist():
        stats["members_total"] += 1
        member = info.filename

        if len(sample_members) < 12:
            sample_members.append(member)

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
        sample = ", ".join(repr(m) for m in sample_members[:8])
        if stats["candidates"] == 0:
            raise ValueError(
                "ZIP_NO_IMAGES: o ZIP abriu, mas nenhuma entrada "
                ".png/.jpg/.jpeg foi encontrada. "
                f"Membros={stats['members_total']}, "
                f"meta={stats['skipped_meta']}. "
                f"Exemplos: [{sample}]. "
                "Envie o ZIP das frames Photron, não o ZIP do repositório GitHub."
            )
        raise ValueError(
            "ZIP_UNREADABLE_IMAGES: candidatas encontradas, mas nenhuma "
            "foi legível pelo PIL. "
            f"candidatas={stats['candidates']}, ilegíveis={stats['unreadable']}."
        )

    frames.sort(key=lambda item: natural_frame_key(item[0]))
    return frames, stats


def decode_image(data: bytes) -> Image.Image:
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
# Source: ZIP upload (mobile-friendly)
# ---------------------------------------------------------------------------
st.sidebar.header("Source")

st.sidebar.caption(
    "Celular: envie **um** arquivo só, no Wi‑Fi se possível. "
    "Apague itens vermelhos antes de tentar de novo."
)

# No strict extension filter — some mobile browsers report odd MIME types.
# We validate ZIP content ourselves via magic bytes + zipfile.
uploaded_zip = st.sidebar.file_uploader(
    "Upload ZIP with frames",
    type=None,
    accept_multiple_files=False,
    help=(
        "Arquivo ZIP com frame_000.png … frame_129.png. "
        "No celular: um arquivo por vez, preferencialmente no Wi‑Fi."
    ),
)

frames: List[Tuple[str, bytes]] = []
source_label = "none"
load_error: Optional[str] = None
stats: Dict[str, int] = {}

if uploaded_zip is not None:
    try:
        # Read once into memory; show size for mobile debugging
        zip_bytes = uploaded_zip.getvalue()
        size_mb = len(zip_bytes) / (1024 * 1024)
        st.sidebar.write(
            f"Recebido: `{uploaded_zip.name}` · **{size_mb:.2f} MB** "
            f"({len(zip_bytes)} bytes)"
        )

        if len(zip_bytes) < 100:
            raise ValueError(
                "ZIP_EMPTY: upload parece incompleto "
                f"({len(zip_bytes)} bytes). Apague o item e envie de novo."
            )

        # Soft warning if name has no .zip (mobile sometimes strips / renames)
        name_lower = (uploaded_zip.name or "").lower()
        if name_lower and not (
            name_lower.endswith(".zip")
            or name_lower.endswith(".png")
            or _looks_like_zip(zip_bytes)
        ):
            st.sidebar.warning(
                "O arquivo não tem extensão .zip; tentando abrir mesmo assim…"
            )

        frames, stats = load_frames_from_zip(zip_bytes)
        source_label = f"ZIP · {uploaded_zip.name}"
    except ValueError as exc:
        load_error = str(exc)
    except Exception as exc:
        load_error = f"ZIP_ERROR: {type(exc).__name__}: {exc}"

if load_error:
    st.error(load_error)
    st.warning(
        "Dica (celular): remova os arquivos com ícone vermelho (X) → "
        "conecte no Wi‑Fi → envie **apenas um** `photron_frames.zip` → "
        "aguarde o upload terminar antes de tocar na tela."
    )

if not frames:
    st.info(
        "Faça upload do ZIP **das frames Photron** na barra lateral.\n\n"
        "O arquivo deve conter `frame_000.png` … `frame_129.png`.\n\n"
        "**Não** envie o ZIP do repositório GitHub (`ballistic-tracking-poc.zip`).\n\n"
        "No **celular**: um arquivo por vez, no Wi‑Fi, e apague itens vermelhos "
        "antes de reenviar."
    )
    st.stop()

st.success(f"✓ {len(frames)} frames loaded")
if uploaded_zip is not None:
    st.caption(
        f"Archive: `{uploaded_zip.name}` · "
        f"images: {len(frames)} · "
        f"members scanned: {stats.get('members_total', '?')} · "
        f"bytes: {stats.get('bytes', '?')}"
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
