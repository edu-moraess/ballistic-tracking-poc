"""
Ballistic Tracking — Sequence Explorer (visualization only).

This app does not train YOLO, does not run the Kalman filter, and does not
modify experimental results. It only loads and displays existing Photron frames.
"""

from __future__ import annotations

import os
import tempfile
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

# Dark professional theme (works even when Streamlit theme is light)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    h1, h2, h3, h4 {
        color: #f0f6fc !important;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .frame-caption {
        font-size: 0.75rem;
        color: #8b949e;
        text-align: center;
    }
    div[data-testid="stAlert"] {
        background-color: #21262d;
        border: 1px solid #30363d;
    }
    .stButton > button {
        background-color: #238636;
        color: white;
        border: 1px solid #2ea043;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎞️ Ballistic Tracking — Sequence Explorer")
st.caption(
    "Visualização somente. Este app **não** treina YOLO, **não** executa o Kalman "
    "e **não** altera os resultados do experimento."
)

# ---------------------------------------------------------------------------
# Path resolution (local mount / secrets / env) — no credentials in source
# ---------------------------------------------------------------------------

# Known locations where the 130 Photron frames may appear when Drive is mounted
# or when the user points a local mirror at the extracted sequence.
CANDIDATE_DIRS = [
    Path("/content/drive/MyDrive/ballistic_tracking/extracted/photron"),
    Path("/content/drive/MyDrive/ballistic_tracking/extracted/photron/"),
    Path("data/extracted/photron"),
    Path("data/photron"),
    Path("extracted/photron"),
]


def _secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read from Streamlit secrets without crashing when secrets are absent."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def resolve_frames_dir() -> Tuple[Optional[Path], str]:
    """
    Resolve the directory that contains frame_*.png.

    Priority:
      1. Streamlit secret  frames_dir
      2. Environment var   BALLISTIC_FRAMES_DIR
      3. Known candidate paths (Colab Drive mount, local mirrors)
    """
    secret_dir = _secret("frames_dir")
    if secret_dir:
        p = Path(secret_dir).expanduser()
        if p.is_dir():
            return p, "streamlit secrets (frames_dir)"

    env_dir = os.environ.get("BALLISTIC_FRAMES_DIR")
    if env_dir:
        p = Path(env_dir).expanduser()
        if p.is_dir():
            return p, "environment variable BALLISTIC_FRAMES_DIR"

    for candidate in CANDIDATE_DIRS:
        if candidate.is_dir():
            return candidate, f"local path ({candidate})"

    return None, "not found"


def list_frame_paths(directory: Path) -> List[Path]:
    """Return sorted frame paths (png/jpg) from a directory."""
    paths = sorted(
        [
            *directory.glob("frame_*.png"),
            *directory.glob("frame_*.jpg"),
            *directory.glob("*.png"),
            *directory.glob("*.jpg"),
            *directory.glob("*.jpeg"),
        ]
    )
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in paths:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    return unique


@st.cache_data(show_spinner="Carregando frames…")
def load_frames_from_dir(dir_str: str) -> List[Tuple[str, Image.Image]]:
    directory = Path(dir_str)
    paths = list_frame_paths(directory)
    frames: List[Tuple[str, Image.Image]] = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            frames.append((path.name, img))
        except Exception:
            continue
    return frames


@st.cache_data(show_spinner=False)
def decode_uploaded_frames(files) -> List[Tuple[str, Image.Image]]:
    frames: List[Tuple[str, Image.Image]] = []
    for file in files:
        try:
            image = Image.open(file).convert("RGB")
            frames.append((file.name, image))
        except Exception:
            continue
    return sorted(frames, key=lambda item: item[0])


# ---------------------------------------------------------------------------
# Sidebar — source selection
# ---------------------------------------------------------------------------
st.sidebar.header("Sequence source")

frames_dir, source_label = resolve_frames_dir()

if frames_dir is not None:
    st.sidebar.success(f"Diretório detectado\n\n`{frames_dir}`")
    st.sidebar.caption(f"Origem: {source_label}")
else:
    st.sidebar.warning(
        "Nenhum diretório de frames encontrado automaticamente.\n\n"
        "Opções:\n"
        "1. Monte o Google Drive (Colab) em "
        "`/content/drive/MyDrive/ballistic_tracking/extracted/photron`\n"
        "2. Defina o secret `frames_dir` no Streamlit\n"
        "3. Defina a variável de ambiente `BALLISTIC_FRAMES_DIR`\n"
        "4. Faça upload manual das PNGs abaixo"
    )

uploaded = st.sidebar.file_uploader(
    "Upload manual das frames PNG/JPG",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Selecione as 130 frames extraídas do Photron (frame_000.png …).",
)

# ---------------------------------------------------------------------------
# Load frames
# ---------------------------------------------------------------------------
frames: List[Tuple[str, Image.Image]] = []
source = "none"

if uploaded:
    frames = decode_uploaded_frames(uploaded)
    source = "upload"
elif frames_dir is not None:
    frames = load_frames_from_dir(str(frames_dir))
    source = source_label
else:
    frames = []
    source = "none"

if not frames:
    st.error(
        "### Frames não disponíveis\n\n"
        "As 130 imagens Photron **não** estão versionadas no GitHub "
        "(política de tamanho / dados brutos). Elas residem no Google Drive:\n\n"
        "`ballistic_tracking/extracted/photron/`\n\n"
        "**Como conectar este app às frames:**\n\n"
        "1. **Colab / ambiente com Drive montado** — o caminho "
        "`/content/drive/MyDrive/ballistic_tracking/extracted/photron` "
        "é detectado automaticamente.\n"
        "2. **Streamlit secrets** — adicione em `.streamlit/secrets.toml`:\n"
        "   ```toml\n"
        "   frames_dir = \"/caminho/local/para/extracted/photron\"\n"
        "   ```\n"
        "3. **Variável de ambiente** — `BALLISTIC_FRAMES_DIR=/caminho/para/photron`\n"
        "4. **Upload manual** — use o seletor na barra lateral.\n\n"
        "Nenhuma credencial ou token deve ser colocado no código-fonte."
    )
    st.stop()

st.sidebar.success(f"**{len(frames)}** frame(s) carregadas")
st.sidebar.caption(f"Fonte: `{source}`")

# ---------------------------------------------------------------------------
# View controls
# ---------------------------------------------------------------------------
st.sidebar.header("Visualização")
view = st.sidebar.radio(
    "Modo",
    ["Grade", "Frame selecionada"],
    index=0,
)

show_fusion = st.sidebar.checkbox(
    "Tentar visualização Fusion",
    value=False,
    help="Fusion só aparece se existirem overlays/resultados adicionais. "
    "No experimento atual a visualização principal é Raw.",
)

# ---------------------------------------------------------------------------
# Main views
# ---------------------------------------------------------------------------
if view == "Frame selecionada":
    index = (
        st.sidebar.slider(
            "Frame",
            min_value=1,
            max_value=len(frames),
            value=min(max(1, (len(frames) + 1) // 2), len(frames)),
        )
        - 1
    )
    name, image = frames[index]

    col_img, col_meta = st.columns([3, 1])
    with col_img:
        st.subheader("Raw")
        st.image(image, caption=name, use_container_width=True)

        if show_fusion:
            st.subheader("Fusion")
            st.info(
                "Nenhum overlay de fusion (detecção YOLO / Kalman) está "
                "disponível como arquivo estático por frame neste repositório. "
                "Use `docs/demo_tracking.mp4` para a visualização combinada "
                "gerada pelo pipeline de tracking."
            )

    with col_meta:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Frame", f"{index + 1} / {len(frames)}")
        st.write(f"**Arquivo:** `{name}`")
        st.write(f"**Resolução:** `{image.width} × {image.height}`")
        st.write(f"**Modo:** Raw")
        st.write(f"**Origem:** `{source}`")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    columns = st.sidebar.slider("Colunas", 2, 10, 5)
    thumbnail_width = st.sidebar.slider("Largura da thumbnail", 80, 280, 140)

    st.subheader(f"Todas as frames — {len(frames)}")
    st.caption(
        "Ordem determinada pelo nome do arquivo (`frame_000` …), "
        "preservando a sequência temporal original do shot."
    )

    for start in range(0, len(frames), columns):
        row = frames[start : start + columns]
        cols = st.columns(columns)
        for col, (name, image) in zip(cols, row):
            with col:
                st.image(image, width=thumbnail_width)
                st.markdown(
                    f'<p class="frame-caption">{name}</p>',
                    unsafe_allow_html=True,
                )

st.divider()
st.caption(
    "Somente visualização · código científico (`src/`), pesos (`models/best.pt`) "
    "e resultados experimentais permanecem inalterados."
)
