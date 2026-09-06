"""
Ballistic Tracking — Sequence Explorer (visualization only).

Loads the 130 Photron frames from Google Drive via the Drive API
(service account credentials provided exclusively through Streamlit Secrets).

This app does not train YOLO, does not run the Kalman filter, and does not
modify experimental results.
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

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
        font-size: 0.75rem;
        color: #8b949e;
        text-align: center;
    }
    div[data-testid="stAlert"] {
        background-color: #21262d;
        border: 1px solid #30363d;
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

DEFAULT_PHOTRON_FOLDER_ID = "1Bj_5IX9h9GATjQbU6D2ouU2BQSHkZA65"

LOCAL_CANDIDATE_DIRS = [
    Path("/content/drive/MyDrive/ballistic_tracking/extracted/photron"),
    Path("data/extracted/photron"),
    Path("data/photron"),
    Path("extracted/photron"),
]

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _secret_get(key: str, default: Any = None) -> Any:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _service_account_info() -> Optional[Dict[str, Any]]:
    try:
        sa = st.secrets.get("gcp_service_account")
        if sa is None:
            return None
        info = dict(sa)
        if "private_key" in info and isinstance(info["private_key"], str):
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        return info
    except Exception:
        return None


def _folder_id() -> str:
    return str(_secret_get("gdrive_photron_folder_id", DEFAULT_PHOTRON_FOLDER_ID))


def _build_drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    info = _service_account_info()
    if not info:
        raise RuntimeError(
            "Streamlit secret [gcp_service_account] nao encontrado. "
            "Configure a service account em Secrets (Streamlit Cloud)."
        )

    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=SCOPES
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _natural_frame_key(name: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", name)
    if m:
        return (int(m.group(1)), name.lower())
    return (10**9, name.lower())


@st.cache_data(show_spinner="Listando frames no Google Drive…", ttl=600)
def list_drive_frames(folder_id: str) -> List[Dict[str, str]]:
    service = _build_drive_service()
    query = (
        f"'{folder_id}' in parents and trashed = false and ("
        "mimeType = 'image/png' or mimeType = 'image/jpeg' or "
        "mimeType = 'image/jpg')"
    )
    files: List[Dict[str, str]] = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=200,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for f in response.get("files", []):
            files.append({"id": f["id"], "name": f["name"]})
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    files.sort(key=lambda f: _natural_frame_key(f["name"]))
    return files


@st.cache_data(show_spinner=False, ttl=3600)
def download_drive_image(file_id: str, name: str) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload

    service = _build_drive_service()
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def make_thumbnail(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    ratio = max_width / float(image.width)
    size = (max_width, max(1, int(image.height * ratio)))
    return image.resize(size, Image.Resampling.BILINEAR)


def resolve_local_frames_dir() -> Optional[Path]:
    secret_dir = _secret_get("frames_dir")
    if secret_dir:
        p = Path(str(secret_dir)).expanduser()
        if p.is_dir():
            return p
    env_dir = os.environ.get("BALLISTIC_FRAMES_DIR")
    if env_dir:
        p = Path(env_dir).expanduser()
        if p.is_dir():
            return p
    for candidate in LOCAL_CANDIDATE_DIRS:
        if candidate.is_dir():
            return candidate
    return None


@st.cache_data(show_spinner="Carregando frames locais…")
def list_local_frames(dir_str: str) -> List[Dict[str, str]]:
    directory = Path(dir_str)
    paths = sorted(
        set(
            list(directory.glob("frame_*.png"))
            + list(directory.glob("*.png"))
            + list(directory.glob("*.jpg"))
            + list(directory.glob("*.jpeg"))
        ),
        key=lambda p: _natural_frame_key(p.name),
    )
    return [{"id": str(p.resolve()), "name": p.name} for p in paths]


def load_local_image_bytes(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


st.sidebar.header("Sequence source")

drive_error: Optional[str] = None
frame_index: List[Dict[str, str]] = []
source = "none"
source_mode = "none"

sa_info = _service_account_info()
folder_id = _folder_id()

if sa_info is not None:
    try:
        frame_index = list_drive_frames(folder_id)
        if frame_index:
            source = f"Google Drive API (folder `{folder_id}`)"
            source_mode = "drive"
            st.sidebar.success(
                f"Google Drive conectado\n\n**{len(frame_index)}** frames"
            )
            st.sidebar.caption(f"Pasta: `{folder_id}`")
        else:
            drive_error = (
                f"A pasta do Drive (`{folder_id}`) esta acessivel, "
                "mas nenhuma imagem PNG/JPEG foi encontrada. "
                "Verifique se a service account tem permissao de leitura "
                "na pasta ballistic_tracking/extracted/photron."
            )
    except Exception as exc:
        drive_error = (
            f"Falha ao acessar o Google Drive: "
            f"`{type(exc).__name__}: {exc}`"
        )
else:
    drive_error = (
        "Secret [gcp_service_account] nao configurado. "
        "No Streamlit Cloud, adicione a service account em Settings → Secrets."
    )

if not frame_index:
    local_dir = resolve_local_frames_dir()
    if local_dir is not None:
        frame_index = list_local_frames(str(local_dir))
        if frame_index:
            source = f"local path ({local_dir})"
            source_mode = "local"
            st.sidebar.success(
                f"Frames locais\n\n**{len(frame_index)}** frames"
            )
            st.sidebar.caption(str(local_dir))

if not frame_index:
    st.error(
        "### Frames nao disponiveis\n\n"
        "As 130 imagens Photron residem no Google Drive em "
        "`ballistic_tracking/extracted/photron/` e **nao** estao no GitHub.\n\n"
        f"**Status do Drive:** {drive_error or 'desconhecido'}\n\n"
        "### Configuracao necessaria (Streamlit Cloud)\n\n"
        "1. Crie uma **Google Cloud service account** e compartilhe a pasta "
        "`extracted/photron` (ou `ballistic_tracking`) com o e-mail da "
        "service account como **Viewer** (somente leitura).\n"
        "2. Em **Settings → Secrets** do app, adicione o bloco "
        "`[gcp_service_account]` com type, project_id, private_key_id, "
        "private_key, client_email, client_id e token_uri.\n"
        "3. Opcional: `gdrive_photron_folder_id = "
        '"1Bj_5IX9h9GATjQbU6D2ouU2BQSHkZA65"`.\n\n'
        "Nenhuma credencial e armazenada no repositorio GitHub."
    )
    st.stop()

st.sidebar.caption(f"Fonte: `{source}`")

st.sidebar.header("Visualizacao")
view = st.sidebar.radio("Modo", ["Grade", "Frame selecionada"], index=0)
show_fusion = st.sidebar.checkbox(
    "Tentar visualizacao Fusion",
    value=False,
    help=(
        "Fusion so aparece se existirem overlays por frame. "
        "No experimento atual a visualizacao principal e Raw."
    ),
)


def fetch_image(entry: Dict[str, str]) -> Image.Image:
    if source_mode == "drive":
        data = download_drive_image(entry["id"], entry["name"])
    else:
        data = load_local_image_bytes(entry["id"])
    return load_image_from_bytes(data)


if view == "Frame selecionada":
    index = (
        st.sidebar.slider(
            "Frame",
            min_value=1,
            max_value=len(frame_index),
            value=min(max(1, (len(frame_index) + 1) // 2), len(frame_index)),
        )
        - 1
    )
    entry = frame_index[index]

    with st.spinner(f"Carregando {entry['name']}…"):
        image = fetch_image(entry)

    col_img, col_meta = st.columns([3, 1])
    with col_img:
        st.subheader("Raw")
        st.image(image, caption=entry["name"], use_container_width=True)
        if show_fusion:
            st.subheader("Fusion")
            st.info(
                "Nenhum overlay de fusion (deteccao YOLO / Kalman) esta "
                "disponivel como arquivo estatico por frame neste repositorio. "
                "Use `docs/demo_tracking.mp4` para a visualizacao combinada."
            )
    with col_meta:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Frame", f"{index + 1} / {len(frame_index)}")
        st.write(f"**Arquivo:** `{entry['name']}`")
        st.write(f"**Resolucao:** `{image.width} × {image.height}`")
        st.write("**Modo:** Raw (resolucao completa)")
        st.write(f"**Origem:** `{source_mode}`")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    columns = st.sidebar.slider("Colunas", 2, 10, 5)
    thumbnail_width = st.sidebar.slider("Largura da thumbnail", 80, 280, 140)

    st.subheader(f"Todas as frames — {len(frame_index)}")
    st.caption(
        "Ordem numerica pelo nome (`frame_000` … `frame_129`), "
        "preservando a sequencia temporal do shot."
    )

    progress = st.progress(0, text="Carregando thumbnails…")
    thumbs: List[Tuple[str, Image.Image]] = []
    total = len(frame_index)
    for i, entry in enumerate(frame_index):
        try:
            img = fetch_image(entry)
            thumbs.append((entry["name"], make_thumbnail(img, thumbnail_width)))
        except Exception:
            placeholder = Image.new(
                "RGB",
                (thumbnail_width, max(1, thumbnail_width // 2)),
                (40, 40, 40),
            )
            thumbs.append((entry["name"], placeholder))
        if total:
            progress.progress((i + 1) / total, text=f"Thumbnail {i + 1}/{total}")
    progress.empty()

    for start in range(0, len(thumbs), columns):
        row = thumbs[start : start + columns]
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
    "Somente visualizacao · codigo cientifico (`src/`), pesos (`models/best.pt`) "
    "e resultados experimentais permanecem inalterados · "
    f"{len(frame_index)} frames · fonte: {source_mode}"
)
