from pathlib import Path
from typing import List, Tuple

import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Ballistic Tracking — Sequence Explorer",
    page_icon="🎞️",
    layout="wide",
)

st.title("Ballistic Tracking — Sequence Explorer")
st.caption(
    "Visualização somente. Este app não treina YOLO, não executa o Kalman e não altera os resultados do experimento."
)


DEFAULT_FRAME_DIRS = [
    Path("data/extracted/photron"),
    Path("data/photron"),
]


def load_local_frames() -> List[Tuple[str, Image.Image]]:
    """Load existing PNG/JPG frames without modifying the experiment."""
    for directory in DEFAULT_FRAME_DIRS:
        if directory.exists():
            paths = sorted(
                [
                    *directory.glob("*.png"),
                    *directory.glob("*.jpg"),
                    *directory.glob("*.jpeg"),
                ]
            )
            if paths:
                return [(path.name, Image.open(path).convert("RGB")) for path in paths]
    return []


@st.cache_data(show_spinner=False)
def decode_uploaded_frames(files):
    frames = []
    for file in files:
        try:
            image = Image.open(file).convert("RGB")
            frames.append((file.name, image))
        except Exception:
            continue
    return sorted(frames, key=lambda item: item[0])


local_frames = load_local_frames()

st.sidebar.header("Sequence")
uploaded = st.sidebar.file_uploader(
    "Carregue as frames PNG/JPG",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Você pode selecionar as 130 frames de uma vez.",
)

if uploaded:
    frames = decode_uploaded_frames(uploaded)
    source = "upload"
elif local_frames:
    frames = local_frames
    source = "repository"
else:
    frames = []
    source = "none"

if not frames:
    st.info(
        "Nenhuma frame foi encontrada automaticamente. "
        "Selecione as imagens extraídas do Photron no uploader da barra lateral."
    )
    st.stop()

st.sidebar.success(f"{len(frames)} frame(s) carregadas")

view = st.sidebar.radio(
    "Visualização",
    ["Grade", "Frame selecionada"],
    index=0,
)

if view == "Frame selecionada":
    index = st.sidebar.slider("Frame", 1, len(frames), min(1, (len(frames) + 1) // 2)) - 1
    name, image = frames[index]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.image(image, caption=name, width="stretch")
    with col2:
        st.metric("Frame", f"{index + 1}/{len(frames)}")
        st.write(f"**Arquivo:** `{name}`")
        st.write(f"**Resolução:** `{image.width} × {image.height}`")
        st.write(f"**Origem:** `{source}`")

else:
    columns = st.sidebar.slider("Colunas", 2, 8, 5)
    thumbnail_width = st.sidebar.slider("Largura da thumbnail", 80, 260, 150)

    st.subheader(f"Todas as frames — {len(frames)}")
    st.caption("A ordem é determinada pelo nome do arquivo, preservando a sequência original.")

    for start in range(0, len(frames), columns):
        row = frames[start : start + columns]
        cols = st.columns(columns)
        for col, (name, image) in zip(cols, row):
            with col:
                st.image(image, caption=name, width=thumbnail_width)

st.divider()
st.caption(
    "Somente visualização: os arquivos do experimento e a lógica de YOLO/Kalman permanecem inalterados."
)
