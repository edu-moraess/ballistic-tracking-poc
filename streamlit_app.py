"""Ballistic Tracking — Sequence Explorer (visualization only)."""
from __future__ import annotations
import io
import re
import zipfile
from typing import Dict, List, Tuple
import streamlit as st
from PIL import Image

st.set_page_config(page_title="BALLISTIC TRACKING // SEQUENCE EXPLORER", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root{--bg:#05070b;--panel:#0b1018;--panel2:#0f1621;--line:#26384f;--text:#e8f1f7;--muted:#78909f;--accent:#5aa9ff}
.stApp{background:radial-gradient(circle at 50% -10%,rgba(52,108,180,.18),transparent 35%),linear-gradient(180deg,#05070b 0%,#070b11 55%,#05070b 100%);color:var(--text)}
[data-testid="stHeader"]{background:transparent}.block-container{max-width:1500px;padding-top:2rem}h1,h2,h3,h4{color:var(--text)!important;letter-spacing:.04em}
.hero{border:1px solid var(--line);background:linear-gradient(135deg,rgba(12,22,31,.96),rgba(7,11,17,.96));border-radius:18px;padding:28px 30px 24px;margin-bottom:18px;box-shadow:0 0 45px rgba(50,110,190,.10),inset 0 1px 0 rgba(255,255,255,.025)}
.eyebrow,.media-title{color:var(--accent);font-size:.72rem;letter-spacing:.2em;font-weight:700}.hero-title{font-size:clamp(1.65rem,4vw,3rem);font-weight:800;margin:5px 0;letter-spacing:.08em}.hero-sub{color:var(--muted);font-size:.9rem}
.status{display:inline-block;border:1px solid #315b86;color:var(--accent);background:rgba(64,130,210,.09);border-radius:999px;padding:5px 11px;font-size:.72rem;letter-spacing:.08em}
.upload-box{border:1px dashed #31577d;background:rgba(8,18,26,.8);border-radius:16px;padding:10px;margin:12px 0 22px}.metric-card{background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;padding:16px;min-height:92px;margin-bottom:12px}.metric-label{color:var(--muted);font-size:.68rem;letter-spacing:.14em}.metric-value{color:var(--text);font-size:1.25rem;font-weight:700;margin-top:5px}.frame-caption{font-size:.68rem;color:#78909f;text-align:center;margin:3px 0 10px}
div[data-testid="stFileUploader"]{background:rgba(9,16,24,.72);border:1px solid #26384f;border-radius:12px;padding:8px}div[data-testid="stFileUploaderDropzone"]{background:transparent}button[kind="secondary"]{border-color:#31516d}
@media(max-width:700px){.block-container{padding:1rem .75rem 2rem}.hero{padding:20px 18px;border-radius:14px}.hero-sub{font-size:.78rem}}
</style>
""",unsafe_allow_html=True)

st.markdown("""<div class="hero"><div class="eyebrow">COMPUTER VISION × STATE ESTIMATION</div><div class="hero-title">BALLISTIC TRACKING</div><div class="hero-sub">SEQUENCE EXPLORER // VISUALIZATION CONSOLE</div><br><span class="status">● LOCAL MEDIA MODE</span></div>""",unsafe_allow_html=True)

IMAGE_EXTENSIONS=(".png",".jpg",".jpeg",".webp",".bmp")
VIDEO_EXTENSIONS=(".mp4",".mov",".avi",".webm",".mkv",".mpeg",".mpg")

def _basename(name:str)->str:
    name=name.replace("\\","/").rstrip("/")
    return name.rsplit("/",1)[-1]

def _is_skippable_member(name:str)->bool:
    norm=name.replace("\\","/"); base=_basename(norm)
    return not norm or norm.endswith("/") or norm.startswith("__MACOSX/") or "/__MACOSX/" in norm or not base or base.startswith(".") or base.startswith("._")

def _is_image_member(name:str)->bool:return _basename(name).lower().endswith(IMAGE_EXTENSIONS)

def natural_frame_key(name:str)->Tuple[int,str]:
    stem=name.rsplit(".",1)[0] if "." in name else name; m=re.search(r"(\d+)",stem)
    return (int(m.group(1)),name.lower()) if m else (10**9,name.lower())

def _validate_image_bytes(data:bytes)->bool:
    if not data or len(data)<24:return False
    try:
        with Image.open(io.BytesIO(data)) as im: im.verify()
        with Image.open(io.BytesIO(data)) as im: im.load()
        return True
    except Exception:return False

def _looks_like_zip(data:bytes)->bool:return len(data)>=4 and data[:2]==b"PK"

@st.cache_data(show_spinner="Reading sequence…",ttl=3600)
def load_frames_from_zip(zip_bytes:bytes)->Tuple[List[Tuple[str,bytes]],Dict[str,int]]:
    stats={"members_total":0,"skipped_meta":0,"candidates":0,"unreadable":0,"duplicates":0,"loaded":0,"bytes":len(zip_bytes)}
    if not zip_bytes:raise ValueError("ZIP_EMPTY: o arquivo enviado está vazio.")
    if not _looks_like_zip(zip_bytes):raise ValueError("ZIP_OPEN_FAILED: o conteúdo recebido não parece um ZIP.")
    try:zf=zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:raise ValueError(f"ZIP_OPEN_FAILED: arquivo ZIP inválido ou incompleto ({exc}).") from exc
    try:
        bad=zf.testzip()
        if bad is not None:raise ValueError(f"ZIP_CORRUPT: entrada corrompida: {bad}")
    except ValueError:raise
    except Exception:pass
    frames=[];seen=set()
    for info in zf.infolist():
        stats["members_total"]+=1;member=info.filename
        if info.is_dir() or _is_skippable_member(member):stats["skipped_meta"]+=1;continue
        if not _is_image_member(member):continue
        stats["candidates"]+=1;base=_basename(member)
        if base in seen:stats["duplicates"]+=1;continue
        try:data=zf.read(info)
        except Exception:stats["unreadable"]+=1;continue
        if not _validate_image_bytes(data):stats["unreadable"]+=1;continue
        seen.add(base);frames.append((base,data));stats["loaded"]+=1
    if not frames:
        if stats["candidates"]==0:raise ValueError("ZIP_NO_IMAGES: nenhuma imagem PNG/JPG/JPEG foi encontrada no ZIP.")
        raise ValueError("ZIP_UNREADABLE_IMAGES: as imagens encontradas não puderam ser lidas.")
    frames.sort(key=lambda x:natural_frame_key(x[0]));return frames,stats

def decode_image(data:bytes)->Image.Image:
    im=Image.open(io.BytesIO(data));im.load();return im.convert("RGB")

def make_thumbnail(image:Image.Image,max_width:int)->Image.Image:
    if image.width<=max_width:return image
    ratio=max_width/float(image.width);return image.resize((max_width,max(1,int(image.height*ratio))),Image.Resampling.BILINEAR)

def detect_media_kind(name:str,data:bytes)->str:
    lower=(name or "").lower()
    if _looks_like_zip(data):return "zip"
    if data.startswith(b"\x89PNG") or data.startswith(b"\xff\xd8\xff") or data.startswith(b"RIFF"):
        if _validate_image_bytes(data):return "image"
    if data.startswith(b"\x00\x00\x00") and b"ftyp" in data[:64]:return "video"
    if data.startswith(b"\x1aE\xdf\xa3"):return "video"
    if any(lower.endswith(x) for x in IMAGE_EXTENSIONS):return "image"
    if any(lower.endswith(x) for x in VIDEO_EXTENSIONS):return "video"
    if _validate_image_bytes(data):return "image"
    return "unknown"

st.markdown('<div class="media-title">MEDIA INPUT</div>',unsafe_allow_html=True)
st.markdown('<div class="upload-box">',unsafe_allow_html=True)
uploads=st.file_uploader("Arraste arquivos para cá ou toque em Procurar arquivos",type=None,accept_multiple_files=True,help="Aceita imagens soltas, ZIPs com frames e vídeos. O conteúdo é validado pelo app.")
st.markdown('</div>',unsafe_allow_html=True)
if not uploads:
    st.info("Nenhuma mídia carregada. Envie PNG/JPG, um ZIP de frames ou um vídeo para iniciar a visualização.");st.stop()

all_frames=[];videos=[];unknown=[];zip_sources=[]
for uploaded in uploads:
    data=uploaded.getvalue();kind=detect_media_kind(uploaded.name,data)
    if kind=="zip":
        try:frames,stats=load_frames_from_zip(data);all_frames.extend(frames);zip_sources.append(f"{uploaded.name} · {len(frames)} frames")
        except ValueError as exc:st.error(f"{uploaded.name}: {exc}")
    elif kind=="image":
        if _validate_image_bytes(data):all_frames.append((_basename(uploaded.name),data))
        else:unknown.append(uploaded.name)
    elif kind=="video":videos.append((uploaded.name,data))
    else:unknown.append(uploaded.name)

seen_names=set();unique_frames=[]
for name,data in all_frames:
    key=(name.lower(),len(data))
    if key not in seen_names:seen_names.add(key);unique_frames.append((name,data))
all_frames=sorted(unique_frames,key=lambda x:natural_frame_key(x[0]))
if unknown:st.warning("Não foi possível identificar: "+", ".join(unknown))

stat_cols=st.columns(4)
for col,(label,value) in zip(stat_cols,[("FRAMES",str(len(all_frames))),("VIDEOS",str(len(videos))),("SOURCES",str(len(uploads))),("MODE","VISUAL")]):
    with col:st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)
if zip_sources:st.caption("ZIP sources: "+" · ".join(zip_sources))
for video_name,video_bytes in videos:
    st.markdown(f'<div class="media-title">VIDEO // {video_name}</div>',unsafe_allow_html=True);st.video(video_bytes)

if all_frames:
    st.markdown('<div class="media-title">FRAME VIEWER</div>',unsafe_allow_html=True)
    # Streamlit does not allow min_value == max_value on sliders.
    if len(all_frames)==1:
        selected_index=0
    else:
        selected_index=st.slider("Frame",min_value=1,max_value=len(all_frames),value=min(max(1,(len(all_frames)+1)//2),len(all_frames)),label_visibility="collapsed")-1
    name,raw=all_frames[selected_index];image=decode_image(raw)
    col_img,col_meta=st.columns([3,1])
    with col_img:
        st.markdown(f"### FRAME {selected_index+1:03d} / {len(all_frames)}");st.image(image,caption=name,use_container_width=True)
    with col_meta:
        st.markdown(f'<div class="metric-card"><div class="metric-label">SELECTED FRAME</div><div class="metric-value">{selected_index+1:03d}</div></div><div class="metric-card"><div class="metric-label">FILE</div><div class="metric-value" style="font-size:.85rem;word-break:break-all">{name}</div></div><div class="metric-card"><div class="metric-label">RESOLUTION</div><div class="metric-value">{image.width} × {image.height}</div></div>',unsafe_allow_html=True)
    st.markdown("---");st.markdown('<div class="media-title">SEQUENCE // ALL FRAMES</div>',unsafe_allow_html=True)
    grid_columns=min(8,max(3,len(all_frames)))
    for start in range(0,len(all_frames),grid_columns):
        row=all_frames[start:start+grid_columns];cols=st.columns(grid_columns)
        for col,(fname,fbytes) in zip(cols,row):
            with col:
                st.image(make_thumbnail(decode_image(fbytes),150),use_container_width=True)
                m=re.search(r"(\d+)",fname.rsplit(".",1)[0]);label=f"{int(m.group(1)):03d}" if m else fname
                st.markdown(f'<p class="frame-caption">[{label}]</p>',unsafe_allow_html=True)

st.divider();st.caption("VISUALIZATION ONLY · YOLO, Kalman, training code, weights and experimental results are not modified or executed here.")
