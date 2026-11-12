"""Streamlit app: label-free mitochondria segmentation from refractive-index (RI) images."""
import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import segment as S

st.set_page_config(page_title='Mitochondria segmentation (RI)', layout='wide')

MODES = {  # value: fast flag passed to S.segment
    'Accurate: 3-fold ensemble + mirroring (as in the paper)': False,
    'Fast: single model, no mirroring': True,
}
EXAMPLES = Path(__file__).parent / 'examples'
MAX_FILES = 20


@st.cache_resource(show_spinner='Downloading model weights (first start only)...')
def model_dir():
    return S.download_model()


@st.cache_resource(show_spinner='Loading model...')
def predictor():
    return S.load_predictor(model_dir())


def read_image(data: bytes, name: str):
    """Return (8-bit grayscale array, list of warnings)."""
    img = Image.open(io.BytesIO(data))
    notes = []
    if img.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(arr, (0.5, 99.5))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        notes.append(f'{name}: {img.mode} image rescaled to 8 bit (0.5-99.5 percentile). The model was '
                     'trained on 8-bit exports; results may differ from the paper.')
        gray = arr.astype(np.uint8)
    else:
        if img.mode not in ('L', 'P', '1'):
            rgb = np.asarray(img.convert('RGB')).astype(int)
            if np.abs(rgb[..., 0] - rgb[..., 1]).max() > 2 or np.abs(rgb[..., 1] - rgb[..., 2]).max() > 2:
                notes.append(f'{name}: colour image converted to grayscale. The model expects a '
                             'single-channel RI image, not fluorescence or a colour overlay.')
        gray = S.to_gray(img)
    h, w = gray.shape
    if max(h, w) < 256 or max(h, w) > 2048:
        notes.append(f'{name}: {w}x{h} px. The model was trained on 512x512 tiles at the native pixel '
                     'size; strongly different sizes or pixel sizes reduce accuracy.')
    return gray, notes


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()


# ---------------------------------------------------------------- UI
st.title('Label-free mitochondria segmentation')
st.markdown(
    'Segments mitochondria in **refractive-index (RI) images from holotomography** with an nnU-Net '
    'trained on fluorescence-guided labels of HUVECs. Upload 8-bit grayscale RI frames '
    '(512x512 px is what the model was trained on).')

with st.expander('Scope and limitations', expanded=False):
    st.markdown(
        '- Trained on HUVEC RI images from one holotomographic microscope. Held-out test Dice '
        '0.850 ± 0.064 (same cell type, unseen dish).\n'
        '- On a different microscope and cell type (iPSCs) agreement drops (Dice 0.731 vs. a reference '
        'method). Check results visually before quantifying anything.\n'
        '- Not suitable for fluorescence, brightfield or phase-contrast images.\n'
        '- Uploaded images are processed in memory and not stored.')

mode = st.radio('Mode', list(MODES), index=0,
                help='Accurate: ~5-10 s per 512x512 image on the free CPU. Fast: under 1 s, '
                     'slightly lower accuracy.')

files = st.file_uploader('RI images', type=['png', 'tif', 'tiff', 'jpg', 'jpeg'],
                         accept_multiple_files=True)
inputs = [(f.name, f.getvalue()) for f in files or []]

examples = sorted(p for ext in ('*.png', '*.tif', '*.tiff') for p in EXAMPLES.glob(ext)) \
    if EXAMPLES.exists() else []
if not inputs and examples:
    st.markdown('**No images? Try the examples:**')
    thumbs = st.columns(min(len(examples), 6))
    for i, p in enumerate(examples):
        thumbs[i % len(thumbs)].image(S.to_gray(Image.open(p)), caption=p.stem,
                                      use_container_width=True)
    picked = st.multiselect('Examples to segment', [p.stem for p in examples],
                            default=st.session_state.get('picked', []))
    c1, c2 = st.columns([1, 5])
    if c1.button('Run selected', disabled=not picked):
        st.session_state['picked'] = picked
    if c2.button('Run all examples'):
        st.session_state['picked'] = [p.stem for p in examples]
    chosen = set(st.session_state.get('picked', []))
    inputs = [(p.name, p.read_bytes()) for p in examples if p.stem in chosen]

if len(inputs) > MAX_FILES:
    st.warning(f'Processing the first {MAX_FILES} images only.')
    inputs = inputs[:MAX_FILES]

if inputs:
    pred = predictor()
    results = []
    progress = st.progress(0.0, text='Segmenting...')
    for i, (name, data) in enumerate(inputs):
        try:
            gray, notes = read_image(data, name)
        except Exception as e:  # unreadable file
            st.error(f'{name}: could not read image ({e}).')
            continue
        for n in notes:
            st.warning(n)
        mask = S.segment(pred, gray, fast=MODES[mode])
        results.append((name, gray, mask))
        progress.progress((i + 1) / len(inputs), text=f'Segmented {i + 1}/{len(inputs)}')
    progress.empty()

    if results:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, gray, mask in results:
                z.writestr(f'{Path(name).stem}_mask.png', png_bytes(mask * 255))
        st.download_button(f'Download all masks ({len(results)}, ZIP)', zbuf.getvalue(),
                           file_name='mito_masks.zip', mime='application/zip', type='primary')
        st.caption('Masks are 8-bit PNGs: 255 = mitochondria, 0 = background.')

        for name, gray, mask in results:
            st.subheader(name)
            c1, c2, c3 = st.columns(3)
            c1.image(gray, caption='Input', use_container_width=True, clamp=True)
            c2.image(S.overlay(gray, mask), caption='Overlay', use_container_width=True)
            c3.image(mask * 255, caption=f'Mask ({100 * mask.mean():.1f}% of pixels)',
                     use_container_width=True)
            st.download_button('Download mask', png_bytes(mask * 255),
                               file_name=f'{Path(name).stem}_mask.png', mime='image/png',
                               key=f'dl_{name}')

st.divider()
st.caption('Model: nnU-Net v2 (2D PlainConvUNet, 3 leave-one-dish-out folds). '
           'Citation and code: see the README.')
