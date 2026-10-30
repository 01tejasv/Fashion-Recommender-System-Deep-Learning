
import os
import pickle
import hashlib

import numpy as np
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)
from tensorflow.keras.utils import img_to_array


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Forma | Fashion Discovery",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.pkl")
FILENAMES_FILE = os.path.join(BASE_DIR, "filenames.pkl")
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #20251f;
    --muted: #737970;
    --green: #355b48;
    --green-dark: #243f32;
    --cream: #f7f6f1;
    --line: #e8e8df;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #faf9f6;
    color: var(--ink);
}

header[data-testid="stHeader"] {
    background: rgba(250,249,246,0.9);
}

.block-container {
    max-width: 1240px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1.65rem;
    letter-spacing: -1.2px;
    color: #243f32;
}

.brand span {
    color: #a2a88c;
}

.nav-label {
    text-align: right;
    color: #737970;
    font-size: 0.83rem;
    padding-top: 0.65rem;
    letter-spacing: 0.5px;
}

.hero {
    background: #e9ede4;
    border-radius: 25px;
    padding: 48px 44px;
    margin-top: 14px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    border: 1px solid #e0e6dc;
}

.hero:after {
    content: "✳";
    position: absolute;
    right: 55px;
    top: 8px;
    font-size: 170px;
    color: rgba(53,91,72,0.10);
    line-height: 1;
}

.eyebrow {
    color: #54735f;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 13px;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    color: #203b2e;
    font-size: clamp(2.1rem, 5vw, 3.55rem);
    letter-spacing: -2px;
    line-height: 1.1;
    margin: 0 0 15px 0;
    max-width: 680px;
    font-weight: 800;
}

.hero p {
    color: #59675b;
    font-size: 1rem;
    line-height: 1.8;
    max-width: 590px;
    margin-bottom: 0;
}

.hero-pill {
    display: inline-block;
    margin-top: 24px;
    background: #fffefa;
    border: 1px solid #dce4d8;
    color: #355b48;
    border-radius: 50px;
    padding: 8px 15px;
    font-size: 0.78rem;
    font-weight: 600;
}

.section-kicker {
    color: #7b8276;
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    margin-bottom: 7px;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.8px;
    color: #242b25;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #747b72;
    font-size: 0.92rem;
    margin-bottom: 20px;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1.5px dashed #bdcabb;
    border-radius: 18px;
    padding: 20px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #54735f;
}

[data-testid="stFileUploaderDropzone"] {
    background: #fbfcf9;
    border: none;
}

[data-testid="stFileUploaderDropzone"] button {
    border-radius: 10px;
    border: 1px solid #355b48;
    background: #355b48;
    color: white;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #243f32;
    color: white;
    border-color: #243f32;
}

.stButton > button {
    background: #355b48;
    color: white;
    border: 1px solid #355b48;
    border-radius: 11px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #243f32;
    color: white;
    border-color: #243f32;
    transform: translateY(-1px);
}

.product-card {
    background: #ffffff;
    border: 1px solid #ebebe4;
    border-radius: 16px;
    padding: 12px 14px;
    margin-bottom: 12px;
}

.product-label {
    color: #858a81;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-weight: 700;
    margin-bottom: 5px;
}

.product-title {
    color: #263d2f;
    font-family: 'Manrope', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 4px;
}

.product-detail {
    color: #777e73;
    font-size: 0.78rem;
}

.match-pill {
    display: inline-block;
    background: #edf3e9;
    color: #355b48;
    border-radius: 50px;
    padding: 5px 9px;
    font-size: 0.73rem;
    font-weight: 700;
    margin-top: 10px;
}

[data-testid="stImage"] img {
    border-radius: 13px;
}

.info-panel {
    background: #ffffff;
    border: 1px solid #e9e9e1;
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0 20px 0;
}

.info-panel h4 {
    font-family: 'Manrope', sans-serif;
    color: #263d2f;
    margin: 0 0 7px 0;
}

.info-panel p {
    color: #747b72;
    font-size: 0.88rem;
    line-height: 1.7;
    margin: 0;
}

.footer {
    border-top: 1px solid #e8e8df;
    margin-top: 55px;
    padding-top: 20px;
    color: #858a81;
    font-size: 0.8rem;
    text-align: center;
}

@media (max-width: 700px) {
    .hero {
        padding: 30px 22px;
    }

    .hero:after {
        right: 5px;
        top: 15px;
        font-size: 100px;
    }

    .hero h1 {
        letter-spacing: -1px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

header_left, header_right = st.columns([2, 1])

with header_left:
    st.markdown(
        '<div class="brand">forma<span>.</span></div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div class="nav-label">DISCOVER YOUR NEXT LOOK &nbsp; ✳</div>',
        unsafe_allow_html=True
    )


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR PERSONAL STYLE DISCOVERY</div>
    <h1>Find your next<br>fashion favourite.</h1>
    <p>
        Every great look starts with inspiration. Upload an outfit or
        fashion image and discover pieces with a similar visual style,
        powered by AI.
    </p>
    <div class="hero-pill">✳ &nbsp; AI-powered visual recommendations</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model():
    return ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="max",
        input_shape=(224, 224, 3)
    )


@st.cache_resource
def load_recommender_data():
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)

    with open(FILENAMES_FILE, "rb") as f:
        filenames = pickle.load(f)

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    if len(embeddings) != len(filenames):
        raise ValueError(
            "The number of embeddings does not match the filenames."
        )

    # Ensure embeddings are normalized.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    return embeddings, filenames


@st.cache_resource
def load_nearest_neighbors(_embeddings):
    index = NearestNeighbors(
        n_neighbors=6,
        algorithm="brute",
        metric="cosine",
        n_jobs=-1
    )
    index.fit(_embeddings)
    return index


# =========================================================
# IMAGE HELPERS
# =========================================================

def extract_features(uploaded_image, model):
    img = uploaded_image.convert("RGB").resize((224, 224))

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0)
    features = features / (
        np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
    )

    return features.astype(np.float32)


def get_dataset_image_path(filename):
    """Resolve a stored dataset filename to a local image path."""
    filename = os.path.normpath(str(filename))

    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename

    # Expected paths include images/37294.jpg.
    candidate = os.path.join(BASE_DIR, filename)
    if os.path.isfile(candidate):
        return candidate

    # Fallback for datasets stored in the images directory.
    candidate = os.path.join(IMAGE_FOLDER, os.path.basename(filename))
    if os.path.isfile(candidate):
        return candidate

    return None


def image_hash(img):
    """Hash resized RGB pixels to detect identical image content."""
    img = img.convert("RGB").resize((224, 224))
    return hashlib.md5(img.tobytes()).hexdigest()


def find_recommendations(
    uploaded_image,
    query_features,
    embeddings,
    filenames,
    neighbors_model,
    number_to_show=5
):
    """
    Find similar images while excluding exact duplicates of the upload.
    """
    query_hash = image_hash(uploaded_image)

    # Check more candidates to allow for duplicate images.
    search_count = min(50, len(filenames))

    distances, indices = neighbors_model.kneighbors(
        query_features,
        n_neighbors=search_count
    )

    results = []
    seen_hashes = {query_hash}
    seen_paths = set()

    for idx, distance in zip(indices[0], distances[0]):
        image_path = get_dataset_image_path(filenames[idx])

        if image_path is None:
            continue

        normalized_path = os.path.normcase(os.path.abspath(image_path))

        if normalized_path in seen_paths:
            continue

        seen_paths.add(normalized_path)

        try:
            with Image.open(image_path) as dataset_img:
                current_hash = image_hash(dataset_img)

            # Exclude the uploaded image and any exact duplicate files.
            if current_hash in seen_hashes:
                continue

            seen_hashes.add(current_hash)

        except (OSError, ValueError):
            continue

        results.append((idx, float(distance), image_path))

        if len(results) >= number_to_show:
            break

    return results


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-kicker">01 — START WITH AN IMAGE</div>'
    '<div class="section-title">Show us your inspiration</div>'
    '<div class="section-subtitle">'
    'Choose a fashion image from your device to find similar styles.'
    '</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    uploaded_file = st.file_uploader(
        "Drop your image here or browse files",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG and WEBP."
    )

    st.caption(
        "For best results, use a clear image with the outfit visible."
    )

with right:
    st.markdown("""
    <div class="info-panel">
        <h4>✳ &nbsp; How it works</h4>
        <p>
            <b>01.</b> Upload a fashion image.<br><br>
            <b>02.</b> AI analyses its visual features.<br><br>
            <b>03.</b> Explore five visually similar images
            from the fashion collection.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# RECOMMENDATION SECTION
# =========================================================

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file).convert("RGB")

        st.markdown("---")
        st.markdown(
            '<div class="section-kicker">02 — YOUR STYLE</div>'
            '<div class="section-title">Your inspiration board</div>',
            unsafe_allow_html=True
        )

        preview_col, action_col = st.columns([0.8, 1.2], gap="large")

        with preview_col:
            st.image(
                uploaded_image,
                caption="Your uploaded image",
                use_container_width=True
            )

        with action_col:
            st.markdown("""
            <div class="info-panel">
                <h4>Ready to discover?</h4>
                <p>
                    We'll compare your image with the visual features
                    of items in our fashion collection and find
                    the closest matches.
                </p>
            </div>
            """, unsafe_allow_html=True)

            find_button = st.button(
                "✳  Find Similar Styles",
                use_container_width=True,
                type="primary"
            )

        if find_button:
            with st.spinner(
                "Preparing your personalized style results..."
            ):
                model = load_model()
                embeddings, filenames = load_recommender_data()
                neighbors_model = load_nearest_neighbors(embeddings)

                query_features = extract_features(uploaded_image, model)

                results = find_recommendations(
                    uploaded_image=uploaded_image,
                    query_features=query_features,
                    embeddings=embeddings,
                    filenames=filenames,
                    neighbors_model=neighbors_model,
                    number_to_show=5
                )

            st.markdown("---")
            st.markdown(
                '<div class="section-kicker">03 — CURATED FOR YOU</div>'
                '<div class="section-title">Your style matches ✳</div>'
                '<div class="section-subtitle">'
                'Similar visual styles, selected from your fashion collection.'
                '</div>',
                unsafe_allow_html=True
            )

            if not results:
                st.warning(
                    "No alternative images were found. Try another image "
                    "or check that your dataset images are available."
                )
            else:
                cards = st.columns(5, gap="medium")

                for rank, (idx, distance, image_path) in enumerate(
                    results, start=1
                ):
                    with cards[rank - 1]:
                        st.markdown(
                            f"""
                            <div class="product-card">
                                <div class="product-label">
                                    STYLE MATCH {rank:02d}
                                </div>
                                <div class="product-title">
                                    Look No. {rank}
                                </div>
                                <div class="product-detail">
                                    Visually similar selection
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.image(
                            image_path,
                            use_container_width=True
                        )

                        similarity = max(
                            0.0,
                            min(100.0, (1.0 - distance) * 100)
                        )

                        st.markdown(
                            f'<div class="match-pill">'
                            f'{similarity:.1f}% visual similarity'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                st.markdown("<br>", unsafe_allow_html=True)
                st.success(
                    f"Your discovery board is ready — "
                    f"{len(results)} alternative styles found."
                )

    except Exception as e:
        st.error(f"Unable to generate recommendations: {e}")


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    <div style="font-family:Manrope,sans-serif;font-size:1.2rem;
                font-weight:800;color:#355b48;margin-bottom:8px;">
        forma<span style="color:#a2a88c;">.</span>
    </div>
    Made for the love of personal style.<br>
    AI-powered visual discovery · ResNet50 · Cosine similarity
</div>
""", unsafe_allow_html=True)
