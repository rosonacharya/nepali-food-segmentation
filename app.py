# ================================================================
# NEPALI FOOD SEMANTIC SEGMENTATION
# Streamlit GUI
# Model: SegFormer MiT-B2
# ================================================================

import os
import hashlib

import numpy as np
import streamlit as st

import torch
import torchvision

from PIL import Image, ImageFilter

from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Nepali Food Semantic Segmentation",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ================================================================
# CUSTOM CSS
# Only styles native Streamlit components
# ================================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Main title */
    h1 {
        text-align: center;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }

    /* Section headings */
    h2, h3 {
        font-weight: 650;
    }

    /* Containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 500;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 10px;
    }

    /* Camera input */
    div[data-testid="stCameraInput"] {
        border-radius: 10px;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.25);
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# MODEL CONFIGURATION
# ================================================================
import os

MODEL_DIR = "."

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pth"
)

HF_MODEL_NAME = "nvidia/mit-b2"

IMAGE_SIZE = 512
NUM_CLASSES = 9


# ================================================================
# CLASS NAMES
# ================================================================

CLASS_NAMES = [
    "Background",
    "Chutney",
    "Ghundruk",
    "Lentil (Dal)",
    "Meat Curry",
    "Rice (Bhat)",
    "Salad",
    "Spinach (Saag)",
    "Vegetable Curry",
]


# ================================================================
# EXACT CLASS COLORS
# ================================================================

CLASS_COLORS = [
    (0, 0, 0),          # Background
    (255, 136, 0),      # Chutney
    (200, 174, 74),     # Ghundruk
    (255, 0, 204),      # Lentil
    (255, 0, 0),        # Meat Curry
    (0, 102, 255),      # Rice
    (153, 0, 255),      # Salad
    (0, 204, 68),       # Spinach
    (0, 204, 255),      # Vegetable Curry
]


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# LOAD MODEL
# ================================================================

@st.cache_resource(show_spinner=False)
def load_model():

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    # ------------------------------------------------------------
    # Load official MiT-B2 configuration
    # ------------------------------------------------------------

    config = SegformerConfig.from_pretrained(
        HF_MODEL_NAME
    )

    config.num_labels = NUM_CLASSES

    config.id2label = {
        i: CLASS_NAMES[i]
        for i in range(NUM_CLASSES)
    }

    config.label2id = {
        CLASS_NAMES[i]: i
        for i in range(NUM_CLASSES)
    }

    # ------------------------------------------------------------
    # Build SegFormer MiT-B2
    # ------------------------------------------------------------

    model = SegformerForSemanticSegmentation(
        config
    )

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    # ------------------------------------------------------------
    # Handle possible checkpoint wrappers
    # ------------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # ------------------------------------------------------------
    # Remove common prefixes
    # ------------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned_state_dict[new_key] = value

    # ------------------------------------------------------------
    # Strict loading
    # ------------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    # ------------------------------------------------------------
    # Image processor
    # ------------------------------------------------------------

    processor = SegformerImageProcessor(
        do_resize=True,
        size={
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
        },
        do_normalize=True,
        image_mean=[
            0.485,
            0.456,
            0.406,
        ],
        image_std=[
            0.229,
            0.224,
            0.225,
        ],
    )

    return model, processor


# ================================================================
# IMAGE HASH
# ================================================================

def get_image_hash(image):

    image = image.convert("RGB")

    return hashlib.md5(
        image.tobytes()
    ).hexdigest()


# ================================================================
# PREDICTION
# ================================================================

def predict_image(
    image,
    model,
    processor,
):

    image = image.convert("RGB")

    original_width, original_height = (
        image.size
    )

    # ------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(DEVICE)

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            pixel_values=pixel_values
        )

        logits = outputs.logits

    # ------------------------------------------------------------
    # Resize logits to original image size
    # ------------------------------------------------------------

    logits = torch.nn.functional.interpolate(
        logits,
        size=(
            original_height,
            original_width,
        ),
        mode="bilinear",
        align_corners=False,
    )

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    prediction = (
        prediction[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return prediction


# ================================================================
# CALCULATE DETECTED FOOD CLASSES
# ================================================================

def calculate_detected_classes(
    prediction
):

    detected_classes = []

    # ------------------------------------------------------------
    # Food pixels only
    # ------------------------------------------------------------

    food_mask = prediction != 0

    total_food_pixels = np.sum(
        food_mask
    )

    if total_food_pixels == 0:
        return detected_classes

    # ------------------------------------------------------------
    # Calculate class percentage
    # ------------------------------------------------------------

    for class_id in range(
        1,
        NUM_CLASSES,
    ):

        class_pixels = np.sum(
            prediction == class_id
        )

        if class_pixels == 0:
            continue

        percentage = (
            class_pixels
            / total_food_pixels
            * 100.0
        )

        detected_classes.append(
            {
                "class_id": class_id,
                "name": CLASS_NAMES[
                    class_id
                ],
                "percentage": percentage,
                "pixels": int(
                    class_pixels
                ),
            }
        )

    # ------------------------------------------------------------
    # Largest food region first
    # ------------------------------------------------------------

    detected_classes.sort(
        key=lambda x: x["percentage"],
        reverse=True,
    )

    return detected_classes


# ================================================================
# CREATE COLOR SWATCH
# ================================================================

def create_color_swatch(
    color,
    size=28,
):

    array = np.zeros(
        (
            size,
            size,
            3,
        ),
        dtype=np.uint8,
    )

    array[:, :] = color

    return Image.fromarray(
        array
    )


# ================================================================
# RESIZE IMAGE FOR DISPLAY
# ================================================================

def resize_image_for_display(
    image,
    max_width=650,
):

    image = image.copy()

    width, height = image.size

    if width <= max_width:
        return image

    new_height = int(
        height
        * max_width
        / width
    )

    return image.resize(
        (
            max_width,
            new_height,
        ),
        Image.Resampling.LANCZOS,
    )


# ================================================================
# SELECTED FOOD VISUALIZATION
#
# Selected class remains sharp.
# All other regions are blurred.
# ================================================================

def isolate_food(
    image,
    prediction,
    selected_class,
):

    original = np.array(
        image.convert("RGB")
    )

    # ------------------------------------------------------------
    # Create blurred version
    # ------------------------------------------------------------

    blurred_image = (
        image.convert("RGB")
        .filter(
            ImageFilter.GaussianBlur(
                radius=12
            )
        )
    )

    blurred = np.array(
        blurred_image
    )

    # ------------------------------------------------------------
    # Selected class mask
    # ------------------------------------------------------------

    selected_mask = (
        prediction
        == selected_class
    )

    # ------------------------------------------------------------
    # Start with fully blurred image
    # ------------------------------------------------------------

    result = blurred.copy()

    # ------------------------------------------------------------
    # Restore selected food at original sharpness
    # ------------------------------------------------------------

    result[
        selected_mask
    ] = original[
        selected_mask
    ]

    return Image.fromarray(
        result
    )


# ================================================================
# CLASS LEGEND
# ================================================================

def show_class_legend():

    st.subheader(
        "Food Class Legend"
    )

    columns = st.columns(4)

    for index, class_name in enumerate(
        CLASS_NAMES[1:]
    ):

        class_id = index + 1

        with columns[index % 4]:

            swatch = create_color_swatch(
                CLASS_COLORS[class_id],
                size=24,
            )

            swatch_col, name_col = (
                st.columns(
                    [0.20, 0.80]
                )
            )

            with swatch_col:
                st.image(
                    swatch,
                    width=24,
                )

            with name_col:
                st.caption(
                    class_name
                )


# ================================================================
# DETECTED FOOD SECTION
# ================================================================

def show_detected_food(
    detected_classes
):

    st.subheader(
        "Detected Food"
    )

    if not detected_classes:

        st.info(
            "No food class was detected."
        )

        return

    for item in detected_classes:

        class_id = item[
            "class_id"
        ]

        class_name = item[
            "name"
        ]

        percentage = item[
            "percentage"
        ]

        col1, col2, col3 = st.columns(
            [0.10, 0.68, 0.22]
        )

        # --------------------------------------------------------
        # Color
        # --------------------------------------------------------

        with col1:

            swatch = create_color_swatch(
                CLASS_COLORS[
                    class_id
                ],
                size=28,
            )

            st.image(
                swatch,
                width=28,
            )

        # --------------------------------------------------------
        # Food button
        # --------------------------------------------------------

        with col2:

            clicked = st.button(
                class_name,
                key=f"food_{class_id}",
                use_container_width=True,
            )

            if clicked:

                st.session_state[
                    "selected_class"
                ] = class_id

        # --------------------------------------------------------
        # Percentage
        # --------------------------------------------------------

        with col3:

            st.write(
                f"{percentage:.2f}%"
            )


# ================================================================
# INITIALIZE SESSION STATE
# ================================================================

if "selected_class" not in st.session_state:

    st.session_state[
        "selected_class"
    ] = None


if "image_hash" not in st.session_state:

    st.session_state[
        "image_hash"
    ] = None


if "prediction" not in st.session_state:

    st.session_state[
        "prediction"
    ] = None


if "current_image" not in st.session_state:

    st.session_state[
        "current_image"
    ] = None


# ================================================================
# MAIN TITLE
# ================================================================

st.title(
    "Nepali Food Semantic Segmentation"
)


# ================================================================
# LOAD MODEL
# ================================================================

try:

    with st.spinner(
        "Loading SegFormer MiT-B2..."
    ):

        model, processor = (
            load_model()
        )

except Exception as e:

    st.error(
        "Unable to load the segmentation model."
    )

    st.exception(e)

    st.stop()


# ================================================================
# IMAGE INPUT
# ================================================================

with st.container(
    border=True
):

    st.subheader(
        "Food Image Input"
    )

    upload_col, camera_col = (
        st.columns(2)
    )

    # ------------------------------------------------------------
    # Upload image
    # ------------------------------------------------------------

    with upload_col:

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            label_visibility="visible",
        )

    # ------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------

    with camera_col:

        camera_file = st.camera_input(
            "Capture Photo"
        )


# ================================================================
# SELECT INPUT
# ================================================================

input_file = None

if uploaded_file is not None:

    input_file = uploaded_file

elif camera_file is not None:

    input_file = camera_file


# ================================================================
# PROCESS IMAGE
# ================================================================

if input_file is not None:

    try:

        image = Image.open(
            input_file
        ).convert("RGB")

        current_hash = (
            get_image_hash(
                image
            )
        )

        # --------------------------------------------------------
        # New image
        # --------------------------------------------------------

        if (
            st.session_state[
                "image_hash"
            ]
            != current_hash
        ):

            with st.spinner(
                "Analyzing food image..."
            ):

                prediction = (
                    predict_image(
                        image,
                        model,
                        processor,
                    )
                )

            st.session_state[
                "image_hash"
            ] = current_hash

            st.session_state[
                "prediction"
            ] = prediction

            st.session_state[
                "current_image"
            ] = image

            # Reset selected food
            st.session_state[
                "selected_class"
            ] = None

        # --------------------------------------------------------
        # Existing image
        # --------------------------------------------------------

        else:

            prediction = (
                st.session_state[
                    "prediction"
                ]
            )

            image = (
                st.session_state[
                    "current_image"
                ]
            )

        # ========================================================
        # DETECTED CLASSES
        # ========================================================

        detected_classes = (
            calculate_detected_classes(
                prediction
            )
        )

        # ========================================================
        # ORIGINAL IMAGE + DETECTED FOOD
        # ========================================================

        image_col, food_col = (
            st.columns(
                [1.25, 0.95]
            )
        )

        # --------------------------------------------------------
        # ORIGINAL IMAGE
        # --------------------------------------------------------

        with image_col:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Original Image"
                )

                display_original = (
                    resize_image_for_display(
                        image,
                        max_width=700,
                    )
                )

                st.image(
                    display_original,
                    use_container_width=True,
                )

        # --------------------------------------------------------
        # DETECTED FOOD
        # --------------------------------------------------------

        with food_col:

            with st.container(
                border=True
            ):

                show_detected_food(
                    detected_classes
                )


        # ========================================================
        # SELECTED FOOD
        # ========================================================

        selected_class = (
            st.session_state[
                "selected_class"
            ]
        )

        if selected_class is not None:

            selected_item = None

            for item in detected_classes:

                if (
                    item["class_id"]
                    == selected_class
                ):

                    selected_item = item

                    break

            if selected_item is not None:

                selected_name = (
                    selected_item[
                        "name"
                    ]
                )

                selected_percentage = (
                    selected_item[
                        "percentage"
                    ]
                )

                st.divider()

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "Selected Food"
                    )

                    name_col, percentage_col = (
                        st.columns(
                            [0.75, 0.25]
                        )
                    )

                    # ------------------------------------------------
                    # Selected food name
                    # ------------------------------------------------

                    with name_col:

                        st.write(
                            f"**{selected_name}**"
                        )

                    # ------------------------------------------------
                    # Percentage
                    # ------------------------------------------------

                    with percentage_col:

                        st.write(
                            f"**{selected_percentage:.2f}%**"
                        )

                    # ------------------------------------------------
                    # Create selected-food visualization
                    # ------------------------------------------------

                    selected_image = (
                        isolate_food(
                            image,
                            prediction,
                            selected_class,
                        )
                    )

                    # ------------------------------------------------
                    # Resize for a compact thesis-style display
                    # ------------------------------------------------

                    selected_display = (
                        resize_image_for_display(
                            selected_image,
                            max_width=650,
                        )
                    )

                    # ------------------------------------------------
                    # Center image
                    # ------------------------------------------------

                    left_space, image_space, right_space = (
                        st.columns(
                            [0.20, 0.60, 0.20]
                        )
                    )

                    with image_space:

                        st.image(
                            selected_display,
                            use_container_width=True,
                        )


        # ========================================================
        # FOOD LEGEND
        # ========================================================

        st.divider()

        with st.container(
            border=True
        ):

            show_class_legend()


    except Exception as e:

        st.error(
            "An error occurred while processing the image."
        )

        st.exception(e)


# ================================================================
# EMPTY STATE
# ================================================================

else:

    with st.container(
        border=True
    ):

        st.info(
            "Upload a food image or capture a photo to begin segmentation."
        )


# ================================================================
# FOOTER
# ================================================================

st.markdown(
    """
    <div class="footer-text">
        Roshan Acharya © 2026
    </div>
    """,
    unsafe_allow_html=True,
)