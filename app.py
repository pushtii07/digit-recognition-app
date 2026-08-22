import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas

from image_processing import (
    preprocess_to_grayscale,
    get_bounding_crop,
    pad_to_square,
    resize_image,
    recenter_image
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Digit Lab",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #F5EBDD;
        color: #3B2922;
    }

    /* Main container */
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Headers */
    h1, h2, h3 {
        color: #3B2922 !important;
    }

    /* Normal text */
    p, label, .stMarkdown {
        color: #4B372E;
    }

    /* Cards */
    .card {
        background-color: #FFF9F0;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #E4D3C2;
        box-shadow: 0 4px 15px rgba(59, 41, 34, 0.08);
        margin-bottom: 20px;
    }

    /* Top header */
    .brand {
        font-size: 32px;
        font-weight: 700;
        color: #3B2922;
    }

    .subtitle {
        color: #80695A;
        font-size: 16px;
        margin-top: -10px;
    }

    /* Digit counter */
    .digit-counter {
        background-color: #6B4A3A;
        color: #FFF9F0;
        padding: 15px 22px;
        border-radius: 14px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
    }

    /* Save button */
    .stButton > button {
        background-color: #6B4A3A;
        color: #FFF9F0;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #4F3529;
        color: white;
    }

    /* Metric boxes */
    [data-testid="stMetric"] {
        background-color: #FFF9F0;
        border: 1px solid #E4D3C2;
        padding: 15px;
        border-radius: 14px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "current_digit" not in st.session_state:
    st.session_state.current_digit = 0

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "saved_message" not in st.session_state:
    st.session_state.saved_message = ""


# =========================================================
# LOGIN SCREEN
# =========================================================

if not st.session_state.logged_in:

    st.markdown(
        "<div style='height:80px'></div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown(
            """
            <div class="brand">
                ✦ Digit Lab
            </div>

            <p class="subtitle">
                Handwritten Digit Preprocessing & Dataset Builder
            </p>
            """,
            unsafe_allow_html=True
        )

        name = st.text_input(
            "Enter your name",
            placeholder="Your name"
        )

        if st.button(
            "Enter Digit Lab →",
            use_container_width=True
        ):

            if name.strip():

                st.session_state.user_name = name.strip()
                st.session_state.logged_in = True

                st.rerun()

            else:

                st.warning("Please enter your name.")

    st.stop()


# =========================================================
# HEADER
# =========================================================

header_col1, header_col2 = st.columns([4, 1])

with header_col1:

    st.markdown(
        """
        <div class="brand">
            ✦ Digit Lab
        </div>

        <div class="subtitle">
            Handwritten Digit Collection & Preprocessing
        </div>
        """,
        unsafe_allow_html=True
    )


with header_col2:

    st.markdown(
        f"""
        <div style="
            text-align:right;
            padding-top:10px;
            color:#6B4A3A;
            font-weight:600;
        ">
            Hello, {st.session_state.user_name} 👋
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")


# =========================================================
# DATASET INFORMATION
# =========================================================

DATASET_FILE = "dataset.csv"


if os.path.exists(DATASET_FILE):

    dataset = pd.read_csv(DATASET_FILE)

else:

    dataset = pd.DataFrame()


if not dataset.empty:

    total_samples = len(dataset)

    if "label" in dataset.columns:

        digit_counts = dataset["label"].value_counts()

    else:

        digit_counts = {}

else:

    total_samples = 0
    digit_counts = {}


# =========================================================
# TOP METRICS
# =========================================================

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Total Samples",
        total_samples
    )

with m2:

    st.metric(
        "Current Digit",
        st.session_state.current_digit
    )

with m3:

    st.metric(
        "Features",
        "784 pixels"
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# DIGIT COLLECTION
# =========================================================

st.markdown(
    """
    <div class="card">

    <h2>Digit Collection</h2>

    <p>
    Draw the digit shown below. Your drawing will pass through
    the complete preprocessing pipeline and can then be saved
    directly into <b>dataset.csv</b>.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CURRENT DIGIT
# =========================================================

digit = st.session_state.current_digit

st.markdown(
    f"""
    <div class="digit-counter">
        Draw Digit&nbsp;&nbsp;
        <span style="font-size:32px;">
            {digit}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# CANVAS + PIPELINE
# =========================================================

left_column, right_column = st.columns([1, 1.4])


# =========================================================
# LEFT COLUMN
# =========================================================

with left_column:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("✍️ Drawing Canvas")

    stroke_width = st.slider(
        "Stroke Width",
        min_value=1,
        max_value=30,
        value=10
    )

    if st.button(
        "Clear Canvas",
        use_container_width=True
    ):

        st.session_state.canvas_key += 1

        st.rerun()


    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with right_column:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("⚙️ Preprocessing Pipeline")

    if canvas_result.image_data is not None:

        # -------------------------------------------------
        # STEP 1 + STEP 2
        # -------------------------------------------------

        grayscale_image, binary_image = (
            preprocess_to_grayscale(
                canvas_result.image_data
            )
        )


        # -------------------------------------------------
        # STEP 3
        # -------------------------------------------------

        cropped_image = get_bounding_crop(
            binary_image
        )


        if cropped_image is not None:

            # -------------------------------------------------
            # STEP 4
            # -------------------------------------------------

            padded_image = pad_to_square(
                cropped_image
            )


            # -------------------------------------------------
            # STEP 5
            # -------------------------------------------------

            resized_image = resize_image(
                padded_image,
                28
            )


            # -------------------------------------------------
            # STEP 6
            # -------------------------------------------------

            recentered_image = recenter_image(
                resized_image
            )


            # -------------------------------------------------
            # STEP 7
            # -------------------------------------------------

            final_matrix = np.rint(
                recentered_image
            ).astype(int)


            # =================================================
            # PREPROCESSING STEPS
            # =================================================

            st.caption(
                "Processing flow: Original → Grayscale → "
                "Binary → Crop → Padding → Resize → "
                "Recenter → Final Pixels"
            )


            # =================================================
            # ROW 1 — STEPS 1 TO 4
            # =================================================

            step1, step2, step3, step4 = st.columns(4)


            # -------------------------------------------------
            # STEP 1
            # -------------------------------------------------

            with step1:

                st.markdown(
                    "**Step 1**  \n"
                    "Original"
                )

                st.image(
                    canvas_result.image_data,
                    width=110
                )


            # -------------------------------------------------
            # STEP 2
            # -------------------------------------------------

            with step2:

                st.markdown(
                    "**Step 2**  \n"
                    "Grayscale"
                )

                st.image(
                    grayscale_image,
                    width=110,
                    clamp=True
                )


            # -------------------------------------------------
            # STEP 3
            # -------------------------------------------------

            with step3:

                st.markdown(
                    "**Step 3**  \n"
                    "Binarization"
                )

                st.image(
                    binary_image,
                    width=110,
                    clamp=True
                )


            # -------------------------------------------------
            # STEP 4
            # -------------------------------------------------

            with step4:

                st.markdown(
                    "**Step 4**  \n"
                    "Bounding Box"
                )

                st.image(
                    cropped_image,
                    width=110,
                    clamp=True
                )


            # =================================================
            # SPACE BETWEEN ROWS
            # =================================================

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )


            # =================================================
            # ROW 2 — STEPS 5 TO 8
            # =================================================

            step5, step6, step7, step8 = st.columns(4)


            # -------------------------------------------------
            # STEP 5
            # -------------------------------------------------

            with step5:

                st.markdown(
                    "**Step 5**  \n"
                    "Square Padding"
                )

                st.image(
                    padded_image,
                    width=110,
                    clamp=True
                )


            # -------------------------------------------------
            # STEP 6
            # -------------------------------------------------

            with step6:

                st.markdown(
                    "**Step 6**  \n"
                    "Resize 28 × 28"
                )

                st.image(
                    resized_image,
                    width=110,
                    clamp=True
                )


            # -------------------------------------------------
            # STEP 7
            # -------------------------------------------------

            with step7:

                st.markdown(
                    "**Step 7**  \n"
                    "Recenter"
                )

                st.image(
                    recentered_image,
                    width=110,
                    clamp=True
                )


            # -------------------------------------------------
            # STEP 8
            # -------------------------------------------------

            with step8:

                st.markdown(
                    "**Step 8**  \n"
                    "Final Pixels"
                )

                st.image(
                    final_matrix,
                    width=110,
                    clamp=True
                )


            # =================================================
            # FINAL MATRIX
            # =================================================

            st.markdown("---")

            st.subheader(
                "🔢 Final 28 × 28 Pixel Matrix"
            )

            st.caption(
                "28 × 28 = 784 numerical pixel features"
            )

            with st.expander(
                "View Pixel Matrix"
            ):

                st.dataframe(
                    final_matrix,
                    use_container_width=True
                )


            # =================================================
            # HEATMAP
            # =================================================

            st.subheader(
                "🖼️ Pixel Heatmap"
            )

            fig, ax = plt.subplots(
                figsize=(5, 5)
            )

            heatmap = ax.imshow(
                final_matrix,
                interpolation="nearest",
                cmap="gray"
            )

            ax.set_title(
                "28 × 28 Pixel Representation"
            )

            plt.colorbar(
                heatmap,
                ax=ax
            )

            st.pyplot(fig)

            plt.close(fig)


            # =================================================
            # SAVE SAMPLE
            # =================================================

            st.markdown("---")

            if st.button(
                f"Save Digit {digit} to Dataset",
                use_container_width=True
            ):

                # ---------------------------------------------
                # FLATTEN 28 × 28 → 784
                # ---------------------------------------------

                flattened_pixels = final_matrix.flatten()


                # ---------------------------------------------
                # CREATE DATASET ROW
                # ---------------------------------------------

                row = {
                    "label": digit
                }

                for index, pixel in enumerate(
                    flattened_pixels
                ):

                    row[f"pixel_{index}"] = pixel


                new_row = pd.DataFrame(
                    [row]
                )


                # ---------------------------------------------
                # APPEND TO DATASET
                # ---------------------------------------------

                if os.path.exists(DATASET_FILE):

                    existing_dataset = pd.read_csv(
                        DATASET_FILE
                    )

                    updated_dataset = pd.concat(
                        [
                            existing_dataset,
                            new_row
                        ],
                        ignore_index=True
                    )

                else:

                    updated_dataset = new_row


                updated_dataset.to_csv(
                    DATASET_FILE,
                    index=False
                )


                st.success(
                    f"Digit {digit} saved successfully!"
                )


                # ---------------------------------------------
                # MOVE TO NEXT DIGIT
                # ---------------------------------------------

                if digit < 9:

                    st.session_state.current_digit += 1

                else:

                    st.session_state.current_digit = 0


                # ---------------------------------------------
                # CLEAR CANVAS
                # ---------------------------------------------

                st.session_state.canvas_key += 1

                st.rerun()


        else:

            st.warning(
                "No digit detected. Please draw the digit."
            )

    else:

        st.info(
            "Draw the digit on the canvas to begin."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# DATASET PREVIEW
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="card">

    <h2>📊 Dataset</h2>

    <p>
    Every saved digit is stored as one row in
    <b>dataset.csv</b>.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


if os.path.exists(DATASET_FILE):

    dataset = pd.read_csv(
        DATASET_FILE
    )

    st.write(
        f"**{len(dataset)} samples collected**"
    )

    st.dataframe(
        dataset.head(10),
        use_container_width=True
    )

else:

    st.info(
        "No samples have been collected yet."
    )