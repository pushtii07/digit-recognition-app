import numpy as np
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


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Digit Preprocessing App",
    page_icon="✍️",
    layout="wide"
)


# -------------------------------------------------
# SESSION STATE FOR CANVAS RESET
# -------------------------------------------------

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("✍️ Digit Drawing & MNIST Preprocessing")

st.write(
    "Draw a digit and visualize how it is transformed "
    "step-by-step into an MNIST-style image."
)


# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------

st.sidebar.header("Canvas Controls")

stroke_width = st.sidebar.slider(
    "Stroke Width",
    min_value=1,
    max_value=30,
    value=10
)

target_size = st.sidebar.selectbox(
    "Target Resolution",
    options=[14, 28, 56],
    index=1
)


# -------------------------------------------------
# TWO COLUMN LAYOUT
# -------------------------------------------------

left_column, right_column = st.columns(2)


# -------------------------------------------------
# LEFT COLUMN: DRAWING CANVAS
# -------------------------------------------------

with left_column:

    st.subheader("Draw Your Digit")

    if st.button("🗑️ Clear Canvas"):

        st.session_state.canvas_key += 1

        st.rerun()

    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}"
    )


# -------------------------------------------------
# RIGHT COLUMN: PROCESSING PIPELINE
# -------------------------------------------------

with right_column:

    st.subheader("Image Processing Pipeline")

    if canvas_result.image_data is not None:

        # -----------------------------------------
        # STEP 1: GRAYSCALE + BINARIZATION
        # -----------------------------------------

        grayscale_image, binary_image = (
            preprocess_to_grayscale(
                canvas_result.image_data
            )
        )

        # -----------------------------------------
        # STEP 2: BOUNDING BOX
        # -----------------------------------------

        cropped_image = get_bounding_crop(
            binary_image
        )

        # -----------------------------------------
        # ORIGINAL
        # -----------------------------------------

        st.write("### Step 1: Original Canvas")

        st.image(
            canvas_result.image_data,
            caption="Original RGBA Canvas"
        )

        # -----------------------------------------
        # GRAYSCALE
        # -----------------------------------------

        st.write("### Step 2: Grayscale")

        st.image(
            grayscale_image,
            caption="Grayscale Image",
            clamp=True
        )

        # -----------------------------------------
        # BINARY
        # -----------------------------------------

        st.write("### Step 3: Binarized Image")

        st.image(
            binary_image,
            caption="Binary Image",
            clamp=True
        )

        # -----------------------------------------
        # CHECK IF DIGIT EXISTS
        # -----------------------------------------

        if cropped_image is not None:

            # -------------------------------------
            # PADDING
            # -------------------------------------

            padded_image = pad_to_square(
                cropped_image
            )

            # -------------------------------------
            # RESIZE
            # -------------------------------------

            resized_image = resize_image(
                padded_image,
                target_size
            )

            # -------------------------------------
            # RECENTER
            # -------------------------------------

            recentered_image = recenter_image(
                resized_image
            )

            # -------------------------------------
            # STEP 4: BOUNDING BOX CROP
            # -------------------------------------

            st.write(
                "### Step 4: Bounding Box Crop"
            )

            st.image(
                cropped_image,
                caption="Digit cropped to its bounding box",
                clamp=True
            )

            # -------------------------------------
            # STEP 5: SQUARE PADDING
            # -------------------------------------

            st.write(
                "### Step 5: Square Padding"
            )

            st.image(
                padded_image,
                caption="Digit padded into a square",
                clamp=True
            )

            # -------------------------------------
            # STEP 6: RESIZE
            # -------------------------------------

            st.write(
                f"### Step 6: Resize to "
                f"{target_size} × {target_size}"
            )

            st.image(
                resized_image,
                caption=(
                    f"Pixelated "
                    f"{target_size} × {target_size} Image"
                ),
                clamp=True,
                width=350
            )

            # -------------------------------------
            # STEP 7: RECENTER
            # -------------------------------------

            st.write(
                "### Step 7: Center of Mass Recentring"
            )

            st.image(
                recentered_image,
                caption="Digit shifted to the center",
                clamp=True,
                width=350
            )

            # -------------------------------------
            # STEP 8: FINAL PIXEL MATRIX
            # -------------------------------------

            st.write(
                "### Step 8: Final Pixel Matrix"
            )

            final_matrix = np.rint(
                recentered_image
            ).astype(int)

            st.dataframe(
                final_matrix,
                use_container_width=True
            )

            # -------------------------------------
            # FINAL HEATMAP
            # -------------------------------------

            st.write("### Final Pixel Heatmap")

            fig, ax = plt.subplots()

            heatmap = ax.imshow(
                final_matrix,
                interpolation="nearest",
                cmap="gray"
            )

            ax.set_title(
                f"{target_size} × "
                f"{target_size} Pixel Values"
            )

            plt.colorbar(
                heatmap,
                ax=ax
            )

            st.pyplot(fig)

            plt.close(fig)

        else:

            st.warning(
                "No digit detected. "
                "Please draw a digit on the canvas."
            )

    else:

        st.info(
            "Draw a digit to begin image processing."
        )