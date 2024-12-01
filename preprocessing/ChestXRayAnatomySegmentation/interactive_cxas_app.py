import streamlit as st
import subprocess
import os
from PIL import Image
import cv2
import numpy as np

os.makedirs("tmp", exist_ok=True)


# Helper function to run the segmentation command
def run_segmentation(input_image_path, output_folder, mode="segment", gpu="cpu"):
    command = f"cxas -i {input_image_path} -o {output_folder} --mode {mode} -g {gpu} -s"
    subprocess.run(command, shell=True)
    return output_folder


# Helper function to colorize and outline the binary mask
def colorize_and_outline_mask(mask_image, color=(0, 255, 0)):
    mask_np = np.array(mask_image.convert("L"))  # Ensure it is a grayscale image
    _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    colorized_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    colorized_mask[mask_np == 255] = color  # Apply the color to mask regions
    edges = cv2.Canny(mask_np, 100, 200)  # Detect edges
    colorized_mask[edges == 255] = [255, 255, 255]  # Highlight the edges
    return colorized_mask


# Helper function to overlay mask on the image
def overlay_mask_on_image(input_image, mask_image, alpha=0.5):
    input_image_np = np.array(input_image)
    if len(input_image_np.shape) == 2:  # Convert grayscale to RGB
        input_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_GRAY2RGB)
    mask_image_resized = cv2.resize(
        mask_image, (input_image_np.shape[1], input_image_np.shape[0])
    )
    overlayed_image = cv2.addWeighted(
        input_image_np, 1 - alpha, mask_image_resized, alpha, 0
    )
    return overlayed_image


# Streamlit app
st.title("Image Segmentation Tool")

# Check if session state is initialized
if "input_image" not in st.session_state:
    st.session_state.input_image = None
    st.session_state.output_folder = None
    st.session_state.mask_files = []
    st.session_state.segmentation_done = False
    st.session_state.selected_mask = None  # Store selected mask in session state

# File uploader for user to input image
uploaded_image = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

# If a new image is uploaded, reset the session state
if uploaded_image is not None:
    if not os.path.isdir(
        os.path.join("tmp/output", os.path.splitext(uploaded_image.name)[0])
    ):
        os.makedirs("tmp", exist_ok=True)
        st.session_state.input_image = Image.open(
            uploaded_image
        )  # Store the image in session state
        input_image_path = f"tmp/{uploaded_image.name}"
        st.session_state.input_image.save(input_image_path)

        input_image_name = os.path.splitext(uploaded_image.name)[0]
        output_folder = os.path.join("tmp/output")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        st.session_state.output_folder = output_folder
        st.session_state.mask_files = []
        st.session_state.segmentation_done = False
        st.session_state.selected_mask = None  # Reset mask selection

        st.image(
            st.session_state.input_image,
            caption="Uploaded Image",
            use_column_width=True,
        )

        # Run segmentation if not already done
        if not st.session_state.segmentation_done:
            if st.button("Run Segmentation"):
                with st.spinner("Running segmentation..."):
                    run_segmentation(input_image_path, st.session_state.output_folder)
                st.session_state.output_folder = os.path.join(
                    "tmp/output", input_image_name
                )
                st.success(
                    f"Segmentation completed. Masks saved in {st.session_state.output_folder}"
                )

                st.session_state.mask_files = [
                    f
                    for f in os.listdir(st.session_state.output_folder)
                    if f.endswith(".png")
                ]
                st.session_state.segmentation_done = True

    else:
        input_image_name = os.path.splitext(uploaded_image.name)[0]
        st.session_state.input_image = Image.open(f"tmp/{uploaded_image.name}")
        st.session_state.output_folder = os.path.join("tmp/output", input_image_name)
        st.success(
            f"Segmentation completed. Masks saved in {st.session_state.output_folder}"
        )

        st.session_state.mask_files = [
            f for f in os.listdir(st.session_state.output_folder) if f.endswith(".png")
        ]
        st.session_state.segmentation_done = True


# Display uploaded image
if st.session_state.input_image is not None:

    # Only display dropdown and images if segmentation is done
    if st.session_state.segmentation_done and st.session_state.mask_files:
        # Dropdown to select a mask
        selected_mask = st.selectbox(
            "Select a mask to overlay",
            st.session_state.mask_files,
            index=(
                st.session_state.mask_files.index(st.session_state.selected_mask)
                if st.session_state.selected_mask
                else 0
            ),
        )

        # Save the selected mask in session state
        st.session_state.selected_mask = selected_mask

        # Load the selected mask
        mask_image = Image.open(
            os.path.join(st.session_state.output_folder, selected_mask)
        )

        # Colorize the binary mask and add an outline
        colorized_mask = colorize_and_outline_mask(mask_image)

        # Overlay the selected mask on the input image
        overlayed_image = overlay_mask_on_image(
            st.session_state.input_image, colorized_mask
        )

        # Display the images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                st.session_state.input_image,
                caption="Original Image",
                use_column_width=True,
            )

        with col2:
            st.image(
                overlayed_image,
                caption="Overlayed Image with Mask",
                use_column_width=True,
            )

else:
    st.info("Please upload an image to get started.")
