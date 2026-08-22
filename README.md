# ✍️ Digit Drawing & MNIST Preprocessing App

An interactive Streamlit application that allows users to draw a handwritten digit and visualize the complete image preprocessing pipeline used to transform the drawing into an MNIST-style image.

## 🚀 Live Demo

https://digit-recognition-app-jskheypcbqbhap3b58zznf.streamlit.app/

## 📌 Project Overview

Handwritten digit recognition models require images to be converted into a standardized format before prediction.

This project demonstrates the complete preprocessing pipeline interactively.

The user can draw a digit on the canvas and observe how the image is transformed through multiple stages.

## 🔄 Image Processing Pipeline

The application performs the following steps:

1. **Original Canvas**
   - Captures the handwritten digit drawn by the user.

2. **Grayscale Conversion**
   - Converts the RGBA image into a grayscale image.

3. **Binarization**
   - Converts the grayscale image into a binary image using thresholding.

4. **Bounding Box Cropping**
   - Detects the digit and removes unnecessary background space.

5. **Square Padding**
   - Converts the cropped image into a square while preserving its proportions.

6. **Resizing**
   - Resizes the image to the selected resolution such as 14×14, 28×28, or 56×56.

7. **Center of Mass Recentring**
   - Shifts the digit toward the center of the image.

8. **Final Pixel Matrix**
   - Displays the processed image as a numerical pixel matrix.

9. **Pixel Heatmap**
   - Visualizes the final pixel intensities as a heatmap.

## 🛠️ Technologies Used

- Python
- Streamlit
- NumPy
- OpenCV
- SciPy
- Matplotlib
- Streamlit Drawable Canvas

## 📂 Project Structure

```text
digit-recognition-app/
│
├── app.py
│   └── Streamlit user interface
│
├── image_processing.py
│   └── Image preprocessing functions
│
├── requirements.txt
│   └── Required Python libraries
│
└── README.md
    └── Project documentation
