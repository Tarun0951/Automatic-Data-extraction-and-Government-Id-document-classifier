import streamlit as st
from PIL import Image, ImageDraw
import os
import model
import ocr
import pandas as pd
import uuid
from gtts import gTTS
from io import BytesIO
import base64

# Create a temporary directory if it doesn't exist
TEMP_DIR = 'temp_images'
os.makedirs(TEMP_DIR, exist_ok=True)

# Function to save uploaded file temporarily
def save_uploaded_file(uploaded_file):
    temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

# Function to generate a unique filename
def unique_filename(file_path):
    base_name, ext = os.path.splitext(file_path)
    unique_id = str(uuid.uuid4())[:8]
    unique_name = f"{base_name}_{unique_id}{ext}"
    return unique_name

# Function to extract data using extract.py
def extract_data(img_path):
    from extract import extract_document_data
    extracted_data = extract_document_data(img_path)
    df = pd.DataFrame([extracted_data])
    return df

# Function to highlight text areas in the image
def highlight_text_areas(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline="red", width=3)
    return image

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    audio_data = base64.b64encode(audio_fp.read()).decode()
    return f"data:audio/mp3;base64,{audio_data}"

# Set page config and background
st.set_page_config(
    page_title="Document Classifier",
    page_icon=":file_folder:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #ff8c00, #ffa500, #ffd700);
            color: #333333;
            font-family: Arial, sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #002147;
            color: white;
        }
        .stButton button {
            background-color: #ff8c00;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #ff4500;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Main content
st.title("Document Classifier and Data Extractor")
st.markdown('--------')

# Define sidebar content
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    img_path = save_uploaded_file(uploaded_file)

    # Display the uploaded image
    image = Image.open(img_path)
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(img_path)
        st.image(image, caption='Uploaded Image', width=250)

    with col2:
        # Extract text using OCR
        extracted_text, boxes = ocr.extract_text_with_boxes(img_path)
        highlighted_image = highlight_text_areas(image.copy(), boxes)
        st.image(highlighted_image, caption='Highlighted Image', width=250)

    # Predict label based on extracted text using the model
    predicted_label_model = model.predict_image(img_path)

    # Predict label based on extracted text using OCR
    predicted_label_ocr = ocr.predict_text(extracted_text)

    # Display the extracted text, predicted labels from model and OCR
    st.write("Predicted Label (Model):")
    st.info(predicted_label_model)
    

    # Text-to-Speech for extracted text
    st.audio(text_to_speech(extracted_text), format="audio/mp3")

    # Save the image in respective folders in the temp directory
    if predicted_label_model:
        save_dir = os.path.join(TEMP_DIR, predicted_label_model)
        os.makedirs(save_dir, exist_ok=True)

        unique_name = unique_filename(img_path)
        save_path = os.path.join(save_dir, os.path.basename(unique_name))
        os.rename(img_path, save_path)

    # Button to view extracted data
    if st.button('View Extracted Data', key='view_data'):
        extracted_df = extract_data(save_path)
        st.write("Extracted Data:")
        st.dataframe(extracted_df)

    # Button to export extracted data
    if st.button('Export Extracted Data', key='export_data'):
        extracted_df = extract_data(save_path)
        csv = extracted_df.to_csv(index=False)
        st.download_button(
            label='Download Extracted Data',
            data=csv,
            file_name='extracted_data.csv',
            mime='text/csv',
        )
