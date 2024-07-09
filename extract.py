import pytesseract
import cv2
from PIL import Image
import re

# Function to extract and clean data from Aadhaar card text
def extract_aadhaar_data(text):
    details = {}
    
    # Extract Name
    name_match = re.search(r'\bName\b[:\s]*(\w+\s+\w+)', text, re.IGNORECASE)
    if name_match:
        details['Name'] = name_match.group(1)

    # Extract DOB
    dob_match = re.search(r'\bDOB\b[:\s]*(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
    if dob_match:
        details['DOB'] = dob_match.group(1)

    # Aadhaar specific details
    details['Document Type'] = 'Aadhaar'
    
    # Extract Document Number
    document_no_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', text)
    if document_no_match:
        details['Document Number'] = document_no_match.group()

    # Extract Gender
    gender_match = re.search(r'\b(Male|Female|Transgender)\b', text, re.IGNORECASE)
    if gender_match:
        details['Sex/Gender'] = gender_match.group(1)

    return details

# Function to extract and clean data from PAN card text
def extract_pan_data(text):
    details = {}
    
    # Extract Name
    name_match = re.search(r'\bName\b[:\s]*(\w+\s+\w+)', text, re.IGNORECASE)
    if name_match:
        details['Name'] = name_match.group(1)

    # Extract DOB
    dob_match = re.search(r'\bDOB\b[:\s]*(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
    if dob_match:
        details['DOB'] = dob_match.group(1)

    # PAN specific details
    details['Document Type'] = 'PAN'
    
    # Extract Document Number
    document_no_match = re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text)
    if document_no_match:
        details['Document Number'] = document_no_match.group()

    # Extract Gender
    gender_match = re.search(r'\b(Male|Female|Transgender)\b', text, re.IGNORECASE)
    if gender_match:
        details['Sex/Gender'] = gender_match.group(1)

    return details

# Function to extract and clean data from Driving License text
def extract_license_data(text):
    details = {}
    
    # Extract Name
    name_match = re.search(r'\bName\b[:\s]*(\w+\s+\w+)', text, re.IGNORECASE)
    if name_match:
        details['Name'] = name_match.group(1)

    # Extract DOB
    dob_match = re.search(r'\bDOB\b[:\s]*(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
    if dob_match:
        details['DOB'] = dob_match.group(1)

    # Driving License specific details
    details['Document Type'] = 'Driving License'
    
    # Extract Document Number
    document_no_match = re.search(r'\b[A-Z]{2}\d{2}\s\d{11}\b', text)
    if document_no_match:
        details['Document Number'] = document_no_match.group()

    # Extract Gender
    gender_match = re.search(r'\b(Male|Female|Transgender)\b', text, re.IGNORECASE)
    if gender_match:
        details['Sex/Gender'] = gender_match.group(1)

    return details

# Function to process image using Tesseract
def process_image(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Tesseract to extract text
    text = pytesseract.image_to_string(gray_image)
    
    return text

# Function to identify document type and extract relevant data
def extract_document_data(image_path):
    from model import predict_image
    predict_label=predict_image(image_path)
    if predict_label == 'adhar':
        text = process_image(image_path)
        return extract_aadhaar_data(text)
    elif predict_label == 'pancard':
        text = process_image(image_path)
        return extract_pan_data(text)
    elif predict_label == 'Driving License':
        text = process_image(image_path)
        return extract_license_data(text)
    else:
        return {"Error": "Document type not recognized"}
    

# Example usage
image_path = r'C:\Users\TARUN\mini\augmented_datamini\adhar\aadhar16.jpg'
details = extract_document_data(image_path)
print(details)
