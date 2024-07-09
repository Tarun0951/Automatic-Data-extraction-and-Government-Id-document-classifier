import pytesseract
import cv2
import os
from fuzzywuzzy import fuzz

# Specify the Tesseract executable path if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

# Function to extract text using Tesseract OCR
def extract_text(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(gray)
    return text

def extract_text_with_boxes(image_path):
    # Implement OCR with bounding boxes here
    # For example, using pytesseract:
    from pytesseract import pytesseract, Output
    import cv2

    image = cv2.imread(image_path)
    d = pytesseract.image_to_data(image, output_type=Output.DICT)

    n_boxes = len(d['level'])
    boxes = []
    extracted_text = ""
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        boxes.append((x, y, x+w, y+h))
        extracted_text += d['text'][i] + " "

    return extracted_text, boxes


# Function to predict label based on extracted text
def predict_text(text):
    text_lower = text.lower()
    
    aadhar_keywords = [
        'aadhaar', 'aadhar', 'unique identification authority of india', 
        'uidai', 'aachar', 'aadhar card', 'aadhaar no', 'Government of India'
    ]
    pancard_keywords = [
        'income tax department', 'permanent account number', 'pan card', 
        'income tax'
    ]
    driving_license_keywords = [
        'driving license', 'driving licence', 'dl no', 
        'indian union', 'transport department','union of india', 'union of inoia'    ]
    
    def fuzzy_match(text, keywords):
        for keyword in keywords:
            if fuzz.partial_ratio(text, keyword) > 80:  
                return True
        return False

    if fuzzy_match(text_lower, aadhar_keywords):
        return 'adhar'
    elif fuzzy_match(text_lower, driving_license_keywords):
        return 'Driving License'
    elif fuzzy_match(text_lower, pancard_keywords):
        return 'pancard'
    
    return 'Unknown'
