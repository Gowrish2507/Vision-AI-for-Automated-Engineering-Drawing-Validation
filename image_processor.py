import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re

def annotate_drawing(image):
    """Draws boxes around rooms and dimensions."""
    img_array = np.array(image)
    annotated_img = img_array.copy()
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray_up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh_up = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    try:
        d = pytesseract.image_to_data(thresh_up, output_type=Output.DICT)
        room_keywords = ["KITCHEN", "BEDROOM", "BED", "HALL", "LIVING", "BATH", 
                         "TOILET", "DINING", "PORCH", "VERANDAH", "BALCONY"]
        
        for i in range(len(d['text'])):
            text = d['text'][i].strip().upper()
            conf = int(d['conf'][i])
            if conf > 20 and len(text) > 1: 
                (x, y, w, h) = (d['left'][i]//2, d['top'][i]//2, d['width'][i]//2, d['height'][i]//2)
                if any(keyword in text for keyword in room_keywords):
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                elif 'X' in text or re.search(r'\d+\.\d+', text):
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    except Exception:
        pass
    return annotated_img

def extract_ocr_text(image):
    """Extracts primary dimension number for scale calibration."""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    processed_img = cv2.dilate(thresh, kernel, iterations=1)
    
    try:
        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        numbers = re.findall(r'\d+\.\d+|\d+', text)
        valid_numbers = [n for n in numbers if float(n) > 0.5] 
        
        if not valid_numbers:
            return None, f"OCR Raw Output: '{text.strip()}'\n\nNo valid dimensions found.", processed_img
        return valid_numbers[0], text, processed_img
    except Exception as e:
        return None, f"Tesseract error: {str(e)}", processed_img

def detect_rooms_and_sizes(image):
    """Extracts room names and their dimensions (e.g., 3.00 x 2.00)."""
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        raw_text = pytesseract.image_to_string(gray).upper()
        
        room_keywords = ["KITCHEN", "BEDROOM", "BED ROOM", "HALL", "LIVING", 
                         "BATH", "BATHROOM", "TOILET", "DINING", "PORCH", "VERANDAH"]
        
        detected_data = {}
        dimension_pattern = r'(\d+\.\d+)\s*[xX]\s*(\d+\.\d+)'
        dimensions_found = re.findall(dimension_pattern, raw_text)
        
        dim_index = 0
        for room in room_keywords:
            if room in raw_text:
                clean_name = room.replace("BED ROOM", "BEDROOM").replace("BATHROOM", "BATH")
                if clean_name not in detected_data:
                    area_str = "N/A"
                    area_val = 0.0
                    if dim_index < len(dimensions_found):
                        l, w = dimensions_found[dim_index]
                        try:
                            area_val = float(l) * float(w)
                            area_str = f"{area_val:.2f} m² (from {l}x{w})"
                            dim_index += 1
                        except: pass
                    detected_data[clean_name] = {"string": area_str, "value": area_val}
        return detected_data
    except Exception:
        return {}