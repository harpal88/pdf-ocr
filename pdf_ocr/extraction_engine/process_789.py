import json
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import numpy as np
import cv2
from fuzzywuzzy import fuzz

def detect_boxes(image, min_width=500, min_height=50):
    gray_image = image.convert('L')
    np_image = np.array(gray_image)
    binary = cv2.adaptiveThreshold(np_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [(x, y, w, h) for (x, y, w, h) in [cv2.boundingRect(contour) for contour in contours] if w >= min_width and h >= min_height]
    return sorted(bounding_boxes, key=lambda box: (box[1], box[0]))

def extract_text_from_fields(image, json_data, rectangle_box):
    box_width = rectangle_box[2] - rectangle_box[0]
    box_height = rectangle_box[3] - rectangle_box[1]

    extracted_data = {}

    for field in json_data["boxes"][0]["fields"]:
        rel_top_left = json_data["fields"][field]["relative_top_left"]
        rel_bottom_right = json_data["fields"][field]["relative_bottom_right"]
        
        top_left_x = rectangle_box[0] + int(rel_top_left["x"] * box_width)
        top_left_y = rectangle_box[1] + int(rel_top_left["y"] * box_height)
        bottom_right_x = rectangle_box[0] + int(rel_bottom_right["x"] * box_width)
        bottom_right_y = rectangle_box[1] + int(rel_bottom_right["y"] * box_height)
        
        if top_left_x < bottom_right_x and top_left_y < bottom_right_y:
            field_region = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
            field_text = pytesseract.image_to_string(field_region, lang='eng').strip()
            extracted_data[field] = field_text
        else:
            extracted_data[field] = ""
            print(f"Invalid crop coordinates for field {field}: {(top_left_x, top_left_y, bottom_right_x, bottom_right_y)}")

    return extracted_data

def check_checkbox_status(image, json_data, rectangle_box, threshold=0.5):
    box_width = rectangle_box[2] - rectangle_box[0]
    box_height = rectangle_box[3] - rectangle_box[1]

    checkbox_status = {}

    for field in json_data["boxes"][0]["fields"]:
        if field.startswith("checkbox_"):  # Assuming checkbox fields are named starting with 'checkbox_'
            rel_top_left = json_data["fields"][field]["relative_top_left"]
            rel_bottom_right = json_data["fields"][field]["relative_bottom_right"]
            
            top_left_x = rectangle_box[0] + int(rel_top_left["x"] * box_width)
            top_left_y = rectangle_box[1] + int(rel_top_left["y"] * box_height)
            bottom_right_x = rectangle_box[0] + int(rel_bottom_right["x"] * box_width)
            bottom_right_y = rectangle_box[1] + int(rel_bottom_right["y"] * box_height)
            
            if top_left_x < bottom_right_x and top_left_y < bottom_right_y:
                checkbox_region = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
                checkbox_region_np = np.array(checkbox_region.convert('L'))

                # Focus on the central part of the checkbox
                checkbox_region_center = checkbox_region_np[
                    int(0.25 * checkbox_region_np.shape[0]):int(0.75 * checkbox_region_np.shape[0]),
                    int(0.25 * checkbox_region_np.shape[1]):int(0.75 * checkbox_region_np.shape[1])
                ]

                # Apply Gaussian Blur to reduce noise
                checkbox_region_center = cv2.GaussianBlur(checkbox_region_center, (5, 5), 0)
                
                # Apply adaptive thresholding
                binary = cv2.adaptiveThreshold(checkbox_region_center, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                
                filled_ratio = np.sum(binary) / binary.size
                
                checkbox_status[field] = "Checked" if filled_ratio > threshold else "Unchecked"

    return checkbox_status

def process_page_789(image, json_path, box):
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    rectangles = detect_boxes(image)
    matched_rectangles = []

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    matched_rectangles = [(x, y, x + w, y + h) for x, y, w, h in rectangles if y > box[1]]
    matched_rectangles = matched_rectangles[:len(json_data["boxes"])]

    extracted_data = []

    for rect in matched_rectangles:
        x0, y0, x1, y1 = rect
        if y0 > y1:
            y0, y1 = y1, y0
        if x0 > x1:
            x0, x1 = x1, x0

        extracted_text = extract_text_from_fields(image, json_data, (x0, y0, x1, y1))
        for field, text in extracted_text.items():
            print(f"Field: {field}, Extracted Text: {text}")
            draw.rectangle((x0, y0, x1, y1), outline="blue", width=2)
            draw.text((x0, y0 - 10), f"{field}: {text}", fill="blue", font=font)
            extracted_data.append({
                "unique_id": "1",
                "filing_number": "F12345",
                "filing_date": "2024-07-24",
                "rcs_number": "RCS123",
                "dp_value": text,
                "dp_unique_value": field
            })
        
        checkbox_status = check_checkbox_status(image, json_data, (x0, y0, x1, y1))
        for checkbox, status in checkbox_status.items():
            print(f"Checkbox: {checkbox}, Status: {status}")
            draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
            draw.text((x0, y0 - 10), f"{checkbox}: {status}", fill="red", font=font)
            extracted_data.append({
                "unique_id": "1",
                "filing_number": "F12345",
                "filing_date": "2024-07-24",
                "rcs_number": "RCS123",
                "dp_value": status,
                "dp_unique_value": checkbox
            })

    return extracted_data
