import json
import re
import cv2
import pdfplumber
from PIL import Image
import numpy as np
import pytesseract

from extraction_engine.process_11s import extract_text_from_image_field

def detect_boxes(page_image, min_width=100, min_height=50):
    gray_image = page_image.convert('L')
    np_image = np.array(gray_image)
    np_image = cv2.GaussianBlur(np_image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(np_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [(x, y, w, h) for (x, y, w, h) in [cv2.boundingRect(contour) for contour in contours] if w >= min_width and h >= min_height]
    return sorted(bounding_boxes, key=lambda box: (box[1], box[0]))

def align_boxes(boxes, min_gap=10, max_gap=30):
    if not boxes:
        return []
    boxes.sort(key=lambda box: box[1])
    aligned_boxes = [boxes[0]]
    for box in boxes[1:]:
        last_box = aligned_boxes[-1]
        gap = abs(box[1] - last_box[1])
        if min_gap <= gap <= max_gap:
            aligned_boxes.append((box[0], last_box[1], box[2], box[3]))
        else:
            aligned_boxes.append(box)
    return aligned_boxes

def extract_text_from_image(image, box_coordinates, field_coordinates):
    top_left_x = int(box_coordinates['x'] + field_coordinates['relative_top_left']['x'] * box_coordinates['width'])
    top_left_y = int(box_coordinates['y'] + field_coordinates['relative_top_left']['y'] * box_coordinates['height'])
    bottom_right_x = int(box_coordinates['x'] + field_coordinates['relative_bottom_right']['x'] * box_coordinates['width'])
    bottom_right_y = int(box_coordinates['y'] + field_coordinates['relative_bottom_right']['y'] * box_coordinates['height'])
    cropped_image = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    np_image = np.array(cropped_image.convert('L'))
    custom_config = r'--psm 6'
    text = pytesseract.image_to_string(np_image, config=custom_config).strip()
    return re.sub(r'[^0-9a-zA-ZÀ-ÿ\s.,\'"-/]', '', text).strip()


def check_checkbox_status(image, json_data, aligned_boxes, threshold=0.5):
    checkbox_status = {}
    for i, (box_x, box_y, box_width, box_height) in enumerate(aligned_boxes):
        if i >= len(json_data["boxes"]):
            continue
        for field in json_data["boxes"][i]["fields"]:
            if field.startswith("checkbox_"):
                rel_coordinates = json_data["fields"][field]
                top_left_x = int(box_x + rel_coordinates["relative_top_left"]["x"] * box_width)
                top_left_y = int(box_y + rel_coordinates["relative_top_left"]["y"] * box_height)
                bottom_right_x = int(box_x + rel_coordinates["relative_bottom_right"]["x"] * box_width)
                bottom_right_y = int(box_y + rel_coordinates["relative_bottom_right"]["y"] * box_height)
                checkbox_region = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
                checkbox_region_np = np.array(checkbox_region.convert('L'))
                checkbox_region_center = checkbox_region_np[
                    int(0.25 * checkbox_region_np.shape[0]):int(0.75 * checkbox_region_np.shape[0]),
                    int(0.25 * checkbox_region_np.shape[1]):int(0.75 * checkbox_region_np.shape[1])
                ]
                checkbox_region_center = cv2.GaussianBlur(checkbox_region_center, (5, 5), 0)
                binary = cv2.adaptiveThreshold(checkbox_region_center, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                filled_ratio = np.sum(binary) / binary.size
                checkbox_status[field] = "Checked" if filled_ratio > threshold else "Unchecked"
    return checkbox_status

def draw_rectangles_and_extract_fields(page_image, json_data):
    detected_boxes = detect_boxes(page_image)
    aligned_boxes = align_boxes(detected_boxes)
    fields_data = []
    for i, (box_x, box_y, box_width, box_height) in enumerate(aligned_boxes):
        if i >= len(json_data["boxes"]):
            continue
        box_coordinates = {'x': box_x, 'y': box_y, 'width': box_width, 'height': box_height}
        for field in json_data["boxes"][i]["fields"]:
            field_coordinates = json_data["fields"].get(field, None)
            if field_coordinates:
                field_text = extract_text_from_image(page_image, box_coordinates, field_coordinates)
                fields_data.append((field, field_text))
    return fields_data

def process_page_11(pdf_path, page_number, json_paths):
    with pdfplumber.open(pdf_path) as pdf:
        extracted_data = []
        for i, json_path in enumerate(json_paths):
            page = pdf.pages[page_number + i]
            page_image = page.to_image(resolution=150).original
            
            try:
                with open(json_path, 'r') as file:
                    json_data = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading or decoding JSON from {json_path}: {e}")
                continue

            fields_data = draw_rectangles_and_extract_fields(page_image, json_data)
            checkbox_status = check_checkbox_status(page_image, json_data, detect_boxes(page_image))
            
            for field, text in fields_data:
                print(f"Field: {field}, Extracted Text: {text}")
                extracted_data.append({
                    "unique_id": "1",  # Placeholder, replace with actual unique ID logic
                    "filing_number": "F12345",  # Placeholder, replace with actual filing number
                    "filing_date": "2024-07-24",  # Placeholder, replace with actual filing date
                    "rcs_number": "RCS123",  # Placeholder, replace with actual RCS number
                    "dp_value": text,
                    "dp_unique_value": field
                })
            for checkbox, status in checkbox_status.items():
                print(f"Checkbox: {checkbox}, Status: {status}")
                extracted_data.append({
                    "unique_id": "1",  # Placeholder, replace with actual unique ID logic
                    "filing_number": "F12345",  # Placeholder, replace with actual filing number
                    "filing_date": "2024-07-24",  # Placeholder, replace with actual filing date
                    "rcs_number": "RCS123",  # Placeholder, replace with actual RCS number
                    "dp_value": status,
                    "dp_unique_value": checkbox
                })
                
        return extracted_data

def draw_rectangles_and_extract_fields(page_image, json_data):
    detected_boxes = detect_boxes(page_image)
    aligned_boxes = align_boxes(detected_boxes)
    fields_data = []
    for i, (box_x, box_y, box_width, box_height) in enumerate(aligned_boxes):
        if i >= len(json_data["boxes"]):
            continue
        box_coordinates = {'x': box_x, 'y': box_y, 'width': box_width, 'height': box_height}
        for field in json_data["boxes"][i]["fields"]:
            field_coordinates = json_data["fields"].get(field, None)
            if field_coordinates:
                field_text = extract_text_from_image_field(page_image, box_coordinates, field_coordinates)
                fields_data.append((field, field_text))
    return fields_data

