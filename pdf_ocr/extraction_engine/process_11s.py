import json
import re
from PIL import Image, ImageEnhance, ImageDraw
import pdfplumber
import pytesseract
import numpy as np
import cv2
from fuzzywuzzy import fuzz
import logging

logging.basicConfig(level=logging.INFO)

def calculate_skew_angle(image, min_non_zero_angle_ratio=0.15):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=7)
    
    if lines is None:
        return 0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -0.9 < abs(angle) < 0.9:
            angles.append(angle)

    num_zeros = len([a for a in angles if a == 0])
    num_non_zeros = len(angles) - num_zeros

    if num_non_zeros / num_zeros > min_non_zero_angle_ratio:
        positive_angles = [a for a in angles if a > 0 and a < 0.7]
        negative_angles = [a for a in angles if a > -0.7 and a < 0]
        if len(positive_angles) > len(negative_angles):
            return np.average(positive_angles)
        else:
            return np.average(negative_angles)
    else:
        return 0

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated

def detect_box(image, region, min_width=50, min_height=20, max_width=70, max_height=40):
    cropped_image = image.crop(region)
    gray_image = cropped_image.convert('L')
    np_image = np.array(gray_image)
    blurred_image = cv2.GaussianBlur(np_image, (5, 5), 0)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_width <= w <= max_width and min_height <= h <= max_height:
            return (x + region[0], y + region[1], x + w + region[0], y + h + region[1])
    return None

def preprocess_image(image):
    grayscale_image = image.convert('L')
    enhancer = ImageEnhance.Contrast(grayscale_image)
    enhanced_image = enhancer.enhance(2)
    
    np_image = np.array(enhanced_image)
    
    # Apply median blurring to reduce noise
    np_image = cv2.medianBlur(np_image, 1)
    
    # Use morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    np_image = cv2.morphologyEx(np_image, cv2.MORPH_CLOSE, kernel)
    
    _, binary_image = cv2.threshold(np_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(binary_image)

def extract_text_from_image(image):
    custom_config = '--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(image, config=custom_config).strip()
    return text

def extract_text_from_image_field(image, box_coordinates, field_coordinates):
    top_left_x = int(box_coordinates['x'] + field_coordinates['relative_top_left']['x'] * box_coordinates['width'])
    top_left_y = int(box_coordinates['y'] + field_coordinates['relative_top_left']['y'] * box_coordinates['height'])
    bottom_right_x = int(box_coordinates['x'] + field_coordinates['relative_bottom_right']['x'] * box_coordinates['width'])
    bottom_right_y = int(box_coordinates['y'] + field_coordinates['relative_bottom_right']['y'] * box_coordinates['height'])
    
    cropped_image = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    cropped_image = preprocess_image(cropped_image)
    
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(cropped_image, config=custom_config).strip()
    
    return re.sub(r'[^0-9a-zA-ZÀ-ÿ\s.,\'"-/]', '', text).strip()

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

def draw_rectangles_and_extract_fields(page_image, json_data):
    draw = ImageDraw.Draw(page_image)
    detected_boxes = detect_boxes(page_image)
    aligned_boxes = align_boxes(detected_boxes)
    fields_data = []
    for i, (box_x, box_y, box_width, box_height) in enumerate(aligned_boxes):
        if i >= len(json_data["boxes"]):
            continue
        box_coordinates = {'x': box_x, 'y': box_y, 'width': box_width, 'height': box_height}
        draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], outline="green", width=2)
        for field in json_data["boxes"][i]["fields"]:
            field_coordinates = json_data["fields"].get(field, None)
            if field_coordinates:
                field_text = extract_text_from_image_field(page_image, box_coordinates, field_coordinates)
                fields_data.append((field, field_text))
                # Draw the field rectangle
                top_left_x = int(box_coordinates['x'] + field_coordinates['relative_top_left']['x'] * box_coordinates['width'])
                top_left_y = int(box_coordinates['y'] + field_coordinates['relative_top_left']['y'] * box_coordinates['height'])
                bottom_right_x = int(box_coordinates['x'] + field_coordinates['relative_bottom_right']['x'] * box_coordinates['width'])
                bottom_right_y = int(box_coordinates['y'] + field_coordinates['relative_bottom_right']['y'] * box_coordinates['height'])
                draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline="blue", width=2)
    return fields_data

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

def process_pdf_and_extract_text(pdf_path, search_region, box_width_range, box_height_range, json_paths):
    valid_patterns = re.compile(r"11[1-5]$")
    extracted_info = []
    
    with pdfplumber.open(pdf_path) as pdf:
        first_page_image = pdf.pages[0].to_image(resolution=150).original
        angle = calculate_skew_angle(np.array(first_page_image))
        
        for page_number, page in enumerate(pdf.pages):
            page_image = page.to_image(resolution=150).original
            
            box = detect_box(
                page_image,
                region=search_region,
                min_width=box_width_range[0],
                min_height=box_height_range[0],
                max_width=box_width_range[1],
                max_height=box_height_range[1]
            )
            
            if box:
                cropped_image = page_image.crop(box)
                cropped_image = preprocess_image(cropped_image)
                
                text = extract_text_from_image(cropped_image)
                
                if valid_patterns.match(text):
                    extracted_info.append({
                        "page": page_number + 1,
                        "box": box,
                        "text": text
                    })
                    
                    if angle != 0:
                        rotated_image = rotate_image(np.array(page_image), angle)
                        page_image = Image.fromarray(rotated_image)
                    
                    try:
                        with open(json_paths[0], 'r') as file:
                            json_data = json.load(file)
                        
                        detected_boxes = detect_boxes(page_image)
                        aligned_boxes = align_boxes(detected_boxes)
                        fields_data = draw_rectangles_and_extract_fields(page_image, json_data)
                        checkbox_status = check_checkbox_status(page_image, json_data, aligned_boxes)
                        
                        for field, field_text in fields_data:
                            extracted_info.append({
                                "unique_id": "1",  # Placeholder, replace with actual unique ID logic
                                "filing_number": "F12345",  # Placeholder, replace with actual filing number
                                "filing_date": "2024-07-24",  # Placeholder, replace with actual filing date
                                "rcs_number": "RCS123",  # Placeholder, replace with actual RCS number
                                "dp_value": field_text,
                                "dp_unique_value": field
                            })
                        for checkbox, status in checkbox_status.items():
                            extracted_info.append({
                                "unique_id": "1",  # Placeholder, replace with actual unique ID logic
                                "filing_number": "F12345",  # Placeholder, replace with actual filing number
                                "filing_date": "2024-07-24",  # Placeholder, replace with actual filing date
                                "rcs_number": "RCS123",  # Placeholder, replace with actual RCS number
                                "dp_value": status,
                                "dp_unique_value": checkbox
                            })
                        
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        logging.error(f"Error loading or decoding JSON from {json_paths[0]}: {e}")
                    
                    if page_number + 1 < len(pdf.pages):
                        next_page = pdf.pages[page_number + 1]
                        next_page_image = next_page.to_image(resolution=150).original
                        
                        if angle != 0:
                            rotated_image = rotate_image(np.array(next_page_image), angle)
                            next_page_image = Image.fromarray(rotated_image)
                        
                        try:
                            with open(json_paths[1], 'r') as file:
                                json_data = json.load(file)
                            
                            detected_boxes = detect_boxes(next_page_image)
                            aligned_boxes = align_boxes(detected_boxes)
                            fields_data = draw_rectangles_and_extract_fields(next_page_image, json_data)
                            checkbox_status = check_checkbox_status(next_page_image, json_data, aligned_boxes)
                            
                            for field, field_text in fields_data:
                                extracted_info.append({
                                    "unique_id": "1",  # Placeholder, replace with actual unique ID logic
                                    "filing_number": "F12345",  # Placeholder, replace with actual filing number
                                    "filing_date": "2024-07-24",  # Placeholder, replace with actual filing date
                                    "rcs_number": "RCS123",  # Placeholder, replace with actual RCS number
                                    "dp_value": field_text,
                                    "dp_unique_value": field
                                })
                            for checkbox, status in checkbox_status.items():
                                extracted_info.append({
                                    "unique_id": "1",  # Placeholder, replace with actual unique ID logic
                                    "filing_number": "F12345",  # Placeholder, replace with actual filing number
                                    "filing_date": "2024-07-24",  # Placeholder, replace with actual filing date
                                    "rcs_number": "RCS123",  # Placeholder, replace with actual RCS number
                                    "dp_value": status,
                                    "dp_unique_value": checkbox
                                })
                            
                        except (FileNotFoundError, json.JSONDecodeError) as e:
                            logging.error(f"Error loading or decoding JSON from {json_paths[1]}: {e}")
                        
                        page_number += 1  # Skip the next page as it's already processed
                else:
                    logging.info(f'Text "{text}" on page {page_number + 1} does not match the valid patterns.')
            else:
                logging.info(f'No box detected on page {page_number + 1} within the specified region and dimensions.')
    
    return extracted_info


if __name__ == "__main__":
    pdf_path = 'L090089631 - B146607.pdf'
    search_region = (80, 130, 260, 280)
    box_width_range = (50, 70)
    box_height_range = (30, 40)
    json_paths = ['11.2_1.json', '11.2_2.json']
    extracted_info = process_pdf_and_extract_text(pdf_path, search_region, box_width_range, box_height_range, json_paths)
    
    if extracted_info:
        for info in extracted_info:
            logging.info(f"Page {info['page']} - Box {info['box']}: {info['text']}")
    else:
        logging.warning("No matching anchor found.")