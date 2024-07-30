import pdfplumber
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import pytesseract
import json
import re

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
    np_image = np.array(grayscale_image)
    _, binary_image = cv2.threshold(np_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary_image)

def extract_text_from_image(image):
    custom_config = '--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(image, config=custom_config).strip()
    return text

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

def extract_text_from_fields(image, json_data, aligned_boxes):
    fields_data = []
    for i, (box_x, box_y, box_width, box_height) in enumerate(aligned_boxes):
        if i >= len(json_data["boxes"]):
            continue
        box_coordinates = {'x': box_x, 'y': box_y, 'width': box_width, 'height': box_height}
        for field in json_data["boxes"][i]["fields"]:
            field_coordinates = json_data["fields"].get(field, None)
            if field_coordinates:
                field_text = extract_text_from_image_field(image, box_coordinates, field_coordinates)
                fields_data.append((field, field_text))
    return fields_data

def extract_text_from_image_field(image, box_coordinates, field_coordinates):
    top_left_x = int(box_coordinates['x'] + field_coordinates['relative_top_left']['x'] * box_coordinates['width'])
    top_left_y = int(box_coordinates['y'] + field_coordinates['relative_top_left']['y'] * box_coordinates['height'])
    bottom_right_x = int(box_coordinates['x'] + field_coordinates['relative_bottom_right']['x'] * box_coordinates['width'])
    bottom_right_y = int(box_coordinates['y'] + field_coordinates['relative_bottom_right']['y'] * box_coordinates['height'])
    
    cropped_image = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    enhancer = ImageEnhance.Contrast(cropped_image)
    cropped_image = enhancer.enhance(2).convert('L')
    
    np_image = np.array(cropped_image)
    np_image = cv2.GaussianBlur(np_image, (3, 3), 0)
    _, binary_image = cv2.threshold(np_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            cv2.drawContours(binary_image, [contour], -1, (0, 0, 0), -1)
    
    cleaned_image = Image.fromarray(cv2.bitwise_not(binary_image))
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(cleaned_image, config=custom_config).strip()
    
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

def process_pdf_and_extract_text(pdf_path, search_region, box_width_range, box_height_range, json_paths):
    valid_patterns = re.compile(r"11[1-5]$")
    extracted_info = []
    
    with pdfplumber.open(pdf_path) as pdf:
        page_number = 0
        while page_number < len(pdf.pages):
            page = pdf.pages[page_number]
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
                    
                    try:
                        with open(json_paths[0], 'r') as file:
                            json_data = json.load(file)
                        
                        detected_boxes = detect_boxes(page_image)
                        aligned_boxes = align_boxes(detected_boxes)
                        fields_data = extract_text_from_fields(page_image, json_data, aligned_boxes)
                        checkbox_status = check_checkbox_status(page_image, json_data, aligned_boxes)
                        
                        print(f"Page {page_number + 1} - Box {box}: {text}")
                        for field, field_text in fields_data:
                            print(f"Field: {field}, Extracted Text: {field_text}")
                        for checkbox, status in checkbox_status.items():
                            print(f"Checkbox: {checkbox}, Status: {status}")
                        
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        print(f"Error loading or decoding JSON from {json_paths[0]}: {e}")
                    
                    if page_number + 1 < len(pdf.pages):
                        next_page = pdf.pages[page_number + 1]
                        next_page_image = next_page.to_image(resolution=150).original
                        
                        try:
                            with open(json_paths[1], 'r') as file:
                                json_data = json.load(file)
                            
                            detected_boxes = detect_boxes(next_page_image)
                            aligned_boxes = align_boxes(detected_boxes)
                            fields_data = extract_text_from_fields(next_page_image, json_data, aligned_boxes)
                            checkbox_status = check_checkbox_status(next_page_image, json_data, aligned_boxes)
                            
                            print(f"Page {page_number + 2} - Box N/A: N/A")
                            for field, field_text in fields_data:
                                print(f"Field: {field}, Extracted Text: {field_text}")
                            for checkbox, status in checkbox_status.items():
                                print(f"Checkbox: {checkbox}, Status: {status}")
                            
                        except (FileNotFoundError, json.JSONDecodeError) as e:
                            print(f"Error loading or decoding JSON from {json_paths[1]}: {e}")
                        
                        page_number += 1  # Skip the next page as it's already processed
                else:
                    print(f'Text "{text}" on page {page_number + 1} does not match the valid patterns.')
            else:
                print(f'No box detected on page {page_number + 1} within the specified region and dimensions.')
            
            page_number += 1
    
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
            print(f"Page {info['page']} - Box {info['box']}: {info['text']}")
    else:
        print("No matching anchor found.")
