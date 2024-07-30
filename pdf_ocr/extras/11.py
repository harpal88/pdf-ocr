import pdfplumber
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import cv2
import pytesseract
import json
import re

def find_black_boxes(image, min_gray_value=0, max_gray_value=10, min_width=20, min_height=20, max_width=40, max_height=40, min_fill_ratio=0.9):
    gray = image.convert('L')
    left_side_width = image.width // 6
    left_side = gray.crop((0, 0, left_side_width, image.height))

    np_image = np.array(left_side)
    np_image = cv2.inRange(np_image, min_gray_value, max_gray_value)

    kernel = np.ones((5, 5), np.uint8)
    np_image = cv2.morphologyEx(np_image, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(np_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        rect_area = w * h
        fill_ratio = area / rect_area

        if area > 200 and (min_width <= w <= max_width) and (min_height <= h <= max_height):
            cropped_image = np_image[y:y+h, x:x+w]
            black_pixels = np.sum(cropped_image == 255)
            total_pixels = cropped_image.size
            box_fill_ratio = black_pixels / total_pixels

            if box_fill_ratio > min_fill_ratio:
                boxes.append((x, y, x+w, y+h))
    return boxes

def extract_text_near_black_boxes(image, black_boxes, offset=10):
    text_data = []
    for box in black_boxes:
        x1, y1, x2, y2 = box
        text_region = image.crop((x2 + offset, y1, image.width, y2))
        text = pytesseract.image_to_string(text_region, lang='eng')
        text_data.append((box, text.strip()))
    return text_data

def process_page_for_anchor(page, anchor_words):
    page_image = page.to_image(resolution=150).original
    black_boxes = find_black_boxes(page_image)
    
    if black_boxes:
        text_data = extract_text_near_black_boxes(page_image, black_boxes)
        for box, text in text_data:
            if any(anchor in text for anchor in anchor_words):
                return True, page_image
    return False, page_image

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
    enhancer = ImageEnhance.Contrast(cropped_image)
    cropped_image = enhancer.enhance(3).convert('L')
    np_image = np.array(cropped_image)
    np_image = cv2.threshold(np_image, 160, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(np_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            cv2.drawContours(np_image, [contour], -1, (0, 0, 0), -1)
    cleaned_image = Image.fromarray(cv2.bitwise_not(np_image))
    custom_config = r'--psm 6'
    text = pytesseract.image_to_string(cleaned_image, config=custom_config).strip()
    return re.sub(r'[^0-9a-zA-ZÀ-ÿ\s.,\'"-/]', '', text).strip()

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

def process_pdf_with_json(pdf, json_mappings, anchor_words):
    start_processing = False
    json_paths = []
    page_index_after_anchor = None
    fields_data_all = []
    checkbox_status_all = {}

    for version in json_mappings:
        json_paths.extend(json_mappings[version])

    for page_index, page in enumerate(pdf.pages):
        if not start_processing:
            success, page_image = process_page_for_anchor(page, anchor_words)
            if success:
                start_processing = True
                page_index_after_anchor = page_index
                try:
                    with open(json_paths[0], 'r') as file:
                        json_data = json.load(file)
                    fields_data = draw_rectangles_and_extract_fields(page_image, json_data)
                    checkbox_status = check_checkbox_status(page_image, json_data, detect_boxes(page_image))
                    fields_data_all.extend(fields_data)
                    checkbox_status_all.update(checkbox_status)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error loading or decoding JSON from {json_paths[0]}: {e}")
                continue

        if start_processing and page_index == page_index_after_anchor + 1:
            json_index = 1  # The second JSON path
            if json_index < len(json_paths):
                json_path = json_paths[json_index]
                try:
                    with open(json_path, 'r') as file:
                        json_data = json.load(file)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error loading or decoding JSON from {json_path}: {e}")
                    continue
                page_image = page.to_image(resolution=150).original
                fields_data = draw_rectangles_and_extract_fields(page_image, json_data)
                checkbox_status = check_checkbox_status(page_image, json_data, detect_boxes(page_image))
                fields_data_all.extend(fields_data)
                checkbox_status_all.update(checkbox_status)
                break  # Exit after processing the second JSON file

    return fields_data_all, checkbox_status_all

if __name__ == "__main__":
    pdf_path = 'L090095599 - B146763.pdf'
    json_mappings = {
        "11.1": ['11.1_1.json', '11.1_2.json']
    }
    anchor_words = ["Administrateur"]

    with pdfplumber.open(pdf_path) as pdf:
        fields_data, checkbox_status = process_pdf_with_json(pdf, json_mappings, anchor_words)
        for field, text in fields_data:
            print(f"Field: {field}, Extracted Text: {text}")
        for checkbox, status in checkbox_status.items():
            print(f"Checkbox: {checkbox}, Status: {status}")
