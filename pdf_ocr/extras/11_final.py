import pdfplumber
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pytesseract
import json
import re

def extract_image_from_pdf(pdf_path, page_number):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        page_image = page.to_image(resolution=150)
        return page_image.original

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def calculate_skew_angle(image,min_non_zero_angle_ratio= 0.15):
    gray = preprocess_image(image)
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

    if num_non_zeros / num_zeros >min_non_zero_angle_ratio:
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

def find_boxes(image, min_gray_value=0, max_gray_value=10, min_width=20, min_height=20, max_width=40, max_height=40, min_fill_ratio=0.9, adaptive_threshold=True):
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

def extract_text_near_boxes(image, boxes, offset=10):
    text_data = []
    for box in boxes:
        x1, y1, x2, y2 = box
        text_region = image.crop((x2 + offset, y1, image.width, y2))
        text = pytesseract.image_to_string(text_region, lang='eng')
        text_data.append((box, text.strip()))
    return text_data

def detect_boxes(page_image, min_width=100, min_height=50):
    gray_image = page_image.convert('L')
    np_image = np.array(gray_image)
    np_image = cv2.GaussianBlur(np_image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(np_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [(x, y, w, h) for (x, y, w, h) in [cv2.boundingRect(contour) for contour in contours] if w >= min_width and h >= min_height]
    return sorted(bounding_boxes, key=lambda box: (box[1], box[0]))

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
            json_index = 1
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
                break

    return fields_data_all, checkbox_status_all

def process_page_for_anchor(page, anchor_words):
    page_image = page.to_image(resolution=150).original
    angle = calculate_skew_angle(np.array(page_image))
    if angle != 0:
        rotated_image = rotate_image(np.array(page_image), angle)
        page_image = Image.fromarray(rotated_image)
    boxes = find_boxes(page_image)
    
    if boxes:
        text_data = extract_text_near_boxes(page_image, boxes)
        for box, text in text_data:
            if any(anchor in text for anchor in anchor_words):
                return True, page_image
    return False, page_image

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
