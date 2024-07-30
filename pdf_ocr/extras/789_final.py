import pdfplumber
from fuzzywuzzy import fuzz
from PIL import Image
import numpy as np
import cv2
import pytesseract
import json

def extract_image_from_pdf(pdf_path, page_number):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        page_image = page.to_image(resolution=150)
        return page_image.original

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def calculate_skew_angle(image,min_non_zero_angle_ratio=0.15):
    gray = preprocess_image(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=7)
    
    if lines is None:
        print("No lines detected")
        return 0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        if -0.9 < abs(angle) < 0.9:  # Filter out vertical lines and nearly horizontal lines
            angles.append(angle)

    num_zeros = len([a for a in angles if a == 0])
    num_non_zeros = len(angles) - num_zeros

    print(f"Number of 0-degree angles: {num_zeros}")
    print(f"Number of non-0-degree angles: {num_non_zeros}")

    if num_non_zeros / num_zeros > min_non_zero_angle_ratio:
        positive_angles = [a for a in angles if a > 0 and a < 0.7]
        negative_angles = [a for a in angles if a > -0.7 and a < 0]
        
        if len(positive_angles) > len(negative_angles):
            average_angle = np.average(positive_angles)
            print(f"Average positive angle: {average_angle:.2f} degrees")
            return average_angle
        else:
            median_angle = np.average(negative_angles)
            print(f"Median negative angle: {median_angle:.2f} degrees")
            return median_angle
    else:
        print("Not enough non-zero angles detected")
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

def detect_boxes(image, min_width=600, min_height=50):
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

def process_page(pdf_path, page_number, angle_with_horizontal, anchor_words, json_paths):
    page_image_pil = extract_image_from_pdf(pdf_path, page_number)
    
    # Rotate the page image
    if angle_with_horizontal != 0:
        rotated_image = rotate_image(np.array(page_image_pil), angle_with_horizontal)
        page_image_pil = Image.fromarray(rotated_image)
    
    # Find rectangles after rotation
    rectangles = detect_boxes(page_image_pil)
    
    # Find black boxes in the corrected image
    black_boxes = find_boxes(page_image_pil)
    
    # Extract text near black boxes
    text_data = extract_text_near_boxes(page_image_pil, black_boxes)
    
    all_matched_rectangles = []
    
    for anchor in anchor_words:
        page_image_pil = Image.fromarray(rotate_image(np.array(extract_image_from_pdf(pdf_path, page_number)), angle_with_horizontal))

        for box, text in text_data:
            if fuzz.partial_ratio(anchor, text) > 80:
                # Load corresponding JSON data for the anchor
                json_path = json_paths[anchor]
                with open(json_path, 'r') as file:
                    json_data = json.load(file)
                
                matched_rectangles = [(x, y, x + w, y + h) for x, y, w, h in rectangles if y > box[1]]
                matched_rectangles = matched_rectangles[:len(json_data["boxes"])]
                all_matched_rectangles.extend(matched_rectangles)
                
                for rect in matched_rectangles:
                    # Extract text from fields
                    extracted_text = extract_text_from_fields(page_image_pil, json_data, rect)
                    for field, text in extracted_text.items():
                        print(f"Field: {field}, Extracted Text: {text}")
                    
                    # Check the status of checkboxes
                    checkbox_status = check_checkbox_status(page_image_pil, json_data, rect)
                    for checkbox, status in checkbox_status.items():
                        print(f"Checkbox: {checkbox}, Status: {status}")

    return all_matched_rectangles, angle_with_horizontal

def process_pdf(pdf_path, anchor_words, json_paths, min_non_zero_angle_ratio=0.1):
    all_rectangles = []
    all_angles = []

    # Step 1: Calculate the skew angle using the first page
    first_page_image = extract_image_from_pdf(pdf_path, 0)
    angle_with_horizontal = calculate_skew_angle(first_page_image)
    print(f"Calculated angle with horizontal: {angle_with_horizontal} degrees")

    with pdfplumber.open(pdf_path) as pdf:
        for page_number in range(len(pdf.pages)):
            page_rectangles, angle = process_page(pdf_path, page_number, angle_with_horizontal, anchor_words, json_paths)
            all_rectangles.extend(page_rectangles)
            all_angles.append(angle)
    
    return all_rectangles, all_angles

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

if __name__ == "__main__":
    pdf_path = 'L090089631 - B146607.pdf'
    anchor_words = ["Date de constitution (Griindungsdatum)", "Durée (Dauer der Gesellschaft)", "Exercice social (Geschaftsjahr)"]
    json_paths = {
        "Date de constitution (Griindungsdatum)": '7.json', 
        "Durée (Dauer der Gesellschaft)": '8.json',
        "Exercice social (Geschaftsjahr)": '9.json',
    }
    angles = process_pdf(pdf_path, anchor_words, json_paths)
