from fuzzywuzzy import fuzz
import pdfplumber
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pytesseract
import json
import matplotlib.pyplot as plt

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

def process_pdf_for_anchors(pdf_path, anchor_words):
    all_text_data = []
    anchors = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number in range(len(pdf.pages)):
            page = pdf.pages[page_number]
            page_image = page.to_image(resolution=150)
            page_image_pil = page_image.original

            black_boxes = find_black_boxes(page_image_pil)
            if black_boxes:
                text_data = extract_text_near_black_boxes(page_image_pil, black_boxes)
                for box, text in text_data:
                    all_text_data.append((page_number, box, text))
                    for anchor in anchor_words:
                        if fuzz.partial_ratio(anchor, text) > 80:
                            anchors.append((anchor, box, page_number))
    return anchors, all_text_data

def find_immediate_rectangle_below_anchor(image, anchor_box):
    gray = np.array(image.convert('L'))
    np_image = np.array(gray)

    # Crop the area below the anchor box
    cropped_image = np_image[anchor_box[3]:, :]

    # Apply edge detection
    edges = cv2.Canny(cropped_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour that is immediately below the anchor box
    min_y = float('inf')
    immediate_rectangle = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if y < min_y and y > 0:
            min_y = y
            immediate_rectangle = (x, y + anchor_box[3], x + w, y + h + anchor_box[3])

    return immediate_rectangle

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
def draw_rectangles(image, fields_coordinates, checkbox_fields_coordinates, rectangle_box):
    draw = ImageDraw.Draw(image)

    # Draw the main rectangle box
    draw.rectangle(rectangle_box, outline="green", width=2)
    
    box_width = rectangle_box[2] - rectangle_box[0]
    box_height = rectangle_box[3] - rectangle_box[1]

    # Draw fields
    for field, coordinates in fields_coordinates.items():
        try:
            top_left_x = rectangle_box[0] + int(float(coordinates[0]) * box_width)
            top_left_y = rectangle_box[1] + int(float(coordinates[1]) * box_height)
            bottom_right_x = rectangle_box[0] + int(float(coordinates[2]) * box_width)
            bottom_right_y = rectangle_box[1] + int(float(coordinates[3]) * box_height)
            draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline="blue", width=2)
        except ValueError as e:
            print(f"Error processing field coordinates {field}: {coordinates} - {e}")
    
    # Draw checkboxes
    for checkbox, coordinates in checkbox_fields_coordinates.items():
        try:
            top_left_x = rectangle_box[0] + int(float(coordinates[0]) * box_width)
            top_left_y = rectangle_box[1] + int(float(coordinates[1]) * box_height)
            bottom_right_x = rectangle_box[0] + int(float(coordinates[2]) * box_width)
            bottom_right_y = rectangle_box[1] + int(float(coordinates[3]) * box_height)
            draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline="red", width=2)
        except ValueError as e:
            print(f"Error processing checkbox coordinates {checkbox}: {coordinates} - {e}")

    return image



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

def main():
    pdf_path = 'L090095599 - B146763.pdf'
    json_paths = {
        "Date de constitution (Griindungsdatum)": '7.json',
        "Durée (Dauer der Gesellschaft)": '8.json',
        "Exercice social (Geschaftsjahr)": '9.json',
    }
    
    anchor_words = list(json_paths.keys())
    anchors, all_text_data = process_pdf_for_anchors(pdf_path, anchor_words)

    if anchors:
        with pdfplumber.open(pdf_path) as pdf:
            for detected_anchor, anchor_box, page_number in anchors:
                json_path = json_paths[detected_anchor]
                
                with open(json_path, 'r') as file:
                    json_data = json.load(file)
                
                page = pdf.pages[page_number]
                page_image = page.to_image(resolution=150)
                page_image_pil = page_image.original

                rectangle_box = find_immediate_rectangle_below_anchor(page_image_pil, anchor_box)
                if rectangle_box:
                    print(f"Processing anchor: {detected_anchor} on page {page_number + 1}")
                    # Extract text from fields
                    extracted_text = extract_text_from_fields(page_image_pil, json_data, rectangle_box)
                    for field, text in extracted_text.items():
                        print(f"Field: {field}, Extracted Text: {text}")
                    
                    # Check the status of checkboxes
                    checkbox_status = check_checkbox_status(page_image_pil, json_data, rectangle_box)
                    for checkbox, status in checkbox_status.items():
                        print(f"Checkbox: {checkbox}, Status: {status}")

                    # Draw rectangles on the image for visualization
                    fields_coordinates = {field: (json_data["fields"][field]["relative_top_left"]["x"], 
                                                  json_data["fields"][field]["relative_top_left"]["y"], 
                                                  json_data["fields"][field]["relative_bottom_right"]["x"], 
                                                  json_data["fields"][field]["relative_bottom_right"]["y"]) 
                                          for field in json_data["boxes"][0]["fields"] if not field.startswith("checkbox_")}
                    checkbox_fields_coordinates = {field: (json_data["fields"][field]["relative_top_left"]["x"], 
                                                            json_data["fields"][field]["relative_top_left"]["y"], 
                                                            json_data["fields"][field]["relative_bottom_right"]["x"], 
                                                            json_data["fields"][field]["relative_bottom_right"]["y"]) 
                                                    for field in json_data["boxes"][0]["fields"] if field.startswith("checkbox_")}
                    page_image_pil = draw_rectangles(page_image_pil, fields_coordinates, checkbox_fields_coordinates, rectangle_box)

                    plt.figure(figsize=(10, 15))
                    plt.imshow(page_image_pil)
                    plt.axis('off')
                    plt.show()

                else:
                    print(f"No rectangle found immediately below the anchor: {detected_anchor}")
    else:
        print("No anchor found in the PDF.")

if __name__ == "__main__":
    main()
