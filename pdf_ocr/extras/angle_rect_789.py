from fuzzywuzzy import fuzz
import pdfplumber
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pytesseract
import json
import matplotlib.pyplot as plt

def extract_image_from_pdf(pdf_path, page_number):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        page_image = page.to_image(resolution=150)
        return page_image.original

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

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

def calculate_skew_angle(image, min_non_zero_angle_ratio=0.15):
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

def find_rectangles(image, min_width=600, min_height=50):
    gray_image = image.convert('L')
    np_image = np.array(gray_image)
    binary = cv2.adaptiveThreshold(np_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [(x, y, w, h) for (x, y, w, h) in [cv2.boundingRect(contour) for contour in contours] if w >= min_width and h >= min_height]
    return sorted(bounding_boxes, key=lambda box: (box[1], box[0]))

def extract_text_near_black_boxes(image, black_boxes, offset=10):
    text_data = []
    for box in black_boxes:
        x1, y1, x2, y2 = box
        text_region = image.crop((x2 + offset, y1, image.width, y2))
        text = pytesseract.image_to_string(text_region, lang='eng')
        text_data.append((box, text.strip()))
    return text_data

def process_page(pdf_path, page_number, angle_with_horizontal, anchor_words, processed_anchors):
    page_image_pil = extract_image_from_pdf(pdf_path, page_number)
    
    # Rotate the page image
    if angle_with_horizontal != 0:
        rotated_image = rotate_image(np.array(page_image_pil), angle_with_horizontal)
        page_image_pil = Image.fromarray(rotated_image)
    
    # Find rectangles after rotation
    rectangles = find_rectangles(page_image_pil)
    
    # Find black boxes in the corrected image
    black_boxes = find_black_boxes(page_image_pil)
    
    # Extract text near black boxes
    text_data = extract_text_near_black_boxes(page_image_pil, black_boxes)
    
    for anchor in anchor_words:
        if anchor not in processed_anchors:
            for box, text in text_data:
                if anchor in text:
                    processed_anchors.add(anchor)
                    return [(x, y, x + w, y + h) for x, y, w, h in rectangles if y > box[1]], angle_with_horizontal

    return [], angle_with_horizontal

def process_page(pdf_path, page_number, angle_with_horizontal, anchor_words, processed_anchors):
    page_image_pil = extract_image_from_pdf(pdf_path, page_number)
    
    # Rotate the page image
    if angle_with_horizontal != 0:
        rotated_image = rotate_image(np.array(page_image_pil), angle_with_horizontal)
        page_image_pil = Image.fromarray(rotated_image)
    
    # Find rectangles after rotation
    rectangles = find_rectangles(page_image_pil)
    
    # Find black boxes in the corrected image
    black_boxes = find_black_boxes(page_image_pil)
    
    # Extract text near black boxes
    text_data = extract_text_near_black_boxes(page_image_pil, black_boxes)
    
    for anchor in anchor_words:
        if anchor not in processed_anchors:
            for box, text in text_data:
                if anchor in text:
                    processed_anchors.add(anchor)
                    matched_rectangles = [(x, y, x + w, y + h) for x, y, w, h in rectangles if y > box[1]]
                    
                    # Draw rectangles on the image
                    draw = ImageDraw.Draw(page_image_pil)
                    for rect in matched_rectangles:
                        draw.rectangle(rect, outline="red", width=2)
                    
                    # Display the image
                    plt.figure(figsize=(10, 15))
                    plt.imshow(page_image_pil)
                    plt.axis('off')
                    plt.show()
                    
                    return matched_rectangles, angle_with_horizontal

    return [], angle_with_horizontal

def process_pdf(pdf_path, anchor_words, min_non_zero_angle_ratio=0.1):
    all_rectangles = []
    all_angles = []

    # Step 1: Calculate the skew angle using the first page
    first_page_image = extract_image_from_pdf(pdf_path, 0)
    angle_with_horizontal = calculate_skew_angle(first_page_image)
    print(f"Calculated angle with horizontal: {angle_with_horizontal} degrees")
    
    processed_anchors = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page_number in range(len(pdf.pages)):
            rectangles, angle = process_page(pdf_path, page_number, angle_with_horizontal, anchor_words, processed_anchors)
            all_rectangles.extend(rectangles)
            all_angles.append(angle)
            
            # Stop if all anchors have been processed
            if len(processed_anchors) == len(anchor_words):
                break

    return all_rectangles, all_angles


if __name__ == "__main__":
    pdf_path = 'L090095599 - B146763.pdf'
    anchor_words = ["Date de constitution", "Durée", "Exercice social"]  # Example anchor words
    rectangles, angles = process_pdf(pdf_path, anchor_words)
    print(f"Rectangles: {rectangles}")
