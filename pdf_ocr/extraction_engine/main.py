import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import json
from PIL import Image
import numpy as np
import cv2
import pytesseract
from fuzzywuzzy import fuzz
from datetime import datetime
import logging
from sqlalchemy.orm import Session
from database.db_setup import init_db, SessionLocal
from database.models import DataPoint

# Import functions from other scripts
from extraction_engine.process_789 import process_page_789
from extraction_engine.process_11 import process_page_11
from extraction_engine.process_11s import process_pdf_and_extract_text as process_11s_final

logging.basicConfig(level=logging.INFO)

app = FastAPI()

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
def on_startup():
    init_db()

def save_to_database(data):
    logging.info(f"Saving data to database: {data}")
    db: Session = SessionLocal()
    try:
        db_data = DataPoint(
            unique_id=data.get("unique_id", "default_id"),  # Ensure unique_id is present
            filing_number=data.get("filing_number", ""),
            filing_date=datetime.strptime(data["filing_date"], "%Y-%m-%d") if data.get("filing_date") else None,
            rcs_number=data.get("rcs_number", ""),
            dp_value=data.get("dp_value", ""),
            dp_unique_value=data.get("dp_unique_value", "")
        )
        db.add(db_data)
        db.commit()
        db.refresh(db_data)
        logging.info(f"Data saved to database: {db_data}")
        return db_data
    except Exception as e:
        db.rollback()
        logging.error(f"Error saving to database: {e}")
        raise
    finally:
        db.close()


def extract_image_from_pdf(pdf_path, page_number):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        page_image = page.to_image(resolution=150)
        return page_image.original

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def calculate_skew_angle(image, min_non_zero_angle_ratio=0.15):
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

def process_pdf(pdf_path, anchor_words, json_paths):
    logging.info("Extracting the first page image for skew angle calculation.")
    first_page_image = extract_image_from_pdf(pdf_path, 0)
    angle_with_horizontal = calculate_skew_angle(first_page_image)
    logging.info(f"Calculated angle with horizontal: {angle_with_horizontal} degrees")

    extracted_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number in range(len(pdf.pages)):
            logging.info(f"Processing page {page_number}")
            page_image = extract_image_from_pdf(pdf_path, page_number)

            # Rotate the page image
            if angle_with_horizontal != 0:
                rotated_image = rotate_image(np.array(page_image), angle_with_horizontal)
                page_image = Image.fromarray(rotated_image)

            # Find black boxes
            black_boxes = find_boxes(page_image)
            logging.info(f"Found {len(black_boxes)} black boxes on page {page_number}")

            # Extract text near black boxes
            text_data = extract_text_near_boxes(page_image, black_boxes)

            for anchor in anchor_words:
                for box, text in text_data:
                    if fuzz.partial_ratio(anchor, text) > 80:
                        logging.info(f"Found anchor '{anchor}' on page {page_number}")
                        json_path = json_paths[anchor]
                        try:
                            if anchor in ["Date de constitution (Griindungsdatum)", "Durée (Dauer der Gesellschaft)", "Exercice social (Geschaftsjahr)"]:
                                page_extracted_data = process_page_789(page_image, json_path, box)
                            elif anchor == "Administrateur":
                                page_extracted_data = process_page_11(pdf_path, page_number, json_path)
                                # Process the next page using process_11s_final
                                page_extracted_data_11s = process_11s_final(pdf_path, (80, 130, 260, 280), (50, 70), (30, 40), ['11.2_1.json', '11.2_2.json'])
                                page_extracted_data.extend(page_extracted_data_11s)

                            for data in page_extracted_data:
                                data["unique_id"] = data.get("unique_id", "default_id")  # Ensure unique_id is present
                            extracted_data.extend(page_extracted_data)
                        except Exception as e:
                            logging.error(f"Error processing page {page_number} for anchor '{anchor}': {e}")

    if not extracted_data:
        logging.warning("No data extracted from the PDF.")
    else:
        # Save extracted data to the database only once
        for data in extracted_data:
            save_to_database(data)

    logging.info("PDF processing completed successfully.")
    return extracted_data
