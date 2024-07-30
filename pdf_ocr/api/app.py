import shutil
from pathlib import Path
from extraction_engine.main import process_pdf
from database.db_setup import init_db, SessionLocal
from database.models import DataPoint
from sqlalchemy.orm import Session
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the PDF
        extracted_info = process_pdf(
            str(file_path),
            ["Date de constitution (Griindungsdatum)", "Durée (Dauer der Gesellschaft)", "Exercice social (Geschaftsjahr)", "Administrateur"],
            {
                "Date de constitution (Griindungsdatum)": '7.json',
                "Durée (Dauer der Gesellschaft)": '8.json',
                "Exercice social (Geschaftsjahr)": '9.json',
                "Administrateur": ['11.1_1.json', '11.1_2.json']
            }
        )

        # Clean up the uploaded file
        file_path.unlink()

        return {"info": "File uploaded and processed successfully", "filename": file.filename}
    except PermissionError:
        raise HTTPException(status_code=500, detail="Permission denied: 'uploaded_files'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")

@app.get("/data/{dp_id}")
async def get_data(dp_id: int):
    db: Session = SessionLocal()
    try:
        data_point = db.query(DataPoint).filter(DataPoint.id == dp_id).first()
        if data_point is None:
            raise HTTPException(status_code=404, detail="Data point not found")
        return {
            "id": data_point.id,
            "unique_id": data_point.unique_id,
            "filing_number": data_point.filing_number,
            "filing_date": data_point.filing_date,
            "rcs_number": data_point.rcs_number,
            "dp_value": data_point.dp_value,
            "dp_unique_value": data_point.dp_unique_value
        }

    finally:
        db.close()

@app.get("/data/")
async def get_all_data():
    db: Session = SessionLocal()
    try:
        data_points = db.query(DataPoint).all()
        return [
            {
                "id": dp.id, 
                "unique_id": dp.unique_id, 
                "filing_number": dp.filing_number, 
                "filing_date": dp.filing_date, 
                "rcs_number": dp.rcs_number, 
                "dp_value": dp.dp_value, 
                "dp_unique_value": dp.dp_unique_value
            } for dp in data_points
        ]
    finally:
        db.close()
