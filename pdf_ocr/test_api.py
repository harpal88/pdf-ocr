import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_upload_pdf():
    response = client.post("/upload-pdf/", files={"file": ("test1.pdf", open("path/to/your/test1.pdf", "rb"))})
    assert response.status_code == 200
    assert response.json()["info"] == "File uploaded and processed successfully"

def test_get_data():
    response = client.get("/data/1")
    assert response.status_code == 200
    assert "id" in response.json()

def test_get_all_data():
    response = client.get("/data/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
