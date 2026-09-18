from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }

def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == \
        "CSV Insights API is running"

def test_analyze_csv():
    csv_content = (
        "product,region,sales,quantity\n"
        "Laptop,South,75000,5\n"
        "Mouse,North,5000,20\n"
    )

    response = client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "sales.csv",
                csv_content,
                "text/csv"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "sales.csv"
    assert data["rows"] == 2
    assert data["columns"] == 4
    assert data["numeric_summary"]["sales"]["min"] == 5000
    assert data["numeric_summary"]["sales"]["max"] == 75000

    assert data["numeric_summary"]["quantity"]["min"] == 5
    assert data["numeric_summary"]["quantity"]["max"] == 20

def test_invalid_file_type():
    response = client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "test.txt",
                "hello world",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400

    assert response.json()["detail"] == \
        "Only CSV files are supported"

def test_empty_file():
    response = client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "empty.csv",
                "",
                "text/csv"
            )
        }
    )

    assert response.status_code == 400

    assert response.json()["detail"] == \
        "Uploaded file is empty"


def test_empty_csv():
    csv_content = "product,region,sales,quantity\n"

    response = client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "empty_data.csv",
                csv_content,
                "text/csv"
            )
        }
    )

    assert response.status_code == 400

    assert response.json()["detail"] == \
        "CSV contains no data rows"



def test_copilot_order():
    response = client.post(
        "/copilot",
        params={"ticket": "Where is my order 1001?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "order_status"
    assert data["action"] == "lookup_order"
    assert data["order_id"] == "1001"
    assert "order_lookup" in data["citations"]    