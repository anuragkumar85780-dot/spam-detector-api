import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Spam SMS Classifier API is running"


def test_spam_prediction():
    response = client.post(
        "/predict",
        json={
            "message": "Congratulations! You have won a free prize. Call now to claim."
        }
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "spam_probability" in response.json()


def test_ham_prediction():
    response = client.post(
        "/predict",
        json={
            "message": "Hey, are we meeting for lunch today?"
        }
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "spam_probability" in response.json()