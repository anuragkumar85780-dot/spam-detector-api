from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib


# -----------------------------
# Model Architecture
# -----------------------------
class TextClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super(TextClassifier, self).__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),

            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# -----------------------------
# Load Vectorizer
# -----------------------------
vectorizer = joblib.load("vectorizer.pkl")


# -----------------------------
# Load Model
# -----------------------------
input_size = len(vectorizer.get_feature_names_out())

model = TextClassifier(input_size)

model.load_state_dict(
    torch.load(
        "spam_model_weights.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()
# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Spam SMS Classifier API",
    description="API for detecting Spam and Ham SMS messages",
    version="1.0.0"
)


# -----------------------------
# Request Schema
# -----------------------------
class MessageRequest(BaseModel):
    message: str


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(request: MessageRequest):

    # Convert text to TF-IDF vector
    message_vector = vectorizer.transform([request.message])

    # Convert to PyTorch tensor
    message_tensor = torch.tensor(
        message_vector.toarray(),
        dtype=torch.float32
    )

    # Prediction
    with torch.no_grad():
        output = model(message_tensor)
        probability = torch.sigmoid(output).item()

    # Spam = 1, Ham = 0
    prediction = "Spam" if probability >= 0.5 else "Ham"

    return {
        "message": request.message,
        "prediction": prediction,
        "spam_probability": round(probability, 4)
    }


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Spam SMS Classifier API is running"
    }