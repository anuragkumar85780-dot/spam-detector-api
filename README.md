\# Spam Detector API



A machine learning based spam message detection API built with FastAPI and Docker.



\## Features



\- REST API with `/predict` endpoint

\- Spam/Ham text classification

\- Returns spam probability

\- FastAPI automatic API documentation

\- Docker container support

\- Unit tests using pytest



\## Project Structure



```text

spam-detector-api/

├── app.py

├── Dockerfile

├── requirements.txt

├── spam\_model.pt

├── spam\_model\_weights.pth

├── vectorizer.pkl

├── tests/

│   └── test\_api.py

├── .gitignore

└── README.md

