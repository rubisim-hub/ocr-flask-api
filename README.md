# OCR Flask API

Minimal Flask API that serves a Keras OCR model as a usable REST endpoint.

## Features
- Loads a trained OCR model
- Accepts an image via `POST /predict`
- Returns OCR prediction as JSON
- Includes `/health` endpoint

## Project structure
```text
ocr-flask-api/
├── app.py
├── predictor.py
├── requirements.txt
├── .gitignore
├── README.md
└── models/
    └── final_finetuned_model.keras
