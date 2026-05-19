# OCR Flask API

A lightweight Flask-based inference service that exposes a trained Keras OCR model through a simple REST API. The project is designed to turn a computer vision / OCR model into a usable backend service that accepts an image and returns the predicted reading.

This repository is intentionally minimal and focused on deployment fundamentals:
- loading a trained model
- preprocessing uploaded images
- running inference
- decoding OCR output
- returning predictions through an API endpoint

It is a strong portfolio project for demonstrating practical ML deployment skills beyond model training.

---

## GitHub Project Description

**Short description for GitHub:**

> Flask REST API for serving a custom Keras OCR model with image preprocessing, sequence decoding, and real-time prediction.

Alternative shorter version:

> Lightweight OCR inference API built with Flask and Keras.

---

## Features

- Serves a trained `.keras` OCR model through Flask
- Accepts image uploads via `POST /predict`
- Returns predictions in JSON format
- Includes a health check endpoint
- Supports configurable character mapping through an environment variable
- Simple structure suitable for local development, demos, and portfolio use

---

## Repository Structure

```text
ocr-flask-api/
├── app.py
├── predictor.py
├── requirements.txt
├── .gitignore
├── README.md
└── models/
    └── final_finetuned_model.keras
