# Human Activity Recognition using Machine Learning

## Overview

This project predicts human activities using accelerometer sensor data from the HHAR (Heterogeneity Human Activity Recognition) dataset.

## Features

- Human activity prediction
- Random Forest classifier
- FastAPI backend
- Streamlit frontend
- End-to-end deployment

## Tech Stack

- Python
- Pandas
- Scikit-learn
- FastAPI
- Streamlit
- Joblib

## Dataset

HHAR Dataset (UCI Machine Learning Repository)

## How to Run

1. Train the model:

```bash
python train.py
```

2. Start the backend:

```bash
uvicorn backend.app:app --reload
```

3. Start the frontend:

```bash
streamlit run frontend/app.py
```
##Deploy
Backend:https://mldl-project-1.onrender.com/docs
Frontend:https://mldl-project-sqdmqugfvwzdmjw4cfytvk.streamlit.app/
