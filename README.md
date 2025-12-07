# Weather Prediction Model – Linear Regression

## Overview
This project is a **machine learning weather prediction system** built using **Python, Scikit-Learn, and Streamlit**.  
The model predicts the **temperature (°C)** using weather-related features such as humidity, wind speed, visibility, pressure, and precipitation type.

The pipeline includes:

- Data preprocessing  
- Feature engineering  
- Linear Regression model training  
- Temperature prediction script  
- Interactive Streamlit Web App  

---

## Key Features
- **Temperature Prediction** using Linear Regression  
- **Preprocessing Pipeline** (scaling, encoding, handling missing values)  
- **Time Feature Extraction** (year, month, day, hour, day of week)  
- **Model Evaluation** (MAE, RMSE, R²)  
- **Streamlit Web App** for real-time predictions  
- **Modular Codebase** (clean structure for training & inference)

---

## Tech Stack
- **Language**: Python  
- **ML Framework**: Scikit-Learn  
- **Web App**: Streamlit  
- **Data Handling**: pandas, NumPy  
- **Model Saving**: joblib  

---

## Project Structure

```
Weather_Prediction/
│
├── data/
│   └── weatherHistory.csv          
│
├── models/
│   └── weather_model.joblib        
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                
│   ├── train_model.py              
│   ├── predict_example.py          
│   └── app.py                      
│
├── requirements.txt                
└── README.md
```

---

## Model Performance

After training on the provided dataset:

- **MAE**: ~0.74°C  
- **RMSE**: ~0.95°C  
- **R² Score**: ~0.99  

---

## How the Model Works

### Input Features
- Apparent Temperature  
- Humidity  
- Wind Speed  
- Wind Bearing  
- Visibility  
- Pressure  
- Precipitation Type  
- Derived Date-Time Features (year, month, day, hour, dayofweek)

### Output
- **Predicted Temperature (°C)**

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DulinaSamarathunga/Weather-Prediction-Model.git
cd Weather-Prediction-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python src/train_model.py
```

This trains the Linear Regression model and saves it to:

```
models/weather_model.joblib
```

---

## Make a Prediction (Script)

```bash
python src/predict_example.py
```

Example output:

```
Predicted Temperature (C): 22.65
```

---

## Run the Streamlit Web App

```bash
streamlit run src/app.py
```

App will open at:

```
http://localhost:8501
```

---

## Requirements
- Python 3.12+
- pandas  
- numpy  
- scikit-learn  
- joblib  
- streamlit  

---

## Future Improvements
- Add Random Forest or XGBoost models  
- Deploy Streamlit app online  
- Improve UI with charts  
- Add a REST API endpoint  

---

## Author
**Dulina Samarathunga**  
GitHub: https://github.com/DulinaSamarathunga  
