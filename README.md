# Car Price Prediction

A used-car price prediction system built with **PyTorch**. Predicts the market value of a vehicle from features like year, odometer reading, condition, manufacturer, and more.

---

## Overview

| | |
|---|---|
| **Task** | Regression (predict listing price in USD) |
| **Framework** | PyTorch |
| **Model** | Feed-forward neural network |
| **Preprocessing** | Label encoding (categorical) + Standard scaling (numerical) |
| **Artifacts** | Trained model + encoder + scaler (X and y) |

---

## Project Structure

```
.
├── notebooks/                      # Exploration & experimentation
│   ├── eda.ipynb                   # Exploratory data analysis
│   ├── preprocessing.ipynb         # Data cleaning workflow
│   ├── test_algorithms.ipynb       # Algorithm comparison
│   └── pytorch_model.ipynb         # Final PyTorch model training
├── src/                            # Production pipeline
│   ├── app.py                      # Inference entry point
│   ├── config.py                   # Hyperparameters & paths
│   ├── data_handling.py            # Data loading
│   ├── data_preprocessing.py       # Encoding + scaling
│   ├── model.py                    # PyTorch model definition
│   ├── model_evaluation.py         # Metrics & evaluation
│   ├── prediction.py               # Single-sample inference
│   ├── logger.py                   # Centralized logging
│   └── test_app.py                 # Sanity tests
├── models/                         # Saved artifacts
│   ├── model.pth                   # Trained PyTorch weights
│   ├── encoder.pkl                 # Label encoder
│   ├── scaler_x.pkl                # Feature scaler
│   └── scaler_y.pkl                # Target scaler
├── outputs/graphs/                 # EDA visualizations
│   ├── kde.png
│   ├── priceVSyear.png
│   └── priceVSodometer.png
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Data:** pandas, numpy, scikit-learn (preprocessing)
- **Visualization:** matplotlib, seaborn
- **Notebooks:** Jupyter

---

## Pipeline

1. **EDA** — distribution analysis, missing-value patterns, outlier detection, price-vs-feature relationships (notebooks/eda.ipynb).
2. **Preprocessing** — handle missing values, encode categorical features (manufacturer, condition, etc.), scale numerical features, train/test split.
3. **Modeling** — feed-forward neural net trained with MSE loss; standard scaler applied to both inputs and targets for stable convergence.
4. **Evaluation** — track loss curves, compute regression metrics (MAE, RMSE, R²) on held-out test set.
5. **Inference** — load encoder + scalers + model, transform a single sample, predict price.

---

## How to Run

```bash
pip install -r requirements.txt
python src/app.py
```

For experimentation, open the notebooks in order: `eda.ipynb` → `preprocessing.ipynb` → `pytorch_model.ipynb`.

---

## Engineering Highlights

- **Modular architecture** — strict separation between data handling, preprocessing, model definition, and inference.
- **Reproducible artifacts** — encoder, scaler, and model are all persisted, so inference is decoupled from training.
- **Centralized logging** for run-by-run debug visibility.
- **Target scaling** — `scaler_y.pkl` is preserved so predictions can be inverse-transformed back to dollars.

---

## License

MIT
