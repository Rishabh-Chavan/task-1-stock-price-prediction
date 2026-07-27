# 📈 Task 1: Stock Price Prediction using LSTM

A deep learning project that predicts **Tesla (TSLA) stock closing prices** using a Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras. This was completed as **Task 1** of the Data Science Internship at **Bharat Intern**.

---

## 🧠 Overview

Stock prices are sequential, time-dependent data — making them a great fit for **LSTM (Long Short-Term Memory)** networks, which are designed to learn patterns from historical sequences. This project builds an end-to-end pipeline that:

1. Loads historical TSLA stock data
2. Preprocesses and scales the closing price
3. Builds sequences of the past 60 trading days
4. Trains a stacked LSTM model to predict the next day's closing price
5. Evaluates the model and visualizes predictions against actual prices

---

## 🗂️ Dataset

- **Stock:** Tesla, Inc. (TSLA)
- **Period:** 2018-05-15 to 2023-05-12 (1,258 daily records)
- **Columns used:** `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- **Feature used for prediction:** `Close` price only
- **Split:** 80% training (1,007 records) / 20% testing (251 records)

### Closing Price History
<p align="center">
  <img src="images/closing_price.png" alt="TSLA Closing Price History" width="700"/>
</p>

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Data Handling | `pandas`, `numpy` |
| Data Source | `yfinance`, `pandas_datareader` |
| Deep Learning | `TensorFlow`, `Keras` |
| Preprocessing | `scikit-learn` (`StandardScaler`) |
| Visualization | `matplotlib` |
| Environment | Jupyter Notebook |

---

## 🔄 Workflow / Methodology

1. **Import libraries** — TensorFlow, Keras, pandas, numpy, sklearn, matplotlib
2. **Load data** — read TSLA historical price data from CSV
3. **Visualize** — plot the closing price trend over time
4. **Feature selection** — isolate the `Close` column and convert to a NumPy array
5. **Train/test split** — 80% of the data used for training
6. **Scaling** — `StandardScaler` normalizes the closing prices
7. **Sequence generation** — 60-day rolling windows used as input features (`x_train`), next-day price as the label (`y_train`)
8. **Reshape** — data reshaped into `[samples, timesteps, features]` for LSTM input
9. **Model building, compiling, and training**
10. **Evaluation** — loss and RMSE computed on the test set
11. **Prediction & visualization** — predicted vs. actual closing prices plotted

---

## 🏗️ Model Architecture

A stacked LSTM network with two LSTM layers followed by dense layers:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 60, 100)           40,800
 lstm_1 (LSTM)               (None, 100)               80,400
 dense (Dense)               (None, 50)                5,050
 dense_1 (Dense)             (None, 25)                1,275
 dense_2 (Dense)             (None, 1)                 26
=================================================================
Total params: 127,551
Trainable params: 127,551
Non-trainable params: 0
_________________________________________________________________
```

- **Optimizer:** Adam
- **Loss function:** Mean Squared Error (MSE)
- **Epochs:** 10
- **Batch size:** 32
- **Input window:** 60 previous trading days

---

## 📉 Training Loss

The model's training loss dropped steadily across 10 epochs, from **0.1214** down to **0.0085**, indicating good convergence on the training data.

<p align="center">
  <img src="images/training_loss.png" alt="Training Loss Curve" width="600"/>
</p>

---

## ✅ Results

| Metric | Value |
|---|---|
| Test Loss (MSE, scaled) | 47550.60 |
| **RMSE (actual price scale, $)** | **12.02** |

An RMSE of ~$12 on a stock that traded roughly between $160–$260 during the test window reflects a reasonably close price-level fit, though the model — like most single-feature LSTM price predictors — tends to lag sharp reversals.

### Predicted vs. Actual Closing Price
<p align="center">
  <img src="images/prediction_vs_actual.png" alt="Predicted vs Actual Closing Price" width="700"/>
</p>

The chart above shows the training data, actual test-set prices, and the model's predicted prices, illustrating how closely the LSTM tracks real market movement.

---

## 📁 Project Structure

```
task-1-stock-price-prediction/
├── Task 1.ipynb          # Main Jupyter Notebook (data prep, model, training, evaluation)
├── images/                # Charts used in this README
│   ├── closing_price.png
│   ├── training_loss.png
│   └── prediction_vs_actual.png
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/Rishabh-Chavan/task-1-stock-price-prediction.git
   cd task-1-stock-price-prediction
   ```
2. Install dependencies
   ```bash
   pip install yfinance pandas numpy tensorflow pandas_datareader scikit-learn matplotlib jupyter
   ```
3. Get TSLA historical data (via `yfinance` or download a CSV from [Yahoo Finance](https://finance.yahoo.com/quote/TSLA/history)) and place it in your working directory as `TSLA.csv`.
4. Launch the notebook and run all cells
   ```bash
   jupyter notebook "Task 1.ipynb"
   ```

---

## 🚀 Future Improvements

- Add more features beyond `Close` (e.g., `Volume`, technical indicators like RSI/MACD)
- Use `MinMaxScaler` instead of `StandardScaler` for bounded LSTM inputs
- Experiment with Bidirectional LSTM / GRU / Attention-based architectures
- Hyperparameter tuning (sequence length, units, epochs, dropout)
- Walk-forward validation instead of a single train/test split
- Deploy as an interactive web app (e.g., Streamlit) for live ticker predictions

---

## 👤 Author

**Rishabh Chavan**
Data Science Intern @ Bharat Internship

---

> ⚠️ **Disclaimer:** This project is for educational purposes only and should not be used as financial advice. Stock market prediction is inherently uncertain, and past performance is not indicative of future results.
