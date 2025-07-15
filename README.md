# TensorFlow Lottery Prediction

This repository contains Python scripts for exploring lottery number prediction using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers, built with TensorFlow/Keras.

**Disclaimer:** Lottery outcomes are inherently random and statistically independent events. While these scripts demonstrate machine learning techniques for pattern recognition, they do not guarantee accurate predictions or winning lottery numbers. The purpose is educational and for exploring statistical properties of lottery data.

## Project Structure

```
TensorFlow-Lottery-Prediction/
├── ArchivioSuperAl1801_con7.csv  # Example historical lottery data (replace with your own)
├── createModel.py               # Script for training the LSTM model
├── predict.py                   # Script for making predictions using a trained model
└── README.md                    # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd TensorFlow-Lottery-Prediction
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    pip install numpy pandas scikit-learn tensorflow keras joblib
    ```

## Usage

### 1. `createModel.py` - Training the Prediction Model

This script trains an LSTM model to predict statistical patterns in lottery numbers. Currently, it's configured to predict the counts of even, odd, low, and high numbers in the next lottery draw.

**Input:** A CSV file containing historical lottery draw data.
**Output:** A trained Keras model (`.h5` file), an input data scaler (`.joblib` file), and a counts scaler (`.joblib` file).

**Example Command:**

```bash
python createModel.py --csv_file ArchivioSuperAl1801_con7.csv \
                      --model_output lottery_model_counts.h5 \
                      --scaler_output scaler_input.joblib \
                      --counts_scaler_output scaler_counts.joblib
```

**Arguments:**
*   `--csv_file`: Path to the input CSV file containing lottery data. (Default: `ArchivioSuperAl1801_con7.csv`)
*   `--model_output`: Path to save the trained Keras model. (Default: `lottery_model_counts.h5`)
*   `--scaler_output`: Path to save the fitted `StandardScaler` object for input data. (Default: `scaler_input.joblib`)
*   `--counts_scaler_output`: Path to save the fitted `StandardScaler` object for the predicted counts (even, odd, low, high). (Default: `scaler_counts.joblib`)

### 2. `predict.py` - Making Predictions

This script uses a pre-trained model and scalers to make predictions based on the latest historical data.

**Input:** A trained Keras model (`.h5` file), an input data scaler (`.joblib` file), a counts scaler (`.joblib` file), and the latest historical lottery data CSV.
**Output:** Predicted counts of even, odd, low, and high numbers for the next draw.

**Example Command:**

```bash
python predict.py --model_path lottery_model_counts.h5 \
                  --input_scaler_path scaler_input.joblib \
                  --counts_scaler_path scaler_counts.joblib \
                  --csv_file ArchivioSuperAl1801_con7.csv
```

**Arguments:**
*   `--model_path`: Path to the trained Keras model file. (Default: `lottery_model_counts.h5`)
*   `--input_scaler_path`: Path to the fitted `StandardScaler` object for input data. (Default: `scaler_input.joblib`)
*   `--counts_scaler_path`: Path to the fitted `StandardScaler` object for the counts. (Default: `scaler_counts.joblib`)
*   `--csv_file`: Path to the input CSV file containing lottery data. (Default: `ArchivioSuperAl1801_con7.csv`)

## Code Practices

*   **Docstrings:** Both Python files include comprehensive docstrings explaining their purpose, functions, arguments, and usage.
*   **Comments:** Inline comments are used to clarify complex logic and important sections.
*   **Argparse:** Command-line arguments are used for flexible execution and easy configuration.
*   **Error Handling:** Basic error handling for file operations is included.
*   **Modularity:** Functions are used to encapsulate core logic.

## Further Improvements (Potential)

*   **More Data Preprocessing:** Explore different ways to preprocess the lottery numbers (e.g., one-hot encoding, binning).
*   **Hyperparameter Tuning:** Experiment with different LSTM layers, units, dropout rates, learning rates, and batch sizes.
*   **Different Prediction Targets:** Predict other statistical properties (e.g., sum of first three numbers, range of numbers).
*   **Visualization:** Add scripts to visualize training progress and prediction results.
*   **Unit Testing:** Implement unit tests for data preparation and model components.


