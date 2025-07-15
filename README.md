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

This script trains an LSTM model to predict various statistical patterns in lottery numbers. It supports different prediction types:

*   `raw_numbers`: Predicts the raw scaled lottery numbers.
*   `sum`: Predicts the sum of the lottery numbers.
*   `counts`: Predicts the counts of even, odd, low, and high numbers.

**Input:** A CSV file containing historical lottery draw data.
**Output:** A trained Keras model (`.h5` file), an input data scaler (`.joblib` file), and a target-specific scaler (`.joblib` file).

**Example Commands:**

**Train for `counts` prediction:**
```bash
python createModel.py --csv_file ArchivioSuperAl1801_con7.csv \
                      --prediction_type counts \
                      --model_output lottery_model_counts.h5 \
                      --input_scaler_output scaler_input.joblib \
                      --target_scaler_output scaler_counts.joblib
```

**Train for `sum` prediction:**
```bash
python createModel.py --csv_file ArchivioSuperAl1801_con7.csv \
                      --prediction_type sum \
                      --model_output lottery_model_sum.h5 \
                      --input_scaler_output scaler_input.joblib \
                      --target_scaler_output scaler_sum.joblib
```

**Train for `raw_numbers` prediction:**
```bash
python createModel.py --csv_file ArchivioSuperAl1801_con7.csv \
                      --prediction_type raw_numbers \
                      --model_output lottery_model_raw.h5 \
                      --input_scaler_output scaler_input.joblib \
                      --target_scaler_output scaler_raw.joblib
```

**Arguments:**
*   `--csv_file` (required): Path to the input CSV file containing lottery data. (Default: `ArchivioSuperAl1801_con7.csv`)
*   `--prediction_type` (required): Type of prediction to train for: `raw_numbers`, `sum`, or `counts`.
*   `--model_output`: Path to save the trained Keras model. (Default: `lottery_model_<prediction_type>.h5`)
*   `--input_scaler_output`: Path to save the fitted `StandardScaler` object for input data. (Default: `scaler_input.joblib`)
*   `--target_scaler_output`: Path to save the fitted `StandardScaler` object for the target data. (Default: `scaler_<prediction_type>.joblib`)

### 2. `predict.py` - Making Predictions

This script uses a pre-trained model and scalers to make predictions based on the latest historical data. It also supports different prediction types, which must match the type the model was trained on.

**Input:** A trained Keras model (`.h5` file), an input data scaler (`.joblib` file), a target-specific scaler (`.joblib` file), and the latest historical lottery data CSV.
**Output:** Predicted values based on the chosen prediction type.

**Example Commands:**

**Predict `counts`:**
```bash
python predict.py --csv_file ArchivioSuperAl1801_con7.csv \
                  --prediction_type counts \
                  --model_path lottery_model_counts.h5 \
                  --input_scaler_path scaler_input.joblib \
                  --target_scaler_path scaler_counts.joblib
```

**Predict `sum`:**
```bash
python predict.py --csv_file ArchivioSuperAl1801_con7.csv \
                  --prediction_type sum \
                  --model_path lottery_model_sum.h5 \
                  --input_scaler_path scaler_input.joblib \
                  --target_scaler_path scaler_sum.joblib
```

**Predict `raw_numbers`:**
```bash
python predict.py --csv_file ArchivioSuperAl1801_con7.csv \
                  --prediction_type raw_numbers \
                  --model_path lottery_model_raw.h5 \
                  --input_scaler_path scaler_input.joblib \
                  --target_scaler_path scaler_raw.joblib
```

**Arguments:**
*   `--csv_file` (required): Path to the input CSV file containing lottery data. (Default: `ArchivioSuperAl1801_con7.csv`)
*   `--prediction_type` (required): Type of prediction to perform: `raw_numbers`, `sum`, or `counts`.
*   `--model_path`: Path to the trained Keras model file. (Default: `lottery_model_<prediction_type>.h5`)
*   `--input_scaler_path`: Path to the fitted `StandardScaler` object for input data. (Default: `scaler_input.joblib`)
*   `--target_scaler_path`: Path to the fitted `StandardScaler` object for the target data. (Default: `scaler_<prediction_type>.joblib`)

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