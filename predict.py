"""
predict.py

This script uses a pre-trained TensorFlow/Keras LSTM model to predict statistical patterns
in lottery numbers. Specifically, it predicts the counts of even, odd, low, and high numbers
for the next lottery draw.

The script performs the following steps:
1. Loads a pre-trained Keras model, an input data scaler, and a counts scaler.
2. Reads the latest historical lottery data from a CSV file to prepare input for prediction.
3. Scales the input data using the loaded input scaler.
4. Makes a prediction using the loaded model.
5. Inverse-transforms the scaled prediction using the counts scaler to get the actual counts.
6. Prints the raw and rounded predicted counts for even, odd, low, and high numbers.

Usage:
Run this script from the command line, providing paths to the trained model, scalers, and input CSV.
Example:
    python predict.py --model_path lottery_model_counts.h5 \
                      --input_scaler_path scaler_input.joblib \
                      --counts_scaler_path scaler_counts.joblib \
                      --csv_file ArchivioSuperAl1801_con7.csv
"""

import numpy as np
import pandas as pd
import argparse
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError # Explicitly import for custom_objects
from sklearn.preprocessing import StandardScaler

def predict_lottery(model_path, input_scaler_path, counts_scaler_path, csv_file):
    """
    Loads a trained model and scalers to predict lottery number counts.

    Args:
        model_path (str): Path to the trained Keras model file (.h5).
        input_scaler_path (str): Path to the fitted StandardScaler for input data (.joblib).
        counts_scaler_path (str): Path to the fitted StandardScaler for the predicted counts (.joblib).
        csv_file (str): Path to the input CSV file containing historical lottery data.
    """
    # --- 1. Load Model and Scalers ---
    try:
        # Load the Keras model, explicitly providing MeanSquaredError for custom_objects
        model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    except Exception as e:
        print(f"Error: Could not load model from '{model_path}'. {e}")
        return

    try:
        input_scaler = joblib.load(input_scaler_path)
    except Exception as e:
        print(f"Error: Could not load input scaler from '{input_scaler_path}'. {e}")
        return

    try:
        counts_scaler = joblib.load(counts_scaler_path)
    except Exception as e:
        print(f"Error: Could not load counts scaler from '{counts_scaler_path}'. {e}")
        return

    # --- 2. Load and Prepare Input Data for Prediction ---
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Input CSV file '{csv_file}' not found.")
        return

    # Take the last 'window_length' (7) rows from the CSV as input for prediction
    # This assumes the model was trained with a window_length of 7
    to_predict = df.tail(7)
    print("Input for prediction (last 7 draws):\n", to_predict)
    to_predict = np.array(to_predict)

    # Scale the input data using the loaded input scaler
    scaled_to_predict = input_scaler.transform(to_predict)

    # Reshape for model input: (1, window_length, number_of_features)
    # The model expects a batch dimension, a time step dimension, and a feature dimension
    scaled_to_predict = np.array([scaled_to_predict])

    # --- 3. Make Prediction ---
    y_pred_scaled = model.predict(scaled_to_predict)

    # Inverse transform the scaled prediction using the counts scaler
    # This converts the scaled output back to the original count ranges
    predicted_counts_raw = counts_scaler.inverse_transform(y_pred_scaled)[0]

    # --- 4. Display Predictions ---
    print("\n--- Predicted Counts (Raw and Rounded) ---")
    print(f"Raw predicted counts (float): {predicted_counts_raw}")
    print(f"Rounded to nearest integer: {np.round(predicted_counts_raw).astype(int)}")
    print(f"Rounded up (ceil): {np.ceil(predicted_counts_raw).astype(int)}")
    print(f"Rounded down (floor): {np.floor(predicted_counts_raw).astype(int)}")

    # Assuming the order of counts is [even, odd, low, high] as defined in createModel.py
    print("\n--- Interpreted Predictions (Rounded to Nearest Integer) ---")
    print(f"Predicted Even Count: {int(np.round(predicted_counts_raw[0]))}")
    print(f"Predicted Odd Count: {int(np.round(predicted_counts_raw[1]))}")
    print(f"Predicted Low Count: {int(np.round(predicted_counts_raw[2]))}")
    print(f"Predicted High Count: {int(np.round(predicted_counts_raw[3]))}")

if __name__ == "__main__":
    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Predict lottery numbers using a trained model.")
    parser.add_argument("--model_path", type=str, default="lottery_model_counts.h5",
                        help="Path to the trained Keras model file.")
    parser.add_argument("--input_scaler_path", type=str, default="scaler_input.joblib",
                        help="Path to the fitted StandardScaler object for input data.")
    parser.add_argument("--counts_scaler_path", type=str, default="scaler_counts.joblib",
                        help="Path to the fitted StandardScaler object for the counts.")
    parser.add_argument("--csv_file", type=str, default="ArchivioSuperAl1801_con7.csv",
                        help="Path to the input CSV file containing lottery data.")
    args = parser.parse_args()

    # Call the main prediction function
    predict_lottery(args.model_path, args.input_scaler_path, args.counts_scaler_path, args.csv_file)

