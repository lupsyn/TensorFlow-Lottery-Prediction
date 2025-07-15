
predict.py

This script uses a pre-trained TensorFlow/Keras LSTM model to make predictions based on
different statistical patterns in lottery numbers.

The script supports various prediction types:
- 'raw_numbers': Predicts the raw scaled lottery numbers.
- 'sum': Predicts the sum of the lottery numbers.
- 'counts': Predicts the counts of even, odd, low, and high numbers.

It performs the following steps:
1. Loads a pre-trained Keras model, an input data scaler, and a target-specific scaler.
2. Reads the latest historical lottery data from a CSV file to prepare input for prediction.
3. Scales the input data using the loaded input scaler.
4. Makes a prediction using the loaded model.
5. Inverse-transforms the scaled prediction using the appropriate target scaler.
6. Prints the raw and rounded predicted values based on the chosen prediction type.

Usage:
Run this script from the command line, specifying the prediction type and file paths.
Example (for counts prediction):
    python predict.py --prediction_type counts \
                      --model_path lottery_model_counts.h5 \
                      --input_scaler_path scaler_input.joblib \
                      --target_scaler_path scaler_counts.joblib \
                      --csv_file ArchivioSuperAl1801_con7.csv

Example (for sum prediction):
    python predict.py --prediction_type sum \
                      --model_path lottery_model_sum.h5 \
                      --input_scaler_path scaler_input.joblib \
                      --target_scaler_path scaler_sum.joblib \
                      --csv_file ArchivioSuperAl1801_con7.csv

Example (for raw_numbers prediction):
    python predict.py --prediction_type raw_numbers \
                      --model_path lottery_model_raw.h5 \
                      --input_scaler_path scaler_input.joblib \
                      --target_scaler_path scaler_raw.joblib \
                      --csv_file ArchivioSuperAl1801_con7.csv
"""

import numpy as np
import pandas as pd
import argparse
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError # Explicitly import for custom_objects
from sklearn.preprocessing import StandardScaler

def predict_lottery(prediction_type, model_path, input_scaler_path, target_scaler_path, csv_file):
    """
    Loads a trained model and scalers to predict lottery number patterns.

    Args:
        prediction_type (str): Type of prediction to perform ('raw_numbers', 'sum', 'counts').
        model_path (str): Path to the trained Keras model file (.h5).
        input_scaler_path (str): Path to the fitted StandardScaler for input data (.joblib).
        target_scaler_path (str): Path to the fitted StandardScaler for the target data (.joblib).
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
        target_scaler = joblib.load(target_scaler_path)
    except Exception as e:
        print(f"Error: Could not load target scaler from '{target_scaler_path}'. {e}")
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

    # Inverse transform the scaled prediction using the appropriate target scaler
    predicted_raw = target_scaler.inverse_transform(y_pred_scaled)[0]

    # --- 4. Display Predictions based on prediction_type ---
    print(f"\n--- Predicted {prediction_type.replace('_', ' ').title()} ---")

    if prediction_type == 'raw_numbers':
        print(f"Raw predicted numbers (float): {predicted_raw}")
        print(f"Rounded to nearest integer: {np.round(predicted_raw).astype(int)}")
        print(f"Rounded up (ceil): {np.ceil(predicted_raw).astype(int)}")
        print(f"Rounded down (floor): {np.floor(predicted_raw).astype(int)}")

    elif prediction_type == 'sum':
        predicted_sum_raw = predicted_raw[0] # Sum is a single value
        print(f"Raw predicted sum (float): {predicted_sum_raw:.2f}")
        print(f"Rounded to nearest integer: {int(np.round(predicted_sum_raw))}")
        print(f"Rounded up (ceil): {int(np.ceil(predicted_sum_raw))}")
        print(f"Rounded down (floor): {int(np.floor(predicted_sum_raw))}")

    elif prediction_type == 'counts':
        print(f"Raw predicted counts (float): {predicted_raw}")
        print(f"Rounded to nearest integer: {np.round(predicted_raw).astype(int)}")
        print(f"Rounded up (ceil): {np.ceil(predicted_raw).astype(int)}")
        print(f"Rounded down (floor): {np.floor(predicted_raw).astype(int)}")

        # Assuming the order of counts is [even, odd, low, high] as defined in createModel.py
        print("\n--- Interpreted Counts (Rounded to Nearest Integer) ---")
        print(f"Predicted Even Count: {int(np.round(predicted_raw[0]))}")
        print(f"Predicted Odd Count: {int(np.round(predicted_raw[1]))}")
        print(f"Predicted Low Count: {int(np.round(predicted_raw[2]))}")
        print(f"Predicted High Count: {int(np.round(predicted_raw[3]))}")

    else:
        print(f"Error: Unknown prediction type '{prediction_type}'.")

if __name__ == "__main__":
    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Predict lottery numbers using a trained model.")
    parser.add_argument("--prediction_type", type=str, required=True, choices=['raw_numbers', 'sum', 'counts'],
                        help="Type of prediction to perform: 'raw_numbers', 'sum', or 'counts'.")
    parser.add_argument("--model_path", type=str,
                        help="Path to the trained Keras model file. Default: lottery_model_<type>.h5")
    parser.add_argument("--input_scaler_path", type=str,
                        help="Path to the fitted StandardScaler object for input data. Default: scaler_input.joblib")
    parser.add_argument("--target_scaler_path", type=str,
                        help="Path to the fitted StandardScaler object for the target data. Default: scaler_<type>.joblib")
    parser.add_argument("--csv_file", type=str, default="ArchivioSuperAl1801_con7.csv",
                        help="Path to the input CSV file containing lottery data.")
    args = parser.parse_args()

    # Set default output paths based on prediction_type if not provided
    if not args.model_path:
        args.model_path = f"lottery_model_{args.prediction_type}.h5"
    if not args.input_scaler_path:
        args.input_scaler_path = "scaler_input.joblib"
    if not args.target_scaler_path:
        args.target_scaler_path = f"scaler_{args.prediction_type}.joblib"

    # Call the main prediction function
    predict_lottery(args.prediction_type, args.model_path, args.input_scaler_path, args.target_scaler_path, args.csv_file)


