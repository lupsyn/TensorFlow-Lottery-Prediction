import numpy as np
import pandas as pd
import argparse
import joblib
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_model(csv_file, prediction_type, model_output_path, input_scaler_output_path, target_scaler_output_path):
    """
    Trains an LSTM model to predict various statistical patterns in lottery numbers.

    Args:
        csv_file (str): Path to the input CSV file containing historical lottery data.
        prediction_type (str): Type of prediction to train for ('raw_numbers', 'sum', 'counts').
        model_output_path (str): Path to save the trained Keras model (.h5 file).
        input_scaler_output_path (str): Path to save the fitted StandardScaler for input data (.joblib file).
        target_scaler_output_path (str): Path to save the fitted StandardScaler for the target data (.joblib file).
    """
    # --- 1. Data Loading and Initial Preprocessing ---
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Input CSV file '{csv_file}' not found.")
        return

    print(df.describe()) # Display basic statistics of the loaded data

    # Initialize and fit StandardScaler on the entire dataset for input features
    # This scaler will be used to transform input data for the LSTM model
    input_scaler = StandardScaler().fit(df.values)
    transformed_dataset = input_scaler.transform(df.values)
    transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)

    print(transformed_df.head()) # Display head of the scaled input data

    number_of_rows = df.values.shape[0]
    window_length = 7  # Number of past draws to consider for prediction
    number_of_features = df.values.shape[1] # Number of balls in each draw
    print(f"Number of features (balls per draw): {number_of_features}")

    # --- 2. Target Data Preparation based on prediction_type ---
    y_output_dim = 0 # Initialize output dimension for the model
    target_scaler = None # Initialize target scaler

    if prediction_type == 'raw_numbers':
        y_output_dim = number_of_features
        # For raw numbers, the target is already scaled by input_scaler
        # We'll use a dummy scaler or the input_scaler itself for consistency in saving
        target_scaler = input_scaler # Re-using input_scaler as target_scaler for raw numbers
        print("Prediction Type: Raw Numbers")

    elif prediction_type == 'sum':
        y_output_dim = 1
        # Calculate sums of original numbers for target
        original_sums = np.array([df.iloc[i+window_length : i+window_length+1, 0 : number_of_features].sum(axis=1).values[0] for i in range(0, number_of_rows-window_length)])
        # Scale the sums
        sum_scaler = StandardScaler().fit(original_sums.reshape(-1, 1))
        scaled_sums = sum_scaler.transform(original_sums.reshape(-1, 1))
        target_scaler = sum_scaler
        print("Prediction Type: Sum of Numbers")

    elif prediction_type == 'counts':
        y_output_dim = 4
        # Define ranges for low/high numbers (adjust as per your lottery rules)
        low_threshold = 29 # Numbers 1-29 are considered 'low'

        def calculate_counts(numbers):
            """Calculates even, odd, low, and high counts for a given set of lottery numbers."""
            even_count = sum(1 for num in numbers if num % 2 == 0)
            odd_count = sum(1 for num in numbers if num % 2 != 0)
            low_count = sum(1 for num in numbers if num <= low_threshold)
            high_count = sum(1 for num in numbers if num > low_threshold)
            return [even_count, odd_count, low_count, high_count]

        # Generate the target counts for each prediction window
        original_counts = np.array([calculate_counts(df.iloc[i+window_length : i+window_length+1, 0 : number_of_features].values[0]) for i in range(0, number_of_rows-window_length)])
        # Scale the counts
        counts_scaler = StandardScaler().fit(original_counts)
        scaled_counts = counts_scaler.transform(original_counts)
        target_scaler = counts_scaler
        print("Prediction Type: Counts (Even, Odd, Low, High)")

    else:
        print(f"Error: Invalid prediction_type '{prediction_type}'. Supported types are 'raw_numbers', 'sum', 'counts'.")
        return

    # Prepare X (input sequences) and y (target data)
    X = np.empty([ number_of_rows - window_length, window_length, number_of_features], dtype=float)
    y = np.empty([ number_of_rows - window_length, y_output_dim], dtype=float)

    for i in range(0, number_of_rows-window_length):
        X[i] = transformed_df.iloc[i : i+window_length, 0 : number_of_features] # Scaled input features
        if prediction_type == 'raw_numbers':
            y[i] = transformed_df.iloc[i+window_length : i+window_length+1, 0 : number_of_features]
        elif prediction_type == 'sum':
            y[i] = scaled_sums[i]
        elif prediction_type == 'counts':
            y[i] = scaled_counts[i]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # 80% train, 20% validation

    batch_size = 100 # Number of samples per gradient update

    # --- 3. Model Definition ---
    model = Sequential()
    model.add(Bidirectional(LSTM(240,
                            input_shape = (window_length, number_of_features),
                            return_sequences = True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(240,
                            input_shape = (window_length, number_of_features),
                            return_sequences = True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(240,
                            input_shape = (window_length, number_of_features),
                            return_sequences = True)))

    model.add(Bidirectional(LSTM(240,
                            input_shape = (window_length, number_of_features),
                            return_sequences = False)))
    model.add(Dropout(0.2))

    model.add(Dense(59))
    model.add(Dense(y_output_dim)) # Output layer adjusted based on prediction_type

    # --- 4. Model Compilation ---
    model.compile(optimizer=Adam(learning_rate=0.0001), loss ='mse', metrics=['accuracy'])

    # --- 5. Model Training ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_output_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=300, verbose=2,
              validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

    # --- 6. Model and Scaler Saving ---
    joblib.dump(input_scaler, input_scaler_output_path) # Save the input data scaler
    joblib.dump(target_scaler, target_scaler_output_path) # Save the target-specific scaler

    print(f"Model (best version) saved to {model_output_path}")
    print(f"Input Scaler saved to {input_scaler_output_path}")
    print(f"Target Scaler saved to {target_scaler_output_path}")

if __name__ == "__main__":
    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train an LSTM model for lottery prediction.")
    parser.add_argument("--csv_file", type=str, default="ArchivioSuperAl1801_con7.csv",
                        help="Path to the input CSV file containing lottery data.")
    parser.add_argument("--prediction_type", type=str, required=True, choices=['raw_numbers', 'sum', 'counts'],
                        help="Type of prediction to train for: 'raw_numbers', 'sum', or 'counts'.")
    parser.add_argument("--model_output", type=str,
                        help="Path to save the trained Keras model. Default: lottery_model_<type>.h5")
    parser.add_argument("--input_scaler_output", type=str,
                        help="Path to save the fitted StandardScaler for input data. Default: scaler_input.joblib")
    parser.add_argument("--target_scaler_output", type=str,
                        help="Path to save the fitted StandardScaler for the target data. Default: scaler_<type>.joblib")
    args = parser.parse_args()

    # Set default output paths based on prediction_type if not provided
    if not args.model_output:
        args.model_output = f"lottery_model_{args.prediction_type}.h5"
    if not args.input_scaler_output:
        args.input_scaler_output = "scaler_input.joblib"
    if not args.target_scaler_output:
        args.target_scaler_output = f"scaler_{args.prediction_type}.joblib"

    # Call the main training function
    create_model(args.csv_file, args.prediction_type, args.model_output, args.input_scaler_output, args.target_scaler_output)
