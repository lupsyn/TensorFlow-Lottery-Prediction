import numpy as np
import pandas as pd
import argparse
import joblib
from sklearn.model_selection import train_test_split
# Recurrent Neural Netowrk (RNN) with Long Short Term Memory (LSTM)
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_model(csv_file, model_output_path, scaler_output_path, sum_scaler_output_path):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Input CSV file '{csv_file}' not found.")
        return

    print(df.describe())
    # df.drop(['Data', 'Conc'], axis=1, inplace=True)
    scaler = StandardScaler().fit(df.values)
    transformed_dataset = scaler.transform(df.values)
    transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)

    # Scaled data set
    print(transformed_df.head())

    number_of_rows = df.values.shape[0]
    window_length = 7
    number_of_features = df.values.shape[1]
    print(number_of_features)

    # Define ranges for low/high numbers (adjust as per your lottery rules)
    low_threshold = 29 # Numbers 1-29 are low

    def calculate_counts(numbers):
        even_count = sum(1 for num in numbers if num % 2 == 0)
        odd_count = sum(1 for num in numbers if num % 2 != 0)
        low_count = sum(1 for num in numbers if num <= low_threshold)
        high_count = sum(1 for num in numbers if num > low_threshold)
        return [even_count, odd_count, low_count, high_count]

    # Prepare target data (counts of even, odd, low, high numbers)
    original_counts = np.array([calculate_counts(df.iloc[i+window_length : i+window_length+1, 0 : number_of_features].values[0]) for i in range(0, number_of_rows-window_length)])

    # Scale the counts
    counts_scaler = StandardScaler().fit(original_counts)
    scaled_counts = counts_scaler.transform(original_counts)

    X = np.empty([ number_of_rows - window_length, window_length, number_of_features], dtype=float)
    y = np.empty([ number_of_rows - window_length, 4], dtype=float) # y is now 4-dimensional


    for i in range(0, number_of_rows-window_length):
        X[i] = transformed_df.iloc[i : i+window_length, 0 : number_of_features]
        y[i] = scaled_counts[i] # Assign the scaled counts

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # 80% train, 20% validation

    batch_size = 100


    # Initialising the RNN
    model = Sequential()
    # Adding the input layer and the LSTM layer
    model.add(Bidirectional(LSTM(240,
                            input_shape = (window_length, number_of_features),
                            return_sequences = True)))
    # Adding a first Dropout layer
    model.add(Dropout(0.2))
    # Adding a second LSTM layer
    model.add(Bidirectional(LSTM(240,
                            input_shape = (window_length, number_of_features),
                            return_sequences = True)))
    # Adding a second Dropout layer
    model.add(Dropout(0.2))
    # Adding a third LSTM layer
    model.add(Bidirectional(LSTM(240,
                            input_shape = (window_length, number_of_features),
                            return_sequences = True)))
    # Adding a fourth LSTM layer
    model.add(Bidirectional(LSTM(240,
                            input_shape = (window_length, number_of_features),
                            return_sequences = False)))
    # Adding a fourth Dropout layer
    model.add(Dropout(0.2))
    # Adding the first output layer
    model.add(Dense(59))
    # Adding the last output layer (predicting 4 counts)
    model.add(Dense(4))


    model.compile(optimizer=Adam(learning_rate=0.0001), loss ='mse', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_output_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=300, verbose=2,
              validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

    # The best model is already saved by ModelCheckpoint, so we don't need model.save() here
    joblib.dump(scaler, scaler_output_path)
    joblib.dump(counts_scaler, counts_scaler_output_path)
    print(f"Model (best version) saved to {model_output_path}")
    print(f"Input Scaler saved to {scaler_output_path}")
    print(f"Counts Scaler saved to {counts_scaler_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM model for lottery prediction.")
    parser.add_argument("--csv_file", type=str, default="ArchivioSuperAl1801_con7.csv",
                        help="Path to the input CSV file containing lottery data.")
    parser.add_argument("--model_output", type=str, default="lottery_model_counts.h5",
                        help="Path to save the trained Keras model.")
    parser.add_argument("--scaler_output", type=str, default="scaler_input.joblib",
                        help="Path to save the fitted StandardScaler object for input data.")
    parser.add_argument("--counts_scaler_output", type=str, default="scaler_counts.joblib",
                        help="Path to save the fitted StandardScaler object for the counts.")
    args = parser.parse_args()

    create_model(args.csv_file, args.model_output, args.scaler_output, args.counts_scaler_output)
