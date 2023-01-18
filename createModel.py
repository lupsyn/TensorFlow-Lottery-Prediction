import numpy as np
import pandas as pd
# Recurrent Neural Netowrk (RNN) with Long Short Term Memory (LSTM)
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

df = pd.read_csv("ArchivioSuperAl1801_con7.csv")

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

X = np.empty([ number_of_rows - window_length, window_length, number_of_features], dtype=float)
y = np.empty([ number_of_rows - window_length, number_of_features], dtype=float)


for i in range(0, number_of_rows-window_length):
    X[i] = transformed_df.iloc[i : i+window_length, 0 : number_of_features]
    y[i] = transformed_df.iloc[i+window_length : i+window_length+1, 0 : number_of_features]



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
# Adding the last output layer
model.add(Dense(number_of_features))


model.compile(optimizer=Adam(learning_rate=0.0001), loss ='mse', metrics=['accuracy'])
model.fit(x=X, y=y, batch_size=100, epochs=300, verbose=2)

model.save('180123.h5')