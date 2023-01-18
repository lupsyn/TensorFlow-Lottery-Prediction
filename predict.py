import numpy as np
import pandas as pd
import sys
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

filename = sys.argv[1]

model = load_model(filename)

df = pd.read_csv("ArchivioSuperAl1801_con7.csv")

df.tail(7)

scaler = StandardScaler().fit(df.values)

to_predict = df.tail(7)
print(to_predict)
# to_predict.drop([to_predict.index[-1]],axis=0, inplace=True)
to_predict = np.array(to_predict)
# scaling
scaled_to_predict = scaler.transform(to_predict)
y_pred = model.predict(np.array([scaled_to_predict]))


print("The predicted numbers in the last lottery game are:", scaler.inverse_transform(y_pred).astype(int)[0])
