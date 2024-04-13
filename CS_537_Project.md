---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region jupyter={"outputs_hidden": false} -->
Setup and Preprocessing
<!-- #endregion -->

```python id="29GPqi_NgsAf"
# Imports
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import math

# Tensorflow/keras imports
from keras import Sequential
from keras import layers
from keras import regularizers
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

# Data files
X_train = pd.read_csv("UNSW_NB15_training-set.csv")
X_test = pd.read_csv("UNSW_NB15_testing-set.csv")

# Create train and test dataframes, dropping the id and attack_cat columns from x
# and setting label as y
X_train = X_train.drop(columns=["id", "attack_cat"])
y_train = X_train.pop("label")
X_test = X_test.drop(columns=["id", "attack_cat"])
y_test = X_test.pop("label")
```

```python id="EdBJABlljAD1"
# Function for assigning int values to categorical variables
def handle_non_numerical_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df
```

```python jupyter={"outputs_hidden": false}
# Use the above function to handle categorical data in the training and testing df
X_train = handle_non_numerical_data(X_train)
X_test = handle_non_numerical_data(X_test)

# Scale the column values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Use class weights to balance the classes for the training set
class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))
```

<!-- #region jupyter={"outputs_hidden": false} -->
Model structure and training
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kvLqhGR3hvN5" outputId="c225c7ce-c353-4ba7-a0ac-de7785bd1d8a"
# Hyperparameters
epochs = 10
batch_size = 32
initial_lr = 0.01

# Function which will decrease the lr by 10% every n epochs
def lr_step_decay(epoch, lr):
    epochs_per_drop = 5
    return initial_lr * math.pow(0.9, math.floor(epoch/epochs_per_drop))

# Model
model = Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(42, activation='relu'),
    layers.Dense(42, activation='relu'),
    layers.Dense(12, activation='relu'),
    layers.Dense(12, activation='relu'),
    layers.Dense(1),
    layers.Activation("sigmoid")
])

model.summary()

# Compile model
model.compile(
    optimizer = SGD(learning_rate=initial_lr, momentum=0.9),
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

# Train model with proper callback
model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, class_weight=class_weights, verbose=1, callbacks=[LearningRateScheduler(lr_step_decay, verbose=1)])

# Evaluate on test data
print("\nTest data loss/accuracy:")
model.evaluate(X_test, y_test)
```
