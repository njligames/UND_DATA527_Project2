# project/test.py

import unittest

import csv
import numpy as np
import matplotlib.pyplot as plt

import os.path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def loadRValues(path):
    r_values = None
    if os.path.isfile(path):
        with open(path, 'r') as openfile:
            r_values = json.load(openfile)
    if None == r_values:
        r_values = []


    return r_values

def appendRValue(path, r_values, val):
    r_values.append(val)
    with open(path, "w") as outfile:
        json.dump(r_values, outfile)

def normalizeMinMaxScaling(ary):
    if len(ary) > 0:
        _max = max(ary)
        _min = min(ary)
        new_ary = []
        for item in ary:
            val = float(item)
            if _min == _max:
                new_ary.append(1.0)
            else:
                new_ary.append((val-_min)/(_max-_min))
        return new_ary
    return ary

def writeFile(filename, fields, mydict):
    with open(filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(mydict)

def shouldAdd(field, fields):
    for f in fields:
        if field == f:
            return True
    return False

def parseFields(lines, special_fields):
    fields = []
    mydict = []
    N = 0

    field_dict = {}
    for line in lines:
        row = line.split()
        if [] == fields:
            for field in row:
                fields.append(field)

            for field in fields:
                field_dict[field] = []
        else:
            ary_item = []
            item = {}
            for i in range(len(fields)):
                ary_item.append(row[i])
                item[fields[i]] = row[i]
                field_dict[fields[i]].append(float(row[i]))
            mydict.append(item)
            N = N + 1
    return field_dict, fields, mydict, N

def loadFile(fields_in, fields_out):

    filename = "input_fl_12477"
    lines = []
    with open(filename, 'r') as file:
        lines = file.readlines()

    field_dict, fields, mydict, N = parseFields(lines, fields_in)
    writeFile(filename + ".preProcess.csv", fields, mydict)

    field_in_dict = {}
    field_in = []
    field_out_dict = {}
    field_out = []
    for i in range(len(fields)):
        field_dict[fields[i]] = normalizeMinMaxScaling(field_dict[fields[i]])

        if shouldAdd(fields[i], fields_in):
            field_in_dict[fields[i]] = normalizeMinMaxScaling(field_dict[fields[i]])
            field_in.append(fields[i])
        if shouldAdd(fields[i], fields_out):
            field_out_dict[fields[i]] = normalizeMinMaxScaling(field_dict[fields[i]])
            field_out.append(fields[i])

    X = []
    Y = []
    i = 0
    while i < N:
        item = []
        for field in field_in:
            item.append(field_in_dict[field][i])
        if len(item) == 1:
            X.append(item[0])
        else:
            X.append(item)

        item = []
        for field in field_out:
            item.append(field_out_dict[field][i])
        if len(item) == 1:
            Y.append(item[0])
        else:
            Y.append(item)

        i = i + 1

    mydict = []
    for i in range(N):
        item = {}
        for j in range(len(fields)):
            item[fields[j]] = field_dict[fields[j]][i]
        mydict.append(item)
    writeFile(filename + ".postProcess.csv", fields, mydict)

    mydict = []
    for i in range(N):
        item = {}
        for j in range(len(field_in)):
            item[field_in[j]] = field_in_dict[field_in[j]][i]
        mydict.append(item)
    writeFile(filename + ".X.csv", field_in, mydict)

    mydict = []
    for i in range(N):
        item = {}
        for j in range(len(field_out)):
            item[field_out[j]] = field_out_dict[field_out[j]][i]
        mydict.append(item)
    writeFile(filename + ".Y.csv", field_out, mydict)

    ind = np.array(X)
    dep = np.array([Y]).T
    return ind, dep, X, Y

def calculate_r_squared(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Length of dependant values and predicted values must be the same.")

    # Calculate the mean of the true values
    mean_y_true = sum(y_true) / len(y_true)

    # Calculate the total sum of squares (TSS) without using sum
    tss = 0
    for y in y_true:
        tss += (y - mean_y_true) ** 2

    # Calculate the residual sum of squares (RSS) without using sum
    rss = 0
    for true_val, pred_val in zip(y_true, y_pred):
        rss += (true_val - pred_val) ** 2

    # Calculate R-squared
    r_squared = 1 - (rss / tss)

    return r_squared

def kerasModel(inX, inY):
    X = tf.constant(inX, dtype=tf.float32)
    Y = tf.constant(inY, dtype=tf.float32)

    # Create a new Sequential Model
    model = keras.Sequential()

    model.add(layers.Dense(
        40, # Amount of Neurons
        input_dim=3, # Define an input dimension because this is the first layer
        activation='relu' # Use relu activation function because all inputs are positive
    ))
    model.add(layers.Dense(
        40, # Amount of Neurons. We want one output
        activation='relu' # Use sigmoid because we want to output a binary classification
    ))
    model.add(layers.Dense(
        40, # Amount of Neurons. We want one output
        activation='relu' # Use sigmoid because we want to output a binary classification
    ))
    model.add(layers.Dense(
        40, # Amount of Neurons. We want one output
        activation='relu' # Use sigmoid because we want to output a binary classification
    ))
    model.add(layers.Dense(
        40, # Amount of Neurons. We want one output
        activation='relu' # Use sigmoid because we want to output a binary classification
    ))

    model.add(layers.Dense(
        1, # Amount of Neurons. We want one output
        activation='linear' # Use sigmoid because we want to output a binary classification
    ))
    model.compile(
        loss='mean_squared_error', # The loss function that is being minimized
        optimizer='adam', # Our optimization function
        metrics=['accuracy'] # Metrics are different values that you want the model to track while training
    )

    model.fit(
        X, # Input training data
        Y, # Output training data
        batch_size=3,
        epochs=500, # Amount of iterations we want to train for
        verbose=2 # Amount of detail you want shown in terminal while training
    )
    model.save("mymodel.keras")

class TestCalculations(unittest.TestCase):

    def test_drawloss(self:writeFile):
        mseArray = [0.0164, 0.0135, 0.0129, 0.0123, 0.0118, 0.0111, 0.0111, 0.0110, 0.0109, 0.0107, 0.0102, 0.0102, 0.0100, 0.0097, 0.0096, 0.0095, 0.0095, 0.0096, 0.0096, 0.0095, 0.0093, 0.0092, 0.0094, 0.0090, 0.0092, 0.0089, 0.0090, 0.0091, 0.0091, 0.0089, 0.0086, 0.0092, 0.0085, 0.0086, 0.0088, 0.0085, 0.0085, 0.0085, 0.0082, 0.0086, 0.0083, 0.0082, 0.0082, 0.0084, 0.0083, 0.0080, 0.0083, 0.0083, 0.0082, 0.0080, 0.0083, 0.0080, 0.0081, 0.0080, 0.0080, 0.0078, 0.0081, 0.0077, 0.0078, 0.0078, 0.0078, 0.0079,
    0.0077, 0.0079, 0.0078, 0.0077, 0.0076, 0.0078, 0.0077, 0.0076, 0.0078, 0.0078, 0.0076, 0.0077, 0.0077, 0.0075, 0.0077, 0.0075, 0.0077, 0.0072, 0.0074, 0.0076, 0.0074, 0.0075, 0.0076, 0.0073, 0.0074, 0.0073, 0.0074, 0.0072, 0.0072, 0.0072, 0.0075, 0.0075, 0.0073, 0.0068, 0.0073, 0.0072, 0.0072, 0.0071, 0.0072, 0.0072, 0.0071, 0.0073, 0.0070, 0.0072, 0.0070, 0.0069, 0.0071, 0.0068, 0.0069, 0.0073, 0.0072, 0.0072, 0.0071, 0.0070, 0.0069, 0.0069, 0.0070, 0.0066, 0.0067, 0.0068, 0.0073, 0.0071,
    0.0069, 0.0069, 0.0069, 0.0070, 0.0068, 0.0069, 0.0067, 0.0068, 0.0068, 0.0067, 0.0067, 0.0065, 0.0066, 0.0066, 0.0068, 0.0065, 0.0065, 0.0067, 0.0068, 0.0066, 0.0068, 0.0065, 0.0070, 0.0066, 0.0066, 0.0065, 0.0064, 0.0064, 0.0065, 0.0065, 0.0068, 0.0066, 0.0065, 0.0065, 0.0065, 0.0065, 0.0065, 0.0064, 0.0067, 0.0065, 0.0064, 0.0066, 0.0063, 0.0064, 0.0064, 0.0064, 0.0065, 0.0065, 0.0063, 0.0065, 0.0065, 0.0065, 0.0065, 0.0069, 0.0063, 0.0062, 0.0063, 0.0063, 0.0064, 0.0065, 0.0063, 0.0065,
    0.0063, 0.0067, 0.0063, 0.0063, 0.0063, 0.0064, 0.0065, 0.0064, 0.0066, 0.0063, 0.0064, 0.0066, 0.0065, 0.0062, 0.0063, 0.0062, 0.0061, 0.0061, 0.0062, 0.0066, 0.0061, 0.0062, 0.0063, 0.0062, 0.0061, 0.0062, 0.0064, 0.0061, 0.0064, 0.0062, 0.0061, 0.0064, 0.0060, 0.0061, 0.0062, 0.0061, 0.0062, 0.0061, 0.0060, 0.0062, 0.0060, 0.0061, 0.0060, 0.0063, 0.0061, 0.0062, 0.0060, 0.0060, 0.0061, 0.0062, 0.0062, 0.0063, 0.0060, 0.0063, 0.0063, 0.0059, 0.0060, 0.0063, 0.0059, 0.0060, 0.0060, 0.0061,
    0.0059, 0.0060, 0.0061, 0.0061, 0.0060, 0.0062, 0.0061, 0.0059, 0.0058, 0.0061, 0.0061, 0.0059, 0.0062, 0.0065, 0.0058, 0.0059, 0.0059, 0.0060, 0.0057, 0.0059, 0.0060, 0.0060, 0.0057, 0.0058, 0.0059, 0.0057, 0.0056, 0.0058, 0.0057, 0.0058, 0.0057, 0.0058, 0.0057, 0.0058, 0.0061, 0.0058, 0.0061, 0.0058, 0.0058, 0.0057, 0.0060, 0.0057, 0.0059, 0.0057, 0.0056, 0.0059, 0.0062, 0.0057, 0.0057, 0.0059, 0.0059, 0.0058, 0.0056, 0.0056, 0.0059, 0.0057, 0.0058, 0.0057, 0.0056, 0.0059, 0.0060, 0.0059,
    0.0065, 0.0060, 0.0057, 0.0060, 0.0058, 0.0056, 0.0061, 0.0055, 0.0062, 0.0060, 0.0057, 0.0057, 0.0057, 0.0057, 0.0057, 0.0058, 0.0057, 0.0055, 0.0058, 0.0057, 0.0056, 0.0059, 0.0058, 0.0057, 0.0057, 0.0061, 0.0054, 0.0056, 0.0057, 0.0056, 0.0055, 0.0058, 0.0054, 0.0059, 0.0059, 0.0054, 0.0058, 0.0057, 0.0057, 0.0057, 0.0056, 0.0062, 0.0056, 0.0059, 0.0058, 0.0057, 0.0057, 0.0058, 0.0057, 0.0058, 0.0056, 0.0055, 0.0057, 0.0055, 0.0059, 0.0058, 0.0056, 0.0058, 0.0057, 0.0056, 0.0056, 0.0057,
    0.0053, 0.0056, 0.0057, 0.0054, 0.0053, 0.0058, 0.0055, 0.0060, 0.0055, 0.0061, 0.0055, 0.0057, 0.0055, 0.0056, 0.0055, 0.0055, 0.0056, 0.0055, 0.0056, 0.0056, 0.0053, 0.0056, 0.0058, 0.0058, 0.0056, 0.0054, 0.0056, 0.0056, 0.0054, 0.0056, 0.0056, 0.0059, 0.0057, 0.0056, 0.0055, 0.0053, 0.0058, 0.0057, 0.0062, 0.0057, 0.0057, 0.0054, 0.0058, 0.0055, 0.0053, 0.0054, 0.0055, 0.0060, 0.0055, 0.0052, 0.0054, 0.0058, 0.0056, 0.0055, 0.0055, 0.0055, 0.0053, 0.0058, 0.0053, 0.0054, 0.0055, 0.0053,
    0.0057, 0.0053, 0.0052, 0.0060, 0.0053, 0.0052, 0.0054, 0.0055, 0.0054, 0.0055, 0.0052, 0.0057, 0.0053, 0.0054, 0.0054, 0.0053, 0.0054, 0.0052, 0.0057, 0.0054, 0.0053, 0.0054, 0.0052, 0.0053, 0.0054, 0.0051, 0.0054, 0.0054, 0.0053, 0.0056, 0.0054, 0.0056, 0.0053, 0.0052, 0.0053, 0.0052, 0.0056, 0.0053, 0.0057, 0.0052, 0.0053, 0.0052, 0.0053, 0.0055, 0.0051, 0.0055, 0.0054, 0.0050, 0.0058, 0.0055, 0.0052, 0.0053, 0.0052, 0.0053, 0.0053, 0.0053, 0.0052, 0.0057, 0.0056, 0.0055, 0.0056, 0.0053,
    0.0051, 0.0052, 0.0057, 0.0050]

        print(mseArray)
        plt.plot(mseArray)
        plt.title("Mean Square Errors")
        plt.xlabel("Error Iteration Number")
        plt.ylabel("Error Values")
        plt.savefig('MSE.pdf', dpi=150)
        plt.show()

    def test_load(self):
        fields_in = ["altitude", "indicated_airspeed", "roll"]
        fields_out = ["pitch"]
        ind, dep, X, Y = loadFile(fields_in, fields_out)

        kerasModel(X, Y)

        model = tf.keras.models.load_model("mymodel.keras")

        predictions = []
        for i in range(len(X)):
            inputTens = tf.constant([X[i]], dtype=tf.float32)
            prediction = model.predict(inputTens, verbose=2)[0][0]
            predictions.append(prediction)

        r_squared = calculate_r_squared(Y, predictions)
        print(f"R Squared: {r_squared}")


if __name__ == '__main__':
    unittest.main()

