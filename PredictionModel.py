# Needed imports for this class
import time
import pandas as pd
import numpy as np
from threading import Thread
from sqlalchemy import create_engine
from sqlalchemy import text
from IPython.display import clear_output
import pickle
from pycaret.regression import *
from sklearn.preprocessing import StandardScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from scipy.stats import zscore

"""
In this homework assignment, the goal is to use PyCaret to evaluate regression algorithms and make predictions on 
the next hour's returns using VOL (volatility) and FD (fractal dimension) as features and the return as target variable.

1. On day 1: download the data for the 8 currency pairs for 10 consecutive hours at the 6-minute level.
2. Classify the VOL and FD data into 3 classes (high, medium, and low) for each currency pair.
3. Choose the best sorting method for the dataset for the best results
4. Build a dictionary that maps the original values of VOL and FD to their new classes.
5. Use PyCaret to train a regression model on the data and compare the performance of different algorithms.
6. On day 2: Use the best-performing algorithm to make predictions on the next hour's returns and compare them with 
the actual returns during a second 10 hour interval.
7. Save the predictions and return a CSV file.

The goal is to eventually implement this in the next homework (HWK 5), by using the errors from the predictions to 
improve the results and try to develop a profitable trading strategy.
"""


class PredictionModel():

    def __init__(self):
        self.dict_thresholds = None
        self.best_model = None
        self.sort_option = None

    """
    data_preprocessing - This function reads in a table from a SQL database and preprocesses the data by replacing NaN values
    with zeros and sorting the values in three different ways.

    Assumption: Data is stored in a sql table and the features VOL, FD and target variable return are present.

    There are no guarantees over the sorting of FD or VOL when they come second

    Arguments:
        - curr: tuple of a currency pair

    Returns:
        In a tuple:
        - df_curr1 (pandas DataFrame): A dataframe independetly sorted by VOL and FD
        - df_curr3 (pandas DataFrame): A dataframe sorted first on VOL and after that on FD
        - df_curr4 (pandas DataFrame): A dataframe sorted first on FD and after that VOL
    """

    def data_preprocessing(self, mongo_client, curr, position_type):

        # Get the collection from the NoSQL database
        db = mongo_client["FOREX_traindata3"]
        collection = db[curr[0] + curr[1] + position_type + "_maxmin"]

        # Convert the collection to a DataFrame
        df_curr = pd.DataFrame.from_records(collection.find())

        # Only keep the VOL, FD and returns that are needed for the linear regression model
        df_curr = df_curr.loc[:, ['VOL', 'FD', 'return']]

        # NaN values are present for the first values of FD and return or can be cause when the sanity check discards a value,
        # all these NaN values get replaced with 0 in the dataframe

        # This also best practice 4,
        df_curr = df_curr.replace(np.nan, 0)

        # Sort the values in the three different ways, as proposed during the lecture:
        # Option 1: sort FD and VOL indepedently
        df_curr1 = df_curr.sort_values(by=['VOL'], ascending=True).reset_index()
        df_curr2 = df_curr.sort_values(by=['FD'], ascending=True).reset_index()

        # Option 2: sort first on VOL, after that on FD
        df_curr3 = df_curr.sort_values(by=['VOL', 'FD'], ascending=True).reset_index(drop=True)

        # Option 3: sort first on FD, after that on VOL
        df_curr4 = df_curr.sort_values(by=['FD', 'VOL'], ascending=True).reset_index(drop=True)

        # Divide VOL and FD 3 different classes and label these classes:
        df_curr1.iloc[0:33, 2] = df_curr3.iloc[0:33, 1] = df_curr4.iloc[0:33, 1] = 1  # VOL class 1 = 1
        df_curr1.iloc[33:67, 2] = df_curr3.iloc[33:67, 1] = df_curr4.iloc[33:67, 1] = 2  # VOL class 2 = 2
        df_curr1.iloc[67:, 2] = df_curr3.iloc[67:, 1] = df_curr4.iloc[67:, 1] = 3  # VOL class 2 = 3

        df_curr2.iloc[0:33, 3] = df_curr3.iloc[0:33, 2] = df_curr4.iloc[0:33, 2] = 4  # FD class 1 = 4
        df_curr2.iloc[33:67, 3] = df_curr3.iloc[33:67, 2] = df_curr4.iloc[33:67, 2] = 5  # FD class 2 = 5
        df_curr2.iloc[67:, 3] = df_curr3.iloc[67:, 2] = df_curr4.iloc[67:, 2] = 6  # FD class 2 = 6

        # Combine df_curr1 and df_curr2 for option 1 of sorting
        for i in range(0, df_curr.shape[0]): org_index = df_curr1.iloc[i, 0]; df_curr1.iloc[i, 3] = \
            df_curr2[df_curr2["index"] == org_index]['FD']

        df_curr1 = df_curr1.drop(labels='index', axis=1)

        # Multiply the return target variable with a big number for better results
        df_curr1["return"] = df_curr1["return"] * 100000
        df_curr3["return"] = df_curr1["return"] * 100000
        df_curr4["return"] = df_curr1["return"] * 100000

        return df_curr1, df_curr3, df_curr4

    """
            This function trains a machine learning model on a given dataset using PyCaret. It returns the name of the best performing model,
            chosen based on its R2 value. If there is a tie between models with the same R2 value, the model that appears first 
            in the list will be chosen.
        """

    def train_model(self, mongo_client, curr, position):
        # Do setup for all the 3 sorting options
        # Note that making multiple instances of setup requires the 3.0.0 version of Pycaret

        # Maybe still normalize here first?
        # Get the collection from the NoSQL database
        db = mongo_client["FOREX_traindata3"]
        collection = db[curr[0] + curr[1] + position + "_maxmin"]

        # Convert the collection to a DataFrame
        df_curr = pd.DataFrame.from_records(collection.find())

        # Only keep the VOL, FD and returns that are needed for the linear regression model
        df_curr = df_curr.loc[:, ['VOL', 'FD', 'return']]
        df_curr["return"] = df_curr["return"] * 100000

        scaler = StandardScaler()

        # fit the scaler to the data
        scaler.fit(df_curr[['VOL', 'FD']])

        # transform the data using the scaler
        df_normalized = scaler.transform(df_curr[['VOL', 'FD']])

        df_curr[['VOL', 'FD']] = df_normalized

        setup(df_curr, target='return', silent=True);
        best1 = compare_models(exclude=["lasso", "dummy", "llar"]);

        print(df_curr)

        return best1

    def predict_with_LSTM(self, mongo_client, curr, position):

        with open(curr[0] + curr[1] + position + 'LSMT.pkl', 'rb') as f:
            model = pickle.load(f)

            # Get the collection we want to convert to a DataFrame
            db = mongo_client["FOREX_currencypairs3"]

            collection = db[curr[0] + curr[1] + position + "_maxmin"]

            # Convert the collection to a DataFrame
            df_maxmin = pd.DataFrame.from_records(collection.find())

            # Pick the last 10 values
            df_testset = df_maxmin.tail(10)
            df_testset = df_testset.loc[:, ['VOL', 'FD']]

            scaler = StandardScaler()

            # fit the scaler to the data
            scaler.fit(df_testset[['VOL', 'FD']])

            # transform the data using the scaler
            df_features = scaler.transform(df_testset[['VOL', 'FD']])

            df_features[['VOL', 'FD']] = df_features

            df_features = df_features.tail(10)

            df_return = df_testset['return']
            df_return = df_return.tail(10)

            sequence1, sequence2 = self._build_sequences(self, df, df_features, df_return)
            test1, _ = self._split_sequence(sequence1, sequence2, 4)

            yhat = model.predict(test1, verbose=1)
            return sum(value[0] for value in yhat)

    # Private function
    def _split_sequence(X, y, n_steps):
        X_new, y_new = list(), list()
        for i in range(len(X)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(X) - 1:
                break

            # gather input and output parts of the pattern
            seq_x, seq_y = X[i:end_ix], [sum(val[0] for val in y[i:end_ix])]
            X_new.append(seq_x)
            y_new.append(seq_y)
        return np.array(X_new), np.array(y_new)

    # Private function
    def _build_sequences(self, df, df_features, df_return):
        # Split the data into training and testing sets
        train_size = len(df)
        train = df_features[:train_size]
        train2 = df_return[:train_size]

        # Convert the data to a numpy array
        X_train = train.values
        y_train = train2.values

        sequence1 = [list(row) for row in X_train]
        sequence2 = [list(row) for row in y_train]

        return sequence1, sequence2

    def train_LSMT_model(self, curr, df, df_features, df_return, position):

        sequence1, sequence2 = self._build_sequences(df, df_features, df_return)
        X1, y1 = self._split_sequence(sequence1, sequence2, 4)

        # Build the LSTM model (very basic model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=False, input_shape=(X1.shape[1], y1.shape[2])))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X1, y1, epochs=100, batch_size=32, verbose=1)

        # save the model to a file, best practice 16
        with open(curr[0] + curr[1] + position + 'LSMT.pkl', 'wb') as f:
            pickle.dump(model, f)


    def data_preprocessing2(self, mongo_client, curr, position):

        db = mongo_client["FOREX_traindata3"]
        collection = db[curr[0] + curr[1] + position + "_maxmin"]

        cursor = collection.find({})

        documents = []

        for document in cursor:
            documents.append(document)

        df = pd.DataFrame(documents)

        # Preprocess the data
        # Convert the date column to datetime
        df['inserttime'] = pd.to_datetime(df['inserttime'])

        # Set the date column as the index
        df.set_index('inserttime', inplace=True, drop=True)

        # Remove any missing values
        df.dropna(inplace=True)

        # Scale the data
        scaler = MinMaxScaler()
        df_return = df.loc[:, ['return']]
        df = df.loc[:, ['VOL', 'FD']]
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        df_features = df.loc[:, ['VOL', 'FD']]

        return df, df_features, df_return

    """
        This function trains a machine learning model on a given dataset using PyCaret. It returns the name of the best performing model,
        chosen based on its R2 value. If there is a tie between models with the same R2 value, the model that appears first 
        in the list will be chosen.
    """

    def train_model_with_sorting(self, sorted_datasets):
        # Do setup for all the 3 sorting options
        # Note that making multiple instances of setup requires the 3.0.0 version of Pycaret

        setup(sorted_datasets[0], target='return', categorical_features=["VOL", "FD"], silent=True);
        best1 = compare_models();
        metrics1 = pull()  # Explain pull here

        setup(sorted_datasets[1], target='return', categorical_features=["VOL", "FD"], silent=True);
        best2 = compare_models();
        metrics2 = pull()  # Explain pull here

        setup(sorted_datasets[2], target='return', categorical_features=["VOL", "FD"], silent=True);
        best3 = compare_models();
        metrics3 = pull()

        # I choose the best model based on the best R-squared value
        top_r2_model1 = metrics1['R2'][0]
        top_r2_model2 = metrics2['R2'][0]
        top_r2_model3 = metrics3['R2'][0]

        list_with_best_models = [(best1, top_r2_model1, 1), (best2, top_r2_model2, 2), (best3, top_r2_model3, 3)]
        sorted_list = sorted(list_with_best_models, key=lambda x: x[1], reverse=True)

        # What if we have tie break?
        # Explain that reverse still

        return sorted_list[0]

    def build_dict_thres(self, mongo_client, curr, curr_dict, sort_option):

        dict_thresholds = {}
        db = mongo_client["FOREX_traindata3"]
        collection = db[curr[0] + curr[1] + curr_dict["position"] + "_maxmin"]
        df_curr = pd.DataFrame.from_records(collection.find())

        if sort_option == 1:
            df_curr1 = df_curr.sort_values(by=['VOL'], ascending=True).reset_index()
            df_curr2 = df_curr.sort_values(by=['FD'], ascending=True).reset_index()

            dict_thresholds["VOL_class1-class2"] = df_curr1.loc[33, 'VOL']
            dict_thresholds["VOL_class2-class3"] = df_curr1.loc[67, 'VOL']
            dict_thresholds["FD_class1-class2"] = df_curr2.loc[33, 'FD']
            dict_thresholds["FD_class2-class3"] = df_curr2.loc[67, 'FD']

            return dict_thresholds

        elif sort_option == 2:
            df_curr3 = df_curr.sort_values(by=['VOL', 'FD'], ascending=True).reset_index(drop=True)
            dict_thresholds["VOL_class1-class2"] = dict_thresholds["FD_class1-class2"] = df_curr3.loc[33, 'VOL']
            dict_thresholds["VOL_class2-class3"] = dict_thresholds["FD_class2-class3"] = df_curr3.loc[67, 'VOL']

            return dict_thresholds

        elif sort_option == 3:
            df_curr3 = df_curr.sort_values(by=['FD', 'VOL'], ascending=True).reset_index(drop=True)
            dict_thresholds["VOL_class1-class2"] = dict_thresholds["FD_class1-class2"] = df_curr3.loc[33, 'FD']
            dict_thresholds["VOL_class2-class3"] = dict_thresholds["FD_class2-class3"] = df_curr3.loc[67, 'FD']

        else:
            raise ValueError("Invalid sort options")

        return dict_thresholds

    def predict_return_next_hour(self, mongo_client, curr, position, exact_return_old, exact_return_new):

        # Get the collection we want to convert to a DataFrame
        db = mongo_client["FOREX_currencypairs3"]

        collection = db[curr[0] + curr[1] + position + "_maxmin"]

        # Convert the collection to a DataFrame
        df_maxmin = pd.DataFrame.from_records(collection.find())

        df_testset = df_maxmin.tail(10)
        df_testset = df_testset.loc[:, ['VOL', 'FD']]

        scaler = StandardScaler()

        # fit the scaler to the data
        scaler.fit(df_testset[['VOL', 'FD']])

        # transform the data using the scaler
        df_normalized = scaler.transform(df_testset[['VOL', 'FD']])

        df_testset[['VOL', 'FD']] = df_normalized

        df_testset = df_testset.tail(10)

        result = predict_model(self.best_model, df_testset)

        return result

        # Use dictionary with threshold

        # sums = result.sum(axis=0)
        # hourly_prediction = sums[3] / 100000
        #
        # estimate_new_return = exact_return_old + hourly_prediction
        #
        # # I think the sign is important now, not 100% sure yet how I'm going to do this
        # error = exact_return_new - estimate_new_return

        # return estimate_new_return, error
