#Using XGBoost to predict electrical consumption based on 2009 RECS Survey Data

import sys
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

invalidCol = ["METROMICRO", "UR", "NOCRCASH", "NKRGALNC", "IECC_Climate_Pub"]       # Data in these columns have non-numeric values; removed for prediction

def main():
    for arg in sys.argv[1:]:
        dataSetFileName = arg

    #trainPredictionModel(dataSetFileName)                                          # Method to train the XGBoost model, unnecessary to invoke for each start

    loaded_model = pickle.load(open("pima.pickle.dat", "rb"))                       # Loading trained prediction model
    testData = pd.read_csv (dataSetFileName, low_memory=False)                      # Tuning input data for prediction
    testData = testData.drop(columns=invalidCol)
    if "KWH" in testData.columns:
        testData = testData.drop(columns=["KWH"])

    preds = loaded_model.predict(testData)
    print("Your predicted electrical consumption: ",  preds)



def trainPredictionModel(dataSetFileName):
    dataSet = pd.read_csv (r'recs2009_public.csv', low_memory=False)                    # Reading the dataset          
    dataSet = dataSet.drop(columns=invalidCol)                                          # Remove columns that are not numbers

    parameters, consumtion = dataSet.iloc[:,:-1],dataSet.iloc[:,-1]                     # Separate the data into parameters and electrical consumption
    data_dmatrix = xgb.DMatrix(data=parameters,label=consumtion)                        # Store the data in DMatrix for XGBoost
    X_train, X_test, y_train, y_test = train_test_split(parameters, 
        consumtion, test_size=0.1, random_state=100)
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.6,     # Creating the prediction model
        learning_rate = 0.35, max_depth = 10, alpha = 10, n_estimators = 15)
    model.fit(X_train,y_train)

    pickle.dump(model, open("pima.pickle.dat", "wb"))                                   # Saving the trained model locally
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))                                   # Compute the mean sqaured error for model accuracy adjustment
    print("RMSE: %f" % (rmse))

if __name__ == "__main__":
    main()

