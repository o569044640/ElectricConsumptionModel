#Using XGBoost to predict electrical consumption based on 2009 RECS Survey Data

import sys
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def main():
    for arg in sys.argv[1:]:
        print(arg)

    dataSet = pd.read_csv (r'recs2009_public.csv', low_memory=False)                    # Reading the dataset          
    nonIntFloatCol = []
    for col in dataSet:
        if dataSet[col].dtypes != "float64" and dataSet[col].dtypes != "int64":         # Remove columns that are not numbers
            nonIntFloatCol.append(col)
    dataSet = dataSet.drop(columns=nonIntFloatCol)

    parameters, consumtion = dataSet.iloc[:,:-1],dataSet.iloc[:,-1]                     # Separate the data into parameters and electrical consumption
    data_dmatrix = xgb.DMatrix(data=parameters,label=consumtion)                        # Store the data in DMatrix for XGBoost
    X_train, X_test, y_train, y_test = train_test_split(parameters, 
                consumtion, test_size=0.2, random_state=101)
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.2,
                max_depth = 10, alpha = 10, n_estimators = 10)
    xg_reg.fit(X_train,y_train)

    """
    testData = pd.read_csv (r'aaaa.csv', low_memory=False)          
    invalidCol = ["METROMICRO", "UR", "NOCRCASH", "NKRGALNC", "IECC_Climate_Pub"]
    testData = testData.drop(columns=invalidCol)
    """

    preds = xg_reg.predict(X_test)
    print(preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))


if __name__ == "__main__":
    main()

