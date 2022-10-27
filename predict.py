import pandas as pd

weather = pd.read_csv("weather.csv", index_col="DATE")

#ML models don't like NANs
#Calculate null percentage
#find total number of null values of each column then divide by the occurences of them
null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]

#clean columns with too high null values
#these columns have less than 5% missing values
#having more than 25% nan values under a column is a bad idea
valid_columns = weather.columns[null_pct < .05]

#create a new df to contain only valid columns
weather = weather[valid_columns].copy()

#lowercase column names
weather.columns = weather.columns.str.lower()

#using last non missing value of snow depth to fill in the next missing value
weather = weather.ffill()

#none of the columns have missing values now
weather.apply(pd.isnull).sum()

#object dtype indicates that its str e.g. station & name
weather.dtypes

#need to convert index from obj to datetime
weather.index = pd.to_datetime(weather.index)
weather.index


#this now easily done compared to str
weather.index.year

#ensuring no gaps present in df
#counting unique years w/ record counts in order
weather.index.year.value_counts().sort_index


#snow accumulation overtime
weather["snwd"].plot()


#tmax & tmin temps in F
#goal is to predict tomorrow's tmax & tmin
#Oct 22 is missing so last value of 21st is NaN
weather["target"] = weather.shift(-1)["tmax"]


#replacing the last NaN value
#not technically correct but won't have much impact on the ML
weather = weather.ffill()



from sklearn.linear_model import Ridge

weather.corr()

#ridge regression model penalizes coefficients 
#helps adjust colinearety
#how much coefficients are shrunk is alpha value

rr = Ridge(alpha=.1)

#columns to host predictor values
predictors = weather.columns[~weather.columns.isin(["target","name","station"])]

#estimate error of ML model 
#with time series need to be careful (use backtesting or timeseries cross validation)
#for 10 (365x10) years get predictions for every 90 days
def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []
    
    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]
        
        model.fit(train[predictors],train["target"])
        
        preds = model.predict(test[predictors])
        
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"],preds], axis=1)
        
        combined.columns = ["actual","prediction"]
        
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)
    
    
 predictions = backtest(weather, rr, predictors)
 
#efectiveness of the algorithm: half off, half close - could be better
from sklearn.metrics import mean_absolute_error

mean_absolute_error(predictions["actual"], predictions["prediction"])

#efectiveness of the algorithm 
predictions["diff"].mean()


#improve accuracy by looking into avg temp perticipation and 
#compare how they relate to previous other days
#horizon number of days, col column names

def pct_diff(old, new):
    return (new-old) / old

def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label],weather[col])
    return weather

rolling_horizons = [3, 14]

for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)
        
#iloc is by number, loc is by index (ie in this case by date)
weather = weather.iloc[14:,:]

#missing values means either dividing 0 or dividing by 0
#to fix:
weather = weather.fillna(0)

(predictions["diff"].round().value_counts().sort_index() / predictions.shape[0]).plot()

#goal is to have error < std

weather.describe()
