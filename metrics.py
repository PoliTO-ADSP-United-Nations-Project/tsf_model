import numpy as np
def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def mae(actual, pred):
    forecast_errors = [actual[i]-pred[i] for i in range(len(actual))]
    mean_absolute_error = np.mean( np.abs(forecast_errors) )
    return mean_absolute_error
