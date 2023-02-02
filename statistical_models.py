import argparse
import pandas as pd
import darts
import numpy as np
from darts.metrics import mape
from darts.metrics import mase
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from darts.models import ARIMA,KalmanForecaster
import matplotlib.pyplot as plt
from darts.utils.likelihood_models import QuantileRegression
from metrics import mape
from preprocess import preprocess

def statistical_historical(model_name,load,date_test,Dataset,model_path="Content/model",plot=True,destination_country):


    if model_name=="ARIMA":
        model=ARIMA(p=5,q=7,)
    elif model_name=="KalmanForecaster":
        if load==True:
           model= load(model_path) 
        else:
            model=KalmanForecaster(dim_x=18)
    else:
        raise ValueError("Model not supported.Statistical supported models are ARIMA and KalmanForecaster")
    scaler_3,scaler_4,target_TimeSeries,covariates_TimeSeries,tot_cov,scaled_full,Target,filtered_Target=preprocess(model_name,Dataset,destination_country)
    if model_name in ["ARIMA","KalmanForecaster"]:
        historical_forecast = model.historical_forecasts(scaled_full,past_covariates=tot_cov, num_samples=200 , start=pd.Timestamp(date_test), forecast_horizon=1,stride=1 ,verbose=True)
        ts_pred = scaler_4.inverse_transform(historical_forecast)
        DF_predicted = historical_forecast.quantile_timeseries()
        ts_pred_df = scaler_4.inverse_transform(DF_predicted)
        DF_prova_predicted =  ts_pred_df.pd_dataframe()
        score=mape(filtered_Target["Sum_Inflow"],DF_prova_predicted["Sum_Inflow_0.5"])
    #The if-else part is implemented in case we would like to insert models not supporting quantile_timeseries function
    else:
        historical_forecast = model.historical_forecasts(scaled_full,past_covariates=tot_cov, num_samples=1 , start=pd.Timestamp(date_test), forecast_horizon=1,stride=1 ,verbose=True)
        ts_pred = scaler_4.inverse_transform(historical_forecast)
        model.fit(scaled_full,past_covariates=tot_cov)
        DF_predicted = model.predict(n=10)
        ts_pred_df = scaler_4.inverse_transform(DF_predicted)
        DF_prova_predicted =  ts_pred_df.pd_dataframe()
        score=mape(filtered_Target["Sum_Inflow"],DF_prova_predicted["Sum_Inflow"])
        
    #plot part
    if plot:
        fig, ax = plt.subplots(figsize = (15,9))
        ax.grid(visible=True)
        ax.plot_date(filtered_Target.index,filtered_Target["Sum_Inflow"],linestyle='solid')
        ax.plot_date(filtered_Target.index,DF_prova_predicted["Sum_Inflow_0.5"],linestyle='solid')
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run the models we trained on our dataset')
    
    parser.add_argument('--dataset_dir', action='store', default='dir', 
                        help='root of the dataset')

    # model selection
    parser.add_argument('--model_name',choices=("TFT","KalmanForecaster","ARIMA","'LinearRegressionModel'"), type=str, default="ARIMA",
                        help='model name selection')
    parser.add_argument('--load_model',choices=(True,False), type=bool, default=False,
                        help='load a pre-trained KalmanFilter model or train from scratch')

    
    # data selection for train-test
    parser.add_argument('--split_date', type=str, default="20211201",
                        help='date to split. Format "yyyymmdd".')
    parser.add_argument('--destination_country', type=str, default="ITA",
                        help='Destination Country of the migrants.')

    args = parser.parse_args()
    Dataset=pd.read_csv(args.dataset_dir)
    statistical_historical(args.model_name,args.load_model,date_test=args.split_date,Dataset=Dataset)
