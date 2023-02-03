import pandas as pd
import darts
import torch
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

def preprocess(model_name,Dataset,destination_country):
    Dataset=Dataset[Dataset["Destination_country"]==destination_country][["date","Sum_Inflow"]]
    Dataset=Dataset[:69].set_index("date")
    Dataset.index = pd.to_datetime(Dataset.index)
    if model_name in ["ARIMA","KalmanForecaster"]:
        target_TimeSeries = TimeSeries.from_series(Dataset)
        covariates_TimeSeries = TimeSeries.from_dataframe(Dataset.drop(["Sum_Inflow"], axis=1))
        scaler_3 = Scaler()
        scaler_4 = Scaler()
        tot_cov = covariates_TimeSeries
        scaled_full = scaler_4.fit_transform(target_TimeSeries)
        Target = target_TimeSeries.pd_dataframe()
        filtered_Target = Target.loc["2021-12-01" : "2022-09-01"]
        return scaler_3,scaler_4,target_TimeSeries,covariates_TimeSeries,tot_cov,scaled_full,Target,filtered_Target
    elif model_name== "TFT":
        Dataset_new=Dataset.copy()
        del Dataset_new["Unnamed: 0"]
        Dataset_new["month"]=Dataset_new["month"].map(str)
        return Dataset_new
    else:
        raise ValueError("Not supported model name")
