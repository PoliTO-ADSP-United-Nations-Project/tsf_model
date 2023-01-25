import pandas as pd
import darts
import torch
import numpy as np
from darts.dataprocessing.transformers import Scaler

def preprocess(model_name,Dataset):
    if model_name in ["ARIMA","KalmanFilter"]:
        target_TimeSeries = TimeSeries.from_series(Dataset["Sum_Inflow"])
        covariates_TimeSeries = TimeSeries.from_dataframe(Dataset.drop(["Sum_Inflow"], axis=1))
        scaler_3 = Scaler()
        scaler_4 = Scaler()
        tot_cov = scaler_3.fit_transform(covariates_TimeSeries)
        scaled_full = scaler_4.fit_transform(target_TimeSeries)
        Target = target_TimeSeries.pd_dataframe()
        filtered_Target = Target.loc["2021-12-01" : "2022-09-01"]
        return scaler_3,scaler_4,target_TimeSeries,covariates_TimeSeries,tot_cov,scaled_full,Target,filtered_Target
    elif model_name== "TFT":
        print("We have to implement this...")
    else:
        raise ValueError("Not supported model name")