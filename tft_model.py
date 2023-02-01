import os
import warnings
import random
import argparse

#warnings.filterwarnings("ignore")  # avoid printing out absolute paths

#os.chdir("../../..")
import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from preprocess import preprocess
from metrics import mae,mape


def tft_historical(point_to_predict,input_chunk,output_chunk,n_epochs,Dataset,actual_ita,actual_esp,actual_grc,random_seed,model_path="Content/model",plot=True):
    Dataset = preprocess("TFT", Dataset)
    #Prepare the parameters for the TFT model

    spain_list = ['CIV', 'CMR', 'DZA', 'GIN', 'GMB', 'MAR', 'MLI', 'SEN', 'SYR']
    ita_list = ['AFG', 'BGD', 'CIV', 'DZA', 'EGY', 'ERI', 'GIN', 'GMB', 'IRN', 'IRQ', 'MAR', 'MLI', 'NGA', 'PAK', 'SDN', 'SEN', 'SOM', 'SYR', 'TUN']
    grec_list = ['AFG', 'CMR', 'COD', 'COG', 'IRN', 'IRQ', 'PAK', 'PSE', 'SOM', 'SYR']

    set_countries = set()

    set_countries.update(spain_list)
    set_countries.update(ita_list)
    set_countries.update(grec_list)

    Time_range_day = np.arange(datetime(2017,1,1), datetime(2022,10,1), timedelta(days=1)).astype(datetime)
    Time_range_month =  np.array(Time_range_day, dtype='datetime64[M]')
    set_ = set(Time_range_month)
    Time_range_month = list(set_)
    index_dates = pd.DatetimeIndex(np.array(Time_range_month, dtype='datetime64[M]'))

    max_prediction_length = output_chunk #The model uses a max prediction length of 2. So the model can learn to predict the next point and the one after that. This way the model will also learn the derivative
    max_encoder_length = input_chunk #The input chunk is 24. 24 month is twice the seasonality

    complete = TimeSeriesDataSet(
        Dataset,
        time_idx="time_idx",
        target="Monthly_inflow",
        group_ids=["Departure_country","Destination_country"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Departure_country","Destination_country"],
        static_reals=["Distance_Departure_Destination"],
        time_varying_known_categoricals=["month"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "Monthly_inflow",
            "fatalities",
            "Perc_Change",
            "HDI",
            "Sum_Inflow"
        ],
        target_normalizer=TorchNormalizer(transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if random_seed == None:
        random_seed = np.random.randint(100)

    g = torch.Generator()
    g.manual_seed(random_seed)
    pl.seed_everything(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    Dataset_cutted = Dataset

    poit_to_predict = point_to_predict

    predicted_values_ITA = []
    predicted_values_ESP = []
    predicted_values_GRC = []
    countries = ["ITA","ESP","GRC"]

    for shift in range(1,poit_to_predict+1): 

        max_prediction_length = output_chunk #The model uses a max prediction length of 2. So the model can learn to predict the next point and the one after that. This way the model will also learn the derivative
        max_encoder_length = input_chunk #The input chunk is 24. 24 month is twice the seasonality
        
        training_cutoff = Dataset_cutted["time_idx"].max() - 1 #split in train e validation

        training = TimeSeriesDataSet(
            Dataset_cutted[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="Monthly_inflow",
            group_ids= ["Departure_country","Destination_country"],
            min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals= ["Departure_country","Destination_country"],
            static_reals=["Distance_Departure_Destination"],
            time_varying_known_categoricals=["month"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "Monthly_inflow",
                "fatalities",
                "Perc_Change",
                "HDI",
                "Sum_Inflow"
            ],
            target_normalizer=TorchNormalizer(transformation="softplus"),  # use softplus and normalize by group 
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        batch_size = 64  # set this between 32 to 128

        validation = TimeSeriesDataSet.from_dataset(training, Dataset_cutted, predict=True, stop_randomization=True)
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0,worker_init_fn=seed_worker, generator=g)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0,worker_init_fn=seed_worker, generator=g)
        #train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        #val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
        
        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=60, verbose=False, mode="min") #Early stopping to avoid overfitting
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            accelerator="gpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=30,  # coment in for training, running valiation every 30 batches
            #fast_dev_run=False,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )


        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.0257,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.3,
            hidden_continuous_size=8,
            optimizer='adam',
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            #log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            reduce_on_plateau_patience=4,
        )
        
        #Train the transformer
        trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        )

        sum_ita = 0
        for countrie in countries:
            vector = []

            #Returns the prediction for each starting country
            for ele in set_countries:

                #print(ele)
                raw_prediction, x = tft.predict(
                    complete.filter(lambda x: ( (x.Destination_country == f"{countrie}") & (x.Departure_country == ele) & (x.time_idx_first_prediction == Dataset_cutted["time_idx"].max() ) )),
                    mode="raw",
                    return_x=True,
                )
                    #print( int(raw_prediction["prediction"][0][0][3]))
                vector.append(int(raw_prediction["prediction"][0][0][3]))

                #The final prediction is given by the sum
            vector = np.array(vector)
            sum_ita = vector.sum()
            print(sum_ita)
            if countrie == "ITA":
                predicted_values_ITA.append(vector.sum())
            if countrie == "ESP":
                predicted_values_ESP.append(vector.sum())
            if countrie == "GRC":
                predicted_values_GRC.append(vector.sum())
            if sum_ita == 0:
                print("Errore NEL TRAINING , Training fallito")
        #Reduce the dataset and consider the new month to predict
        Dataset_cutted = Dataset[lambda x: x.time_idx <= training_cutoff]

    predicted_values_ESP = predicted_values_ESP[::-1]
    predicted_values_ITA = predicted_values_ITA[::-1]
    predicted_values_GRC = predicted_values_GRC[::-1]

    time = index_dates[59:69]


    if plot == True:

        fig, ax = plt.subplots(figsize = (15,9))
        ax.grid(visible=True)
        ax.legend()
        ax.plot_date(time,actual_ita,linestyle='solid',linewidth=4,label = "Actual")
        ax.plot_date(time,predicted_values_ITA,linestyle='solid',linewidth=4,label = "Predicted")
        plt.ylabel("Migrants")
        plt.title(f"BackTest ITA ")

        fig1, ax1 = plt.subplots(figsize = (15,9))
        ax1.grid(visible=True)
        ax1.legend()
        ax1.plot_date(time,actual_esp,linestyle='solid',linewidth=4,label = "Actual")
        ax1.plot_date(time,predicted_values_ESP,linestyle='solid',linewidth=4,label = "Predicted")
        plt.ylabel("Migrants")
        plt.title(f"BackTest ESP ")

        fig2, ax2 = plt.subplots(figsize = (15,9))
        ax2.grid(visible=True)
        ax2.legend()
        ax2.plot_date(time,actual_grc,linestyle='solid',linewidth=4,label = "Actual")
        ax2.plot_date(time,predicted_values_GRC,linestyle='solid',linewidth=4,label = "Predicted")
        plt.ylabel("Migrants")
        plt.title(f"BackTest GRC")


    return predicted_values_ITA, predicted_values_ESP,predicted_values_GRC


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run the models we trained on our dataset')
    
    parser.add_argument('--point_to_predict', action='store', default=10, type = int,
                        help='Number of points to predict')

    parser.add_argument('--dataset_dir', action='store', default='dir', 
                        help='root of the dataset')

    #model parameters
    parser.add_argument('--input_chunk', default=24, type = int,
                        help='Max_encoder_length of tft model ')

    parser.add_argument('--output_chunk', default=2, type = int,
                        help='Max_decoder_length of tft model ')

    parser.add_argument('--n_epochs', default=120, type = int,
                        help='Number of training epochs')


    #actual values
    parser.add_argument('--actual_ita', default=[4227.0, 2892.0, 2289.0, 1279.0, 3580.0, 8102.0, 6681.0, 13083.0, 16370.0, 12574.0], type = list,
                        help='Actual value of italian miration')

    parser.add_argument('--actual_esp', default=[3842, 4255.0, 3117, 1388, 1473, 2351, 1540, 2647, 2289, 4190], type = list,
                        help='Actual value of spain miration')

    parser.add_argument('--actual_esp', default=[577.0, 103.0, 49.0, 203.0, 288.0, 215.0, 341.0, 118.0, 698.0, 386.0], type = list,
                        help='Actual value of greece miration')

    
    parser.add_argument('--random_seed', default=None, type = int,
                        help='Random Seed')

    parser.add_argument('--plot', default=False, type = bool,
                        help='Want to plot?')

    args = parser.parse_args()
    Dataset=pd.read_csv(args.dataset_dir)
    tft_historical(args.model_name,args.load_model,date_test=args.split_date,Dataset=Dataset)

    








