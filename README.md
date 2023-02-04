# Time-Series Forecasting for Humanitarian Aid - ***model***

In this repository is possible to find the code built for the purpose of this project. The experiments conducted and the output were driven by the data gathered, through the use of two main technologies:
- [`ARIMA - Autoregressive Integrated Moving Average`](https://github.com/PoliTO-ADSP-United-Nations-Project/tsf_model/blob/main/statistical_models.py): a statistical model which perfectly fits for time series analysis and predictions of futre points (forecasting)
- [`TFT - Temporal Fusion Transformer`](https://github.com/PoliTO-ADSP-United-Nations-Project/tsf_model/blob/main/tft_model.py): attention-based Deep Neural Netwoek, optimized for multivariate time series forecasting.


To use our model, install the requirements and clone the `dataset` repositories:

```bash
git clone https://github.com/PoliTO-ADSP-United-Nations-Project/humanitarian_aid_dataset
```
and `model` repositories:
```bash
git clone https://github.com/PoliTO-ADSP-United-Nations-Project/tsf_model
```

## Execute with Jupyter notebook:
You can find a complete [`Jupyter notebook`](tft_model_jupyter.ipynb) file that shows how run both the models.<br/>
There is the possibility to open it through Google Colab.

## Execute in local:
As an alternative, you can use the operating system terminal to execute the code in this way:

### Create the dataset from scratch: 
```python
pip install -r "./humanitarian_aid_dataset/requirements.txt"
python ./humanitarian_aid_dataset/main.py
```

or download it from [`Figshare`](https://figshare.com/articles/dataset/VAL2G_-_Dataset/22006961).

### Get mandatory packages:
```python
pip install -r "./tsf_model/requirements.txt"
```

### Run **ARIMA** Model:
```python
python ./tsf_model/statistical_models.py --dataset_dir="/content/final_dataset.csv" --destination_country="ITA" --model_name="ARIMA"
```

### Run **TFT** Model:
```python
python ./tsf_model/tft_model.py --dataset_dir="/content/Final_TFT.csv"
```
---

Please, refere to the official paper for further information.
```bibtex
@article{
  title={Time Series Forecasting for Humanitarian Aid},
  author={Bergadano L., Frigiola A., Mantegna G., Scriotino G., Zingarelli V.},
  pages={N/A},
  year={2023}
}
```
