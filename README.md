
# Temporal Event Profiling: TEP-Trans

This is the repository containing the code for the TEP-Trans model described in the Sigir 2021 paper "Temporal Event Profiling based on Multivariate Time Series Analysis over Temporal Document Collections".

## Using the Dataset

The event dataset is stored in the `data/dataset` folder. The dataset is stored as pandas dataframes in pickle file and you can load it as follows:

```
In [1]: import pandas as pd

In [2]: event_data = pd.read_pickle("data/dataset/event_data.pickle")

In [3]: event_data.head()
Out[3]:

                                               event_text       event_time      split
0	NFC Championship, Trans World Dome, St. Louis...	2000-01-23	train
1	Cleveland Indians tie record of scoring eight ...	1995-05-09	train
2	Tammie Green wins LPGA Sprint Titleholders Cha...	1997-05-04	train
3	Mark Davis signs record US$3.25 million per ye...	1989-12-11	train
4	Sociedade Esportiva Palmeiras wins the Liberta...	1999-06-16	train

```

The definitions of the three key columns are:
- `event_text`: The event description.
- `event_time`: The event occurrence time at day granularity.
- `split`: The set that this record belongs to (training set, validation set or test set).

## Quick Start

### Packages
In order to run the model, you need to install a few packages. We recommend using [conda](https://docs.conda.io/en/latest/):
```bash
conda create -n tep-trans python=3.7 anaconda
conda activate tep-trans
git clone https://github.com/WangJiexin/Temporal-Event-Profiling
cd Temporal-Event-Profiling
pip install -r requirements.txt
```

### Download the necessary prepared data for training the model
Running the following command will create a `data/input_data` folder, and download two kinds of prepared data.
```bash
python download_data.py
```

### Train the model
Specify a temporal granularity (Year, Month, Week, Day) and train the model.
```bash
python model.py --temp_granularity Year
```
A `data/models` folder will be created if you run the above code at the first time and the models of specified granularities will be saved in this folder.
Note that if you choose finer granularities (e.g., Week, Day), you need to use larger gpu memory or/and reduce the batch size.

## Prepare Multivariate Time Series And Model Input
Steps to prepare multivariate time series and necessary data input are described in the `data_processing` [README](data_processing/README.md).
