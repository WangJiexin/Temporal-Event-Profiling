
# Temporal Event Profiling: TEP-Trans

This is the repository containing the code for the TEP-Trans model described in the Sigir 2021 paper "Temporal Event Profiling based on Multivariate Time Series Analysis over Temporal Document Collections".

## Using the Dataset

The event dataset is stored in the `data/dataset` folder. The datasets is stored as pandas dataframes in pickle files and you can load them as follows:

```
In [1]: import pandas as pd

In [2]: data = pd.read_pickle("data/dataset/wotd.pkl")

In [3]: data.head()
Out[3]:

                                               event_text       event_time      split
0	NFC Championship, Trans World Dome, St. Louis...	2000-01-23	train
1	Cleveland Indians tie record of scoring eight ...	1995-05-09	train
2	Tammie Green wins LPGA Sprint Titleholders Cha...	1997-05-04	train
3	Mark Davis signs record US$3.25 million per ye...	1989-12-11	train
4	Sociedade Esportiva Palmeiras wins the Liberta...	1999-06-16	train

```

The key key columns are:
- `event_text`: The event description.
- `event_time`: The event occurrence time at day granularity.
- `split`: The set that this record belongs to (training set, validation set or test set).

## Quick Start

### Packages
In order to run the model, you need to install a few packages. We recommend using [conda](https://docs.conda.io/en/latest/):
```bash
conda create -n test python=3.7 anaconda
git clone git@github.com:WangJiexin/Temporal-Event-Profiling.git
cd Temporal-Event-Profiling
pip install -r requirements.txt
```

### Download the necessary prepared data for training the model
Running the following command will create a `data/input_data` folder, and download two kinds of prepared data.
```bash
python download_data.py
```

### Train the model
Specify a granularity (Year, Month, Week, Day) and train the model.
```bash
python model.py --temp_granularity Year
```
A `data/models` folder will be created if you first run the above code, and all the models of specified granularity will be saved in this folder.
Note that if you use finer granularities (e.g., day, week), you need to use larger gpu memory or/and reduce the batch size.

## Multivariate Time Series Calculation And Model Input
