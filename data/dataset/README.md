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
