import pathlib
import pickle

import pandas as pd

base_results_dir = "retinopathy-base-models"

with open(pathlib.Path(base_results_dir, 'logbook.pkl'), 'rb') as f:
    logbook = pickle.load(f)

metrics = []

for data, datalog in logbook.items():
    for model, model_results in datalog.items():
        metrics_df = pd.DataFrame(model_results['metrics_df'])
        total_time = metrics_df.loc['time'].sum()
        metrics_df = metrics_df.append(pd.Series({'tl': total_time, 'fine': total_time}, name='total_time'))

        df = metrics_df.reset_index() \
            .melt(id_vars='index', var_name='step') \
            .rename(columns={'index': 'metric'})

        df.insert(loc=0, column='model', value=model)
        df.insert(loc=0, column='data', value=data)
        metrics.append(df)

metrics = pd.concat(metrics, ignore_index=True)

data_params = metrics.data.str.extract(
    r'ds(?P<dataset>.*)-sp(?P<splits>.*)-is(?P<img_size>.*)-bs(?P<batch_size>.*)-rm(?P<classes>.*)', expand=True)
data_params = data_params.drop(columns=['classes', 'img_size'])

model_params = metrics.model.str.extract(
    r'bm(?P<base_model>.*)-nc(?P<classes>.*)-dr(?P<dropout>.*)-'
    r'im(?P<img_size>.*)-gp(?P<pooling>.*)-da(?P<data_augmentation>.*)--(?P<training_params>.*)',
    expand=True)
training_params = model_params.training_params.str.extract(
    r'tlr(?P<tl_lr>.*)-flr(?P<ft_lr>.*)-'
    r'tep(?P<tl_epochs>.*)-fep(?P<fine_epochs>.*)-(?P<train_layers>.*)',
    expand=True)
model_params = model_params.drop(columns=['training_params'])

metrics = metrics.drop(columns=['data', 'model'])

df = pd.concat([data_params, model_params, training_params, metrics], axis=1)

convert_dict = {
    'batch_size': int,
    'classes': int,
    'dropout': float,
    'img_size': int,
    'data_augmentation': bool
}
df = df.astype(convert_dict)
df.splits = "(" + df.splits.str.replace("_", ", ") + ")"
df = df.drop(['batch_size', 'img_size', 'data_augmentation'], axis=1)
df = df[(df.step == 'fine') & (df.pooling == 'AVG_POOLING')]

results = df[
    ['dataset', 'base_model', 'classes', 'dropout', 'tl_lr',
     'ft_lr', 'train_layers', 'metric', 'value']]

results = results.pivot(index=results.columns[:-2], columns='metric', values='value') \
    .reset_index() \
    .rename_axis(None, axis=1)
