import pathlib
import pickle

import pandas as pd
from deap import base
from deap import creator

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def base_results_df(base_results_dir):
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

    return results


def deap_results_df(experiment_dir):
    runs = list(pathlib.Path(experiment_dir).glob("run-*"))
    rows = []
    experiment_name = pathlib.Path(experiment_dir).name

    with open(pathlib.Path(experiment_dir, "datalog.pkl"), "rb") as f:
        datalog = pickle.load(f)

    if datalog is None:
        raise Exception("Could not read datalog for " + experiment_dir)

    for run_dir in runs:
        run = int(run_dir.name.split('-')[-1])
        checkpoints = list(run_dir.glob("gen_*.pkl"))
        for checkpoint_path in checkpoints:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                population = checkpoint['population']
                generation = checkpoint['generation']

                for individual in population:
                    str_individual = "-".join(map(lambda x: str(x), individual))
                    name = f"{experiment_dir}-{str_individual}"
                    try:
                        log = datalog[name]
                        total_time = log['tl']['time'] + log['fine']['time']
                    except Exception:
                        log = {}
                        total_time = 0

                    rows.append({
                        'run': run,
                        'generation': generation,
                        'dropout': individual[0],
                        'train_layers': individual[1],
                        'tl_lr': individual[2],
                        'ft_lr': individual[3],
                        'fitness': individual.fitness.values[0],
                        **log['fine'],
                        'total_time': total_time
                    })
    df = pd.DataFrame(rows)
    df.insert(0, column='experiment', value=experiment_name)
    return df


if __name__ == '__main__':
    experiments = [
        'deap-retinopathy-v2b-c3-BaseModel.RESNET50_v2',
        'deap-retinopathy-v3-c3-BaseModel.RESNET50_v2'
    ]
    deap_results = [deap_results_df(e) for e in experiments]
    deap_results = pd.concat(deap_results).reset_index()
    base_results = base_results_df('retinopathy-base-models')
