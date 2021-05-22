import pickle

from sklearn.model_selection import ParameterGrid

from lib import model, data, determ
from lib.datasets.retinopathyv2b import RetinopathyV2b
from lib.experiment import DataParams, dataset_defaults, execute_experiment

if __name__ == '__main__':

    SPLITS_NAMES = ["Train", "Validation", "Test"]
    PROJECT_ROOT = "."

    determ.set_global_determinism(42)

    data_grid = {
        'splits': [[0.65, 0.25, 0.15], [0.7, 0.2, 0.1]],
        'image_size': [160]
    }

    dataset_grid = [
        {'dataset': [RetinopathyV2b.name],
         'mapping': [RetinopathyV2b.mapping.c2.name, RetinopathyV2b.mapping.c3.name],
         **data_grid
         },
    ]

    model_grid = {
        'base_model': [m.name for m in model.BaseModel],
        'dropout': [0.2],
        'global_pooling': [model.GlobalPooling.AVG_POOLING.name, model.GlobalPooling.MAX_POOLING.name],
        'tl_learning_rate': [10e-5],
        'tl_epochs': [10],
        'fine_learning_rate': [10e-6],
        'fine_epochs': [5],
        'fine_layers': [30]
    }

    data_params_grid = ParameterGrid(dataset_grid)
    model_params_grid = ParameterGrid(model_grid)

    logbook = {}

    for dcomb in data_params_grid:
        mcomb = dcomb
        data_params = DataParams(
            dataset=dcomb['dataset'],
            splits=dcomb['splits'],
            image_size=dcomb['image_size'],
            remap=dcomb['mapping']
        )

        dataname = data_params.as_name()
        datasets, class_names, images_df = dataset_defaults(PROJECT_ROOT, data_params)
        summary_df = data.summarize_batched_datasets(datasets, SPLITS_NAMES, class_names)

        datalog = {}
        logbook[dataname] = datalog

        print(dataname)
        print("*" * 20)

        for mcomb in model_params_grid:
            model_params = model.ModelParams(
                num_classes=len(class_names),
                base_model=model.BaseModel[mcomb['base_model']],
                image_size=160,
                dropout=mcomb['dropout'],
                global_pooling=model.GlobalPooling[mcomb['global_pooling']],
                use_data_augmentation=True
            )

            training_params = model.TrainingParams(
                tl_learning_rate=mcomb['tl_learning_rate'],
                tl_epochs=mcomb['tl_epochs'],
                fine_learning_rate=mcomb['fine_learning_rate'],
                fine_epochs=mcomb['fine_epochs'],
                fine_layers=mcomb['fine_layers']
            )
            experiment_name = f"{model_params.as_name()}--{training_params.as_name()}"

            print(experiment_name)
            results = execute_experiment(datasets, class_names, model_params, training_params, verbose=2)
            metrics_df, reports, retinopathy_model = results
            datalog[experiment_name] = {'metrics_df': metrics_df, 'reports': reports}

            print(metrics_df)
            print("-" * 20)

    with open("logbook.pkl", "wb") as logfile:
        pickle.dump(logbook, logfile)
