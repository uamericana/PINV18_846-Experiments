from lib import model, data, determ
from lib.datasets.retinopathyv2a import RetinopathyV2a
from lib.experiment import DataParams, dataset_defaults, execute_experiment
from sklearn.model_selection import ParameterGrid

if __name__ == '__main__':

    SPLITS_NAMES = ["Train", "Validation", "Test"]
    PROJECT_ROOT = "."

    determ.set_global_determinism(42)

    data_grid = {
        'splits': [[0.65, 0.25, 0.15], [0.7, 0.2, 0.1]],
        'image_size': [160]
    }

    dataset_grid = [
        {'dataset': [RetinopathyV2a.name],
         'mapping': [RetinopathyV2a.mapping.c2.name, RetinopathyV2a.mapping.c3.name],
         **data_grid
         },
    ]

    model_grid = {
        'base_model': [m.name for m in model.BaseModel],
        'dropout': [0.2],
        'global_pooling': [g.name for g in model.GlobalPooling],
        'tl_learning_rate': [10e-5],
        'tl_epochs': [10],
        'fine_learning_rate': [10e-6],
        'fine_epochs': [5],
        'fine_layers': [30]
    }

    data_params_grid = ParameterGrid(dataset_grid)
    model_params_grid = ParameterGrid(model_grid)

    for dcomb in data_params_grid:
        mcomb = dcomb
        data_params = DataParams(
            dataset=dcomb['dataset'],
            splits=dcomb['splits'],
            image_size=dcomb['image_size'],
            remap=dcomb['mapping']
        )

        # datasets, class_names, images_df = dataset_defaults(PROJECT_ROOT, data_params)
        # summary_df = data.summarize_batched_datasets(datasets, SPLITS_NAMES, class_names)
        print()
        print(data_params)
        print()

        for mcomb in model_params_grid:
            model_params = model.ModelParams(
                num_classes=2,
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

            print(model_params)
            print(training_params)
