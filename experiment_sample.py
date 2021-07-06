from lib import model, data, determ
from lib.datasets.retinopathyv3 import RetinopathyV3
from lib.experiment import DataParams, dataset_defaults, execute_experiment

if __name__ == '__main__':
    SPLITS_NAMES = ["Train", "Validation", "Test"]
    PROJECT_ROOT = '.'

    determ.set_global_determinism(42)

    data_params = DataParams(
        dataset=RetinopathyV3.name,
        remap=RetinopathyV3.mapping.c3.name,
        image_size=160,
        batch_size=32,
        splits=[0.7, 0.2, 0.1]
    )

    datasets, class_names, images_df = dataset_defaults(PROJECT_ROOT, data_params)
    summary_df = data.summarize_batched_datasets(datasets, SPLITS_NAMES, class_names)

    model_params = model.ModelParams(
        base_model=model.BaseModel.RESNET50_v2,
        image_size=160,
        num_classes=len(class_names),
        dropout=0.2,
        global_pooling=model.GlobalPooling.AVG_POOLING,
        use_data_augmentation=True
    )

    training_params = model.TrainingParams(
        tl_learning_rate=0.01,
        tl_epochs=20,
        fine_learning_rate=1e-05,
        fine_epochs=10,
        fine_layers=67
    )

    print(model_params)
    print(training_params)

    metrics, reports, retinopathy_model = execute_experiment(datasets, class_names, model_params, training_params)
