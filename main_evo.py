from lib import model, data, determ

from lib.datasets.retinopathyv2a import RetinopathyV2a

BATCH_SIZE = 32
SPLITS = [0.65, 0.25, 0.15]
SPLITS_NAMES = ["Train", "Validation", "Test"]
PROJECT_ROOT = '.'


def dataset_defaults(
        img_size: int,
        splits=SPLITS,
        remap_with=RetinopathyV2a.mapping.c2,
        shuffle_batches=True,
        reshuffle=False):
    img_size = (img_size, img_size)

    return data.build_dataset(
        RetinopathyV2a,
        PROJECT_ROOT,
        img_size,
        BATCH_SIZE,
        splits,
        remap_classes=remap_with,
        shuffle_split=True,
        shuffle_batches=shuffle_batches,
        reshuffle=reshuffle
    )


if __name__ == '__main__':
    determ.set_global_determinism(42)

    model_params = model.ModelParams(
        base_model=model.BaseModel.RESNET50_v2,
        image_size=160,
        num_classes=2,
        dropout=0.2,
        global_pooling=model.GlobalPooling.AVG_POOLING,
        use_data_augmentation=True
    )

    training_params = model.TrainingParams(
        tl_learning_rate=0.0001,
        tl_epochs=20,
        fine_learning_rate=0.00001,
        fine_epochs=10,
        fine_layers=30
    )

    datasets, class_names, images_df = dataset_defaults(img_size=model_params.image_size)
    train_dataset, validation_dataset, test_dataset = datasets

    retinopathy_resnet50 = model.RetinopathyModel(model_params)
    results = model.transfer_and_fine_tune(
        retinopathy_resnet50, training_params,
        train_dataset, validation_dataset, test_dataset,
        verbose=1)

    print(results)
