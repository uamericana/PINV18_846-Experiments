from lib import model, dataset
import tensorflow as tf

BATCH_SIZE = 32
SPLITS = [0.65, 0.25, 0.15]
SPLITS_NAMES = ["Train", "Validation", "Test"]
PROJECT_ROOT = '.'


def dataset_defaults(
        img_size: int,
        splits=SPLITS,
        remap_with=dataset.RetinopathyV2.mappings['c2'],
        shuffle_batches=True,
        reshuffle=False):
    tf.random.set_seed(42)
    img_size = (img_size, img_size)

    return dataset.build_dataset(
        dataset.RetinopathyV2,
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
        tl_epochs=10,
        fine_learning_rate=0.00001,
        fine_epochs=5,
        fine_layers=5
    )

    print(model_params)
    print(training_params)

    retinopathy_resnet50 = model.RetinopathyModel(model_params)

    datasets, class_names, images_df = dataset_defaults(img_size=model_params.image_size)
    train_dataset, validation_dataset, test_dataset = datasets

    tl_history, tl_model = model.transfer_learn(retinopathy_resnet50,
                                                train_dataset,
                                                validation_dataset,
                                                training_params,
                                                verbose=0)

    tl_loss, tl_accuracy = tl_model.evaluate(test_dataset)

    fine_history, fine_model = model.fine_tune(retinopathy_resnet50,
                                               train_dataset,
                                               validation_dataset,
                                               training_params,
                                               verbose=1)

    fine_loss, fine_accuracy = fine_model.evaluate(test_dataset)