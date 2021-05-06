import json
import time
from enum import Enum
from typing import Callable, Any

import tensorflow as tf

_TL_EPOCHS = 20
_FINE_EPOCHS = 10
_TL_LR = 0.0010
_FINE_LR = 0.00001
_BASE_MODEL_WEIGHTS = 'imagenet'


class MLParamsEncoder(json.JSONEncoder):
    def default(self, obj):
        if issubclass(type(obj), Enum):
            return obj.name
        try:
            return obj.__dict__
        except TypeError:
            pass
        return json.JSONEncoder.default(self, obj)


class BaseModel(Enum):
    RESNET50_v2 = [tf.keras.applications.resnet_v2.ResNet50V2, tf.keras.applications.resnet_v2.preprocess_input]
    RESNET100_v2 = [tf.keras.applications.resnet_v2.ResNet101V2, tf.keras.applications.resnet_v2.preprocess_input]
    XCEPTION = [tf.keras.applications.xception.Xception, tf.keras.applications.xception.preprocess_input]
    MOBILENET_v2 = [tf.keras.applications.mobilenet_v2.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input]


class GlobalPooling(Enum):
    AVG_POOLING = tf.keras.layers.GlobalAvgPool2D
    MAX_POOLING = tf.keras.layers.GlobalMaxPooling2D
    NO_POOLING = None


class MLParams:
    def to_json(self):
        return json.dumps(self, cls=MLParamsEncoder)

    @classmethod
    def from_json(cls, json_st: str):
        d = json.loads(json_st)
        return cls.__init__(**d)

    def __str__(self) -> str:
        return self.to_json()

    def __repr__(self) -> str:
        return f"{type(self)} = {self.__dict__}"


class TrainingParams(MLParams):
    def __init__(self,
                 tl_learning_rate: float, fine_learning_rate: float,
                 tl_epochs: int, fine_epochs: int,
                 fine_layers: int):
        self.tl_learning_rate = tl_learning_rate
        self.fine_learning_rate = fine_learning_rate
        self.tl_epochs = tl_epochs
        self.fine_epochs = fine_epochs
        self.fine_layers = fine_layers

    def as_name(self):
        return f"tlr{self.tl_learning_rate}-" \
               f"flr{self.fine_learning_rate}-" \
               f"tep{self.tl_epochs}-" \
               f"fep{self.fine_epochs}-" \
               f"{self.fine_layers}"


class ModelParams(MLParams):
    def __init__(self,
                 base_model: BaseModel,
                 image_size: int,
                 num_classes: int,
                 dropout: float,
                 global_pooling: GlobalPooling,
                 use_data_augmentation: bool):
        self.base_model = base_model
        self.num_classes = num_classes
        self.dropout = dropout
        self.image_size = image_size
        self.global_pooling = global_pooling
        self.use_data_augmentation = use_data_augmentation

    def as_name(self):
        return f"bm{self.base_model.name}-" \
               f"nc{self.num_classes}-" \
               f"dr{self.dropout}-" \
               f"im{self.image_size}-" \
               f"gp{self.global_pooling.name}-" \
               f"da{self.use_data_augmentation}"

    @classmethod
    def from_json(cls, json_st: str):
        d = json.loads(json_st)
        return ModelParams(
            base_model=BaseModel[d['base_model']],
            num_classes=d['num_classes'],
            dropout=d['dropout'],
            image_size=d['image_size'],
            global_pooling=GlobalPooling[d['global_pooling']],
            use_data_augmentation=d['use_data_augmentation']
        )


def make_model(model_params: ModelParams):
    img_size = (model_params.image_size, model_params.image_size)
    input_shape = img_size + (3,)

    bm_fun, preprocess_fun = model_params.base_model.value
    base_model: tf.keras.Model = bm_fun(input_shape=input_shape,
                                        include_top=False,
                                        weights=_BASE_MODEL_WEIGHTS)

    base_model.trainable = False

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='constant'),
    ]) if model_params.use_data_augmentation else None

    units = 1 if model_params.num_classes <= 2 else model_params.num_classes

    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    if data_augmentation is not None:
        x = data_augmentation(inputs)

    x = preprocess_fun(x)
    x = base_model(x, training=False)

    if model_params.global_pooling is not None:
        pooling_layer: tf.keras.layers.Layer = model_params.global_pooling.value()
        x = pooling_layer(x)

    if model_params.dropout > 0:
        x = tf.keras.layers.Dropout(model_params.dropout)(x)

    outputs = tf.keras.layers.Dense(units)(x)
    model = tf.keras.Model(inputs, outputs, name=model_params.as_name())

    return model, base_model.name


class RetinopathyModel:
    def __init__(self, model_params: ModelParams):
        self.params = model_params
        model, base_model_name = make_model(model_params)
        self.model: tf.keras.Model = model
        self.base_model_name: str = base_model_name


def transfer_learn(retinopathy: RetinopathyModel,
                   train_dataset,
                   validation_dataset,
                   training_params: TrainingParams,
                   callbacks: tf.keras.callbacks.Callback = None,
                   verbose=0):
    """
    Train model using transfer learning
    :param retinopathy: a retinopathy model
    :param train_dataset: training dataset
    :param validation_dataset: validation dataset
    :param training_params: training parameters
    :param callbacks: optional list of callbacks
    :param verbose: verbose logging level
    :return: (training history, fitted model)
    """
    if callbacks is None:
        callbacks = []

    loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True) if retinopathy.params.num_classes <= 2 \
        else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = retinopathy.model

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_params.tl_learning_rate),
                  loss=loss_fun,
                  metrics=['accuracy'])

    history = model.fit(train_dataset,
                        epochs=training_params.tl_epochs,
                        validation_data=validation_dataset,
                        callbacks=callbacks,
                        verbose=verbose)

    return history, model


def fine_tune(retinopathy: RetinopathyModel,
              train_dataset,
              validation_dataset,
              training_params: TrainingParams,
              callbacks: tf.keras.callbacks.Callback = None,
              verbose=0):
    """
    Train model using fine tuning
    :param retinopathy: a retinopathy model (must have been trained with transfer_learn)
    :param train_dataset: training dataset
    :param validation_dataset: validation dataset
    :param training_params: training parameters
    :param callbacks: optional list of callbacks
    :param verbose: verbose logging level
    :return: (training history, fitted model)
    """

    if callbacks is None:
        callbacks = []

    model = retinopathy.model
    base_model_name = retinopathy.base_model_name
    num_classes = retinopathy.params.num_classes

    base_model = model.get_layer(base_model_name)
    base_model.trainable = True

    # Fine-tune n layers from the top
    fine_layers = training_params.fine_layers
    n_layers = len(base_model.layers)
    fine_tune_at = max(n_layers - fine_layers, 0)

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True) if num_classes <= 2 \
        else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss_fun,
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=training_params.fine_learning_rate),
                  metrics=['accuracy'])

    total_epochs = training_params.tl_epochs + training_params.fine_epochs

    history = model.fit(train_dataset,
                        epochs=total_epochs,
                        initial_epoch=training_params.tl_epochs,
                        validation_data=validation_dataset,
                        callbacks=callbacks,
                        verbose=verbose
                        )

    return history, model


def transfer_and_fine_tune(
        retinopathy: RetinopathyModel,
        training_params: TrainingParams,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        metrics_callback: Callable[[tf.keras.Model], Any] = None,
        verbose=0):
    """
    Train model using transfer learning followed by fine tuning
    :param retinopathy: retinopathy model
    :param training_params: training parameters
    :param train_dataset
    :param validation_dataset
    :param test_dataset
    :param metrics_callback:
    :param verbose: verbose logging level
    :return:
    """

    tic = time.perf_counter()
    tl_history, tl_model = transfer_learn(retinopathy,
                                          train_dataset,
                                          validation_dataset,
                                          training_params,
                                          verbose=verbose)
    toc = time.perf_counter()
    tl_time = toc - tic
    tl_loss, tl_accuracy = tl_model.evaluate(test_dataset, verbose=verbose)
    tl_metrics = {}

    if metrics_callback:
        tl_metrics = metrics_callback(tl_model)

    tic = time.perf_counter()
    fine_history, fine_model = fine_tune(retinopathy,
                                         train_dataset,
                                         validation_dataset,
                                         training_params,
                                         verbose=verbose)
    toc = time.perf_counter()
    fine_time = toc - tic
    fine_loss, fine_accuracy = fine_model.evaluate(test_dataset, verbose=verbose)
    fine_metrics = {}

    if metrics_callback:
        fine_metrics = metrics_callback(fine_model)

    return {
        'tl_history': tl_history,
        'tl_model': tl_model,
        'tl_time': tl_time,
        'tl_loss': tl_loss,
        'tl_accuracy': tl_accuracy,
        'tl_metrics': tl_metrics,
        'fine_history': fine_history,
        'fine_model': fine_model,
        'fine_time': fine_time,
        'fine_loss': fine_loss,
        'fine_accuracy': fine_accuracy,
        'fine_metrics': fine_metrics
    }
