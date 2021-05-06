import json
from enum import Enum
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
    def __init__(self, tl_learning_rate: float, fine_learning_rate: float, tl_epochs: int, fine_epochs: int):
        self.tl_learning_rate = tl_learning_rate
        self.fine_learning_rate = fine_learning_rate
        self.tl_epochs = tl_epochs
        self.fine_epochs = fine_epochs

    def as_name(self):
        return f"tlr{self.tl_learning_rate}-" \
               f"flr{self.fine_learning_rate}-" \
               f"tep{self.tl_epochs}-" \
               f"fep{self.fine_epochs}"


class ModelParams(MLParams):
    def __init__(self,
                 base_model: BaseModel,
                 image_size: int,
                 num_classes: int,
                 dropout: float,
                 train_layers: int,
                 global_pooling: GlobalPooling,
                 use_data_augmentation: bool):
        self.base_model = base_model
        self.num_classes = num_classes
        self.train_layers = train_layers
        self.dropout = dropout
        self.image_size = image_size
        self.global_pooling = global_pooling
        self.use_data_augmentation = use_data_augmentation

    def as_name(self):
        return f"bm{self.base_model.name}-" \
               f"nc{self.num_classes}-" \
               f"tl{self.train_layers}-" \
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
            train_layers=d['train_layers'],
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

    return model


class RetinopathyModel:
    def __init__(self, model_params: ModelParams):
        self.model_params = model_params
        self.model = make_model(model_params)
