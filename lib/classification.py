from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import pandas as pd


def classification_report_df(*args, **kwargs):
    kwargs["output_dict"] = True
    report = classification_report(*args, **kwargs)
    df = pd.DataFrame(report).transpose()
    return df


def predictions(model, test_ds, class_names, return_classes=True):
    prediction_logits = model.predict(test_ds)
    num_classes = len(class_names)

    if num_classes <= 2:
        score = tf.nn.sigmoid(prediction_logits)
        y_pred = tf.math.round(score)
    else:
        score = tf.nn.softmax(prediction_logits)
        y_pred = tf.math.argmax(score, axis=1)

    y_pred = tf.squeeze(y_pred).numpy().astype(np.int64)
    y_true = list(
        test_ds.flat_map(
            lambda images, labels: tf.data.Dataset.from_tensor_slices(labels)
        ).as_numpy_iterator()
    )
    y_true = np.array(y_true)

    if not return_classes:
        return y_true, y_pred

    def get_labels(y):
        return np.apply_along_axis(lambda idx: class_names[idx], 0, y)

    return get_labels(y_true), get_labels(y_pred)


def model_classification_report(model, test_ds, class_names):
    y_pred, y_true = predictions(model, test_ds, class_names)
    df = classification_report_df(y_true, y_pred)
    df.loc["accuracy", "support"] = y_pred.shape[0]
    return df


def classification_metrics(report: pd.DataFrame):
    recalls = report[['recall']][:-3]
    n = recalls.shape[0]
    recalls = recalls.transpose()
    recalls['Sensitivity'] = np.nan
    recalls['Specificity'] = np.nan

    if n == 2:
        recalls['Sensitivity'] = recalls[['Has DR Signs']]
        recalls['Specificity'] = recalls[['No DR Signs']]

    return recalls[['Sensitivity', 'Specificity']].iloc[0, :].to_dict()
