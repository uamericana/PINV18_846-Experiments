from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import pandas as pd


def classification_metrics_for_class(matrix, class_value):
    metrics = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0,
    }

    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            if i == j and i == class_value:
                metrics['TP'] += cell
            elif i != class_value and j != class_value:
                metrics['TN'] += cell
            elif i != class_value and j == class_value:
                metrics['FP'] += cell
            elif i == class_value and j != class_value:
                metrics['FN'] += cell

    metrics['Sensitivity'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['Specificity'] = metrics['TN'] / (metrics['TN'] + metrics['FP'])
    return metrics


def classification_metrics_df(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    all_metrics = [{**classification_metrics_for_class(matrix, i)} for i, label in enumerate(labels)]
    df = pd.DataFrame(all_metrics, index=labels)
    return df


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
    y_true, y_pred = predictions(model, test_ds, class_names)
    df = classification_report_df(y_true, y_pred)
    df.loc["accuracy", "support"] = y_pred.shape[0]
    metrics = classification_metrics_df(y_true, y_pred, class_names)
    return df.join(metrics)


def classification_metrics(report: pd.DataFrame):
    series = report.loc['No DR Signs']
    sensitivity = series['Specificity']
    specificity = series['Sensitivity']
    return {'Specificity': specificity, 'Sensitivity': sensitivity}
