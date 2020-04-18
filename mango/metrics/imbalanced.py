import tensorflow as tf
from sklearn.metrics import f1_score


class SparseCategoricalF1Score(tf.keras.metrics.Metric):

    def __init__(self, print_tensor=False, name="sparse_categorical_f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cat_f1 = self.add_weight(name="cf1", initializer="zeros")
        self.print_tensor = print_tensor

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])

        if self.print_tensor:
            tf.print(y_pred, y_true, summarize=-1)

        # y_true, y_pred = tf.make_tensor_proto(y_true), tf.make_tensor_proto(y_pred)
        # y_true, y_pred = tf.make_ndarray(y_true), tf.make_ndarray(y_pred)

        score = f1_score(y_true, y_pred, average='weighted')

        self.cat_f1.assign(score)

    def result(self):
        return self.cat_f1