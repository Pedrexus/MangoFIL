import tensorflow as tf

# make confusion matrix metric
# make sparse categorical Precision, Recall and F1


class SparseCategoricalTruePositives(tf.keras.metrics.Metric):

    def __init__(self, print_tensor=False, name="sparse_categorical_true_positives", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")
        self.print_tensor = print_tensor

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])

        if self.print_tensor:
            tf.print(y_pred, y_true, summarize=-1)

        equal = tf.equal(y_true, y_pred)
        equal_int = tf.cast(equal, dtype=tf.float32)
        true_poss = tf.reduce_sum(equal_int)

        true_float = tf.cast(true_poss, dtype=tf.float32)
        self.cat_true_positives.assign_add(true_float)

    def result(self):
        return self.cat_true_positives


class SparseCategoricalTrueNegatives(tf.keras.metrics.Metric):

    def __init__(self, print_tensor=False, name="sparse_categorical_true_positives", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")
        self.print_tensor = print_tensor

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])

        if self.print_tensor:
            tf.print(y_pred, y_true, summarize=-1)

        equal = tf.equal(y_true, y_pred)
        equal_int = tf.cast(equal, dtype=tf.float32)
        true_poss = tf.reduce_sum(equal_int)

        true_float = tf.cast(true_poss, dtype=tf.float32)
        self.cat_true_positives.assign_add(true_float)

    def result(self):
        return self.cat_true_positives


