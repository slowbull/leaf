import tensorflow as tf

from model_prox import Model


IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(lr)

    def create_model(self):
        """Model function for CNN."""
        is_train = tf.placeholder(tf.bool, name="is_train")
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
		  kernel_initializer=tf.variance_scaling_initializer(),
          activation=tf.nn.relu)
        #conv1 = tf.layers.batch_normalization(conv1, training=is_train)
        #conv1 = tf.nn.relu(conv1)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
		    kernel_initializer=tf.variance_scaling_initializer(),
            activation=tf.nn.relu)
        #conv2 = tf.layers.batch_normalization(conv2, training=is_train)
        #conv2 = tf.nn.relu(conv2)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
        dense = tf.layers.dropout(dense, training=is_train)
        #dense = tf.layers.dense(inputs=dense, units=512, activation=tf.nn.relu)
        #dense = tf.layers.dropout(dense, training=is_train)
        logits = tf.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, is_train, train_op, eval_metric_ops, loss
