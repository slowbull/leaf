"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import os
import sys
import tensorflow as tf
import time

from pgd import PerturbedGradientDescent
from baseline_constants import ACCURACY_KEY, LOSS_KEY

from utils.model_utils import batch_data
from utils.tf_utils import graph_size

#from tensorflow.python.profiler import option_builder
#from tensorflow.python.profiler import model_analyzer



class Model(ABC):

    def __init__(self, lr):
        self.lr = lr
        self._optimizer = None

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.features, self.labels, self.is_train, self.train_op, self.eval_metric_ops, self.loss = self.create_model()
            self.saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.size = graph_size(self.graph)

        #self.profiler = tf.profiler.Profiler(self.sess.graph)
        #self.profiler = model_analyzer.Profiler(self.sess.graph)


        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        self._optimizer = PerturbedGradientDescent(learning_rate=self.lr)

        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 4-tuple consisting of:
                features: A placeholder for the samples' features.
                labels: A placeholder for the samples' labels.
                train_op: A Tensorflow operation that, when run with the features and
                    the labels, trains the model.
                eval_metric_ops: A Tensorflow operation that, when run with features and labels,
                    returns the accuracy of the model.
        """
        return None, None, None, None

    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
	    # intialize as server model.
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for v in all_vars:
                v.load(self.init_vals[v.name], self.sess)

        with self.graph.as_default():
            init_values = [self.sess.run(v) for v in tf.trainable_variables()]

        batched_x, batched_y = batch_data(data, batch_size)
        #run_metadata = tf.RunMetadata()
        for _ in range(num_epochs):
            for i, raw_x_batch in enumerate(batched_x):
                input_data = self.process_x(raw_x_batch)
                raw_y_batch = batched_y[i]
                target_data = self.process_y(raw_y_batch)
                with self.graph.as_default():

                    self.sess.run(
                        self.train_op,
                        feed_dict={self.features: input_data, self.labels: target_data, self.is_train:True},
					    #options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
					    #run_metadata=run_metadata
                        )
                    #self.profiler.add_step(i, run_metadata)
                    #self.profiler.profile_graph(options=opts)
                    #opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
                    #profiler.profile_operations(options=opts)

        #profile_code_opt_builder.select(['micros','occurrence'])
        #profile_code_opt_builder.order_by('micros')
        #profile_code_opt_builder.with_max_depth(10)
        #self.profiler.profile_operations(profile_code_opt_builder.build())
        #self.profiler.profile_python(profile_code_opt_builder.build())

                    
        with self.graph.as_default():
            update = [self.sess.run(v) for v in tf.trainable_variables()]
            #update = [np.subtract(update[i], init_values[i]) for i in range(len(update))]
        comp = num_epochs * len(batched_y) * batch_size * self.flops
        return comp, update

    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        with self.graph.as_default():
            tot_acc, tot_loss = self.sess.run(
                [self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels, self.is_train:False}
            )
        acc = float(tot_acc) / x_vecs.shape[0]
        return {ACCURACY_KEY: acc, LOSS_KEY: tot_loss}

    def close(self):
        self.sess.close()

    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return np.asarray(raw_x_batch)

    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return np.asarray(raw_y_batch)


class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        tmp_vals = []
        with self.model.graph.as_default():
            all_vars = tf.trainable_variables()
            for v in all_vars:
                val = self.model.sess.run(v)
                var_vals[v.name] = val
                tmp_vals.append(val)

        for c in clients:
            c.model.init_vals = var_vals
            with c.model.graph.as_default():
                all_vars = tf.trainable_variables()
                for v in all_vars:
                    v.load(var_vals[v.name], c.model.sess)

            with c.model.graph.as_default():
                c.model._optimizer.set_params(tmp_vals, c.model)


    def update(self, updates):
        """Updates server model using given client updates.

        Args:
            updates: list of (num_samples, update), where num_samples is the
                number of training samples corresponding to the update, and update
                is a list of variable weights
        """
        tot_samples = np.sum([u[0] for u in updates])

        weighted_vals = [np.zeros(np.shape(v), dtype=float) for v in updates[0][1]]

        for i, update in enumerate(updates):
            for j, weighted_val in enumerate(weighted_vals):
                weighted_vals[j] = np.add(weighted_val, update[0] * update[1][j])

        weighted_updates = [v / tot_samples for v in weighted_vals]

        with self.model.graph.as_default():
            all_vars = tf.trainable_variables()
            for i, v in enumerate(all_vars):
                init_val = self.model.sess.run(v)
                #v.load(np.add(init_val, weighted_updates[i]), self.model.sess)
                v.load(weighted_updates[i], self.model.sess)

    def save(self, path='checkpoints/model.ckpt'):
        return self.model.saver.save(self.model.sess, path)

    def close(self):
        self.model.close()

    def get_model(self):
        with self.model.graph.as_default():
            init_values = [self.model.sess.run(v) for v in tf.trainable_variables()] 
        return init_values
