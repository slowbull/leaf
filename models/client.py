import random
import warnings

from utils.language_utils import letter_to_vec, word_to_indices

class Client:
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id # integer
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data

        if isinstance(train_data['x'][0], str):
            self.train_data['x'], self.train_data['y'] = self.process_x(self.train_data['x']), self.process_y(self.train_data['y'])
            self.eval_data['x'], self.eval_data['y'] = self.process_x(self.eval_data['x']), self.process_y(self.eval_data['y'])

    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        if minibatch is None:
            data = self.train_data
            comp, update = self.model.train(data, num_epochs, batch_size)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {"x": xs, "y": ys}
            comp, update = self.model.train(data, num_epochs, num_data)
        num_train_samples = len(data)
        return comp, num_train_samples, update

    def test(self, model, train=True):
        """Tests self.model on self.eval_data.

        Return:
            dict of metrics returned by the model.
        """
        if train:
            return model.test(self.train_data)
        else:
            return model.test(self.eval_data)


    @property
    def num_test_samples(self):
        return len(self.eval_data['y'])

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model

    def process_x(self, raw_x_batch):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        return y_batch
