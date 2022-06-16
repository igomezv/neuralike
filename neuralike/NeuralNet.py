"""neural networks to neuralike.
Author: Isidro Gómez-Vargas (igomez@icf.unam.mx)
Date: April 2022
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from time import time
import math

import torch
from torch import nn
from torchinfo import summary
from torch_optimizer import AdaBound
# from pytorchtools import EarlyStopping


class NeuralNet:
    def __init__(self, load=False, model_path=None, X=None, Y=None, topology=None, **kwargs):
        """
        Read the network params
        Parameters
        -----------
        load: bool
            if True, then use an existing model
        X, Y: numpy array
            Data to train

        """
        self.load = load
        self.model_path = model_path
        self.topology = topology
        self.dims = topology[0]
        self.epochs = kwargs.pop('epochs', 50)
        self.learning_rate = kwargs.pop('learning_rate', 5e-4)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.patience = kwargs.pop('patience', 5)
        psplit = kwargs.pop('psplit', 0.7)
        if load:
            self.model = self.load_model()
            self.model.summary()
        else:

            # percentage = 0.05
            nrows, ncols = X.shape
            # noise_x = np.zeros(X.shape)
            # for col in range(ncols):
            #     noise_x[:, col] = np.random.normal(0, X[:, col].std(), nrows) * percentage
            # noise_y = np.random.normal(0, Y.std(), Y.shape) * percentage
            # X_r = X + noise_x
            # Y_r = Y + noise_y
            # X = np.concatenate((X_r, X), axis=0)
            # Y = np.concatenate((Y_r, Y), axis=0)
            ntrain = int(psplit * len(X))
            indx = [ntrain]
            shuffle = np.random.permutation(len(X))
            X = X[shuffle]
            Y = Y[shuffle]
            self.X_train, self.X_test = np.split(X, indx)
            self.Y_train, self.Y_test = np.split(Y, indx)

            ntrain_valtest = int(0.5 * len(self.Y_test))
            indx_valtest = [ntrain_valtest]
            self.X_val, self.X_test = np.split(self.X_test, indx_valtest)
            self.Y_val, self.Y_test = np.split(self.Y_test, indx_valtest)
            # Initialize the MLP
            self.model = MLP(self.dims, self.topology[-1])
            self.model.float()
        print("Neuralike: Shape of X dataset: {} | Shape of Y dataset: {}".format(X.shape, Y.shape))
        print("Neuralike: Shape of X_val dataset: {} | Shape of X_test dataset: {}".format(self.X_val.shape, self.X_test.shape))

    def train(self):
        dataset_train = LoadDataSet(self.X_train, self.Y_train)
        dataset_val = LoadDataSet(self.X_val, self.Y_val)

        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=1)
        validloader = torch.utils.data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=True, num_workers=1)

        # Define the loss function and optimizer
        # loss_function = nn.L1Loss()
        loss_function = nn.MSELoss()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate, weight_decay=0.05)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-5)
        optimizer = AdaBound(self.model.parameters(), lr=self.learning_rate, final_lr=0.01, weight_decay=1e-10)
        # optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
        #                                 lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.05, patience=5)
        # try:
        summary(self.model)
        t0 = time()
        # Run the training loop
        history_train = np.empty((1,))
        history_val = np.empty((1,))
        for epoch in range(0, self.epochs):
            # Print epoch
            # print(f'Starting epoch {epoch + 1}', end=' ')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):

                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)
                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 10 == 0:
                    # print('Loss after mini-batch %5d: %.3f' %
                    #       #                 (i + 1, current_loss / 500))
                    #       (i + 1, loss.item()), end='\r')
                    current_loss = 0.0
            history_train = np.append(history_train, current_loss)

            valid_loss = 0.0
            self.model.eval()  # Optional when not using Model Specific layer
            for i, data in enumerate(validloader, 0):
                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))

                output_val = self.model(inputs)
                valid_loss = loss_function(output_val, targets)
                valid_loss += loss.item()
                # scheduler.step(valid_loss)

            history_val = np.append(history_val, valid_loss.item())
            print('Epoch: {}/{} | Training Loss: {:.5f} | Validation Loss:'
                  '{:.5f}'.format(epoch+1, self.epochs, loss.item(), valid_loss.item()), end='\r')
        # Process is complete.
        tf = time() - t0
        print('\nTraining process has finished in {:.3f} minutes.'.format(tf/60))
        self.history = {'loss': history_train, 'val_loss': history_val}
        self.loss_val = history_val[-5:]
        self.loss_train = history_train[-5:]
        return self.history

    def predict(self, x):
        x = torch.from_numpy(x).float()
        prediction = self.model.forward(x)
        return prediction.detach().numpy()

    def plot(self, save=False, figname=False, ylogscale=False, show=False):
        plt.plot(self.history['loss'], label='training set')
        plt.plot(self.history['val_loss'], label='validation set')
        if ylogscale:
            plt.yscale('log')
        plt.title('MSE train: {:.4f} | MSE val: {:.4f} | '
                  'MSE test: {:.4f}'.format(self.loss_train[-1],
                                             self.loss_val[-1],
                                             self.test_mse()))
        plt.ylabel('loss function')
        plt.xlabel('epoch')
        plt.xlim(0, self.epochs)
        plt.legend(['train', 'val'], loc='upper left')
        if save and figname:
            plt.savefig(figname)
        if show:
            plt.show()
        return True

    def test_mse(self):
        y_pred = self.predict(self.X_test)
        mse = ((self.Y_test - y_pred) ** 2).mean()
        return mse

    # def tunning(self):


class LoadDataSet:
    def __init__(self, X, y, scale_data=False):
        """
        Prepare the dataset for regression
        """
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # # Apply scaling if necessary
            # if scale_data:
            #     X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MLP(nn.Module):
    def __init__(self, ncols, noutput, numneurons=200, dropout=0.1):
        """
            Multilayer Perceptron for regression.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(ncols, numneurons),
            # nn.SELU(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(numneurons, numneurons),
            # nn.SELU(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(numneurons, numneurons),
            # nn.SELU(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(numneurons, noutput)
        )
        # use the modules apply function to recursively apply the initialization
        self.model.apply(self.init_normal)

    def forward(self, x):
        '''
          Forward pass
        '''
        return self.model(x)

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)


class L1(torch.nn.Module):
    def __init__(self, module, weight_decay=1e-5):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)