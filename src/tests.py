"""
tests.py
Tests for Musical Robots
"""
import unittest
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

module_path = os.path.abspath(os.path.join('./src/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import ResNet

# import warnings

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #set some variables that will always be the same
        cls.n_points = 10
        cls.n_timesteps = 500
        cls.max_dim = 128
        cls.n_inputs = 3

        #make a resnet object that will be used throughout
        cls.model = ResNet.ResNet(torch.ones((cls.n_points, cls.n_timesteps, 1, 1)),
                                  torch.ones((cls.n_points, cls.n_timesteps, 1, 1)),
                                  n_inputs = cls.n_inputs)

    def test_resnet_make(self):
        #test a few things that resnet is right
        assert self.model.train_data.shape == (self.n_points, self.n_timesteps, 1, 1)
        assert self.model.train_data.shape == self.model.val_data.shape

        #these will test form_data
        assert self.model.inputs.shape ==  (self.n_points *( self.n_timesteps-self.n_inputs), self.n_inputs)
        assert self.model.outputs.shape ==  (self.n_points *( self.n_timesteps-self.n_inputs), 1)

    def test_forward(self):
        #check that forward works and gets right shape
        n_points = 2
        x = torch.ones((n_points, self.n_inputs))
        x_forward = self.model(x)
        assert x_forward.shape == (n_points, 1)

    def test_train_below_threshold(self):
        #check that the error gets smaller than threshold
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        self.model.train_model(optimizer, loss_func, n_no_improve = 1)
        predict = self.model(self.model.inputs)
        end_loss = loss_func(predict, self.model.outputs)
        assert end_loss < self.model.threshold


    def test_predict_mse(self):
        #test that the prediction and mse work

        #need to train first
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        self.model.train_model(optimizer, loss_func, n_no_improve = 1)

        predicted, mse = self.model.predict_mse()
        assert mse < 1e-6
        assert predicted.shape == self.model.train_data.shape[:2]
