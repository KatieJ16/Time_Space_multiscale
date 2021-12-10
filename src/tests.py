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
import utils
import training_class

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

        np.save('train_data.npy', np.ones((1,1,4,4)))
        np.save('val_data.npy', np.ones((2,1,4,4)))
#===============================================================================
#tests for ResNet
    def test_resnet_make(self):
        #test a few things that resnet is right
        self.assertTrue(self.model.train_data.shape == (self.n_points, self.n_timesteps, 1, 1))
        self.assertTrue(self.model.train_data.shape == self.model.val_data.shape)

        #these will test form_data
        self.assertTrue(self.model.inputs.shape ==  (self.n_points * (self.n_timesteps-self.n_inputs), self.n_inputs))
        self.assertTrue(self.model.outputs.shape ==  (self.n_points * (self.n_timesteps-self.n_inputs), 1))

    def test_forward(self):
        #check that forward works and gets right shape
        n_points = 2
        x = torch.ones((n_points, self.n_inputs))
        x_forward = self.model(x)
        self.assertTrue(x_forward.shape == (n_points, 1))

    def test_train_below_threshold(self):
        #check that the error gets smaller than threshold
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        self.model.train_model(optimizer, loss_func, n_no_improve = 1)
        predict = self.model(self.model.inputs)
        end_loss = loss_func(predict, self.model.outputs)
        self.assertTrue(end_loss < self.model.threshold)


    def test_predict_mse(self):
        #test that the prediction and mse work

        #need to train first
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        self.model.train_model(optimizer, loss_func, n_no_improve = 1)

        predicted, mse = self.model.predict_mse()
        self.assertTrue(mse < 1e-6)
        self.assertTrue(predicted.shape == self.model.train_data.shape[:2])
#===============================================================================
#Tests for utils
    def test_power_2(self):
        #check the power of two function
        self.assertTrue(utils.isPowerOfTwo(128))
        self.assertFalse(utils.isPowerOfTwo(127))

    def test_average_one_level(self):
        averaged = utils.ave_one_level(torch.ones((1,1,2,2)))
        self.assertAlmostEqual(averaged[0,0,0,0],1)

    def test_make_dict(self):
        data = torch.ones((1,1,4,4))
        dict = utils.make_dict_all_sizes(data)
        self.assertTrue( len(dict) == 3)
        self.assertTrue(dict['4'].shape == (1,1,4,4))
        self.assertTrue(dict['2'].shape == (1,1,2,2))
        self.assertTrue(dict['1'].shape == (1,1,1,1))

    def test_find_error_4(self):
        resolved, loss, unresolved = utils.find_error_4(torch.ones((self.n_points, self.n_timesteps, 1, 1)),
                                                        self.model,
                                                        torch.ones((self.n_points, self.n_timesteps, 2, 2)))
        self.assertFalse(resolved)
        self.assertTrue( loss.shape == unresolved.shape)
        self.assertTrue(loss.shape == (2,2))

        resolved, loss, unresolved = utils.find_error_4(torch.ones((self.n_points, self.n_timesteps, 1, 1)),
                                                        self.model,
                                                        torch.ones((self.n_points, self.n_timesteps, 2, 2)), tol=10)
        self.assertTrue( resolved)

        #should throw an error when dim of data is larger than dim of truth data
        with self.assertRaises(Exception):
            data = torch.zeros((1,1,2,2))
            data = torch.zeros((1,1,1,1))
            find_error_4(data, self.model, truth_data)

    def test_mse(self):
        mse1 = utils.mse(torch.ones((20,500,2,2)),torch.ones((20,500,2,2)))
        self.assertTrue(mse1.shape == (2,2))
        self.assertAlmostEqual(mse1[0,0],0)
        self.assertAlmostEqual(mse1[0,1],0)

        mse1 = utils.mse(torch.ones((20,500,1,1)),torch.ones((20,500,2,2))*2)

        self.assertTrue(mse1.shape == (2,2))
        self.assertAlmostEqual(mse1[0,0],1)
        self.assertAlmostEqual(mse1[0,1],1)

    def test_grow(self):
        data = torch.ones(self.n_points, self.n_timesteps, 1, 1)
        x = utils.grow(data, self.max_dim)
        self.assertTrue(x.shape == (self.n_points, self.n_timesteps, self.max_dim, self.max_dim))

        data = torch.as_tensor([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]])
        x = utils.grow(data, self.max_dim)
        self.assertTrue( x.shape == (1, 1, self.max_dim, self.max_dim))
        x1 = np.random.randint(self.max_dim)
        x2 = np.random.randint(self.max_dim)
        self.assertTrue( x[0, 0, x1, x2] == data[0, 0, x1 // 32, x2 // 32])

    def test_find_error_1(self):
        data = torch.ones([10,100,1,1])
        mse, resolved = utils.find_error_1(data, self.model)

        #either form can be inputed
        data = torch.ones([10,100])
        mse, resolved = utils.find_error_1(data, self.model)

        #check with bad form of input
        with self.assertRaises(Exception):
            utils.find_error_1(torch.ones([1,2,3]), self.model)
            utils.find_error_1(torch.ones([1]), self.model)
            utils.find_error_1(torch.ones([1,2,3,4,5]), self.model)

    def test_make_and_load(self):
        train_dict, val_dict = utils.load_and_make_dict('.')

        self.assertTrue( len(train_dict) == 3)
        self.assertTrue(train_dict['4'].shape == (1,1,4,4))
        self.assertTrue(train_dict['2'].shape == (1,1,2,2))
        self.assertTrue(train_dict['1'].shape == (1,1,1,1))

        self.assertTrue( len(val_dict) == 3)
        self.assertTrue(val_dict['4'].shape == (2,1,4,4))
        self.assertTrue(val_dict['2'].shape == (2,1,2,2))
        self.assertTrue(val_dict['1'].shape == (2,1,1,1))
#==============================================================================
#tests for training_class

    def test_training_class(self):
        tol = 1e-8
        data_dir = '.'
        model_dir = '.'
        result_dir = '.'

        tc = training_class.training_class(data_dir, model_dir, result_dir, tol, n_inputs=3)

        current_size = 1
        tc, resolved = training_class.train_one_step(tc, current_size, make_new=True,
                                                     verbose=False,
                                                     start_k=2, largest_k=3,
                                                     plot_all_timesteps=False)

        current_size = 2
        tc, resolved = training_class.train_next_step(tc, current_size,
                                                      verbose=False, make_new=True,
                                                      start_k=2, largest_k=3)
#===============================================================================
#clean up after all tests are run
    @classmethod
    def tearDownClass(cls):
        os.remove('model.pt')
        os.remove('train_data.npy')
        os.remove('val_data.npy')
