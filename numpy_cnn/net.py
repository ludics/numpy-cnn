#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8


import numpy as np
import copy
from .layers import *


class Net(Layer):
    def __init__(self, layers):
        super().__init__("Net")
        self.layers = layers
        self.parameters = []
        for layer in self.layers:
            self.get_params(layer)
        self._phase = "TRAIN"

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        layer_grads = []
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params(self, layer):
        if layer.name == "Linear":
            self.parameters.append(layer.params['w'])
            self.parameters.append(layer.params['b'])

    def __repr__(self):
        ret_str = '\n\tNet arch:\n'
        for layer in self.layers:
            ret_str += '\t\t' + layer.__repr__() + '\n'
        return ret_str


