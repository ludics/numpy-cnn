#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8

import numpy as np
import os
from utils.load_mnist import get_mnist
from utils.mylog import Logger


if not os.path.exists('./logs'):
    os.makedirs('./logs')
log = Logger('./logs/train.log',level='debug').logger

if __name__ == "__main__":

    log.info('Just test')

