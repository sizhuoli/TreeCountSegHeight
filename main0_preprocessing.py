#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:19:53 2021

@author: sizhuo
"""


from config_cisong.Preprocessing import Configuration
from core2.preprocessing import Processor

<<<<<<< HEAD
config = Configuration()
prep = Processor(config, boundary = True, aux = True)
=======
config = Preprocessing.Configuration()

prep = Processor(config,boundary = True, aux = True)
>>>>>>> b82a14d0f9e8e5af0a94c0928df298d9a7940e77
prep.extract_training_sets()

# # no boundary
# prep = processor(config, boundary = False, aux = True)
# prep.extract_training_sets()

# # svls (Spatially Varying Label Smoothing)
# prep = processor(config, boundary = False, aux = True)
# prep.extract_svls

