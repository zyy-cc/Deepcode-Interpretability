# Deepcode-Interpretability

## Overview
This repository contains the codes for the paper "Interpreting Deepcode, a learned feedback code". It is a non-linear interpretable model for AWGN channel with feedback.

## Structure

### noiseless_feedback
The `noiseless_feedback` folder contains:

- `deepcode` - Implementation of the Deepcode using Pytorch based on Tensorflow Deepcode.
- `interpretable` - Interpretable model based on Deepcode: 5 hidden states (encoder + decoder) / 7 hidden states (encoder)
- `equivalent` - Contains equivalent expression of interpretable model

### noisy_feedback
The `noisy_feedback` folder contains the model when the feedback is noisy.

## Prerequisites
Python environment:
- Python 3.10.8
- numpy: 1.23.4
- pytorch: 1.12.1

original TensorFlow Deepcode: https://github.com/hyejikim1/Deepcode

