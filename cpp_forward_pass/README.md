## C++ Forward Pass

This folder contains the floating-point C++ reference implementation of the LSTM model used for ECG waveform segmentation.

The purpose of this stage was to reproduce the MATLAB reference forward pass in C++ and validate the model behavior before applying quantization and mapping the design to Vitis HLS.

The testbench loads ECG samples and ground-truth labels from binary files and evaluates the model by generating per-class accuracy and a confusion matrix.