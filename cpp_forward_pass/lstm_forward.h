// lstm_forward.h
//
// Floating-point reference implementation of the LSTM model used for
// ECG waveform segmentation. This version was used to validate the
// network behavior before quantization and FPGA deployment.
#ifndef LSTM_FORWARD_H
#define LSTM_FORWARD_H

#include <vector>

//-------------------------------------------------------------------------
// Model Dimensions
//-------------------------------------------------------------------------
constexpr int INPUT_SIZE   = 1;
constexpr int HIDDEN_SIZE  = 200;
constexpr int OUTPUT_SIZE  = 4;

//-------------------------------------------------------------------------
// Include all LSTM weights and biases
//-------------------------------------------------------------------------
// Contains all trained LSTM weights and biases exported from MATLAB
#include "lstm_weights.h"  // Defines Wi_input, Wi_recurrent, Wi_bias, etc., and W_fc, b_fc

//-------------------------------------------------------------------------
// Forward declaration of the pure-C++ LSTM forward pass
//-------------------------------------------------------------------------
// ecg: input signal of length numSamples
// pred: output vector (will be resized to numSamples)
// numSamples: total number of samples to process
// chunkSize: processing chunk size (ping-pong buffer granularity)
void lstm_forward(
    const std::vector<float>& ecg,
    std::vector<int>&         pred,
    int                       numSamples,
    int                       chunkSize
);

#endif // LSTM_FORWARD_H

