// lstm_forward.h
// Type definitions and top-level interface for the HLS LSTM accelerator.

#ifndef LSTM_FORWARD_H
#define LSTM_FORWARD_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

//-------------------------------------------------------------------------
// Dimensions 
//-------------------------------------------------------------------------
#define INPUT_SIZE   1
#define HIDDEN_SIZE  200
#define OUTPUT_SIZE  4
//-------------------------------------------------------------------------
// Quantized and fixed-point data types
//-------------------------------------------------------------------------
typedef ap_int<8>     input_t;       // raw int8 ECG samples
// Use ap_fixed to retain fractional bits for interpolation
typedef ap_fixed<16,6> state_t;         // hidden / cell state with 10 fractional bits
typedef ap_fixed<32,24> acc_t;         // MAC accumulator
typedef ap_fixed<26,14> cell_t;       // cell-state accumulator/storage
// Wider fixed-point for sum in softmax (more fraction bits)
typedef ap_fixed<24,6> sum_t; // softmax accumulation
// Define a 8-bit AXI-Stream packet with no user/id/dest bits
typedef ap_axiu<8,0,0,0>  in_pkt_t;		    // 8-bit data, no side-channels
typedef ap_axiu<8, 0, 0, 0>  out_pkt_t;    // 8-bit AXI packet carrying the predicted class label
// Top-level streaming forward pass
void lstm_forward(
    hls::stream<in_pkt_t>&  ecgSignal,
    hls::stream<out_pkt_t>& predictedLabels,
    int                     numSamples,
    int                     chunkSize);

#endif // LSTM_FORWARD_H
