# HLS Kernel

This folder contains the Vitis HLS implementation of the LSTM accelerator used for ECG waveform segmentation on FPGA.

The HLS design was derived from the validated floating-point C++ reference model, then restructured for hardware synthesis using fixed-point arithmetic, AXI streaming, LUT-based nonlinearities, and explicit HLS optimization directives.

## Main Design Features

- AXI-Stream input/output interfaces
- AXI-Lite control interface
- Fixed-point arithmetic using `ap_fixed`
- Separate datatypes for input, hidden state, cell state, accumulation, and softmax sum
- LUT-based approximations for:
  - sigmoid
  - tanh
  - exp
- `DATAFLOW` decomposition into reader / compute / writer stages
- Explicit use of HLS directives such as:
  - `PIPELINE`
  - `UNROLL`
  - `ARRAY_PARTITION`
  - `STREAM`
  - `bind_storage`

## File Overview

- `lstm_forward.cpp`  
  Main HLS kernel implementation

- `lstm_forward.h`  
  Top-level interface, datatype definitions, and AXI packet types

- `testbench.cpp`  
  Vitis HLS testbench used to feed binary ECG data, collect predictions, and compute per-class accuracy and a confusion matrix

- `tanh_lut.h`, `sigmoid_lut.h`, `exp_lut.h`  
  Lookup tables used to approximate nonlinear activation functions efficiently in hardware

## Top-Level Architecture

The design is organized into three streaming stages:

1. **unpack_axis**  
   Reads ECG samples from the AXI input stream and converts them to the internal sample stream

2. **compute**  
   Executes the fixed-point LSTM forward pass and fully connected classification layer

3. **pack_axis**  
   Packs predicted labels into AXI output packets

These stages are connected through internal HLS streams and wrapped inside a `DATAFLOW` region.

## Numerical Design

The floating-point model was converted into a fixed-point representation for FPGA deployment.

The implementation uses:

- quantized input samples
- fixed-point hidden and cell states
- fixed-point accumulators
- LUT-based nonlinearities instead of direct floating-point function evaluation

This design choice reduces hardware cost while preserving acceptable segmentation accuracy.

## Memory Mapping

The exported network parameters are stored as constant arrays and mapped to FPGA memory resources using HLS storage directives.

The implementation binds:

- LSTM weights
- LSTM biases
- fully connected weights
- fully connected biases

to ROM/BRAM structures for efficient hardware access.

## Verification

The kernel was verified in Vitis HLS using a dedicated testbench that:

- loads quantized ECG input samples from binary files
- loads ground-truth labels
- feeds the input AXI stream
- reads predicted output labels
- computes per-class accuracy
- generates a row-normalized confusion matrix

## Notes

- Input dataset binaries are not included in the public repository
- Generated weight headers may be omitted or reduced in the public version depending on repository size
- This folder focuses on the FPGA/HLS implementation stage of the complete project pipeline