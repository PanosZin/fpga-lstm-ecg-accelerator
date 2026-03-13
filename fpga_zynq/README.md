# FPGA Zynq Integration

This folder contains the bare-metal application used to run the LSTM accelerator on the Zynq UltraScale+ ZCU104 platform.

The application runs on the Cortex-A53 processing system and performs the system-level control required to execute the FPGA accelerator.

## Main Responsibilities

- Configure the HLS-generated LSTM IP through its AXI-Lite control interface
- Stream quantized ECG input samples to the accelerator using AXI-DMA MM2S
- Receive predicted labels from the accelerator using AXI-DMA S2MM
- Manage cache coherency for DDR-resident input/output buffers
- Measure total inference time using the ARM generic timer
- Compute and print a confusion matrix and per-class accuracy on hardware results

## Deployment Flow

For each burst:

1. Flush input buffers and invalidate output buffers
2. Configure `numSamples` and `chunkSize`
3. Start the HLS accelerator
4. Send input samples through DMA
5. Receive predictions through DMA
6. Wait for the accelerator to return to idle
7. Repeat until all samples are processed

## Notes

- The application targets a bare-metal environment on the Zynq A53
- Dataset binaries stored in DDR are not included in the public repository
- This folder represents the final on-board execution stage of the full project pipeline