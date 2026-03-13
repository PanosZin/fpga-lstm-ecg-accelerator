# FPGA Implementation

This folder contains documentation and figures related to the FPGA deployment of the LSTM ECG segmentation accelerator.

The accelerator was implemented using **Vitis HLS** and integrated into a **Zynq UltraScale+ MPSoC (ZCU104)** platform.

The system streams ECG samples through AXI DMA into the programmable logic, where the LSTM inference accelerator processes the data.

---

## System Architecture

The FPGA design integrates the HLS LSTM accelerator with the Zynq processing system using AXI interfaces.

![Zynq Block Design](images/zynq_block_design.jpg)

---

## HLS Dataflow Architecture

The accelerator uses a streaming pipeline implemented using the HLS `DATAFLOW` pragma.

![Dataflow](images/dataflow.jpg)

---

## FPGA Execution Results

The system was evaluated using 14,175,000 ECG samples (2835 segments × 5000 samples per segment).

![FPGA results](images/fpga_results.jpg)

The results include:

- per-class accuracy
- confusion matrix
- total inference runtime