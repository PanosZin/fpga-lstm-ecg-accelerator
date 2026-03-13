## Quantization-Aware Training Pipeline

The script `train_qat_and_export.py` performs quantization-aware training (QAT) of the LSTM model using PyTorch FX.  
Its purpose is to prepare the neural network for efficient fixed-point inference on FPGA.

The pipeline includes:

- Loading ECG training and test segments from `.mat` datasets
- Applying weighted sampling to balance waveform classes
- Training an LSTM model using FX-based quantization-aware training
- Evaluating the trained model in:
  - floating-point mode
  - fake-quantized mode
  - INT8 quantized mode
- Collecting activation and cell-state statistics
- Suggesting a suitable `ap_fixed<N,I>` datatype for FPGA deployment
- Exporting trained parameters as HLS-compatible C++ header files

These exported headers are used directly by the FPGA inference implementation.

---

## Output Files

After training, the script generates the following files inside the output directory:

| File | Description |
|-----|-------------|
| `W_all_fixed.h` | Combined LSTM input and recurrent weights |
| `B_all_fixed.h` | LSTM bias vector |
| `W_fc_fixed.h` | Fully-connected layer weights |
| `b_fc_fixed.h` | Fully-connected layer bias |
| `qat_stats.json` | Activation and cell-state statistics |
| `act_ranges.json` | Recorded activation ranges |
| `datatype_suggestion.txt` | Suggested fixed-point datatype for FPGA |

---

## Required Input Files

The following files are required but **not included in this repository**:

- `rawNetWeights.mat` – MATLAB-exported LSTM weights  
- `qat_data.mat` – Preprocessed ECG dataset

Update the paths in the configuration block before running the script:

```python
class Cfg:
    WEIGHTS_MAT = "path/to/rawNetWeights.mat"
    CLEAN_DATA  = "path/to/qat_data.mat"