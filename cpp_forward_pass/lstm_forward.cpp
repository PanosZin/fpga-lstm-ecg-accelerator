// lstm_forward.cpp
// Floating-point C++ reference implementation of the LSTM forward pass
// used to reproduce the MATLAB ECG segmentation model before quantization
// and FPGA deployment.
#include "lstm_forward.h"
#include <algorithm>
#include <cmath>

//-------------------------------------------------------------------------
// Activation functions
//-------------------------------------------------------------------------
static float sigmoid_fast(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float tanh_fast(float x) {
    return std::tanh(x);
}

//-------------------------------------------------------------------------
// Softmax + argmax
//-------------------------------------------------------------------------
static void softmax(const float in[OUTPUT_SIZE], float out[OUTPUT_SIZE]) {
    float m = *std::max_element(in, in + OUTPUT_SIZE);
    float sum = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        out[i] = std::exp(in[i] - m);
        sum += out[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        out[i] /= sum;
    }
}

static int argmax(const float out[OUTPUT_SIZE]) {
    return static_cast<int>(std::distance(out, std::max_element(out, out + OUTPUT_SIZE)));
}

//-------------------------------------------------------------------------
// Fully-connected + softmax
//-------------------------------------------------------------------------
static void fully_connected_and_softmax(const float hid[HIDDEN_SIZE],
                                        float       out[OUTPUT_SIZE])
{
    float tmp[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        float acc = b_fc[i];
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            acc += W_fc[i][j] * hid[j];
        }
        tmp[i] = acc;
    }
    softmax(tmp, out);
    
}

//-------------------------------------------------------------------------
// Single LSTM cell (I-F-O-G gate order)
//-------------------------------------------------------------------------
static void lstm_cell(const float input[INPUT_SIZE],
                      const float hid_i[HIDDEN_SIZE],
                      const float cell_i[HIDDEN_SIZE],
                            float hid_o[HIDDEN_SIZE],
                            float cell_o[HIDDEN_SIZE])
{
    float x = input[0];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        // Input gate
        float gi = Wi_bias[i] + Wi_input[i][0] * x;
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            gi += Wi_recurrent[i][j] * hid_i[j];
        gi = sigmoid_fast(gi);

        // Forget gate
        float gf = Wf_bias[i] + Wf_input[i][0] * x;
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            gf += Wf_recurrent[i][j] * hid_i[j];
        gf = sigmoid_fast(gf);

        // Output gate
        float go = Wo_bias[i] + Wo_input[i][0] * x;
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            go += Wo_recurrent[i][j] * hid_i[j];
        go = sigmoid_fast(go);

        // Cell candidate (G)
        float gg = Wc_bias[i] + Wc_input[i][0] * x;
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            gg += Wc_recurrent[i][j] * hid_i[j];
        gg = tanh_fast(gg);

        // Update cell state and hidden state
        float next_c = gf * cell_i[i] + gi * gg;
        cell_o[i]    = next_c;
        hid_o[i]     = go * tanh_fast(next_c);

       
    }
}

//-------------------------------------------------------------------------
// Top-level forward pass implementation
//-------------------------------------------------------------------------
void lstm_forward(const std::vector<float>& ecg,
                  std::vector<int>&         pred,
                  int                       numSamples,
                  int                       chunkSize)
{
    pred.resize(numSamples);
    std::vector<float> hid_0(HIDDEN_SIZE, 0.0f), hid_1(HIDDEN_SIZE, 0.0f);
    std::vector<float> cell_0(HIDDEN_SIZE, 0.0f), cell_1(HIDDEN_SIZE, 0.0f);

    int done = 0;
    while (done < numSamples) {
        int thisChunk = std::min(chunkSize, numSamples - done);
        for (int t = 0; t < thisChunk; ++t) {
            float inb[INPUT_SIZE] = { ecg[done + t] };
            float logits[OUTPUT_SIZE];
            if (((done + t) & 1) == 0) {
                lstm_cell(inb,
                          hid_0.data(), cell_0.data(),
                          hid_1.data(), cell_1.data());
                fully_connected_and_softmax(hid_1.data(), logits);
            } else {
                lstm_cell(inb,
                          hid_1.data(), cell_1.data(),
                          hid_0.data(), cell_0.data());
                fully_connected_and_softmax(hid_0.data(), logits);
            }
            pred[done + t] = argmax(logits);
        }
        done += thisChunk;
    }
}

