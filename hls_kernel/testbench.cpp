// testbench.cpp
// Vitis HLS testbench for the streaming LSTM accelerator.
// Loads quantized ECG input samples and ground-truth labels from binary files,
// drives the AXI-stream interface, collects predictions, and reports
// per-class accuracy and a confusion matrix.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "hls_stream.h"
#include "lstm_forward.h"

using namespace std;

//-----------------------------------------------------------------------------  
// Input files are not included in the public repository.
// Update these paths locally before running the testbench.
static const char* INPUT_BIN = "data/raw_testData_int8.bin";
static const char* GROUND_TRUTH = "data/raw_GroundTruth.bin";
static const int   SEGMENT_LEN  = 5000;       // samples per segment
static const int   CHUNK_SIZE   = 5000;      // value sent to the new port

// Load quantized ECG samples from a binary file
bool load_bin(const char *path, vector<int8_t> &out) {
    ifstream f(path, ios::binary);
    if (!f) return false;
    out.assign(istreambuf_iterator<char>(f), {});
    return true;
}

// Load ground-truth labels from a binary file
bool load_gt_bin(const char *path, vector<uint8_t> &out) {
    ifstream f(path, ios::binary);
    if (!f) return false;
    out.assign(istreambuf_iterator<char>(f), {});
    return true;
}

int main() {
    // how many segments to run?
    const int nSegments = 2835; // number of 5000-sample ECG segments to test

    // 1) load ECG test data
    vector<int8_t> data;
    if (!load_bin(INPUT_BIN, data)) {
        cerr << "Cannot open INPUT_BIN: " << INPUT_BIN << "\n";
        return 1;
    }

    // 2) load GT labels
    vector<uint8_t> gt;
    if (!load_gt_bin(GROUND_TRUTH, gt)) {
        cerr << "Cannot open GROUND_TRUTH: " << GROUND_TRUTH << "\n";
        return 1;
    }

    // verify we have enough samples
    int total_samples = nSegments * SEGMENT_LEN;
    if ((int)data.size() < total_samples || (int)gt.size() < total_samples) {
        cerr << "ERROR: not enough data/GT for " << nSegments << " segments\n";
        return 1;
    }

    // 3) prepare HLS streams
    hls::stream<in_pkt_t>  istream; // input stream
    hls::stream<out_pkt_t> ostream; // output stream

    // feed input stream
    for (int s = 0; s < nSegments; ++s) {
        for (int i = 0; i < SEGMENT_LEN; ++i) {
            in_pkt_t pkt;
            // cast int8_t -> ap_uint<8> (preserving two's-complement)
            pkt.data = (ap_uint<8>)(uint8_t)data[s*SEGMENT_LEN + i];
            pkt.keep = (ap_uint<8>)(-1);
            pkt.strb = (ap_uint<8>)(-1);
            pkt.last = (i == SEGMENT_LEN-1) ? 1 : 0;
            istream.write(pkt);
        }
    }

    // 4) invoke DUT with chunkSize
    lstm_forward(istream, ostream, total_samples, CHUNK_SIZE);

    // 5) collect predictions
    vector<int> pred(total_samples);
    for (int i = 0; i < total_samples; ++i) {
        auto o = ostream.read();
        pred[i] = (int)o.data.to_uint();
    }

    // --- print first/last 100 predicted labels ---
    cout << "First 100 predictions:\n";
    for (int i = 0; i < 100 && i < total_samples; ++i) {
        cout << pred[i] << ' ';
    }
    cout << "\n\nLast 100 predictions:\n";
    for (int i = max(0, total_samples-100); i < total_samples; ++i) {
        cout << pred[i] << ' ';
    }
    cout << "\n\n";

    // 6) compute confusion + accuracy
    const int CL = OUTPUT_SIZE;
    vector<vector<int>> cm(CL, vector<int>(CL, 0));
    vector<int> counts(CL, 0);
    for (int i = 0; i < total_samples; ++i) {
        int a = gt[i], p = pred[i];
        cm[a][p]++;
        counts[a]++;
    }

    // 7) write results to file
    ofstream r("results_confusion.txt");
    r << fixed << setprecision(2);

    r << "Per-class accuracy:\n";
    const char* names[4] = {"n/a","P","QRS","T"};
    for (int c = 0; c < CL; ++c) {
        int correct = cm[c][c];
        int tot     = counts[c];
        double pct  = tot ? (100.0 * correct / tot) : 0.0;
        r << setw(4) << names[c] << " : "
          << setw(6) << pct << "%  ("
          << correct << "/" << tot << ")\n";
    }

    r << "\nConfusion Matrix (Actual vs. Predicted) - Row-Normalized [%]:\n";
    r << "               ";
    for (int c = 0; c < CL; ++c) r << setw(8) << names[c];
    r << "\n" << "--------------------------------------------------------\n";
    for (int a = 0; a < CL; ++a) {
        r << "Actual " << setw(3) << names[a] << " |";
        for (int p = 0; p < CL; ++p) {
            double pct = counts[a] ? (100.0 * cm[a][p] / counts[a]) : 0.0;
            r << setw(8) << pct << "%";
        }
        r << "\n";
    }
    r.close();

    cout << "[TB] Done. See results_confusion.txt\n";
    return 0;
}
