// testbench.cpp
// Pure C++ testbench for the floating-point LSTM reference implementation.
// Loads ECG input samples and ground-truth labels from binary files,
// runs inference segment by segment with LSTM state reset per segment,
// and reports per-class accuracy, confusion matrix.
#include "lstm_forward.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdint>

using namespace std;

//-----------------------------------------------------------------------------  
// Input dataset binaries are not included in the public repository.
// Update these paths locally before running the testbench.
static const char* INPUT_BIN = "data/cleanTestData_float32.bin";
static const char* GROUND_TRUTH = "data/raw_GroundTruth.bin";
static const int   SEGMENT_LEN = 5000;   // samples per segment
static const int   CHUNK_SIZE = 5000;   // as used in forward pass
static const int   N_SEGMENTS = 2835;   // number of segments

// load raw float32 data into vector<float>
bool load_bin(const char* path, vector<float>& out) {
    ifstream f(path, ios::binary | ios::ate);
    if (!f) return false;
    auto size = f.tellg();
    f.seekg(0, ios::beg);
    size_t n = size / sizeof(float);
    out.resize(n);
    f.read(reinterpret_cast<char*>(out.data()), n * sizeof(float));
    return true;
}

// load ground truth uint8 labels
bool load_gt_bin(const char* path, vector<uint8_t>& out) {
    ifstream f(path, ios::binary);
    if (!f) return false;
    out.assign((istreambuf_iterator<char>(f)), {});
    return true;
}

int main() {
    // 1) load ECG test data as floats
    vector<float> ecg;
    if (!load_bin(INPUT_BIN, ecg)) {
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
    int total_samples = N_SEGMENTS * SEGMENT_LEN;
    if ((int)ecg.size() < total_samples || (int)gt.size() < total_samples) {
        cerr << "ERROR: not enough data/GT for " << N_SEGMENTS << " segments\n";
        return 1;
    }

    // 3) run each segment separately to reset LSTM state per segment
    vector<int> all_pred;
    all_pred.reserve(total_samples);
    for (int seg = 0; seg < N_SEGMENTS; ++seg) {
        auto start = ecg.begin() + seg * SEGMENT_LEN;
        auto end = start + SEGMENT_LEN;
        vector<float> slice(start, end);

        vector<int> pred_seg;
        lstm_forward(slice, pred_seg, SEGMENT_LEN, CHUNK_SIZE);

        all_pred.insert(all_pred.end(), pred_seg.begin(), pred_seg.end());
    }



    // --- print first/last 100 predicted labels ---
    cout << "First 100 predictions:\n";
    for (int i = 0; i < 100 && i < total_samples; ++i) {
        cout << all_pred[i] << ' ';
    }
    cout << "\n\nLast 100 predictions:\n";
    for (int i = max(0, total_samples - 100); i < total_samples; ++i) {
        cout << all_pred[i] << ' ';
    }
    cout << "\n\n";

    // 4) compute confusion + accuracy
    const int CL = OUTPUT_SIZE;
    vector<vector<int>> cm(CL, vector<int>(CL, 0));
    vector<int> counts(CL, 0);
    for (int i = 0; i < total_samples; ++i) {
        int a = static_cast<int>(gt[i]);
        int p = all_pred[i];
        cm[a][p]++;
        counts[a]++;
    }

    // 5) write results to file
    ofstream r("results_confusion.txt");
    r << fixed << setprecision(2);

    r << "Per-class accuracy:\n";
    const char* names[4] = { "n/a","P","QRS","T" };
    for (int c = 0; c < CL; ++c) {
        int correct = cm[c][c];
        int tot = counts[c];
        double pct = tot ? (100.0 * correct / tot) : 0.0;
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

