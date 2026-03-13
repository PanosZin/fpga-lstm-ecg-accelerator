// lstm_forward.cpp
// HLS implementation of the LSTM accelerator for ECG waveform segmentation.
// Uses AXI-Stream interfaces, fixed-point arithmetic, LUT-based nonlinearities,
// and dataflow decomposition for FPGA synthesis.

#include "lstm_forward.h"
#include "hls_math.h"
#include "W_all_fixed.h"
#include "B_all_fixed.h"
#include "W_fc_fixed.h"
#include "b_fc_fixed.h"
#include "tanh_lut.h"
#include "exp_lut.h"
#include "sigmoid_lut.h"

// ───────────────────── forward declarations ────────────────────────────
static state_t sigmoid_fast(acc_t x);
static state_t tanh_fast   (acc_t x);
static state_t exp_fast    (acc_t x);

static void softmax(const state_t in[OUTPUT_SIZE], state_t out[OUTPUT_SIZE]);
static int  argmax (const state_t out[OUTPUT_SIZE]);

static void fully_connected_and_softmax(const state_t hid[HIDDEN_SIZE],
                                        state_t       out[OUTPUT_SIZE]);

static void lstm_cell(const input_t  input[INPUT_SIZE],
                      const state_t  hid_i[HIDDEN_SIZE],
                      const cell_t   cell_i[HIDDEN_SIZE],
                            state_t  hid_o[HIDDEN_SIZE],
                            cell_t   cell_o[HIDDEN_SIZE]);


// ─────────────────────── AXI helpers ──────────────────────────────────
static void unpack_axis(hls::stream<in_pkt_t>& axis_in,
                        hls::stream<input_t>&  raw,
                        int                    nSteps)
{
READ_LOOP:
    for (int i = 0; i < nSteps; ++i) {
	#pragma HLS PIPELINE II=1
        in_pkt_t p = axis_in.read();
        raw.write((input_t)p.data);
    }
}

static void pack_axis(hls::stream<int>&       lbl,
                      hls::stream<out_pkt_t>& axis_out,
                      int                     nSteps)
{
WRITE_LOOP:
    for (int i = 0; i < nSteps; ++i) {
	#pragma HLS PIPELINE II=1
        out_pkt_t p;
        p.data = (ap_uint<3>)lbl.read();
        p.keep = p.strb = 1;
        p.last = (i == nSteps-1);
        axis_out.write(p);
    }
}

//────────────────────── compute() ────────────────────────────────────────────
static void compute(hls::stream<input_t>& raw,
                    hls::stream<int>&     labels,
                    int                   nSteps)
{
    static state_t hid_0[HIDDEN_SIZE], hid_1[HIDDEN_SIZE];
    static cell_t  cell_0[HIDDEN_SIZE], cell_1[HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=hid_0  complete
#pragma HLS ARRAY_PARTITION variable=hid_1  complete
#pragma HLS ARRAY_PARTITION variable=cell_0 complete
#pragma HLS ARRAY_PARTITION variable=cell_1 complete
#pragma HLS reset variable=hid_0
#pragma HLS reset variable=cell_0
#pragma HLS reset variable=hid_1
#pragma HLS reset variable=cell_1

RESET_LOOP:
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
	#pragma HLS UNROLL
        hid_0[i]  = hid_1[i]  = 0;
        cell_0[i] = cell_1[i] = 0;
    }

TIME_LOOP:
    for (int t = 0; t < nSteps; ++t) {
        // Full II=1 pipelining of TIME_LOOP was explored but is not currently feasible
        // with the present resource/performance tradeoff.
        input_t inb[1];
        inb[0] = raw.read();

        state_t logits[OUTPUT_SIZE];
        if ((t & 1) == 0) {
            lstm_cell(inb, hid_0, cell_0, hid_1, cell_1);
            fully_connected_and_softmax(hid_1, logits);
        } else {
            lstm_cell(inb, hid_1, cell_1, hid_0, cell_0);
            fully_connected_and_softmax(hid_0, logits);
        }
        labels.write(argmax(logits));
    }
}

// ────────────────────── data-flow wrapper ─────────────────────────────────
static void lstm_forward_chunk(hls::stream<in_pkt_t>&  ecg,
                               hls::stream<out_pkt_t>& pred,
                               int                     nSteps)
{
#pragma HLS DATAFLOW
    hls::stream<input_t> s_raw("raw");
    hls::stream<int>     s_lbl("lbl");
#pragma HLS STREAM variable=s_raw depth=256
#pragma HLS STREAM variable=s_lbl depth=256

    unpack_axis(ecg,  s_raw, nSteps);
    compute    (s_raw,s_lbl, nSteps);
    pack_axis  (s_lbl,pred,  nSteps);
}

// ───────────────────────── Top Function ────────────────────────────────────
void lstm_forward(hls::stream<in_pkt_t>&  ecg,
                  hls::stream<out_pkt_t>& pred,
                  int                     numSamples,
                  int                     chunkSize)
{
#pragma HLS INTERFACE axis      register both port=ecg
#pragma HLS INTERFACE axis      register both port=pred
#pragma HLS INTERFACE s_axilite port=numSamples bundle=control
#pragma HLS INTERFACE s_axilite port=chunkSize  bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

//──────────────────── constant ROM & LUT bindings ──────────────────────────
#pragma HLS bind_storage variable=W_all_fixed type=ROM_2P impl=BRAM
#pragma HLS bind_storage variable=B_all_fixed type=ROM_1P impl=BRAM
#pragma HLS bind_storage variable=W_fc_fixed  type=ROM_2P impl=BRAM
#pragma HLS bind_storage variable=b_fc_fixed  type=ROM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=tanh_lut    complete
#pragma HLS ARRAY_PARTITION variable=sigmoid_lut complete
#pragma HLS ARRAY_PARTITION variable=exp_lut     complete

    int done = 0;
BURST_LOOP:
    while (done < numSamples) {
        int thisChunk = (done + chunkSize <= numSamples) ? chunkSize
                                                         : (numSamples - done);
        lstm_forward_chunk(ecg, pred, thisChunk);
        done += thisChunk;
    }
}

//──────────────────────────────────────────────────────────────────────────────
//  		Fast sigmoid / tanh / exp with LUT
//──────────────────────────────────────────────────────────────────────────────
static state_t sigmoid_fast(acc_t x)
{
#pragma HLS INLINE
    const acc_t  MIN = (acc_t)-32.0;
    const acc_t  MAX = (acc_t)+32.0;
    if (x <= MIN) return sigmoid_lut[0];
    if (x >= MAX) return sigmoid_lut[SIGMOID_LUT_SIZE-1];

    acc_t   s_f = (x - MIN) / (acc_t)SIGMOID_LUT_STEP;   // wide type
    int     idx = s_f.to_int();                          // LUT base index
    state_t f   = s_f - (acc_t)idx;                      // fractional interpolation offset
    return sigmoid_lut[idx] + f * (sigmoid_lut[idx+1] - sigmoid_lut[idx]);
}

static state_t tanh_fast(acc_t x)
{
#pragma HLS INLINE
    const acc_t  MIN = (acc_t)-32.0;
    const acc_t  MAX = (acc_t)+32.0;
    if (x <= MIN) return (state_t)-1;
    if (x >= MAX) return (state_t)+1;

    acc_t   s_f = (x - MIN) / (acc_t)TANH_LUT_STEP;
    int     idx = s_f.to_int();
    state_t f   = s_f - (acc_t)idx;
    return tanh_lut[idx] + f * (tanh_lut[idx+1] - tanh_lut[idx]);
}

static state_t exp_fast(acc_t x)
{
#pragma HLS INLINE
    const acc_t MIN = (acc_t)EXP_LUT_MIN;   // -16
    if (x <= MIN) return exp_lut[0];
    if (x >= (acc_t)0) return exp_lut[EXP_LUT_SIZE-1];

    acc_t delta = x - MIN;
    acc_t idx_f = delta / (acc_t)EXP_LUT_STEP;           // floating-point LUT position
    int   idx   = idx_f.to_int();
    state_t frac = idx_f - (acc_t)idx;
    return exp_lut[idx] + frac * (exp_lut[idx+1] - exp_lut[idx]);
}

// ──────────────────── softmax & argmax ──────────────────────────────────────────
static void softmax(const state_t in[OUTPUT_SIZE], state_t out[OUTPUT_SIZE])
{
    state_t m = in[0];
    for (int i = 1; i < OUTPUT_SIZE; ++i) {
	#pragma HLS PIPELINE II=1
        if (in[i] > m) m = in[i];
    }
    sum_t sum = 0;
    state_t buf[OUTPUT_SIZE];
	#pragma HLS ARRAY_PARTITION variable=buf complete
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
	#pragma HLS PIPELINE II=1
        buf[i] = exp_fast(acc_t(in[i]) - acc_t(m));
        sum   += buf[i];
    }
    // <——— GUARD HERE to avoid divide-by-zero ———>
    if (sum == (sum_t)0) sum = (sum_t)1;

    // ────────────── normalize ──────────────
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
	#pragma HLS UNROLL
        out[i] = buf[i] / sum;
    }
}

static int argmax(const state_t out[OUTPUT_SIZE])
{
    int best = 0; state_t mv = out[0];
    for (int i = 1; i < OUTPUT_SIZE; ++i) {
	#pragma HLS UNROLL
        if (out[i] > mv) { mv = out[i]; best = i; }
    }
    return best;
}

static void fully_connected_and_softmax(const state_t hid[HIDDEN_SIZE],
                                        state_t       out[OUTPUT_SIZE])
{
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
	#pragma HLS PIPELINE II=1
        acc_t acc = (acc_t)b_fc_fixed[i];
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
		#pragma HLS UNROLL
            acc += (acc_t)W_fc_fixed[i][j] * hid[j];
        }
        out[i] = acc;
    }
    softmax(out, out);
}

// ───────────────────── lstm_cell ────────────────────────────
static void lstm_cell(const input_t  input[INPUT_SIZE],
                      const state_t  hid_i[HIDDEN_SIZE],
                      const cell_t   cell_i[HIDDEN_SIZE],
                            state_t  hid_o[HIDDEN_SIZE],
                            cell_t   cell_o[HIDDEN_SIZE])
{
    // ────────────────── Concatenate x_t and h_{t-1} ──────────────────────
    acc_t combined[INPUT_SIZE + HIDDEN_SIZE];
	#pragma HLS ARRAY_PARTITION variable=combined complete
    combined[0] = input[0];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
	#pragma HLS UNROLL
        combined[INPUT_SIZE + i] = hid_i[i];
    }

    acc_t gates[4 * HIDDEN_SIZE];
	#pragma HLS ARRAY_PARTITION variable=gates block factor=4

MAC_LOOP:
    for (int i = 0; i < 4 * HIDDEN_SIZE; ++i) {
	#pragma HLS PIPELINE II=1
        acc_t acc = (acc_t)B_all_fixed[i];
        for (int j = 0; j < INPUT_SIZE + HIDDEN_SIZE; ++j) {
		#pragma HLS UNROLL
            acc += (acc_t)W_all_fixed[i][j] * combined[j];
        }
        gates[i] = acc;
    }

    // ────────────── Apply non-linearities and update cell / hid ──────────────
UPD_LOOP:
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
	#pragma HLS PIPELINE II=1
    	state_t iv = sigmoid_fast(gates[i]);                   // I
    	state_t fv = sigmoid_fast(gates[i +   HIDDEN_SIZE]);   // F
    	state_t cv = tanh_fast   (gates[i + 2*HIDDEN_SIZE]);   // **G**
    	state_t ov = sigmoid_fast(gates[i + 3*HIDDEN_SIZE]);   // **O**

        cell_t next_c = fv * cell_i[i] + iv * cv;
        cell_o[i]     = next_c;
        hid_o[i]      = ov * tanh_fast(acc_t(next_c));
    }
}
