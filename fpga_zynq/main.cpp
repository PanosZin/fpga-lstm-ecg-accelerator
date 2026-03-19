/***********************************************************************
 *  Bare-metal demo – ZCU104 (A-53)  –  LSTM inference accelerator
 *
 *  • Streams ECG   int8   samples through AXI-DMA MM2S
 *  • Collects  uint8 predictions through AXI-DMA S2MM
 *  • Sends <numSamples,chunkSize> to the HLS core every burst
 *  • Prints a 4×4 confusion matrix + per-class accuracy
 *  • Measures total inference time via the ARM generic timer
 ************************************************************************/

#include <cstdint>
#include <cstdio>
#include "xil_io.h"
#include "xil_cache.h"
#include "xlstm_forward.h"          // generated driver
#include "xparameters.h"            // base addresses & timestamp freq
#include "xtime_l.h"                // XTime_GetTime()

 /*---------------------------------------------------------------------------*/
 /* ** DDR buffer base addresses ** **/
 /*---------------------------------------------------------------------------*/
#define ECG_BASE_PHYS      0x10000000u      // int8_t  ECG samples
#define GT_BASE_PHYS       0x20000000u      // uint8_t ground-truth labels
#define PR_BASE_PHYS       0x30000000u      // uint8_t predictions out

/*---------------------------------------------------------------------------*/
/* ** Dataset & chunk parameters **                                          */
/*---------------------------------------------------------------------------*/
#define TOTAL_SAMPLES      14175000u        //  (2835 segments × 5000)
#define CHUNK_SAMPLES      5000u            //  MUST match software & RTL

/*---------------------------------------------------------------------------*/
/* ** DMA register map (suitable for both MM2S & S2MM) **                    */
/*---------------------------------------------------------------------------*/
#define DMA_MM2S_BASE      XPAR_AXI_DMA_0_BASEADDR
#define DMA_S2MM_BASE      (DMA_MM2S_BASE + 0x30)

#define DMACR          0x00u
#define DMASR          0x04u
#define SA_LSB         0x18u
#define SA_MSB         0x1Cu
#define LENGTH         0x28u

#define DMACR_RS       0x00000001u
#define DMASR_HALTED   0x00000001u
#define DMASR_IDLE     0x00000002u
#define DMASR_IOC_IRQ  0x00001000u

/*---------------------------------------------------------------------------*/
/*  Very small helper:  wait for HLS core to finish current chunk             */
/*---------------------------------------------------------------------------*/
static void wait_core_idle(XLstm_forward* ip)
{
    while (!XLstm_forward_IsIdle(ip));
}

/*---------------------------------------------------------------------------*/
/*  MM2S one-shot DMA                                                        */
/*---------------------------------------------------------------------------*/
static void dma_mm2s(uintptr_t srcPhys, uint32_t bytes)
{
    Xil_Out32(DMA_MM2S_BASE + DMACR, 0);
    while (!(Xil_In32(DMA_MM2S_BASE + DMASR) & DMASR_HALTED));
    Xil_Out32(DMA_MM2S_BASE + DMASR, 0x00007000u);
    Xil_Out32(DMA_MM2S_BASE + SA_LSB, (uint32_t)(srcPhys));
    Xil_Out32(DMA_MM2S_BASE + SA_MSB, (uint32_t)(srcPhys >> 32));
    Xil_Out32(DMA_MM2S_BASE + DMACR, DMACR_RS);
    Xil_Out32(DMA_MM2S_BASE + LENGTH, bytes);
    while (!(Xil_In32(DMA_MM2S_BASE + DMASR) & DMASR_IOC_IRQ));
}

/*---------------------------------------------------------------------------*/
/*  S2MM one-shot DMA                                                        */
/*---------------------------------------------------------------------------*/
static void dma_s2mm(uintptr_t dstPhys, uint32_t bytes)
{
    Xil_Out32(DMA_S2MM_BASE + DMACR, 0);
    while (!(Xil_In32(DMA_S2MM_BASE + DMASR) & DMASR_HALTED));
    Xil_Out32(DMA_S2MM_BASE + DMASR, 0x00007000u);
    Xil_Out32(DMA_S2MM_BASE + SA_LSB, (uint32_t)(dstPhys));
    Xil_Out32(DMA_S2MM_BASE + SA_MSB, (uint32_t)(dstPhys >> 32));
    Xil_Out32(DMA_S2MM_BASE + DMACR, DMACR_RS);
    Xil_Out32(DMA_S2MM_BASE + LENGTH, bytes);
    while (!(Xil_In32(DMA_S2MM_BASE + DMASR) & DMASR_IOC_IRQ));
}

/*---------------------------------------------------------------------------*/
/*  Confusion-matrix utils (4-class), reordered P, QRS, T, n/a               */
/*---------------------------------------------------------------------------*/
static void print_confusion(const uint8_t* gt,
                            const uint8_t* pr,
                            uint32_t       n)
{
    // Class order: P=1, QRS=2, T=3, n/a=0
    constexpr int C = 4;
    uint32_t cm[C][C] = {};
    uint32_t cnt[C] = {};

    for (uint32_t i = 0; i < n; ++i) {
        uint8_t a = gt[i], p = pr[i];
        if (a < C && p < C) {
            cm[a][p]++; cnt[a]++;
        }
    }

    // Names in display order
    const char* names[C] = { "P","QRS","T","n/a" };
    // Mapping from display index to actual label
    const int label_map[C] = { 1, 2, 3, 0 };

    printf("\n Per-class accuracy\n");
    for (int d = 0; d < C; ++d) {
        int c = label_map[d];
        double pct = cnt[c] ? 100.0 * cm[c][c] / cnt[c] : 0.0;
        printf("  %-3s : %6.2f %%  (%u / %u)\n",
            names[d], pct, cm[c][c], cnt[c]);
    }

    printf("\n Confusion matrix (row-norm. %%)\n       ");
    for (int d = 0; d < C; ++d) {
        printf("%8s", names[d]);
    }
    printf("\n ---------------------------------------------------------\n");

    for (int da = 0; da < C; ++da) {
        int a = label_map[da];
        printf(" %3s |", names[da]);
        for (int dp = 0; dp < C; ++dp) {
            int p = label_map[dp];
            double pct = cnt[a] ? 100.0 * cm[a][p] / cnt[a] : 0.0;
            printf("%8.2f", pct);
        }
        printf("\n");
    }
}

int main()
{
    printf("\n=== LSTM ECG FORWARD PASS ===\n");
    printf("Samples  : %u\nChunk sz : %u\n", TOTAL_SAMPLES, CHUNK_SAMPLES);

    // Initialize HLS driver
    XLstm_forward ip;
    if (XLstm_forward_Initialize(&ip, XPAR_XLSTM_FORWARD_0_DEVICE_ID)) {
        printf("ERROR: cannot init XLstm_forward driver\n");
        return -1;
    }

    // DDR buffer pointers
    volatile uint8_t* const ecg = reinterpret_cast<volatile uint8_t*>(ECG_BASE_PHYS);
    volatile uint8_t* const gt = reinterpret_cast<volatile uint8_t*>(GT_BASE_PHYS);
    volatile uint8_t* const pr = reinterpret_cast<volatile uint8_t*>(PR_BASE_PHYS);

    // Start timer
    XTime tStart, tStop;
    XTime_GetTime(&tStart);

    // Burst loop
    uint32_t produced = 0;
    while (produced < TOTAL_SAMPLES) {
        uint32_t nThis = (produced + CHUNK_SAMPLES <= TOTAL_SAMPLES)
            ? CHUNK_SAMPLES
            : (TOTAL_SAMPLES - produced);

        Xil_DCacheFlushRange((uintptr_t)&ecg[produced], nThis);
        Xil_DCacheInvalidateRange((uintptr_t)&pr[produced], nThis);

        XLstm_forward_Set_numSamples(&ip, nThis);
        XLstm_forward_Set_chunkSize(&ip, nThis);
        XLstm_forward_Start(&ip);

        dma_mm2s((uintptr_t)&ecg[produced], nThis);
        dma_s2mm((uintptr_t)&pr[produced], nThis);

        wait_core_idle(&ip);

        produced += nThis;
        if ((produced & 0xFFFFF) == 0)
            printf("  streamed %u / %u\r", produced, TOTAL_SAMPLES);
    }

    // Stop timer
    XTime_GetTime(&tStop);
    double elapsed_s = (double)(tStop - tStart)
        / (double)XPAR_CPU_CORTEXA53_0_TIMESTAMP_CLK_FREQ;

    printf("\nStreaming done – %u samples processed.\n", produced);
    printf("Total inference time: %.3f seconds\n", elapsed_s);

    // Ensure predictions are coherent
    Xil_DCacheInvalidateRange(PR_BASE_PHYS, TOTAL_SAMPLES);

    // Confusion & accuracy
    print_confusion((const uint8_t*)gt,
        (const uint8_t*)pr,
        TOTAL_SAMPLES);

    printf("\nLSTM Forward pass finished.\n");
    return 0;
}
