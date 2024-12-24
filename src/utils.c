#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "config.h" 
#include "model.h"

#define SCALE_SHIFT 8  // Scale factor (2^6 = 64)
#define LUT_SIZE 128   // Size of the exp lookup table
#define LUT_MAX 6      // Maximum value for the LUT range
#define MAX_INT8 127
#define MIN_INT8 -128

int8_t quantize(float value, float scale) {
    return (int8_t)(value * scale);
}

float dequantize(int8_t value, float scale) {
    return value / scale;
}

// void matmul(int8_t* A, int8_t* B, int8_t* C, int M, int N, int K, float scale) {
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             int32_t sum = 0;
//             for (int k = 0; k < K; k++) {
//                 sum += A[i * K + k] * B[k * N + j];
//             }
//             C[i * N + j] = quantize(sum, scale);
//         }
//     }
// }

// void matmul(int8_t* A, int8_t* B, int8_t* C, int M, int N, int K, float scale) {
//     // // Parallelize the outer loops if applicable
//     // #pragma omp parallel for collapse(2)
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             int32_t sum = 0;
//             for (int k = 0; k < K; k++) {
//                 int8_t a_val = A[i * K + k];
//                 int8_t b_val = B[k * N + j];  // Direct access without transposing
//                 int32_t product = a_val * b_val;
//                 sum += product;
//             }

//             // Fixed-point quantization with clamping
//             int32_t scaled_sum = sum * (int32_t)(scale * (1 << SCALE_SHIFT));  // Scale sum
//             int8_t result = (int8_t)(scaled_sum >> SCALE_SHIFT);

//             // Clamp to int8_t range
//             if (result > 127) result = 127;
//             if (result < -128) result = -128;

//             C[i * N + j] = result;

//             // Debugging output (conditional)
//             #ifdef DEBUG
//             printf("C[%d][%d] = quantize(%d, scale) = %d\n", i, j, sum, result);
//             #endif
//         }
//     }
// }

#include <assert.h>

void matmul(int8_t* A, int8_t* B, int8_t* C, int M, int N, int K, float scale) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                int a_idx = i * K + k;
                int b_idx = k * N + j;
                // Assert indices are within bounds
                assert(a_idx < (M * K));
                assert(b_idx < (K * N));
                int8_t a_val = A[a_idx];
                int8_t b_val = B[b_idx];
                int32_t product = a_val * b_val;
                sum += product;
            }

            // Fixed-point quantization with clamping
            int32_t scaled_sum = sum * (int32_t)(scale * (1 << SCALE_SHIFT));  // Scale sum
            int8_t result = (int8_t)(scaled_sum >> SCALE_SHIFT);

            // Clamp to int8_t range
            if (result > 127) result = 127;
            if (result < -128) result = -128;

            C[i * N + j] = result;

            // Debugging output (conditional)
            #ifdef DEBUG
            printf("C[%d][%d] = quantize(%d, scale) = %d\n", i, j, sum, result);
            #endif
        }
    }
}


// Precompute a LUT for exp(-x) in the range [0, LUT_MAX]
int16_t exp_lut[LUT_SIZE];  // Store as fixed-point values
void initialize_exp_lut() {
    for (int i = 0; i < LUT_SIZE; i++) {
        float x = (float)i / (LUT_SIZE - 1) * LUT_MAX;  // Map i to [0, LUT_MAX]
        exp_lut[i] = (int16_t)((1.0f / expf(x)) * (1 << SCALE_SHIFT));  // Fixed-point reciprocal of exp(x)
    }
}

// Fast exp(-x) using LUT
inline int16_t fast_exp_neg(int8_t x) {
    // Clamp input to the LUT range
    if (x >= LUT_MAX * (1 << SCALE_SHIFT)) return 1;  // exp(0) = 1 in fixed-point
    if (x <= -LUT_MAX * (1 << SCALE_SHIFT)) return exp_lut[LUT_SIZE - 1];

    // Map x to LUT index
    int idx = ((-x * LUT_SIZE) / (LUT_MAX * (1 << SCALE_SHIFT))) % LUT_SIZE;
    return exp_lut[idx];
}

// Integer SiLU
void apply_silu(int8_t* tensor, int batch, int seqlen, int channels) {
    for (int b = 0; b < batch; b++) {
        for (int l = 0; l < seqlen; l++) {
            for (int c = 0; c < channels; c++) {
                int idx = b * seqlen * channels + l * channels + c;

                // Dequantize value (interpreted as fixed-point)
                int16_t val = tensor[idx] << SCALE_SHIFT;  // Convert to fixed-point

                // Compute exp(-x) using LUT
                int16_t exp_val = fast_exp_neg(val);

                // Compute SiLU: val / (1 + exp(-val))
                int16_t silu_val = (val * (1 << SCALE_SHIFT)) / ((1 << SCALE_SHIFT) + exp_val);

                // Requantize to int8_t
                silu_val >>= SCALE_SHIFT;  // Scale back to integer range
                if (silu_val > MAX_INT8) silu_val = MAX_INT8;
                if (silu_val < MIN_INT8) silu_val = MIN_INT8;

                tensor[idx] = (int8_t)silu_val;
            }
        }
    }
}


// void apply_silu(int8_t* tensor, int batch, int seqlen, int channels, float scale) {
//     for (int b = 0; b < batch; b++) {
//         for (int l = 0; l < seqlen; l++) {
//             for (int c = 0; c < channels; c++) {
//                 int idx = b * seqlen * channels + l * channels + c;

//                 // Dequantize
//                 float val = dequantize(tensor[idx], scale);

//                 // Debug: Print dequantized value
//                 // printf("Dequantized value at idx %d: %f\n", idx, val);

//                 // Clamp value to avoid extreme ranges
//                 val = fmaxf(fminf(val, 6.0f), -6.0f);  // Clamp to [-6, 6]

//                 // Apply SiLU activation
//                 float activated_val = val / (1.0f + expf(-val));  // SiLU

//                 // Debug: Print activated value
//                 // printf("Activated value at idx %d: %f\n", idx, activated_val);

//                 // Quantize back to int8
//                 tensor[idx] = quantize(activated_val, scale);

//                 // Debug: Print quantized value
//                 // printf("Quantized value at idx %d: %d\n", idx, tensor[idx]);
//             }
//         }
//     }
// }

// void split_xz(int8_t* xz, int8_t* x, int8_t* z, int batch, int seqlen, int channels) {
//     for (int b = 0; b < batch; b++) {
//         for (int l = 0; l < seqlen; l++) {
//             for (int c = 0; c < channels; c++) {
//                 x[b * seqlen * channels + l * channels + c] = xz[b * seqlen * channels * 2 + l * channels * 2 + c];
//                 z[b * seqlen * channels + l * channels + c] = xz[b * seqlen * channels * 2 + l * channels * 2 + c + channels];
//             }
//         }
//     }
// }

void split_xz(int8_t* xz, int8_t* x, int8_t* z, int a, int b, int c) {
    int half = D_INNER / 2;
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            for (int d = 0; d < half; d++) {
                int idx = b * SEQ_LEN * D_INNER + s * D_INNER + d;
                x[b * SEQ_LEN * half + s * half + d] = xz[idx];
                z[b * SEQ_LEN * half + s * half + d] = xz[idx + half];
            }
        }
    }
}


void conv1d(int8_t* input, int8_t* weights, int8_t* output, int channels, int width, int kernel_size, float scale) {
    int pad = (kernel_size - 1);

    for (int c = 0; c < channels; c++) {
        for (int w = 0; w < width; w++) {
            int32_t sum = 0;
            for (int k = 0; k < kernel_size; k++) {
                int input_idx = w - pad + k;
                if (input_idx >= 0 && input_idx < width) { // Handle padding
                    sum += input[c * width + input_idx] * weights[c * kernel_size + k];
                }
            }

            // Apply logarithmic scaling for large sums
            if (sum > 127) {
                sum = (int32_t)(127 + log2(sum - 126));
            } else if (sum < -128) {
                sum = (int32_t)(-128 - log2(-sum - 127));
            }

            // Quantize and store in output
            int32_t quantized = round(sum * scale);
            output[c * width + w] = (int8_t)quantized;
        }
    }
}



void load_weights(const char* weight_file, int8_t* weights, int size) {
    FILE* wf = fopen(weight_file, "r");
    if (!wf) {
        fprintf(stderr, "Error opening weight file: %s\n", weight_file);
        perror("File error");
        exit(1);
    }

    int count = 0; // Track the number of weights read
    for (int i = 0; i < size; i++) {
        if (fscanf(wf, "%hhd", &weights[i]) != 1) {
            fprintf(stderr, "Error reading weight at index %d from file: %s (Read %d weights so far)\n", i, weight_file, count);
            fclose(wf);
            exit(1);
        }
        count++;
    }

    fclose(wf);
}

float softplus(float x) {
    return logf(1.0f + expf(x));
}