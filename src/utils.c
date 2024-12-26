#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "config.h" 
#include "model.h"
#include <string.h>

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
int16_t fast_exp_neg(int8_t x) {
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
    int half = D_INNER;
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


// void conv1d(int8_t* input, int8_t* weights, int8_t* output, int channels, int width, int kernel_size, float scale) {
//     int pad = (kernel_size - 1);

//     for (int c = 0; c < channels; c++) {
//         for (int w = 0; w < width; w++) {
//             int32_t sum = 0;
//             for (int k = 0; k < kernel_size; k++) {
//                 int input_idx = w - pad + k;
//                 if (input_idx >= 0 && input_idx < width) { // Handle padding
//                     sum += input[c * width + input_idx] * weights[c * kernel_size + k];
//                 }
//             }

//             // Apply logarithmic scaling for large sums
//             if (sum > 127) {
//                 sum = (int32_t)(127 + log2(sum - 126));
//             } else if (sum < -128) {
//                 sum = (int32_t)(-128 - log2(-sum - 127));
//             }

//             // Quantize and store in output
//             int32_t quantized = round(sum * scale);
//             output[c * width + w] = (int8_t)quantized;
//         }
//     }
// }

void conv1d(int8_t* input, int8_t* weights, int8_t* output, int batch_size, int channels, int width, int kernel_size, float scale) {
    int pad = (kernel_size - 1);

    for (int b = 0; b < batch_size; b++) {          // Loop over batch
        for (int c = 0; c < channels; c++) {       // Loop over channels
            for (int w = 0; w < width; w++) {      // Loop over width
                int32_t sum = 0;
                for (int k = 0; k < kernel_size; k++) { // Apply kernel
                    int input_idx = b * channels * width + c * width + w - pad + k;
                    if (input_idx >= b * channels * width + c * width && input_idx < b * channels * width + (c + 1) * width) { // Handle padding
                        sum += input[input_idx] * weights[c * kernel_size + k];
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
                output[b * channels * width + c * width + w] = (int8_t)quantized;
            }
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

void matmul_and_concatenate(
    int8_t* input,          // Input matrix (e.g., `conv_output`)
    int8_t* x_proj_weights, // Weights for `DT`, `B`, and `C` (entire `x_proj.weight`)
    int8_t* output,         // Output matrix (result: concatenated `DT`, `B`, and `C`)
    int batch_size,
    int seq_len,
    int input_dim,          // Input dimension (e.g., `D_INNER`)
    int dt_rank,
    int d_state,
    int scale_shift         // Scaling factor for quantization
) {
    int output_dim = dt_rank + 2 * d_state; // Total output dimension (X_PROJ_OUT)

    // Allocate intermediate buffers for DT, B, and C
    int8_t* dt_output = (int8_t*)malloc(batch_size * seq_len * dt_rank * sizeof(int8_t));
    int8_t* b_output = (int8_t*)malloc(batch_size * seq_len * d_state * sizeof(int8_t));
    int8_t* c_output = (int8_t*)malloc(batch_size * seq_len * d_state * sizeof(int8_t));

    // Perform matrix multiplication for DT, B, and C
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int j = 0; j < dt_rank; j++) { // Compute DT projection
                int32_t sum = 0;
                for (int k = 0; k < input_dim; k++) {
                    sum += input[b * seq_len * input_dim + s * input_dim + k] *
                           x_proj_weights[j * input_dim + k]; // First `DT_RANK` columns
                }
                sum = sum >> scale_shift; // Apply quantization scaling
                if (sum > 127) sum = 127;
                if (sum < -128) sum = -128;
                dt_output[b * seq_len * dt_rank + s * dt_rank + j] = (int8_t)sum;
            }

            for (int j = 0; j < d_state; j++) { // Compute B projection
                int32_t sum = 0;
                for (int k = 0; k < input_dim; k++) {
                    sum += input[b * seq_len * input_dim + s * input_dim + k] *
                           x_proj_weights[(dt_rank + j) * input_dim + k]; // Next `D_STATE` columns
                }
                sum = sum >> scale_shift; // Apply quantization scaling
                if (sum > 127) sum = 127;
                if (sum < -128) sum = -128;
                b_output[b * seq_len * d_state + s * d_state + j] = (int8_t)sum;
            }

            for (int j = 0; j < d_state; j++) { // Compute C projection
                int32_t sum = 0;
                for (int k = 0; k < input_dim; k++) {
                    sum += input[b * seq_len * input_dim + s * input_dim + k] *
                           x_proj_weights[(dt_rank + d_state + j) * input_dim + k]; // Last `D_STATE` columns
                }
                sum = sum >> scale_shift; // Apply quantization scaling
                if (sum > 127) sum = 127;
                if (sum < -128) sum = -128;
                c_output[b * seq_len * d_state + s * d_state + j] = (int8_t)sum;
            }
        }
    }

    // Concatenate DT, B, and C into the final output
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            // Copy DT
            memcpy(&output[b * seq_len * output_dim + s * output_dim],
                   &dt_output[b * seq_len * dt_rank + s * dt_rank],
                   dt_rank * sizeof(int8_t));

            // Copy B
            memcpy(&output[b * seq_len * output_dim + s * output_dim + dt_rank],
                   &b_output[b * seq_len * d_state + s * d_state],
                   d_state * sizeof(int8_t));

            // Copy C
            memcpy(&output[b * seq_len * output_dim + s * output_dim + dt_rank + d_state],
                   &c_output[b * seq_len * d_state + s * d_state],
                   d_state * sizeof(int8_t));
        }
    }

    // Free intermediate buffers
    free(dt_output);
    free(b_output);
    free(c_output);
}

void matmul_expand(int8_t* A, int8_t* B, int8_t* C, 
                   int M, int N, int K, int expand_factor, float scale) {
    // M: Batch size (BATCH_SIZE)
    // N: Sequence length (SEQ_LEN)
    // K: Model dimension (D_MODEL)
    // expand_factor: Expansion factor (2 for D_INNER * 2)
    // B is not batched; its shape is [K, K * expand_factor]
    // Output C will have dimensions [M, N, K * expand_factor]
    int expanded_k = K * expand_factor;

    for (int i = 0; i < M; i++) {             // Loop over batches
        for (int j = 0; j < N; j++) {         // Loop over sequence length
            for (int e = 0; e < expanded_k; e++) {  // Loop over expanded D_INNER*2
                int32_t sum = 0;
                for (int k = 0; k < K; k++) {  // Loop over model dimension
                    int a_idx = i * N * K + j * K + k;      // Input A index
                    int b_idx = k * expanded_k + e;         // Weights B index
                    assert(a_idx < (M * N * K));            // Bounds check for A
                    assert(b_idx < (K * expanded_k));       // Bounds check for B

                    int8_t a_val = A[a_idx];
                    int8_t b_val = B[b_idx];
                    sum += a_val * b_val;
                }

                // Apply scaling and quantization
                int32_t scaled_sum = sum * (int32_t)(scale * (1 << SCALE_SHIFT));  
                int8_t result = (int8_t)(scaled_sum >> SCALE_SHIFT);

                // Clamp to int8_t range
                if (result > 127) result = 127;
                if (result < -128) result = -128;

                int c_idx = i * N * expanded_k + j * expanded_k + e;  // Output C index
                assert(c_idx < (M * N * expanded_k));  // Bounds check for C
                C[c_idx] = result;

#ifdef DEBUG
                printf("C[%d][%d][%d] = quantize(%d, scale) = %d\n", i, j, e, sum, result);
#endif
            }
        }
    }
}