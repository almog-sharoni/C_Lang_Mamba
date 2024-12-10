#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "config.h" 
#include "model.h"

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

void matmul(int8_t* A, int8_t* B, int8_t* C, int M, int N, int K, float scale) {
    int8_t* B_transposed = (int8_t*)malloc(N * K * sizeof(int8_t));
    if (!B_transposed) {
        perror("Failed to allocate memory for B_transposed");
        exit(1);
    }

    // Transpose B
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            B_transposed[n * K + k] = B[k * N + n];
        }
    }

    // Matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                int8_t a_val = A[i * K + k];
                int8_t b_val = B_transposed[j * K + k];
                int32_t product = a_val * b_val;
                sum += product;
            }
            // Quantize and assign to C
            int8_t result = (int8_t)(sum * scale);
            if (result > 127) result = 127;
            if (result < -128) result = -128;
            C[i * N + j] = result;

            // Debugging output
            // printf("C[%d][%d] = quantize(%d, scale) = %d\n", i, j, sum, result);
        }
    }

    // printf("Printing A : \n");
    // for (int i = 0; i < N; i++) {
    //     for (int k = 0; k < K; k++) {
    //         printf("A[%d][%d] = %d ", i, k, A[i * D_INNER + k]);
    //     }
    //     printf("\n");
    // }

    // printf("Printing B for all rows:\n");
    // for (int k = 0; k < D_INNER; k++) {
    //     for (int j = 0; j < D_STATE; j++) {
    //         printf("B[%d][%d] = %d ", k, j, B[k * D_STATE + j]);
    //     }
    //     printf("\n");
    // }

    // // break;
    // exit(1);

    free(B_transposed);
}


void apply_silu(int8_t* tensor, int batch, int seqlen, int channels, float scale) {
    for (int b = 0; b < batch; b++) {
        for (int l = 0; l < seqlen; l++) {
            for (int c = 0; c < channels; c++) {
                int idx = b * seqlen * channels + l * channels + c;

                // Dequantize
                float val = dequantize(tensor[idx], scale);

                // Debug: Print dequantized value
                // printf("Dequantized value at idx %d: %f\n", idx, val);

                // Clamp value to avoid extreme ranges
                val = fmaxf(fminf(val, 6.0f), -6.0f);  // Clamp to [-6, 6]

                // Apply SiLU activation
                float activated_val = val / (1.0f + expf(-val));  // SiLU

                // Debug: Print activated value
                // printf("Activated value at idx %d: %f\n", idx, activated_val);

                // Quantize back to int8
                tensor[idx] = quantize(activated_val, scale);

                // Debug: Print quantized value
                // printf("Quantized value at idx %d: %d\n", idx, tensor[idx]);
            }
        }
    }
}

void split_xz(int8_t* xz, int8_t* x, int8_t* z, int batch, int seqlen, int channels) {
    for (int b = 0; b < batch; b++) {
        for (int l = 0; l < seqlen; l++) {
            for (int c = 0; c < channels; c++) {
                x[b * seqlen * channels + l * channels + c] = xz[b * seqlen * channels * 2 + l * channels * 2 + c];
                z[b * seqlen * channels + l * channels + c] = xz[b * seqlen * channels * 2 + l * channels * 2 + c + channels];
            }
        }
    }
}


void conv1d(int8_t* input, int8_t* weights, int8_t* output, int channels, int width, int kernel_size, float scale) {
    int pad = (kernel_size - 1) / 2; // Symmetric padding
    for (int c = 0; c < channels; c++) {
        for (int w = 0; w < width; w++) {
            int32_t sum = 0;
            for (int k = 0; k < kernel_size; k++) {
                int input_idx = w - pad + k;
                if (input_idx >= 0 && input_idx < width) { // Handle padding
                    sum += input[c * width + input_idx] * weights[c * kernel_size + k];
                }
            }
            // Quantize and store in output
            int32_t quantized = round(sum * scale); // Scale and round
            output[c * width + w] = (int8_t)fmax(fmin(quantized, 127), -128); // Clamp to int8_t range
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
