#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <math.h>
#include "config.h" 
#include "model.h"



// Macros
#define SILU(x) ((x) / (1 + fabs(x))) // SiLU approximation

// Function Declarations
int8_t quantize(float value, float scale);
float dequantize(int8_t value, float scale);
float softplus(float x);
void matmul(int8_t* A, int8_t* B, int8_t* C, int M, int N, int K, float scale);
void apply_silu(int8_t* tensor, int batch, int seqlen, int channels);
// void conv1d(int8_t* input, int8_t* weights, int8_t* output, int channels, int width, int kernel_size, float scale);
void conv1d(int8_t* input, int8_t* weights, int8_t* output, int batch_size, int channels, int width, int kernel_size, float scale) ;

void split_xz(int8_t* xz, int8_t* x, int8_t* z, int batch, int seqlen, int channels);
void load_weights(const char* weight_file, int8_t* weights, int size) ;
void initialize_exp_lut();
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
);
void matmul_expand(int8_t* A, int8_t* B, int8_t* C, 
                   int M, int N, int K, int expand_factor, float scale) ;
#endif // UTILS_H
