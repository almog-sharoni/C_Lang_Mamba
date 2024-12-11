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
void apply_silu(int8_t* tensor, int batch, int seqlen, int channels, float scale);
void conv1d(int8_t* input, int8_t* weights, int8_t* output, int channels, int width, int kernel_size, float scale);
void split_xz(int8_t* xz, int8_t* x, int8_t* z, int batch, int seqlen, int channels);
void load_weights(const char* weight_file, int8_t* weights, int size) ;

#endif // UTILS_H
