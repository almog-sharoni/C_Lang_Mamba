#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "config.h" 


// // Input Projection Weights and Bias
// int8_t in_proj_weights[D_MODEL][D_INNER * 2]; // [128][512]
// int8_t in_proj_bias[D_INNER * 2];            // [512]

// // Convolution Weights and Bias
// int8_t conv1d_weights[D_INNER][D_CONV];      // [256][64]
// int8_t conv1d_bias[D_INNER];                 // [256]

// // Output Projection Weights and Bias
// int8_t out_proj_weights[D_INNER][D_MODEL];   // [256][128]
// int8_t out_proj_bias[D_MODEL];               // [128]

// // Additional Layer Weights
// int8_t A_log[D_INNER];                       // [256]
// int8_t D[D_INNER];                           // [256]
// int8_t x_proj_weights[D_INNER][136];         // [256][136]
// int8_t dt_proj_weights[256][8];              // [256][8]
// int8_t dt_proj_bias[256];                    // [256]

// Global Variables for Weights and Biases
extern int8_t in_proj_weights[D_MODEL][D_INNER * 2];
extern int8_t in_proj_bias[D_INNER * 2];
extern int8_t conv1d_weights[D_INNER][D_CONV];
extern int8_t conv1d_bias[D_INNER];
extern int8_t out_proj_weights[D_INNER][D_MODEL];
extern int8_t out_proj_bias[D_MODEL];
extern int8_t A_log[D_INNER];
extern int8_t D[D_INNER];
extern int8_t x_proj_weights[D_INNER][136];
extern int8_t dt_proj_weights[256][8];
extern int8_t dt_proj_bias[256];


// Function Declaration
void initialize_weights();

#endif // MODEL_H
