#include "model.h"
#include "utils.h"
#include "config.h" 
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// #define D_MODEL 128       // Input dimension (number of features)
// #define D_INNER 256       // Intermediate dimension
// #define D_CONV 64         // Convolution kernel size

// // Input Projection Weights and Bias
// int8_t in_proj_weights[D_INNER][D_MODEL]; // [256][128]
// int8_t in_proj_bias[D_INNER * 2];            // [512]

// // Convolution Weights and Bias
// int8_t conv1d_weights[D_INNER][D_CONV];      // [256][64]
// int8_t conv1d_bias[D_INNER];                 // [256]

// // Output Projection Weights and Bias
// int8_t out_proj_weights[D_INNER][D_MODEL];   // [256][128]
// int8_t out_proj_bias[D_MODEL];               // [128]

// // Additional Layer Weights
// int8_t A_log[D_INNER][D_STATE];              // [256][64]
// int8_t D[D_INNER];                           // [256]
// int8_t x_proj_weights[D_INNER][X_PROJ_OUT];         // [256][136]
// int8_t dt_proj_weights[256][8];              // [256][8]
// int8_t dt_proj_bias[256];                    // [256]


// Corrected Weight Definitions
int8_t in_proj_weights[D_INNER][D_MODEL];   // [256][128]
int8_t in_proj_bias[D_INNER];               // [256]

// Convolution Weights and Bias
int8_t conv1d_weights[D_INNER][D_CONV];      // [256][64]
int8_t conv1d_bias[D_INNER];                 // [256]

// Output Projection Weights and Bias
int8_t out_proj_weights[D_INNER][D_MODEL];   // [256][128]
int8_t out_proj_bias[D_MODEL];               // [128]

// Additional Layer Weights
int8_t A_log[D_INNER][D_STATE];              // [256][64]
int8_t D[D_INNER];                           // [256]
int8_t x_proj_weights[D_INNER][X_PROJ_OUT];  // [256][136]
int8_t dt_proj_weights[D_INNER][DT_RANK];  // [256][DT_PROJ_N] // Adjust DT_PROJ_N as needed
int8_t dt_proj_bias[D_INNER];                // [256]


void initialize_weights() {
    // Load weights for A_log
    load_weights("../weights/A_log_weights.txt", &A_log[0][0], D_INNER * D_STATE);

    // Load weights for D
    load_weights("../weights/D_weights.txt", D, D_INNER);

    // Load weights for in_proj
    load_weights("../weights/in_proj.weight_weights.txt", &in_proj_weights[0][0], D_INNER * D_MODEL);
    // load_weights("../weights/in_proj.bias_weights.txt", in_proj_bias, D_INNER);

    // Load weights for conv1d
    load_weights("../weights/conv1d.weight_weights.txt", &conv1d_weights[0][0], D_INNER * D_CONV);
    load_weights("../weights/conv1d.bias_weights.txt", conv1d_bias, D_INNER);

    // Load weights for x_proj
    load_weights("../weights/x_proj.weight_weights.txt", &x_proj_weights[0][0], D_INNER * X_PROJ_OUT);

    // Load weights for dt_proj
    load_weights("../weights/dt_proj.weight_weights.txt", &dt_proj_weights[0][0], D_INNER * DT_RANK);
    load_weights("../weights/dt_proj.bias_weights.txt", dt_proj_bias, D_INNER);

    // Load weights for out_proj
    load_weights("../weights/out_proj.weight_weights.txt", &out_proj_weights[0][0], D_INNER * D_MODEL);
    // load_weights("../weights/out_proj.bias_weights.txt", out_proj_bias, D_MODEL);

    // Optionally, log successful loading
    printf("All weights loaded successfully.\n");
}


// void initialize_weights() {
//     // Load weights for A_log
//     load_weights("../weights/A_log_weights.txt", &A_log[0][0], D_INNER * D_STATE);

//     // Load weights for D
//     load_weights("../weights/D_weights.txt", D, D_INNER);

//     // Load weights for in_proj
//     load_weights("../weights/in_proj.weight_weights.txt", &in_proj_weights[0][0], D_INNER * D_MODEL);

    
//     // Load weights for conv1d
//     load_weights("../weights/conv1d.weight_weights.txt", &conv1d_weights[0][0], D_INNER * D_CONV);
//     load_weights("../weights/conv1d.bias_weights.txt", conv1d_bias, D_INNER);

//     // Load weights for x_proj
//     load_weights("../weights/x_proj.weight_weights.txt", &x_proj_weights[0][0], D_INNER * 136);

//     // Load weights for dt_proj
//     load_weights("../weights/dt_proj.weight_weights.txt", &dt_proj_weights[0][0], 256 * 8);
//     load_weights("../weights/dt_proj.bias_weights.txt", dt_proj_bias, 256);

//     // Load weights for out_proj
//     load_weights("../weights/out_proj.weight_weights.txt", &out_proj_weights[0][0], D_INNER * D_MODEL);

//     // Optionally, log successful loading
//     printf("All weights loaded successfully.\n");
// }

