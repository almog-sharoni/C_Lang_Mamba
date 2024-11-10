// mamba_model.h

#ifndef MAMBA_MODEL_H
#define MAMBA_MODEL_H

#include <stdint.h>

// Define model dimensions
#define BATCH_SIZE 1
#define SEQ_LENGTH 2
#define INPUT_DIM 4
#define OUTPUT_DIM 2
#define D_MODEL 4
#define D_STATE 2
#define D_CONV 4
#define EXPAND 2
#define D_INNER (EXPAND * D_MODEL) // 8
#define DT_RANK 1
#define STATE_DIM (D_INNER * D_STATE) // 16
#define FIXED_POINT_SHIFT 4
#define FIXED_POINT_SCALE (1 << FIXED_POINT_SHIFT)

// Define the Mamba model structure
typedef struct {
    // Linear projections
    float in_proj_weight[D_INNER * 2][D_MODEL]; // [16][4]
    float in_proj_bias[D_INNER * 2];           // [16]

    // Convolutional layer (depthwise)
    float conv1d_weight[D_INNER][1][D_CONV];   // [8][1][4]
    float conv1d_bias[D_INNER];                // [8]

    // Intermediate projections
    float x_proj_weight[DT_RANK + D_STATE * 2][D_INNER]; // [5][8]
    float dt_proj_weight[D_INNER][DT_RANK];              // [8][1]
    float dt_proj_bias[D_INNER];                          // [8]

    // State-space parameters
    float A_log[D_INNER][D_STATE]; // [8][2]
    float D_param[D_INNER];        // [8]

    // Output projection
    float out_proj_weight[OUTPUT_DIM][D_INNER]; // [2][8]
    float out_proj_bias[OUTPUT_DIM];            // [2]

    // State vectors
    float state[BATCH_SIZE][STATE_DIM]; // [1][16]

    // Convolution buffers
    float conv_buffers[D_INNER][D_CONV]; // [8][4]
} MambaModel;

// Function prototypes
void initialize_model(MambaModel* model);
float silu(float x);
int8_t fixed_point_mult(float a, float b);
void linear_forward(float output[D_INNER * 2], float input[D_MODEL], float weight[D_INNER * 2][D_MODEL], float bias[D_INNER * 2]);
void depthwise_conv1d(float output[D_INNER], float input[D_INNER], float weight[D_INNER][1][D_CONV], float bias[D_INNER], float conv_buffers[D_INNER][D_CONV]);
void state_update(MambaModel* model, float delta_val[D_INNER], float new_state[D_INNER]);
void output_projection(int8_t output_int8[OUTPUT_DIM], float state[D_INNER * D_STATE], float weight[OUTPUT_DIM][D_INNER], float bias[OUTPUT_DIM]);
void mamba_forward(MambaModel* model, int8_t input_int8[BATCH_SIZE][SEQ_LENGTH][INPUT_DIM], int8_t output_int8[BATCH_SIZE][SEQ_LENGTH][OUTPUT_DIM]);

#endif // MAMBA_MODEL_H
