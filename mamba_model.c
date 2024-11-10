// mamba_model.c

#include "mamba_model.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// Activation function: SiLU
float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Fixed-point multiplication with scaling and saturation
int8_t fixed_point_mult(float a, float b) {
    float result = a * b;
    // Scale the result
    result = result / FIXED_POINT_SCALE;
    // Clamp to int8_t range
    if (result > 127.0f) result = 127.0f;
    if (result < -128.0f) result = -128.0f;
    return (int8_t)roundf(result);
}

// Initialize the Mamba model with fixed values
void initialize_model(MambaModel* model) {
    // Initialize in_proj_weight and in_proj_bias
    // For demonstration, set weights to 1.0f and biases to 0.0f
    for (int i = 0; i < D_INNER * 2; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            model->in_proj_weight[i][j] = 1.0f; // Simple initialization
        }
        model->in_proj_bias[i] = 0.0f;
    }

    // Initialize conv1d_weight and conv1d_bias
    for (int i = 0; i < D_INNER; i++) {
        for (int j = 0; j < 1; j++) {
            for (int k = 0; k < D_CONV; k++) {
                model->conv1d_weight[i][j][k] = 1.0f; // Simple initialization
            }
        }
        model->conv1d_bias[i] = 0.0f;
    }

    // Initialize x_proj_weight
    for (int i = 0; i < DT_RANK + D_STATE * 2; i++) {
        for (int j = 0; j < D_INNER; j++) {
            model->x_proj_weight[i][j] = 1.0f; // Simple initialization
        }
    }

    // Initialize dt_proj_weight and dt_proj_bias
    for (int i = 0; i < D_INNER; i++) {
        for (int j = 0; j < DT_RANK; j++) {
            model->dt_proj_weight[i][j] = 1.0f; // Simple initialization
        }
        model->dt_proj_bias[i] = 0.0f;
    }

    // Initialize A_log and D_param
    for (int i = 0; i < D_INNER; i++) {
        for (int j = 0; j < D_STATE; j++) {
            if (i == j) {
                model->A_log[i][j] = 0.0f; // log(1) = 0 for identity
            } else {
                model->A_log[i][j] = -1000.0f; // A large negative value to simulate log(0)
            }
        }
        model->D_param[i] = 1.0f; // Initialize D to 1.0f
    }

    // Initialize out_proj_weight and out_proj_bias
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < D_INNER; j++) {
            model->out_proj_weight[i][j] = 1.0f; // Simple initialization
        }
        model->out_proj_bias[i] = 0.0f;
    }

    // Initialize state to zero
    memset(model->state, 0, sizeof(model->state));

    // Initialize conv_buffers to zero
    for (int c = 0; c < D_INNER; c++) {
        for (int k = 0; k < D_CONV; k++) {
            model->conv_buffers[c][k] = 0.0f;
        }
    }
}

// Linear projection: output = weight * input + bias
void linear_forward(float output[D_INNER * 2], float input[D_MODEL], float weight[D_INNER * 2][D_MODEL], float bias[D_INNER * 2]) {
    for (int i = 0; i < D_INNER * 2; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < D_MODEL; j++) {
            output[i] += weight[i][j] * input[j];
        }
        output[i] += bias[i];
    }
}

// Depthwise 1D Convolution: output[D_INNER] = conv1d_weight[D_INNER][1][D_CONV] * conv_buffers[D_INNER][D_CONV] + conv1d_bias[D_INNER]
void depthwise_conv1d(float output[D_INNER], float input[D_INNER], float weight[D_INNER][1][D_CONV], float bias[D_INNER], float conv_buffers[D_INNER][D_CONV]) {
    for (int c = 0; c < D_INNER; c++) {
        // Shift buffer to left
        for (int k = 0; k < D_CONV - 1; k++) {
            conv_buffers[c][k] = conv_buffers[c][k + 1];
        }
        // Insert new input at the end
        conv_buffers[c][D_CONV - 1] = input[c];
        // Perform convolution
        output[c] = 0.0f;
        for (int k = 0; k < D_CONV; k++) {
            output[c] += weight[c][0][k] * conv_buffers[c][k];
        }
        // Add bias
        output[c] += bias[c];
    }
}

// State-space update: state = delta_val * (A * state_prev + D * new_state)
void state_update(MambaModel* model, float delta_val[D_INNER], float new_state[D_INNER]) {
    for (int i = 0; i < D_INNER; i++) {
        float updated_state = 0.0f;
        for (int j = 0; j < D_STATE; j++) {
            float exp_val = expf(-model->A_log[i][j]); // exp(0)=1 for j=0, exp(1000)=inf for j=1
            if (isfinite(exp_val)) {
                updated_state += exp_val * model->state[0][i * D_STATE + j];
            }
        }
        updated_state += model->D_param[i] * new_state[i];
        // Apply delta_val and fixed-point scaling
        float scaled_state = updated_state * delta_val[i];
        // Convert to fixed-point and clamp
        if (scaled_state > 127.0f) scaled_state = 127.0f;
        if (scaled_state < -128.0f) scaled_state = -128.0f;
        model->state[0][i] = scaled_state;
    }
}

// Output projection: output_int8 = out_proj_weight * state + out_proj_bias, scaled and clamped
void output_projection(int8_t output_int8[OUTPUT_DIM], float state[D_INNER * D_STATE], float weight[OUTPUT_DIM][D_INNER], float bias[OUTPUT_DIM]) {
    for (int o = 0; o < OUTPUT_DIM; o++) {
        float out = 0.0f;
        for (int j = 0; j < D_INNER; j++) {
            for (int k = 0; k < D_STATE; k++) {
                out += weight[o][j] * state[j * D_STATE + k];
            }
        }
        out += bias[o];
        // Scale and clamp
        out = out * FIXED_POINT_SCALE;
        if (out > 127.0f) out = 127.0f;
        if (out < -128.0f) out = -128.0f;
        output_int8[o] = (int8_t)roundf(out);
    }
}

// Complete forward pass for the Mamba model
void mamba_forward(MambaModel* model, int8_t input_int8[BATCH_SIZE][SEQ_LENGTH][INPUT_DIM], int8_t output_int8[BATCH_SIZE][SEQ_LENGTH][OUTPUT_DIM]) {
    // Iterate over each time step
    for (int t = 0; t < SEQ_LENGTH; t++) {
        // Prepare input for linear projection
        float input_float[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) {
            input_float[i] = (float)input_int8[0][t][i] / FIXED_POINT_SCALE;
        }

        // Linear projection
        float xz[D_INNER * 2];
        linear_forward(xz, input_float, model->in_proj_weight, model->in_proj_bias);

        // Split into x and z
        float x[D_INNER];
        float z[D_INNER];
        for (int i = 0; i < D_INNER; i++) {
            x[i] = xz[i];
            z[i] = xz[D_INNER + i];
        }

        // Convolution and activation
        float conv_output[D_INNER];
        depthwise_conv1d(conv_output, x, model->conv1d_weight, model->conv1d_bias, model->conv_buffers);
        for (int i = 0; i < D_INNER; i++) {
            conv_output[i] = silu(conv_output[i]);
        }

        // Intermediate projection
        float x_db[D_INNER];
        for (int i = 0; i < D_INNER; i++) {
            x_db[i] = 0.0f;
            for (int j = 0; j < D_INNER; j++) {
                x_db[i] += model->x_proj_weight[i][j] * conv_output[j];
            }
            // No bias for x_proj
        }

        // Split into dt, B, C
        float dt[DT_RANK];
        float B[D_STATE];
        float C[D_STATE];
        for (int i = 0; i < DT_RANK; i++) {
            dt[i] = x_db[i];
        }
        for (int i = 0; i < D_STATE; i++) {
            B[i] = x_db[DT_RANK + i];
            C[i] = x_db[DT_RANK + D_STATE + i];
        }

        // dt projection
        float dt_proj_output[D_INNER];
        for (int i = 0; i < D_INNER; i++) {
            dt_proj_output[i] = 0.0f;
            for (int j = 0; j < DT_RANK; j++) {
                dt_proj_output[i] += model->dt_proj_weight[i][j] * dt[j];
            }
            dt_proj_output[i] += model->dt_proj_bias[i];
        }

        // Apply softplus to dt
        float dt_softplus[D_INNER];
        for (int i = 0; i < D_INNER; i++) {
            dt_softplus[i] = logf(1.0f + expf(dt_proj_output[i]));
        }

        // State update
        state_update(model, dt_softplus, conv_output);

        // Output projection
        int8_t out_int8[OUTPUT_DIM];
        output_projection(out_int8, model->state[0], model->out_proj_weight, model->out_proj_bias);

        // Assign to output
        for (int o = 0; o < OUTPUT_DIM; o++) {
            output_int8[0][t][o] = out_int8[o];
        }
    }
}
