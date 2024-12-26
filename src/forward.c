#include "forward.h"
#include "utils.h"
#include "state.h"
#include "config.h"
#include "model.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>




#define CHECK_ALLOC(ptr) if (!(ptr)) { perror("Memory allocation failed"); exit(EXIT_FAILURE); }


#define SCALE_SHIFT 4  // Scale factor (2^6 = 64)
#define MAX_INT8 127
#define MIN_INT8 -128

// Integer approximation of exp function using a simple lookup table
int8_t approximate_exp(int8_t x) {
    if (x < MIN_INT8) return 0;  // Minimum value for exp
    if (x > MAX_INT8) return MAX_INT8;  // Clamp to max int8
    return (int8_t)((1 + (x >> 4)) << SCALE_SHIFT);  // Simplistic scaling
}

// Integer softplus approximation
int8_t approximate_softplus(int8_t x) {
    if (x < -64) return 0;  // Approximation for very small values
    if (x > 64) return x;   // Approximation for large positive values
    return (x + (SCALE_SHIFT / 2));  // Linear softplus
}

void forward(int8_t* hidden_states, int8_t* output) {
    printf("Forward pass: BATCH_SIZE=%d, SEQ_LEN=%d, D_MODEL=%d, D_INNER=%d\n", BATCH_SIZE, SEQ_LEN, D_MODEL, D_INNER);

    int8_t* xz = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_INNER * 2 * sizeof(int8_t));
    int8_t* x = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_INNER * sizeof(int8_t));
    int8_t* z = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_INNER * sizeof(int8_t));
    int8_t* conv_output = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_INNER * sizeof(int8_t));
    int8_t* x_proj_output = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * X_PROJ_OUT * sizeof(int8_t));
    int8_t* ssm_state = (int8_t*)calloc(BATCH_SIZE * SEQ_LEN * D_STATE, sizeof(int8_t));
    int8_t* dt = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * DT_RANK * sizeof(int8_t));
    int8_t* dA = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_STATE * sizeof(int8_t));
    int8_t* dB = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_STATE * sizeof(int8_t));
    int8_t* B = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_STATE * sizeof(int8_t));
    int8_t* C = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_STATE * sizeof(int8_t));

    CHECK_ALLOC(dt);
    CHECK_ALLOC(dA);
    CHECK_ALLOC(dB);
    CHECK_ALLOC(B);
    CHECK_ALLOC(C);
    CHECK_ALLOC(ssm_state);
    CHECK_ALLOC(xz);
    CHECK_ALLOC(x);
    CHECK_ALLOC(z);
    CHECK_ALLOC(conv_output);
    CHECK_ALLOC(x_proj_output);


//     printf("Expanding hidden_states...\n");
//     // for (int b = 0; b < BATCH_SIZE; b++) {
//     //     for (int s = 0; s < SEQ_LEN; s++) {
//     //         for (int d = 0; d < D_MODEL; d++) {
//     //             for (int e = 0; e < EXPAND; e++) {
//     //                 xz[b * SEQ_LEN * D_INNER + s * D_INNER + d * EXPAND + e] =
//     //                     hidden_states[b * SEQ_LEN * D_MODEL + s * D_MODEL + d];
//     //             }
//     //         }
//     //     }
//     // }

//     #include <assert.h>

// // Within the expansion loop
// for (int b = 0; b < BATCH_SIZE; b++) {
//     for (int s = 0; s < SEQ_LEN; s++) {
//         for (int d = 0; d < D_MODEL; d++) {
//             for (int e = 0; e < EXPAND; e++) {
//                 int xz_idx = b * SEQ_LEN * D_INNER + s * D_INNER + d * EXPAND + e;
//                 int hidden_idx = b * SEQ_LEN * D_MODEL + s * D_MODEL + d;
//                 assert(xz_idx < (BATCH_SIZE * SEQ_LEN * D_INNER));
//                 assert(hidden_idx < (BATCH_SIZE * SEQ_LEN * D_MODEL));
//                 xz[xz_idx] = hidden_states[hidden_idx];
//             }
//         }
//     }
// }
//     printf("First few values of xz after expansion:\n");
//     for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_INNER; i++) {
//         printf("%d ", xz[i]);
//     }
//     printf("\n");

    printf("Performing matmul for in_proj...\n");
    // matmul(hidden_states, (int8_t*)in_proj_weights, xz, BATCH_SIZE, SEQ_LEN, D_INNER*2, SCALE_SHIFT);
    matmul_expand(hidden_states, (int8_t*)in_proj_weights, xz, 
              BATCH_SIZE, SEQ_LEN, D_INNER , 2, SCALE_SHIFT);
    printf("First few values of xz after matmul:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", xz[i]);
    }
    printf("\n");

    printf("Splitting xz into x and z...\n");
    split_xz(xz, x, z, BATCH_SIZE, SEQ_LEN, D_INNER);
    printf("First few values of x:\n");
    for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_INNER; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");
    printf("First few values of z:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", z[i]);
    }
    printf("\n");

    //  !! PROBLEM: The following code is not working as expected. The output is not as expected. !!
    printf("Performing conv1d...\n");
    // conv1d(x, (int8_t*)conv1d_weights, conv_output, D_INNER, SEQ_LEN, D_CONV, SCALE_SHIFT);
    conv1d(
    x,                       // Input tensor (e.g., `conv_output` from the previous step)
    (int8_t*)conv1d_weights, // Weights for conv1d (flattened)
    conv_output,             // Output tensor (pre-allocated)
    BATCH_SIZE,              // Number of batches
    D_INNER,                 // Number of input channels (D_INNER = D_MODEL * EXPAND)
    SEQ_LEN,                 // Sequence length
    D_CONV,                  // Kernel size for conv1d
    SCALE_SHIFT              // Scaling factor for quantization
);
    printf("First few values of conv_output after conv1d:\n");
    for (int i = 0; i < 16384; i++) {
        printf("%d ", conv_output[i]);
    }
    printf("\n");

    printf("Applying SiLU activation...\n");
    apply_silu(conv_output, BATCH_SIZE, SEQ_LEN, D_INNER);
    printf("First few values of conv_output after SiLU:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", conv_output[i]);
    }
    printf("\n");
    //random x_proj_output
    for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_INNER; i++) {
        x_proj_output[i] = -1;
    }

    printf("Projecting conv_output with dt_proj_weights...\n");
    // matmul(conv_output, (int8_t*)dt_proj_weights, x_proj_output, BATCH_SIZE, SEQ_LEN, D_MODEL, SCALE_SHIFT);
    // matmul_and_concatenate(conv_output, x_proj_weights, x_proj_output, 
    //                    BATCH_SIZE, SEQ_LEN, D_INNER, DT_RANK, D_STATE, SCALE_SHIFT);
    matmul_and_concatenate(conv_output, (int8_t *)x_proj_weights, x_proj_output,
                       BATCH_SIZE, SEQ_LEN, D_INNER, DT_RANK, D_STATE, SCALE_SHIFT);
    printf("First few values of x_proj_output after dt_proj_weights matmul:\n");
    for (int i = 0; i < (256); i++) {
        printf("%d ", x_proj_output[i]);
    }
    printf("\n");

    printf("Adding dt_proj_bias and computing dt using approximate_softplus...\n");
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            for (int i = 0; i < DT_RANK; i++) {
                int idx = b * SEQ_LEN * DT_RANK + s * DT_RANK + i;
                dt[idx] = approximate_softplus(x_proj_output[idx] + dt_proj_bias[i]);
            }
        }
    }
    printf("First few values of dt:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", dt[i]);
    }
    printf("\n");

    printf("Splitting x_proj_output into B and C...\n");
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            for (int i = 0; i < D_STATE; i++) {
                B[b * SEQ_LEN * D_STATE + s * D_STATE + i] =
                    x_proj_output[b * SEQ_LEN * (DT_RANK + 2 * D_STATE) + s * (DT_RANK + 2 * D_STATE) + DT_RANK + i];
                C[b * SEQ_LEN * D_STATE + s * D_STATE + i] =
                    x_proj_output[b * SEQ_LEN * (DT_RANK + 2 * D_STATE) + s * (DT_RANK + 2 * D_STATE) + DT_RANK + D_STATE + i];
            }
        }
    }
    printf("First few values of B:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", B[i]);
    }
    printf("\n");
    printf("First few values of C:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");

    // Continue updating `dA`, `dB`, and `ssm_state` similarly with integer arithmetic.
    printf("Computing dA and dB...\n");
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            for (int i = 0; i < D_STATE; i++) {
                // int inner_idx = (b * SEQ_LEN + s) % D_INNER; // Determine the corresponding inner index
                int inner_idx = s % D_INNER;
                // Compute dA using integer approximation
                int16_t scaled_exp_A = approximate_exp(A_log[inner_idx][i]); // Scaled exponential value of A_log
                int16_t value = (dt[b * SEQ_LEN * DT_RANK + s * DT_RANK] * scaled_exp_A) >> SCALE_SHIFT;
                dA[b * SEQ_LEN * D_STATE + s * D_STATE + i] = approximate_exp(value);

                // Compute dB as scaled multiplication
                dB[b * SEQ_LEN * D_STATE + s * D_STATE + i] = 
                    (dt[b * SEQ_LEN * DT_RANK + s * DT_RANK] * B[b * SEQ_LEN * D_STATE + s * D_STATE + i]) >> SCALE_SHIFT;
            }
        }
    }


    // Debug: Print first few values of dA and dB
    printf("First few values of dA:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", dA[i]);
    }
    printf("\n");
    printf("First few values of dB:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", dB[i]);
    }
    printf("\n");

printf("Updating ssm_state...\n");

// // first few values of ssm_state
// printf("First few values of ssm_state before update:\n");
// for (int i = 0; i < 1000; i++) {
//     int SEQ_LEN = 128;
//     int idx = SEQ_LEN / 2 + i;
//     printf("%d ", ssm_state[idx]);

// }
// printf("\n");

for (int b = 0; b < BATCH_SIZE; b++) {
    for (int s = 0; s < SEQ_LEN; s++) {
        for (int i = 0; i < D_STATE; i++) {
            // Fetch indices
            int idx = b * SEQ_LEN * D_STATE + s * D_STATE + i;

            // Calculate state_term
            int32_t state_mult = ssm_state[idx] * dA[idx];
            int16_t state_term = state_mult >> SCALE_SHIFT;

            // // Debug: Print intermediate state_term
            // if (b == 0 && s < 2 && i < 5) { // Example: Debug the first few states
            //     printf("state_mult[%d][%d][%d]: %d, state_term: %d\n", b, s, i, state_mult, state_term);
            // }

            // Calculate input_term
            int32_t input_mult = conv_output[idx] * dB[idx];
            int16_t input_term = input_mult >> SCALE_SHIFT;

            // // Debug: Print intermediate input_term
            // if (b == 0 && s < 2 && i < 5) { // Example: Debug the first few inputs
            //     printf("input_mult[%d][%d][%d]: %d, input_term: %d\n", b, s, i, input_mult, input_term);
            // }

            // Compute the updated ssm_state
            int32_t updated_state = state_term + input_term;

            // Clamp to int8_t range
            if (updated_state > 127) updated_state = 127;
            if (updated_state < -128) updated_state = -128;

            ssm_state[idx] = (int8_t)updated_state;

            // // Debug: Print the updated state
            // if (b == 0 && s < 5 && i < 5) { // Example: Debug the first few updated states
            //     printf("updated_state[%d][%d][%d]: %d\n", b, s, i, updated_state);
            // }
        }
    }
}


    // Debug: Print first few values of ssm_state after update
    printf("First few values of ssm_state after update:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", ssm_state[i]);
    }
    printf("\n");

  printf("Computing final output...\n");

    // Initialize variables to find the max and min values
    int16_t max_value = 127;
    int16_t min_value = -128;

    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            for (int i = 0; i < D_MODEL; i++) {
                // Compute output as scaled multiplication of ssm_state, C, and z
                int16_t state_term = (ssm_state[b * SEQ_LEN * D_STATE + s * D_STATE + i] * C[b * SEQ_LEN * D_STATE + s * D_STATE + i]) >> SCALE_SHIFT;
                int16_t final_output = (state_term * z[b * SEQ_LEN * D_INNER + s * D_INNER + i]) >> SCALE_SHIFT;

                // Update max and min values
                if (final_output > max_value) max_value = final_output;
                if (final_output < min_value) min_value = final_output;
            }
        }
    }

    // Calculate the scaling factor
    int16_t range = max_value - min_value;
    float scale_factor = (range > 255) ? 255.0f / range : 1.0f;

    printf("Dynamic scaling factor: %f\n", scale_factor);

    // // debug: fill output with -1
    // for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; i++) {
    //     output[i] = -1;
    // }

    // // printf("First few values of output:\n");
    // // for (int i = 0; i < 500000; i++) {
    // //     printf("%d ", output[i]);
    // // }
    // // printf("\n");



    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            for (int i = 0; i < D_MODEL; i++) {
                // Compute output as scaled multiplication of ssm_state, C, and z
                int16_t state_term = (ssm_state[b * SEQ_LEN * D_STATE + s * D_STATE + i] * C[b * SEQ_LEN * D_STATE + s * D_STATE + i]) >> SCALE_SHIFT;
                int16_t final_output = (state_term * z[b * SEQ_LEN * D_INNER + s * D_INNER + i]) >> SCALE_SHIFT;

                // Apply dynamic scaling
                final_output = (int16_t)(final_output * scale_factor);

                // Clamp to int8_t range after scaling
                if (final_output > 127) final_output = 127;
                if (final_output < -128) final_output = -128;

                output[b * SEQ_LEN * D_MODEL + s * D_MODEL + i] = (int8_t)final_output;
            }
        }
    }


    // Debug: Print first few values of output
    printf("First few values of output:\n");
    for (int i = 0; i < 1000; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");



    CHECK_ALLOC(dt);
    CHECK_ALLOC(dA);
    CHECK_ALLOC(dB);
    CHECK_ALLOC(B);
    CHECK_ALLOC(C);
    CHECK_ALLOC(ssm_state);
    CHECK_ALLOC(xz);
    CHECK_ALLOC(x);
    CHECK_ALLOC(z);
    CHECK_ALLOC(conv_output);
    CHECK_ALLOC(x_proj_output);

    // Free allocated memory
    free(xz);
    free(x);
    free(z);
    free(conv_output);
    free(x_proj_output);
    free(dt);
    free(dA);
    free(dB);
    free(B);
    free(C);
    free(ssm_state);
    printf("Memory freed successfully.\n");
}
