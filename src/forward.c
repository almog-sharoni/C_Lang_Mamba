#include "forward.h"
#include "utils.h"
#include "state.h"
#include "config.h" 
#include "model.h"

#include <stdlib.h>
#include <stdio.h>

void forward(int8_t* hidden_states, int8_t* output, int batch, int seqlen) {
    // Debug: Print dimensions
    printf("Forward pass: batch=%d, seqlen=%d, D_MODEL=%d, D_INNER=%d\n", batch, seqlen, D_MODEL, D_INNER);

    // Dynamically allocate memory for large arrays
    int8_t* xz = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_INNER * EXPAND * sizeof(int8_t));
    int8_t* x = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_INNER * sizeof(int8_t));
    int8_t* z = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_INNER * sizeof(int8_t));
    int8_t* conv_output = (int8_t*)malloc(BATCH_SIZE * SEQ_LEN * D_INNER * sizeof(int8_t));
    int8_t* x_proj_output = (int8_t*)malloc(batch * seqlen * X_PROJ_OUT * sizeof(int8_t));
    // Initialize state buffers (global or passed in as arguments)
    int8_t* conv_state = (int8_t*)calloc(BATCH_SIZE * D_INNER * D_CONV, sizeof(int8_t));
    float* ssm_state = (float*)calloc(BATCH_SIZE * D_INNER * D_STATE, sizeof(float));
    float* dt = (float*)malloc(batch * seqlen * DT_RANK * sizeof(float));
    float* dA = (float*)malloc(batch * seqlen * D_STATE * sizeof(float));
    float* dB = (float*)malloc(batch * seqlen * D_STATE * sizeof(float));
    int8_t* B = (int8_t*)malloc(batch * seqlen * D_STATE * sizeof(int8_t));
    int8_t* C = (int8_t*)malloc(batch * seqlen * D_STATE * sizeof(int8_t));

    if (!dt || !dA || !dB || !B || !C) {
        perror("Failed to allocate memory for dt, dA, dB, B, or C");
        exit(1);
    }
    
    if (!conv_state || !ssm_state) {
        perror("Failed to allocate memory for states");
        exit(1);
    }
    if (x_proj_output == NULL) {
        perror("Failed to allocate memory for x_proj_output");
        exit(1);
    }

    if (xz == NULL || x == NULL || z == NULL || conv_output == NULL) {
        perror("Failed to allocate memory");
        exit(1);
    }
    printf("Memory allocated successfully.\n");

    // // print all values of hidden_states
    // for (int i = 0; i < batch * seqlen * D_MODEL; i++) {
    //     printf("%d ", hidden_states[i]);
    // }
    // printf("\n");




    // Expand hidden_states to match D_MODEL * EXPAND
    printf("Expanding hidden_states...\n");
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seqlen; s++) {
            for (int d = 0; d < D_MODEL; d++) {
                for (int e = 0; e < EXPAND; e++) {
                    xz[b * seqlen * D_INNER + s * D_INNER + d * EXPAND + e] =
                        hidden_states[b * seqlen * D_MODEL + s * D_MODEL + d];
                }
            }
        }
    }
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seqlen; s++) {
            for (int d = 0; d < D_INNER; d++) {
                ssm_state[b * D_INNER * D_STATE + d] = ((float)xz[b * seqlen * D_INNER + s * D_INNER + d]) / SCALE_WEIGHTS;
            }
        }
    }
    printf("First few values of ssm_state after initialization:\n");
    
    printf("First few values of hidden_states:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", hidden_states[i]);
    }
    printf("\n");

    // Debug: Print first few values of dA, dB, and ssm_state
    // maybe problem with dA
    printf("First few values of dA:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", dA[i]);
    }
    printf("\n");
    printf("First few values of dB:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", dB[i]);
    }
    printf("\n");
    printf("First few values of ssm_state:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", ssm_state[i]);
    }
    printf("\n");
    printf("**********\n");
    
    printf("Expansion completed.\n");

    // // print all values of hidden_states
    // for (int i = 0; i < batch * seqlen * D_INNER; i++) {
    //     printf("%d ", hidden_states[i]);
    // }
    // printf("\n");
    printf("First few values of in_proj_weights:\n");
    for (int i = 0; i < 10; i++) {
        printf("%hhd ", ((int8_t*)in_proj_weights)[i]);
    }
    printf("\n");
    // Use matmul with explicit casting
    printf("Performing matmul for in_proj...\n");
    matmul(hidden_states, (int8_t*)in_proj_weights, xz, batch, seqlen, D_INNER, SCALE_WEIGHTS);
    printf("Matmul for in_proj completed.\n");

    // Debug: Print first few values of xz
    printf("First few values of xz:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", xz[i]);
    }
    printf("\n");

    // Use split_xz with explicit casting
    printf("Splitting xz into x and z...\n");
    split_xz(xz, x, z, batch, seqlen, D_INNER);
    printf("Splitting completed.\n");

    // Debug: Print first few values of z
    printf("First few values of z:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", z[i]);
    }
    
    printf("\n");


    // Debug: Print first few values of x
    printf("First few values of x:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");

    // Use conv1d with explicit casting
    printf("Performing conv1d...\n");
    conv1d(x, (int8_t*)conv1d_weights, conv_output, D_INNER, seqlen, D_CONV, SCALE_WEIGHTS);
    printf("conv1d completed.\n");

    // Debug: Print first few values of conv_output
    printf("First few values of conv_output:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", conv_output[i]);
    }
    printf("\n");

    // Use apply_silu directly (already flat pointer)
    printf("Applying SiLU activation...\n");
    apply_silu(conv_output, batch, seqlen, D_INNER, SCALE_WEIGHTS);
    printf("SiLU activation applied.\n");

    // Debug: Print first few values of activated conv_output
    printf("First few values of activated conv_output:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", conv_output[i]);
    }
    printf("\n");

    // Use matmul for output projection with explicit casting
    printf("Performing matmul for x_proj...\n");
    matmul(conv_output, (int8_t*)x_proj_weights, x_proj_output, batch, seqlen, X_PROJ_OUT, SCALE_WEIGHTS);
    printf("Matmul for x_proj completed.\n");

    // Debug: Print first few values of output
    printf("First few values of x_proj_output:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", x_proj_output[i]);
    }
    printf("\n");

    // Split x_proj_output into dt, B, and C
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seqlen; s++) {
            for (int i = 0; i < DT_RANK; i++) {
                dt[b * seqlen * DT_RANK + s * DT_RANK + i] = 
                    softplus(x_proj_output[b * seqlen * (DT_RANK + 2 * D_STATE) + 
                                        s * (DT_RANK + 2 * D_STATE) + i] * SCALE_WEIGHTS + 
                            dt_proj_bias[i]);
            }
            for (int i = 0; i < D_STATE; i++) {
                B[b * seqlen * D_STATE + s * D_STATE + i] = 
                    x_proj_output[b * seqlen * (DT_RANK + 2 * D_STATE) + 
                                s * (DT_RANK + 2 * D_STATE) + DT_RANK + i];

                C[b * seqlen * D_STATE + s * D_STATE + i] = 
                    x_proj_output[b * seqlen * (DT_RANK + 2 * D_STATE) + 
                                s * (DT_RANK + 2 * D_STATE) + DT_RANK + D_STATE + i];
            }
        }
    }

    srand(time(NULL)); // Seed for randomness
    for (int i = 0; i < D_STATE; i++) {
        B[i] = (rand() % (64 + 64 + 1)) - 64;
    }
    // Debug: Print first few values of B
    printf("First few values of B after initialization:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", B[i]);
    }
    printf("\n");

    printf("First few values of A_log:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", A_log[i]);
    }
    printf("\n");

    // Compute dA and dB
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seqlen; s++) {
            for (int i = 0; i < D_STATE; i++) {
                dA[b * seqlen * D_STATE + s * D_STATE + i] = 
                    expf(dt[b * seqlen * DT_RANK + s * DT_RANK] * expf(A_log[i]));

                dB[b * seqlen * D_STATE + s * D_STATE + i] = 
                    dt[b * seqlen * DT_RANK + s * DT_RANK] * 
                    B[b * seqlen * D_STATE + s * D_STATE + i];
            }
        }
    }

    // Update ssm_state
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seqlen; s++) {
            for (int i = 0; i < D_STATE; i++) {
                ssm_state[b * seqlen * D_STATE + s * D_STATE + i] =
                    ssm_state[b * seqlen * D_STATE + s * D_STATE + i] * 
                    dA[b * seqlen * D_STATE + s * D_STATE + i] +
                    conv_output[b * seqlen * D_STATE + s * D_STATE + i] * 
                    dB[b * seqlen * D_STATE + s * D_STATE + i];
            }
        }
    }
    // Debug: Print first few values of dt
    printf("First few values of dt:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", dt[i]);
    }
    printf("\n");
    // Debug: Print first few values of B
    printf("First few values of B:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", B[i]);
    }
    printf("\n");
    // Debug: Print first few values of C
    printf("First few values of C:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");



    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seqlen; s++) {
            for (int i = 0; i < D_MODEL; i++) {
                output[b * seqlen * D_MODEL + s * D_MODEL + i] =
                    ssm_state[b * seqlen * D_STATE + s * D_STATE + i] *
                    C[b * seqlen * D_STATE + s * D_STATE + i] *
                    z[b * seqlen * D_INNER + s * D_INNER + i];
            }
        }
    }

    // Debug: Print first few values of output
    printf("First few values of output:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", output[i]);
    }

    // printf("First few values of ssm_state before final output:\n");
    // for (int i = 0; i < 10; i++) {
    //     printf("%f ", ssm_state[i]);
    // }
    // printf("\n");
    // printf("First few values of C before final output:\n");
    // for (int i = 0; i < 10; i++) {
    //     printf("%d ", C[i]);
    // }
    // printf("\n");
    // printf("First few values of z before final output:\n");
    // for (int i = 0; i < 10; i++) {
    //     printf("%d ", z[i]);
    // }
    // printf("\n");

    // Free dynamically allocated memory
    free(xz);
    free(x);
    free(z);
    free(conv_output);
    free(dt);
    free(dA);
    free(dB);
    free(B);
    free(C);
    printf("Memory freed successfully.\n");
}

// void forward(int8_t* hidden_states, int8_t* output, int8_t* in_proj_weights, int8_t* conv1d_weights, 
//              int8_t* x_proj_weights, float* dt_proj_bias, float* A_log, int8_t* ssm_state,
//              int batch, int seqlen) {
//     // Allocate memory for intermediate arrays
//     int8_t* xz = (int8_t*)malloc(batch * seqlen * D_INNER * 2 * sizeof(int8_t));
//     int8_t* x = (int8_t*)malloc(batch * seqlen * D_INNER * sizeof(int8_t));
//     int8_t* z = (int8_t*)malloc(batch * seqlen * D_INNER * sizeof(int8_t));
//     int8_t* conv_output = (int8_t*)malloc(batch * seqlen * D_INNER * sizeof(int8_t));
//     int8_t* x_proj_output = (int8_t*)malloc(batch * seqlen * (DT_RANK + 2 * D_STATE) * sizeof(int8_t));
//     float* dt = (float*)malloc(batch * seqlen * DT_RANK * sizeof(float));
//     float* dA = (float*)malloc(batch * seqlen * D_STATE * sizeof(float));
//     float* dB = (float*)malloc(batch * seqlen * D_STATE * sizeof(float));
//     int8_t* B = (int8_t*)malloc(batch * seqlen * D_STATE * sizeof(int8_t));
//     int8_t* C = (int8_t*)malloc(batch * seqlen * D_STATE * sizeof(int8_t));

//     if (!xz || !x || !z || !conv_output || !x_proj_output || !dt || !dA || !dB || !B || !C) {
//         perror("Failed to allocate memory");
//         exit(1);
//     }

//     // Step 1: Input Projection
//     printf("Performing input projection...\n");
//     matmul(hidden_states, in_proj_weights, xz, batch, seqlen, D_INNER * 2, SCALE_WEIGHTS);

//     // Split xz into x and z
//     printf("Splitting xz into x and z...\n");
//     for (int b = 0; b < batch; b++) {
//         for (int s = 0; s < seqlen; s++) {
//             for (int i = 0; i < D_INNER; i++) {
//                 x[b * seqlen * D_INNER + s * D_INNER + i] = xz[b * seqlen * D_INNER * 2 + s * D_INNER * 2 + i];
//                 z[b * seqlen * D_INNER + s * D_INNER + i] = xz[b * seqlen * D_INNER * 2 + s * D_INNER * 2 + D_INNER + i];
//             }
//         }
//     }

//     // Step 2: Convolution
//     printf("Performing 1D convolution...\n");
//     conv1d(x, conv1d_weights, conv_output, D_INNER, seqlen, D_CONV, SCALE_WEIGHTS);

//     // Apply SiLU activation
//     printf("Applying SiLU activation...\n");
//     apply_silu(conv_output, batch, seqlen, D_INNER, SCALE_WEIGHTS);

//     // Step 3: Feature Projection
//     printf("Performing feature projection...\n");
//     matmul(conv_output, x_proj_weights, x_proj_output, batch, seqlen, DT_RANK + 2 * D_STATE, SCALE_WEIGHTS);

//     // Split x_proj_output into dt, B, and C
//     printf("Splitting feature projection output into dt, B, and C...\n");
//     for (int b = 0; b < batch; b++) {
//         for (int s = 0; s < seqlen; s++) {
//             for (int i = 0; i < DT_RANK; i++) {
//                 dt[b * seqlen * DT_RANK + s * DT_RANK + i] = softplus(
//                     x_proj_output[b * seqlen * (DT_RANK + 2 * D_STATE) + s * (DT_RANK + 2 * D_STATE) + i] * SCALE_WEIGHTS + dt_proj_bias[i]);
//             }
//             for (int i = 0; i < D_STATE; i++) {
//                 B[b * seqlen * D_STATE + s * D_STATE + i] = x_proj_output[b * seqlen * (DT_RANK + 2 * D_STATE) + s * (DT_RANK + 2 * D_STATE) + DT_RANK + i];
//                 C[b * seqlen * D_STATE + s * D_STATE + i] = x_proj_output[b * seqlen * (DT_RANK + 2 * D_STATE) + s * (DT_RANK + 2 * D_STATE) + DT_RANK + D_STATE + i];
//             }
//         }
//     }

//     // Step 4: State Updates
//     printf("Updating state...\n");

//     // Compute dA
//     for (int b = 0; b < batch; b++) {
//         for (int s = 0; s < seqlen; s++) {
//             for (int i = 0; i < D_STATE; i++) {
//                 dA[b * seqlen * D_STATE + s * D_STATE + i] = expf(dt[b * seqlen * DT_RANK + s * DT_RANK] * expf(A_log[i]));
//             }
//         }
//     }

//     // Compute dB
//     for (int b = 0; b < batch; b++) {
//         for (int s = 0; s < seqlen; s++) {
//             for (int i = 0; i < D_STATE; i++) {
//                 dB[b * seqlen * D_STATE + s * D_STATE + i] = dt[b * seqlen * DT_RANK + s * DT_RANK] * B[b * seqlen * D_STATE + s * D_STATE + i];
//             }
//         }
//     }

//     // Update ssm_state
//     for (int b = 0; b < batch; b++) {
//         for (int s = 0; s < seqlen; s++) {
//             for (int i = 0; i < D_STATE; i++) {
//                 ssm_state[b * seqlen * D_STATE + s * D_STATE + i] =
//                     ssm_state[b * seqlen * D_STATE + s * D_STATE + i] * dA[b * seqlen * D_STATE + s * D_STATE + i] +
//                     conv_output[b * seqlen * D_STATE + s * D_STATE + i] * dB[b * seqlen * D_STATE + s * D_STATE + i];
//             }
//         }
//     }

//     // Step 5: Output Projection
//     printf("Applying SiLU activation to z...\n");
//     // Apply SiLU to the entire z tensor
//     apply_silu(z, batch, seqlen, D_INNER, SCALE_WEIGHTS);

//     printf("Performing output projection...\n");
//     for (int b = 0; b < batch; b++) {
//         for (int s = 0; s < seqlen; s++) {
//             for (int i = 0; i < D_MODEL; i++) {
//                 output[b * seqlen * D_MODEL + s * D_MODEL + i] =
//                     ssm_state[b * seqlen * D_STATE + s * D_STATE + i] * 
//                     C[b * seqlen * D_STATE + s * D_STATE + i] *
//                     z[b * seqlen * D_INNER + s * D_INNER + i];  // Use activated z
//             }
//         }
//     }

//     // Free allocated memory
//     printf("Freeing allocated memory...\n");
//     free(xz);
//     free(x);
//     free(z);
//     free(conv_output);
//     free(x_proj_output);
//     free(dt);
//     free(dA);
//     free(dB);
//     free(B);
//     free(C);
//     printf("Forward pass completed.\n");
// }