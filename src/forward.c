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
    printf("Expansion completed.\n");

    // // print all values of hidden_states
    // for (int i = 0; i < batch * seqlen * D_INNER; i++) {
    //     printf("%d ", hidden_states[i]);
    // }
    // printf("\n");

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
    for (int i = 0; i < batch * seqlen * D_INNER; i++) {
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

    // Free dynamically allocated memory
    free(xz);
    free(x);
    free(z);
    free(conv_output);
    printf("Memory freed successfully.\n");
}
