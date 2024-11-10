// forward.c

#include <stdio.h>
#include "mamba_model.h"

int main() {
    int8_t input[BATCH_SIZE][SEQ_LENGTH][INPUT_DIM];
    int8_t output[BATCH_SIZE][SEQ_LENGTH][OUTPUT_DIM];

    // Initialize input with some values
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int t = 0; t < SEQ_LENGTH; t++) {
            for (int i = 0; i < INPUT_DIM; i++) {
                input[b][t][i] = (int8_t)(i + t);
            }
        }
    }

    // Initialize the model
    mamba_init();

    // Run the forward pass
    mamba_forward(input, output);

    // Print the output
    for (int b = 0; b < BATCH_SIZE; b++) {
        printf("Batch %d Output:\n", b);
        for (int t = 0; t < SEQ_LENGTH; t++) {
            printf("Time Step %d: ", t);
            for (int i = 0; i < OUTPUT_DIM; i++) {
                printf("%d ", output[b][t][i]);
            }
            printf("\n");
        }
    }

    return 0;
}
