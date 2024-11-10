// test_mamba_model.c

#include "mamba_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Function to compare outputs with expected values
int compare_outputs(int8_t custom_output[OUTPUT_DIM], int8_t expected_output[OUTPUT_DIM], float tolerance) {
    int pass = 1;
    for(int o = 0; o < OUTPUT_DIM; o++) {
        float diff = fabsf((float)custom_output[o] - (float)expected_output[o]);
        if(diff > tolerance) {
            printf("Mismatch at Output %d: Custom=%d, Expected=%d\n", o, custom_output[o], expected_output[o]);
            pass = 0;
        }
    }
    return pass;
}

int main() {
    // Initialize the Mamba model
    MambaModel model;
    initialize_model(&model);

    // Define test inputs with positive sums to avoid zero outputs
    // Example Test Cases:
    // Test Case 1: [[1, 1, 1, 1], [2, 2, 2, 2]]
    // Test Case 2: [[1, 2, 3, 4], [5, 6, 7, 8]]
    // Test Case 3: [[0, 1, 0, 1], [3, 3, 3, 3]]
    // Test Case 4: [[1, -1, 2, 2], [4, 4, 4, 4]]
    // Test Case 5: [[10, 10, 10, 10], [20, 20, 20, 20]]

    int8_t test_inputs[5][BATCH_SIZE][SEQ_LENGTH][INPUT_DIM] = {
        {   // Test Case 1
            {
                {1, 1, 1, 1},  // Sum = 4
                {2, 2, 2, 2}   // Sum = 8
            }
        },
        {   // Test Case 2
            {
                {1, 2, 3, 4},  // Sum = 10
                {5, 6, 7, 8}   // Sum = 26
            }
        },
        {   // Test Case 3
            {
                {0, 1, 0, 1},  // Sum = 2
                {3, 3, 3, 3}   // Sum = 12
            }
        },
        {   // Test Case 4
            {
                {1, -1, 2, 2}, // Sum = 4
                {4, 4, 4, 4}    // Sum = 16
            }
        },
        {   // Test Case 5
            {
                {10, 10, 10, 10}, // Sum = 40
                {20, 20, 20, 20}   // Sum = 80
            }
        }
    };

    // Define expected outputs
    // Given FIXED_POINT_SHIFT=4, and output_projection scales by 16
    // After state update: y[o] = sum_j(weight[o][j] * state[j * D_STATE + k])
    // With weights=1.0f, state[j * D_STATE +k]=~2.49, sum_j=8*2.49≈19.92
    // Scaled y[o]=19.92 *16≈318.72, clamped to 127
    // Thus, expected_outputs should be [127,127] for all test cases

    int8_t expected_outputs[5][OUTPUT_DIM] = {
        {127, 127}, // Expected Output for Test Case 1
        {127, 127}, // Test Case 2
        {127, 127}, // Test Case 3
        {127, 127}, // Test Case 4
        {127, 127}  // Test Case 5
    };

    // Iterate through test cases
    for(int tc = 0; tc < 5; tc++) {
        printf("Running Test Case %d:\n", tc+1);

        // Reset model state to zero before each test case
        for(int i = 0; i < D_INNER * D_STATE; i++) {
            model.state[0][i] = 0.0f;
        }

        // Reset convolution buffers to zero
        for(int c = 0; c < D_INNER; c++) {
            for(int k = 0; k < D_CONV; k++) {
                model.conv_buffers[c][k] = 0.0f;
            }
        }

        // Prepare input for current test case
        int8_t input[BATCH_SIZE][SEQ_LENGTH][INPUT_DIM];
        memcpy(input, test_inputs[tc], sizeof(input));

        // Prepare output buffer
        int8_t output[BATCH_SIZE][SEQ_LENGTH][OUTPUT_DIM];
        memset(output, 0, sizeof(output));

        // Run forward pass
        mamba_forward(&model, input, output);

        // Print outputs
        printf("Output from mamba_forward:\n");
        for(int b = 0; b < BATCH_SIZE; b++) {
            for(int t = 0; t < SEQ_LENGTH; t++) {
                printf("[");
                for(int o = 0; o < OUTPUT_DIM; o++) {
                    printf("%d", output[b][t][o]);
                    if(o < OUTPUT_DIM -1) printf(" ");
                }
                printf("]\n");
            }
        }

        // Compare with expected outputs
        int test_pass = 1;
        for(int b = 0; b < BATCH_SIZE; b++) {
            for(int t = 0; t < SEQ_LENGTH; t++) {
                if(!compare_outputs(output[b][t], expected_outputs[tc], 1.0f)) {
                    test_pass = 0;
                }
            }
        }

        if(test_pass) {
            printf("Test Case %d Passed.\n\n", tc+1);
        } else {
            printf("Test Case %d Failed.\n\n", tc+1);
        }
    }

    return 0;
}
