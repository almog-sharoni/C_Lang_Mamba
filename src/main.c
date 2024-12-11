#include "forward.h"
#include "model.h"
#include "config.h" 
#include "utils.h"


// int main2() {
//     // Initialize model weights and states
//     initialize_weights();
    
//     // Test forward pass with dummy input   
//     // Allocate inputs and outputs
//         // int batch = 1; // Batch size
//         // int seqlen = 64; // Sequence length
//         int8_t input[BATCH_SIZE * SEQ_LEN * D_MODEL]; // Input tensor
//         int8_t output[BATCH_SIZE * SEQ_LEN * D_MODEL]; // Output tensor

//         // Initialize input with random values
//         for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; i++) {
//             // input[i] = i % 127; // Simple pattern
//             input[i] = (int8_t)(rand() % 256 - 128);
//         }
        
        
//         // Perform forward pass
//         forward(input, output, BATCH_SIZE, SEQ_LEN);

//         // // Print first few outputs for verification
//         // printf("Output:\n");
//         // for (int i = 0; i < 10; i++) {
//         //     printf("%d ", output[i]);
//         // }
//         // printf("\n");

//         // print all values of output
//         for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; i++) {
//             printf("%d ", output[i]);
//         }

//     return 0;
// }

// File I/O functions
void read_input_from_file(const char* filename, int8_t* input, int size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%hhd", &input[i]) != 1) {  // Check fscanf return value
            fprintf(stderr, "Error reading input at index %d\n", i);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
    printf("Input successfully loaded from file: %s\n", filename);
}


void write_output_to_file(const char* filename, int8_t* output, int size) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Failed to open output file");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%d\n", output[i]);
    }
    fclose(file);
}

// Main function
int main() {
    int batch = BATCH_SIZE;
    int seqlen = SEQ_LEN;
    int input_size = batch * seqlen * D_MODEL;
    int output_size = batch * seqlen * D_INNER;

    int8_t* input = malloc(input_size * sizeof(int8_t));
    int8_t* output = malloc(output_size * sizeof(int8_t));

    // print input length
    printf("Input size: %d\n", input_size);



    if (!input || !output) {
        perror("Failed to allocate memory");
        return 1;
    }

    read_input_from_file("data/input.txt", input, input_size);
    // Print first few inputs for verification
    printf("First few inputs:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", input[i]);
    }
    printf("\n");

    initialize_weights();

    forward(input, output, batch, seqlen);

    write_output_to_file("data/output_c.txt", output, output_size);

    // // print all values of output
    // for (int i = 0; i < 4096; i++) {
    //     printf("%d ", output[i]);
    // }

    free(input);
    free(output);

    return 0;
}