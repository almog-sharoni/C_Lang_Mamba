#include "state.h"
#include "config.h" 

void update_state(int8_t* input, int8_t* state, int channels, int kernel_size) {
    for (int c = 0; c < channels; c++) {
        for (int k = kernel_size - 1; k > 0; k--) {
            state[c * kernel_size + k] = state[c * kernel_size + k - 1];
        }
        state[c * kernel_size] = input[c];
    }
}
