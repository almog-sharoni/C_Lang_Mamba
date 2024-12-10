#ifndef STATE_H
#define STATE_H

#include <stdint.h>
#include "config.h" 


// Function Declaration
void update_state(int8_t* input, int8_t* state, int channels, int kernel_size);

#endif // STATE_H
