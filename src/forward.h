#ifndef FORWARD_H
#define FORWARD_H

#include <stdint.h>
#include "config.h"
#include "model.h"


// Function Declaration
void forward(int8_t* input, int8_t* output, int batch, int seqlen);

#endif // FORWARD_H
