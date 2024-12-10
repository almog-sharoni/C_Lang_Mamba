#ifndef CONFIG_H
#define CONFIG_H

// Model Parameters
#define D_MODEL 128
#define EXPAND 2
#define D_INNER (D_MODEL * EXPAND)
#define D_STATE 64
#define D_CONV 64
#define BATCH_SIZE 2
#define SEQ_LEN 128

// New Parameters
#define DT_RANK 9   // Update as per the Python calculation
#define X_PROJ_OUT (DT_RANK + 2 * D_STATE)  // Output features of x_proj


// Quantization Scales
#define SCALE_IN 0.1f
#define SCALE_WEIGHTS 10.0f
#define SCALE_OUTPUT 0.1f

// Standard Library Headers
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#endif // CONFIG_H
