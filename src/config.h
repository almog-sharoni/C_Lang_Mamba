#ifndef CONFIG_H
#define CONFIG_H

// Model Parameters
#define D_MODEL 128       // Feature size of the model
#define EXPAND 2          // Expansion factor
#define D_INNER (D_MODEL * EXPAND) // Expanded inner dimension
#define D_STATE 64        // State size for SSM
#define D_CONV 64         // Kernel size for Conv1d
#define BATCH_SIZE 32     // Batch size
#define SEQ_LEN 1024      // Sequence length
#define X_PROJ_OUT (DT_RANK + 2 * D_STATE)      // Output dimension of x_proj

// Additional Parameters
#define DT_RANK 8         // Rank for dt_proj computation

// Quantization Scales
#define SCALE_IN 1.0f
#define SCALE_WEIGHTS 1.0f
#define SCALE_OUTPUT 1.0f

// Standard Library Headers
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#endif // CONFIG_H
