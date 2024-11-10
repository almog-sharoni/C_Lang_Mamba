
# Mamba-C: C Language Implementation of the Mamba Model for Keyword Spotting



## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Compiling the Project](#compiling-the-project)
  - [Running the Testbench](#running-the-testbench)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Overview

**Mamba-C** is a C language implementation of the Mamba model, originally developed in Python for efficient keyword spotting (KWS) applications. This project aims to provide a lightweight, high-performance alternative suitable for embedded systems and resource-constrained environments. By leveraging fixed-point arithmetic and optimized algorithms, Mamba-C achieves real-time keyword detection with minimal computational overhead.

## Features

- **Efficient Fixed-Point Arithmetic**: Utilizes fixed-point operations to reduce computational complexity and memory usage.
- **Depthwise Convolution**: Implements depthwise 1D convolutions for effective feature extraction.
- **State-Space Model Integration**: Incorporates state-space dynamics for temporal dependency modeling.
- **Modular Design**: Organized into separate modules for ease of understanding and maintenance.
- **Comprehensive Testing**: Includes a testbench to validate model outputs against expected results.
- **Cross-Platform Compatibility**: Designed to compile and run on various Unix-like systems.

## Requirements

- **Compiler**: GCC (GNU Compiler Collection) or any C99-compatible compiler.
- **Libraries**:
  - **Standard C Libraries**: `math.h`, `stdio.h`, `stdlib.h`, etc.
- **Development Environment**: Unix-like operating system (Linux, macOS).

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/Mamba-C.git
   cd Mamba-C/8bit/mamba8bit
   ```

2. **Initialize Submodules (if any)**

   If your project uses Git submodules, initialize them:

   ```bash
   git submodule update --init --recursive
   ```


## Usage

### Compiling the Project

To compile the Mamba-C project and the testbench, follow these steps:

1. **Navigate to the Project Directory**

   ```bash
   cd ~/KWS-on-MAMBA/8bit/mamba8bit
   ```

2. **Compile the Model and Testbench**

   Use `gcc` to compile the C source files. Ensure that all source files are in the current directory.

   ```bash
   gcc -o test_mamba_model forward.c mamba_model.c test_mamba_model.c -lm
   ```

   - `-o test_mamba_model`: Specifies the output executable name.
   - `forward.c`, `mamba_model.c`, `test_mamba_model.c`: Source files.
   - `-lm`: Links the math library required for mathematical functions like `expf` and `fabsf`.

### Running the Testbench

After successful compilation, execute the testbench to validate the model's functionality.

```bash
./test_mamba_model
```



## Project Structure

```
mamba8bit/
├── forward.c
├── mamba_model.c
├── mamba_model.h
├── README.md
├── test_mamba_model.c
└── test_mamba_model.exe (Compiled Executable)
```

- **forward.c**: Contains functions related to the forward pass of the Mamba model.
- **mamba_model.c**: Implements the core Mamba model functionalities, including initialization and state updates.
- **mamba_model.h**: Header file declaring the Mamba model structure and function prototypes.
- **README.md**: Project documentation (this file).
- **test_mamba_model.c**: Testbench to validate the Mamba model's output.
- **test_mamba_model.exe**: Compiled executable for the testbench.

## Testing

The project includes a testbench (`test_mamba_model.c`) designed to validate the correctness of the Mamba model's implementation.

### Defining Test Cases

Test cases are defined within the `test_mamba_model.c` file. Each test case provides specific input sequences to evaluate the model's response.

```c
int8_t test_inputs[5][BATCH_SIZE][SEQ_LENGTH][INPUT_DIM] = {
    {   // Test Case 1
        {
            {1, 1, 1, 1},  // Sum = 4
            {2, 2, 2, 2}   // Sum = 8
        }
    },
    // Additional Test Cases...
};
```

### Expected Outputs

Expected outputs are placeholders and should be replaced with accurate values generated from a reference implementation (e.g., Python-based Mamba model).

```c
int8_t expected_outputs[5][OUTPUT_DIM] = {
    {16, 16}, // Test Case 1
    {32, 32}, // Test Case 2
    {48, 48}, // Test Case 3
    {64, 64}, // Test Case 4
    {80, 80}  // Test Case 5
};
```

### Running Tests

Execute the testbench and observe whether the model's outputs match the expected values.

```bash
./test_mamba_model
```


**Interpreting Results:**

- **Passed**: Indicates that the model's output aligns with expectations.
- **Failed**: Highlights discrepancies between the model's output and expected values, suggesting potential issues in the implementation.

## Troubleshooting

### Common Issues and Solutions

1. **All Outputs Are Zeros or Maxed Out (`127`)**

   - **Cause**: 
     - Incorrect weight or bias initialization.
     - Errors in fixed-point arithmetic or scaling.
     - Flaws in convolution or state update logic.
   
   - **Solution**:
     - Verify that all weights and biases are initialized with non-zero values.
     - Ensure that fixed-point scaling is consistently applied and that clamping logic is correctly implemented.
     - Add debugging `printf` statements to trace intermediate values.


