
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


