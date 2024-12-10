# Calculate the size of the input
D_MODEL = 128
EXPAND = 2
D_INNER = D_MODEL * EXPAND
D_STATE = 64
D_CONV = 64
BATCH_SIZE = 8
SEQ_LEN = 128

# Total size of the input
total_size = BATCH_SIZE * SEQ_LEN * D_MODEL

# Generate the input file content
import numpy as np

# Generate random int8 data within range [-128, 127]
input_data = np.random.randint(-128, 128, size=total_size, dtype=np.int8)

# Save the generated input data to a text file
input_file_path = '../data/input.txt'
np.savetxt(input_file_path, input_data, fmt='%d')

total_size, input_file_path
