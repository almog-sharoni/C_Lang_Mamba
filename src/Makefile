CC = gcc
CFLAGS = -Wall -O3

SRC = main.c forward.c utils.c state.c model.c
OBJ = $(SRC:.c=.o)
TARGET = mamba_model

all: $(TARGET)

$(TARGET): $(OBJ)
    $(CC) $(CFLAGS) -o $@ $^  # This line must be indented with a tab

%.o: %.c
    $(CC) $(CFLAGS) -c $<     # This line must also be indented with a tab

clean:
    rm -f $(OBJ) $(TARGET)     # This line also requires a tab
