# Makefile for CUDA vector addition

# CUDA compiler
NVCC = nvcc

# Compiler flags
# -arch=sm_70 specifies the GPU architecture (adjust for your GPU)
# sm_70 works for V100, sm_80 for A100, sm_86 for RTX 30-series
NVCCFLAGS = -arch=sm_70

# Target executable name
TARGET = vector_add

# Source file
SRC = vector_add.cu

# Default target: compile the program
all: $(TARGET)

# Rule to compile the CUDA program
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

# Clean up compiled files
clean:
	rm -f $(TARGET)

# Run the program (for local testing)
run: $(TARGET)
	./$(TARGET)