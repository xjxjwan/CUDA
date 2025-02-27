#############################
# 1. Basic Configuration
#############################

# Compiler
CUDA_COMPILER := nvcc

# Executable name
EXECUTABLE := main

# Source files
SOURCE_FILES := try.cu

# Object files (replace .cu with .o)
OBJECT_FILES := $(SOURCE_FILES:.cu=.o)


#############################
# 2. Compilation Options
#############################

# (a) Common compilation flags:
#    -std=c++14: Use C++14 standard
#    -arch=sm_80: Target NVIDIA Ampere architecture (e.g., A100, RTX 30 series)
#    -gencode=arch=compute_80,code=sm_80: Generate code for sm_80
COMMON_FLAGS := -std=c++14 -arch=sm_80 -gencode=arch=compute_80,code=sm_80

# (b) Release mode optimization flags
RELEASE_FLAGS := -O3 -Xptxas -O3

# (c) Debug mode compilation flags
#    -G: Generate device debug code (affects performance)
#    -O0: Disable optimization
DEBUG_FLAGS := -G -O0

# (d) Strict floating-point precision options (optional):
#    --fmad=false: Disable FMA (fused multiply-add) to reduce rounding differences
#    -prec-div=true / -prec-sqrt=true: Improve precision of division and square root
# Enable only when necessary as it impacts performance
PRECISION_FLAGS := --fmad=false -prec-div=true -prec-sqrt=true


#############################
# 3. Build Mode Selection
#############################
# Select the build mode via command-line arguments:
#   make          (default: release mode)
#   make DEBUG=1  (enable debug mode)
#   make ACCURATE=1 (enable strict floating-point precision)
#
# Combination example:
#   make DEBUG=1 ACCURATE=1
# (Enables debugging and strict floating-point precision, but is slow)
#############################

# Default to release mode
ifeq ($(DEBUG),1)
  COMPILER_FLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS)
else
  COMPILER_FLAGS := $(COMMON_FLAGS) $(RELEASE_FLAGS)
endif

# Append floating-point precision flags if ACCURATE=1 is set
ifeq ($(ACCURATE),1)
  COMPILER_FLAGS += $(PRECISION_FLAGS)
endif


#############################
# 4. Build Rules
#############################

# 4.1 Build the executable
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECT_FILES)
	$(CUDA_COMPILER) $(COMPILER_FLAGS) -o $@ $^

# 4.2 Compile .cu -> .o
%.o: %.cu
	$(CUDA_COMPILER) $(COMPILER_FLAGS) -c $< -o $@

# 4.3 Clean up
clean:
	rm -f $(OBJECT_FILES) $(EXECUTABLE)
