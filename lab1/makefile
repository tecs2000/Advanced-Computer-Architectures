# =======================
# Makefile for HPC Lab - Followers Aggregation (Master-Worker)
# =======================

# Compiler
CXX = g++

# Compiler flags
# -Wall: show all warnings
# -O2: optimize for speed
# -pthread: enable POSIX threads
CXXFLAGS = -Wall -O2 -pthread -fopenmp

# Directories
SRC_DIR = src
BUILD_DIR = build

# Source and output files
SRC = $(SRC_DIR)/social_media.cpp $(SRC_DIR)/generation.cpp
TARGET = $(BUILD_DIR)/social_media

# Default target: compile and link
all: $(TARGET)

# How to build the target (compile all sources)
$(TARGET): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Clean up build files
clean:
	rm -rf $(BUILD_DIR)

# Phony targets (not files)
.PHONY: all run clean
