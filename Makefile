CXX := g++
CXXFLAGS := -std=c++17 -Wall -I./eigen

OBJ_DIR := obj
TARGET := nn

SOURCES := $(wildcard *.cpp)
OBJECTS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SOURCES))

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)