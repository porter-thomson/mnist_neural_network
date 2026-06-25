CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -I../eigen

SRCS_COMMON := ../Layer.cpp ../Network.cpp ../DataLoader.cpp

.PHONY: all test clean

all: test_layer test_network test_dataloader

test_layer: test_layer.cpp $(SRCS_COMMON)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_network: test_network.cpp $(SRCS_COMMON)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_dataloader: test_dataloader.cpp $(SRCS_COMMON)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Runs all three test binaries concurrently (backgrounded in the same shell,
# then waited on individually so each exit code can be checked).
test: all
	@echo "Running all tests in parallel..."
	@./test_layer > layer.log 2>&1 & pid1=$$!; \
	 ./test_network > network.log 2>&1 & pid2=$$!; \
	 ./test_dataloader > dataloader.log 2>&1 & pid3=$$!; \
	 wait $$pid1; s1=$$?; \
	 wait $$pid2; s2=$$?; \
	 wait $$pid3; s3=$$?; \
	 echo "=== Layer ==="; cat layer.log; \
	 echo "=== Network ==="; cat network.log; \
	 echo "=== DataLoader ==="; cat dataloader.log; \
	 rm -f layer.log network.log dataloader.log; \
	 exit $$(( s1 || s2 || s3 ))

clean:
	rm -f test_layer test_network test_dataloader *.log