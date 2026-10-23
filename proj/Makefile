NVCC   = nvcc
ARCH   = native
FLAGS  = -O3 -std=c++17 -arch=$(ARCH)
SRC    = test3.cu
TARGET = test3

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(FLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).exe $(TARGET).exp $(TARGET).lib

.PHONY: all run clean
