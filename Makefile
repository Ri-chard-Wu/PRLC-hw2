CXX = mpicxx
CC  = mpicc
CPPFLAGS = -I/home/ipc23/share/hw2/lodepng
LDFLAGS = -pthread -fopenmp -lm 
CXXFLAGS = -ltbb -msse2 -std=c++17 -O3
TARGET = hw2
HW2 = hw2.cc
LODEPNG = /home/ipc23/share/hw2/lodepng/lodepng.cpp

all: $(TARGET)

$(TARGET): $(HW2) $(LODEPNG)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(HW2) $(LODEPNG) -o $(TARGET)

.PHONY: clean
clean:
	rm -r hw2	




