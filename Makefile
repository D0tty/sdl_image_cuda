CXX=g++
CPPFLAGS=-D_GNU_SOURCE=1 \
	 `pkg-config --cflags cuda` \
	 `pkg-config --cflags sdl2` \
	 `pkg-config --cflags SDL2_image`
CXXFLAGS=-Wall -Wextra -std=c++17 -Og -g
LDLIBS=`pkg-config --libs cuda` -lcudart \
       `pkg-config --libs sdl2` \
       `pkg-config --libs SDL2_image`
BIN=image
OBJS=main.o
CUSRC=gpu.cu
CUOBJS=$(CUSRC:.cu=.o)

all:$(BIN)

dbg:
	echo $(CUOBJS)
	echo $(OBJS)

$(BIN):$(OBJS) $(CUOBJS)
	$(LINK.cc) -o $@ $^ $(LDLIBS)

cuda:$(CUOBJS)

$(CUOBJS):$(CUSRC)
	nvcc -c $(CUSRC)

clean:
	$(RM) $(OBJS) $(CUOBJS) $(BIN)

.PHONY: all clean cuda
