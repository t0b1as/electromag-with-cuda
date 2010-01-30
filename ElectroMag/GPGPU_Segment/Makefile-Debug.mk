# Makefile for creating debug versions
include Makefile


# add debugging sopport to compiler flags
CXXFLAGS+= -g
CCFLAGS+= -g