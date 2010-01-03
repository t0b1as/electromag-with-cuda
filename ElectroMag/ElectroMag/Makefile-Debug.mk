# Makefile for creating debug versions
include Makefile.mk


# add debugging sopport to compiler flags
CXXFLAGS+= -g
CCFLAGS+= -g