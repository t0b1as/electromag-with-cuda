# generic makefile for ElectroMag library
include ../Config.mk

# Change this to Makefile-GNU.mk if the Intel compiler is unavailable,
# but by all means, use the Intel compiler if at all posible
include Makefile-Intel.mk
#include Makefile-GNU.mk # AKA The Headache Compiler

# C++ Sources
CXXsources= src/ElectroMag.cpp src/CPU_Implement.cpp src/stdafx.cpp\
	src/CPUID/CPUID.cpp src/Graphics/Renderer.cpp src/Graphics/FieldRender.cpp

OBJDIR= obj
OBJS= $(OBJDIR)/ElectroMag.o $(OBJDIR)/CPU_Implement.o $(OBJDIR)/stdafx.o $(OBJDIR)/CPUID.o \
	$(OBJDIR)/Renderer.o $(OBJDIR)/FieldRender.o $(OBJDIR)/FrontendGUI.o


# Explicit
all: $(TARGET_DIR)/$(TARGET)

run: $(TARGET_DIR)/$(TARGET)
	LD_LIBRARY_PATH+=;$(CUDA_LIB_PATH)/
	$(TARGET_DIR)/$(TARGET)

pre-build:
	mkdir -p $(OBJDIR)

# Rule for linking all objects
$(TARGET_DIR)/$(TARGET): $(OBJS)
	@echo
	@echo Linking Everything
	@echo
	mkdir -p $(TARGET_DIR)
	$(CXX) -o $(TARGET_DIR)/$(TARGET)  $(OBJS) $(LDFLAGS)
	@echo
	@echo Done Linking Everything
	@echo

# The heavy processing part
$(OBJDIR)/ElectroMag.o: pre-build src/ElectroMag.cpp
	$(CXX) -c $(CXXFLAGS) src/ElectroMag.cpp -o $(OBJDIR)/ElectroMag.o
$(OBJDIR)/CPU_Implement.o: pre-build src/CPU_Implement.cpp
	$(CXX) -c $(CXXFLAGS) src/CPU_Implement.cpp -o $(OBJDIR)/CPU_Implement.o
$(OBJDIR)/stdafx.o: pre-build src/stdafx.cpp
	$(CXX) -c $(CXXFLAGS) src/stdafx.cpp -o $(OBJDIR)/stdafx.o
$(OBJDIR)/Renderer.o: pre-build src/Graphics/Renderer.cpp
	$(CXX) -c $(CXXFLAGS) src/Graphics/Renderer.cpp -o $(OBJDIR)/Renderer.o
$(OBJDIR)/FieldRender.o: pre-build src/Graphics/FieldRender.cpp
	$(CXX) -c $(CXXFLAGS) src/Graphics/FieldRender.cpp -o $(OBJDIR)/FieldRender.o
$(OBJDIR)/CPUID.o: pre-build src/CPUID/CPUID.cpp
	$(CXX) -c $(CXXFLAGS) src/CPUID/CPUID.cpp -o $(OBJDIR)/CPUID.o
$(OBJDIR)/FrontendGUI.o: pre-build src/Graphics/FrontendGUI.cpp
	$(CXX) -c $(CXXFLAGS) src/Graphics/FrontendGUI.cpp -o $(OBJDIR)/FrontendGUI.o

#Merciless seek and delete
clean:
	rm -r -f $(TARGET_DIR)
	rm -r -f $(OBJDIR)



