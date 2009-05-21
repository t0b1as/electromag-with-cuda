# generic makefile for ElectroMag library
include ../Config.mk

# Change this to Makefile-GNU.mk if the Intel compuler is unavailable,
# but by all means, use the Intel compiler if at all posible
include Makefile-Intel.mk
#include Makefile-GNU.mk # AKA The Headache Compiler

# C++ Sources
CXXsources= src/ElectroMag.cpp src/CPU_Implement.cpp src/stdafx.cpp\
	src/CPUID/CPUID.cpp src/Graphics/Renderer.cpp src/Graphics/FieldRender.cpp
		
CXXFLAGS+=-g

OBJDIR= obj
OBJS= $(OBJDIR)/ElectroMag.o $(OBJDIR)/CPU_Implement.o $(OBJDIR)/stdafx.o $(OBJDIR)/CPUID.o \
	$(OBJDIR)/Renderer.o $(OBJDIR)/FieldRender.o $(OBJDIR)/FrontendGUI.o


# Explicit
all: $(TARGET_DIR)/$(TARGET)

run: $(TARGET_DIR)/$(TARGET)
	export LD_LIBRARY_PATH=$(CUDA_LIB_PATH)/
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

#Merciless seek and delete
clean:
	rm -r -f $(TARGET_DIR)
	rm -r -f $(OBJDIR)

# The heavy procesing part
$(OBJDIR)/ElectroMag.o: pre-build
	$(CXX) -c $(CXXFLAGS) src/ElectroMag.cpp -o $(OBJDIR)/ElectroMag.o
$(OBJDIR)/CPU_Implement.o: pre-build
	$(CXX) -c $(CXXFLAGS) src/CPU_Implement.cpp -o $(OBJDIR)/CPU_Implement.o
$(OBJDIR)/stdafx.o: pre-build
	$(CXX) -c $(CXXFLAGS) src/stdafx.cpp -o $(OBJDIR)/stdafx.o
$(OBJDIR)/Renderer.o: pre-build
	$(CXX) -c $(CXXFLAGS) src/Graphics/Renderer.cpp -o $(OBJDIR)/Renderer.o
$(OBJDIR)/FieldRender.o: pre-build
	$(CXX) -c $(CXXFLAGS) src/Graphics/FieldRender.cpp -o $(OBJDIR)/FieldRender.o
$(OBJDIR)/CPUID.o: pre-build
	$(CXX) -c $(CXXFLAGS) src/CPUID/CPUID.cpp -o $(OBJDIR)/CPUID.o
$(OBJDIR)/FrontendGUI.o: pre-build
	$(CXX) -c $(CXXFLAGS) src/Graphics/FrontendGUI.cpp -o $(OBJDIR)/FrontendGUI.o


