# generic makefile for ElectroMag library
include ../Config.mk


# C++ Sources
SRCDIR= src
CXXsources= $(SRCDIR)/ElectroMag.cpp $(SRCDIR)/CPU_Implement.cpp $(SRCDIR)/stdafx.cpp\
	$(SRCDIR)/CPUID/CPUID.cpp $(SRCDIR)/Graphics/Renderer.cpp $(SRCDIR)/Graphics/FieldRender.cpp

OBJDIR= obj
OBJS= $(OBJDIR)/ElectroMag.o $(OBJDIR)/CPU_Implement.o $(OBJDIR)/stdafx.o $(OBJDIR)/CPUID.o \
	$(OBJDIR)/Renderer.o $(OBJDIR)/FieldRender.o $(OBJDIR)/FrontendGUI.o\
	$(OBJDIR)/freeglut_dynlink.o  $(OBJDIR)/glew.o

# Explicit
all: post-build

post-build: $(TARGET_DIR)/$(TARGET)
	#copy .ptx files, as those contain the GPU kernel code
	cp -f $(GPU_LIB_PATH)/*.ptx $(TARGET_DIR)

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
	$(CXX) -o $(TARGET_DIR)/$(TARGET)  $(OBJS) $(LDFLAGS) obj/glew.o
	@echo
	@echo Done Linking Everything
	@echo

#1 The heavy processing part
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp pre-build
	$(CXX) $(CXXFLAGS) -c $< -o $@
# Rule for files that are located in src/somedir/
$(OBJDIR)/%.o: $(SRCDIR)/*/%.cpp pre-build
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for files that are located in src/somedir/more_nested_dir/
$(OBJDIR)/%.o: $(SRCDIR)/*/*/%.c pre-build
	$(CXX) $(CXXFLAGS) -c $< -o $@

#Merciless seek and delete
clean:
	rm -r -f $(TARGET_DIR)
	rm -r -f $(OBJDIR)



