# generic makefile for ElectroMag library
include ../Config.mk


# C++ Sources
SRCDIR= src
CXXsources= $(SRCDIR)/ElectroMag.cpp $(SRCDIR)/CPU_Implement.cpp $(SRCDIR)/stdafx.cpp\
	$(SRCDIR)/CPUID/CPUID.cpp $(SRCDIR)/Graphics/Renderer.cpp $(SRCDIR)/Graphics/FieldRender.cpp

OBJDIR= obj
OBJS= $(OBJDIR)/ElectroMag.o $(OBJDIR)/CPU_Implement.o $(OBJDIR)/stdafx.o\
	$(OBJDIR)/CPUID.o $(OBJDIR)/Graphics_dynlink.o\
	$(OBJDIR)/Particle_System.o

.PHONY: clean all

# Explicit
all: $(TARGET_DIR)/$(TARGET)

# Rule for linking all objects
$(TARGET_DIR)/$(TARGET): $(OBJS) $(LIB_PATH)/$(GPGPU_LIB)
	@echo =======================================================
	@echo = Linking Everything                                  =
	@echo =======================================================
	@mkdir -p $(TARGET_DIR)
	@$(CXX) -o $@  $(OBJS) $(LDFLAGS)
	@echo =======================================================
	@echo = Done Linking Everything                             =
	@echo =======================================================

#1 The heavy processing part
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -c $< -o $@

#Rules for files in src/somedir
$(OBJDIR)/%.o: $(SRCDIR)/*/%.cpp
	@mkdir -p $(OBJDIR)
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for C files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	@echo Compiling $<
	@$(CC) $(CXXFLAGS) -c $< -o $@

#Merciless seek and delete
clean:
	rm -r -f $(OBJDIR)
	rm -r -f $(TARGET_DIR)/$(TARGET)
