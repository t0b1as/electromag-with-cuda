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

.PHONY: clean all

# Explicit
all: $(TARGET_DIR)/$(TARGET)

# Rule for linking all objects
$(TARGET_DIR)/$(TARGET): $(OBJS)
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
# Rule for files that are located in src/somedir/
$(OBJDIR)/%.o: $(SRCDIR)/*/%.cpp
	@mkdir -p $(OBJDIR)
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for files that are located in src/somedir/more_nested_dir/
$(OBJDIR)/%.o: $(SRCDIR)/*/*/%.c
	@mkdir -p $(OBJDIR)
	@echo Compiling $<
	@$(CC) $(CXXFLAGS) -c $< -o $@

#Merciless seek and delete
clean:
	rm -r -f $(OBJDIR)
	rm -r -f $(TARGET_DIR)/$(TARGET)
