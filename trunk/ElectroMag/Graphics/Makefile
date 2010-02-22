# Makefile for Graphics library for ElectroMag

include ../Config.mk

#Include directories
#CPP_INCLUDE = -I $(CUDA_INC_PATH) -I $(CUDA_SDK_INC_PATH)
#CPP_INCLUDE += -I. -I../ElectroMag/src

#Library options
# Overwrite LDFLAGS and only link to the dependencies needed by this module
# This will allow runtime linking to this library to fail if the dependencies
# are not available, but the main app will be able to continue execution
LDFLAGS = -lglut -lGLEW -lGLU

OBJDIR=obj
SRCDIR=src

# Target produced by this makefie
LOCAL_TARGET= $(LIB_PATH)/$(GRAPHICS_LIB)$(GRAPHICS_LIB_POSTFIX)

# Required objects
OBJS= $(OBJDIR)/Renderer_Interface.o\
	 $(OBJDIR)/FieldRender.o $(OBJDIR)/FieldRender_Interface.o\
	 $(OBJDIR)/FrontendGUI.o $(OBJDIR)/MainGUI.o



.PHONY: all clean

# All for dummies
all: $(LOCAL_TARGET)

# Rule for compiling C and C++ files and creating library
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	@echo Compiling $<
	@$(CXX) -c $(CXXFLAGS) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	@echo Compiling $<
	@$(CC) -c $(CCFLAGS) $< -o $@

$(LOCAL_TARGET): $(OBJS)
	@mkdir -p $(LIB_PATH)
	@echo =======================================================
	@echo = Linking Everything                                  =
	@echo =======================================================
	@$(CXX) -shared -o $@ $(OBJS) $(LDFLAGS)
	@echo =======================================================
	@echo = Done Linking Everything                             =
	@echo =======================================================
	

#Merciless seek and delete
clean:
	@rm -f $(LOCAL_TARGET)
	@rm -r -f $(OBJDIR)
 
