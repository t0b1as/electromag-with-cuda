#pragma once

#define BLOCK_X 64
#define KERNEL_STEPS 32
#define MAX_CMEM_SP_CHARGES 511

#define BLOCK_X_MT 32
#define BLOCK_Y_MT 4
#define BLOCK_DIM_MT (BLOCK_X_MT * BLOCK_Y_MT)
//#define MT_OCCUPANCY 4

#define CoreFunctor electroPartField
#define CoreFunctorFLOP electroPartFieldFLOP

#define CalcField_kernelFLOP(n,p) ( n * (p *(CoreFunctorFLOP + 3) + 13) )
#define CalcField_kernelFLOP_Curvature(n,p) ( n * (p *(CoreFunctorFLOP + 3) + 45) )

