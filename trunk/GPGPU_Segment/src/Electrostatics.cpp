/***********************************************************************************************
Copyright (C) 2009-2010 - Alexandru Gagniuc - <http:\\g-tech.homeserver.com\HPC.htm>
 * This file is part of ElectroMag.

    ElectroMag is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ElectroMag is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ElectroMag.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************************************/
#include "CL Electrostatics.hpp"

/*////////////////////////////////////////////////////////////////////////////////
Define specializations of the field function that call the generic template
These guarantee that the specializations will be compiled and included in the
export library
 */////////////////////////////////////////////////////////////////////////////////

int CalcField(Array<Vector3<float> >& fieldLines, Array<electro::pointCharge<float> >& pointCharges,
              size_t n, float resolution, perfPacket& perfData, bool useCurvature)
{
    return -1;
}

int CalcField(Array<Vector3<double> >& fieldLines, Array<electro::pointCharge<double> >& pointCharges,
              size_t n, double resolution, perfPacket& perfData, bool useCurvature)
{
    return -1;
}

void TestCL(Vector3<Array<float> >& fieldLines, Array<electro::pointCharge<float> >& pointCharges,
            size_t n, float resolution,  perfPacket& perfData, bool useCurvature)
{
    CLElectrosFunctor<float>::BindDataParams dataParams = {&fieldLines, &pointCharges, n, resolution, perfData, useCurvature};
    CLtest.BindData((void*) &dataParams);
    
    CLtest.Run();
    
}



