At the present moment, the application only displays static electric field lines generated from a static distribution of charges. It uses all available GPUs in the system to calculate the field lines, then displays them using OpenGL.

However, the goal of ElectroMag is to evolve into a modular, high-performance framework for physics simulations. The functionality of Electromag is provided in several flavours:
  * A generic C++ implementation, which can be used on virtually any architecture with a C++ compiler.
  * A highly optimized SSE variant for x86-64 processors (AVX-256 support is planed as well).
  * An OpenCL variant, to take advantage of other GPUs or devices that can be used with OpenCL.


Besides the obvious extension of functionality, and new algorithms ,the current to do list includes:
  * MPI support, for usability in clusters.

Besides that, ElectroMag is licensed under the GNU General Public License v3, and is open to everyone interested to contribute.