# This should build the entire project

all: Electromag
clean: Electromag-clean GPGPU_segment-clean
	rm -r -f bin
debug: Electromag-debug GPGPU_segment-debug
Electromag-debug: GPGPU_segment-debug
run: Electromag-run

Electromag: GPGPU_segment
	make --directory=ElectroMag

GPGPU_segment:
	make --directory=GPGPU_Segment
		
Electromag-clean:
	make clean --directory=ElectroMag

GPGPU_segment-clean:
	make clean --directory=GPGPU_Segment

Electromag-debug:
	make --directory=ElectroMag --file=Makefile-Debug.mk

GPGPU_segment-debug:
	make --directory=GPGPU_Segment --file=Makefile-Debug.mk
		
Electromag-run: GPGPU_segment
	make run --directory=ElectroMag

