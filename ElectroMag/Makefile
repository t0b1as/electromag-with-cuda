# This should build the entire project


all: Electromag
clean: Electromag-clean GPGPU_segment-clean
run: Electromag-run

Electromag: GPGPU_segment
	make --directory=ElectroMag

GPGPU_segment:
	make --directory=GPGPU\ Segment
		
Electromag-clean: GPGPU_segment
	make clean --directory=ElectroMag

GPGPU_segment-clean:
	make clean --directory=GPGPU\ Segment
		
Electromag-run: GPGPU_segment
	make run --directory=ElectroMag

