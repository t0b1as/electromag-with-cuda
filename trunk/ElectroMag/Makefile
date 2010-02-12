# This should build the entire project
MAKE = make

all: Electromag
clean: Electromag-clean GPGPU_segment-clean
	rm -r -f bin

debug: Electromag-debug GPGPU_segment-debug
	
Electromag-debug: GPGPU_segment-debug
run: Electromag-run

Electromag: GPGPU_segment
	$(MAKE) --directory=ElectroMag

GPGPU_segment:
	$(MAKE) --directory=GPGPU_Segment
		
Electromag-clean:
	$(MAKE) clean --directory=ElectroMag

GPGPU_segment-clean:
	$(MAKE) clean --directory=GPGPU_Segment

Electromag-debug:
	$(MAKE) --directory=ElectroMag --file=Makefile-Debug.mk

GPGPU_segment-debug:
	$(MAKE) --directory=GPGPU_Segment --file=Makefile-Debug.mk
		
Electromag-run: GPGPU_segment
	$(MAKE) run --directory=ElectroMag

