# This should build the entire project
MAKE = make

all: Electromag
clean: Electromag-clean GPGPU_segment-clean
	rm -r -f bin

debug: Electromag-debug GPGPU_segment-debug
	
Electromag-debug: GPGPU_segment-debug
run: Electromag-run

Electromag: GPGPU_segment
	$(MAKE) -C ElectroMag

GPGPU_segment:
	$(MAKE) -C GPGPU_Segment
		
Electromag-clean:
	$(MAKE) clean -C ElectroMag

GPGPU_segment-clean:
	$(MAKE) clean -C GPGPU_Segment

Electromag-debug:
	$(MAKE) -C ElectroMag --file=Makefile-Debug.mk

GPGPU_segment-debug:
	$(MAKE) -C GPGPU_Segment --file=Makefile-Debug.mk
		
Electromag-run: GPGPU_segment
	$(MAKE) run -C ElectroMag

