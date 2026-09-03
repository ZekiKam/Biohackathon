all: coalescence

coalescence: droplet_coalescence.c
	qcc -O2 -Wall droplet_coalescence.c -o coalescence -lm
	mkdir -p snapshots

run: coalescence
	./coalescence

clean:
	rm -f coalescence
	rm -rf snapshots log.csv

.PHONY: all run clean
