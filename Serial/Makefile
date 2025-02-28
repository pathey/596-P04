all:	P04_Serial

P04_Serial: P04_Serial.o
	gcc -fopenmp P04_Serial.c -o P04_Serial -lm

run:	P04_Serial
	nohup ./P04_Serial > output.log 2>&1 &
