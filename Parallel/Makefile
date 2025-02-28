all:	P04_Parallel_CPU

P04_Parallel_CPU: P04_Parallel_CPU.o
	gcc -fopenmp P04_Parallel_CPU.c -o P04_Parallel_CPU -lm

run_2:	P04_Parallel_CPU
	nohup ./P04_Parallel_CPU 2 > output2.log 2>&1 &

run_4:	P04_Parallel_CPU
	nohup ./P04_Parallel_CPU 4 > output4.log 2>&1 &

run_8:	P04_Parallel_CPU
	nohup ./P04_Parallel_CPU 8 > output8.log 2>&1 &

run_16:	P04_Parallel_CPU
	nohup ./P04_Parallel_CPU 16 > output16.log 2>&1 &

run_32:	P04_Parallel_CPU
	nohup ./P04_Parallel_CPU 32> output32.log 2>&1 &

run_64:	P04_Parallel_CPU
	nohup ./P04_Parallel_CPU 64 > output64.log 2>&1 &

run_128:	P04_Parallel_CPU
	nohup ./P04_Parallel_CPU 128 > output128.log 2>&1 &

run_256:	P04_Parallel_CPU
	nohup ./P04_Parallel_CPU 256 > output256.log 2>&1 &

run_512:	P04_Parallel_CPU
	nohup ./P04_Parallel_CPU 512 > output512.log 2>&1 &

run_1024:	P04_Parallel_CPU
	nohup ./P04_Parallel_CPU 1024 > output1024.log 2>&1 &
