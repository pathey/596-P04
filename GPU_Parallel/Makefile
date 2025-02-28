all:	P04_Parallel_GPU

CC = nvc
CFLAGS = -mp=gpu -gpu=cc80

P04_Parallel_GPU: P04_Parallel_GPU.o
	 $(CC) $(CFLAGS) P04_Parallel_GPU.c -o P04_Parallel_GPU -lm

run_2:	P04_Parallel_GPU
	nohup ./P04_Parallel_GPU 2 > output2_GPU_no_vector.log 2>&1 &

run_4:	P04_Parallel_GPU
	nohup ./P04_Parallel_GPU 4 > output4_GPU_no_vector.log 2>&1 &

run_8:	P04_Parallel_GPU
	nohup ./P04_Parallel_GPU 8 > output8_GPU_no_vector.log 2>&1 &

run_16:	P04_Parallel_GPU
	nohup ./P04_Parallel_GPU 16 > output16_GPU_no_vector.log 2>&1 &

run_32:	P04_Parallel_GPU
	nohup ./P04_Parallel_GPU 32> output32_GPU_no_vector.log 2>&1 &

run_64:	P04_Parallel_GPU
	nohup ./P04_Parallel_GPU 64 > output64_GPU_no_vector.log 2>&1 &

run_128:	P04_Parallel_GPU
	nohup ./P04_Parallel_GPU 128 > output128_GPU_no_vector.log 2>&1 &

run_256:	P04_Parallel_GPU
	nohup ./P04_Parallel_GPU 256 > output256_GPU_no_vector.log 2>&1 &

run_512:	P04_Parallel_GPU
	nohup ./P04_Parallel_GPU 512 > output512_GPU_no_vector.log 2>&1 &

run_1024:	P04_Parallel_GPU
	nohup ./P04_Parallel_GPU 1024 > output1024_GPU_no_vector.log 2>&1 &
