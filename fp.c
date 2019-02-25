#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include <CL/cl.h>

//#define str(x) #x

#define xstr(s) str(s)
#define str(s) #s

#define SIGM(x) (1/(1+exp(-x)))


void check_error(int err, char *str){
	if(err != CL_SUCCESS){
		printf("ERROR: '%s': ", str);
		switch(err){
			case CL_INVALID_PROGRAM:
				printf("CL_INVALID_PROGRAM.\n");
				break;
			case CL_INVALID_PROGRAM_EXECUTABLE:
				printf("CL_INVALID_PROGRAM_EXECUTABLE.\n");
				break;
			case CL_INVALID_KERNEL_NAME:
				printf("CL_INVALID_KERNEL_NAME.\n");
				break;
			case CL_INVALID_KERNEL_DEFINITION:
				printf("CL_INVALID_KERNEL_DEFINITION.\n");
				break;
			case CL_INVALID_VALUE:
				printf("CL_INVALID_VALUE.\n");
				break;
			case CL_OUT_OF_HOST_MEMORY:
				printf("CL_OUT_OF_HOST_MEMORY.\n");
				break;
      case CL_INVALID_COMMAND_QUEUE:
        printf("CL_INVALID_COMMAND_QUEUE.\n");
        break;
      case CL_INVALID_KERNEL:
        printf("CL_INVALID_KERNEL.\n");
        break;
      case CL_INVALID_CONTEXT:
        printf("CL_INVALID_CONTEXT.\n");
        break;
      case CL_INVALID_KERNEL_ARGS:
        printf("CL_INVALID_KERNEL_ARGS.\n");
        break;
      case CL_INVALID_WORK_DIMENSION:
        printf("CL_INVALID_WORK_DIMENSION.\n");
        break;
      case CL_INVALID_WORK_GROUP_SIZE:
        printf("CL_INVALID_WORK_GROUP_SIZE.\n");
        break;
      case CL_INVALID_WORK_ITEM_SIZE:
        printf("CL_INVALID_WORK_ITEM_SIZE.\n");
        break;
      case CL_INVALID_GLOBAL_OFFSET:
        printf("CL_INVALID_GLOBAL_OFFSET.\n");
        break;
			default:
				printf("default err.\n");
				break;
		}
		exit(1);
	}
}

void error(char *str){
	printf("%s\n", str);
	exit(1);
}

float uniform(float minimum, float maximum){
	float center = minimum + (maximum - minimum)/2;
	float max_mag = maximum - center;
	if(rand()&1)
		return center + ((((float)rand())/RAND_MAX)) * max_mag;
	else
		return center - ((((float)rand())/RAND_MAX)) * max_mag;
}

void mult_cpu(const float *x, const float *w, float *dest, size_t size, size_t inputsize){
	for(int i = 0; i < size; i++){
		dest[i] = 0;
		for(int j = 0; j < inputsize; j++){
			dest[i] += w[i*inputsize + j] * x[j];
		}
	}
}

int main(){
	int status = 0;


	float x1 = 1;
	printf("stringifying macro: '%s' and '%s'\n", str(SIGM(x1)), xstr(SIGM(x1)));
	/* Start setup */
	cl_uint num_platforms, num_devices;
	status = clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id platforms[num_platforms];

	if(status != CL_SUCCESS) error("could not get number of plat ids.");

	status = clGetPlatformIDs(num_platforms, platforms, NULL);

	if(status != CL_SUCCESS) error("could not get plat ids.");

	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);

	if(status != CL_SUCCESS) error("could not find gpu on platform.");
	
	cl_device_id devices[num_devices];

	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

	if(status != CL_SUCCESS) error("failed to acquire gpu device id.");

	const cl_context_properties context_cfg[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties) platforms[0],
		0,
		0
	};
	
	cl_context context = clCreateContext(context_cfg, num_devices, devices, NULL, NULL, &status);

	if(status != CL_SUCCESS) error("failed to create context.");

	FILE *fp = fopen("mult.cl", "rb");
	if(!fp) error("couldn't open cl file\n");
	fseek(fp, 0, SEEK_END);
	size_t datafilelen = ftell(fp);
	fclose(fp);
	printf("cl file was %lu chars long.\n", datafilelen);

	fp = fopen("mult.cl", "rb");
	char *clfile = (char*)malloc(sizeof(char) * (datafilelen + 1));
	for(int i = 0; i < datafilelen; i++)
		clfile[i] = fgetc(fp);

	clfile[datafilelen] = '\0';

	printf("got: '%s'\n", clfile);

	cl_program prog = clCreateProgramWithSource(context, 1, (const char**)&clfile, NULL, NULL);

	check_error(clBuildProgram(prog, 0, NULL, NULL, NULL, NULL), "couldn't build program.");

	cl_kernel kernel = clCreateKernel(prog, "MULT", &status);

	check_error(status, "couldn't build kernel");

	cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &status);

	check_error(status, "couldn't make command queue.");


	/* End setup */
	size_t input_size = 10000;
	size_t neurons = 10000;

  size_t local_item_size = 100;
	
	float *x = (float*)malloc(sizeof(float) * input_size);
	float *w = (float*)malloc(sizeof(float) * neurons * input_size);
	float *y_gpu = (float*)malloc(sizeof(float) * neurons);
	float *y_cpu = (float*)malloc(sizeof(float) * neurons);

	for(int i = 0; i < 10; i++){
		for(int j = 0; j < input_size; j++){
			x[j] = uniform(-1, 1);
		}
		for(int j = 0; j < input_size * neurons; j++){
			w[j] = uniform(-1, 1);
		}
		cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_size, x, &status);
		check_error(status, "failed to create clmem from x");

		cl_mem weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_size * neurons, w, &status);
		check_error(status, "failed to create clmem from weights");

		cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * neurons, NULL, &status);
		check_error(status, "failed to create clmem from output buf");

		clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight_buffer);
		clSetKernelArg(kernel, 3, sizeof(int), &input_size);

    clock_t start = clock();
		check_error(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &neurons, &local_item_size, 0, NULL, NULL), "couldn't enqueue kernel.");
    check_error(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(float) * neurons, y_gpu, 0, NULL, NULL), "couldn't read kernel output");
    float gpu_time = ((float)(clock() - start)) / CLOCKS_PER_SEC;

    
    start = clock();
    mult_cpu(x, w, y_cpu, neurons, input_size);
    float cpu_time = ((float)(clock() - start)) / CLOCKS_PER_SEC;
		float sqe = 0;
    for(int i = 0; i < neurons; i++){
			sqe += sqrt(y_gpu[i]*y_gpu[i] - y_cpu[i]*y_cpu[i]);
    }
    printf("trial %d: gpu took %6.5f seconds, cpu took %6.5f seconds (gpu %f times faster), mean error %f\n", i+1, gpu_time, cpu_time, cpu_time/gpu_time, sqe);
    check_error(clFlush(queue), "flushing cmd queue");
    check_error(clReleaseMemObject(input_buffer), "releasing input buffer");
    check_error(clReleaseMemObject(weight_buffer), "releasing weight buffer");
    check_error(clReleaseMemObject(output_buffer), "releasing output buffer");

	}
  free(x);
  free(w);
  free(y_gpu);
  free(y_cpu);

  check_error(clFinish(queue), "done with cmd queue");
  check_error(clReleaseKernel(kernel), "releasing kernel");
  check_error(clReleaseProgram(prog), "releasing program");
  check_error(clReleaseCommandQueue(queue), "releasing queue");
  check_error(clReleaseContext(context), "releasing context");

	printf("completed.\n");
}
