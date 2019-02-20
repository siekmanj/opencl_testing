#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

void check_error(int err, char *str){
	if(err != CL_SUCCESS){
		printf("ERROR: '%s'.\n", str);
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

void mult_cpu(const float *x, const float **w, float *dest, size_t size, size_t inputsize){
	for(int i = 0; i < size; i++){
		dest[i] = 0;
		for(int j = 0; j < inputsize; j++){
			dest[i] += w[i][j] * x[j];
		}
	}
}

int main(){
	int status = 0;

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


	//clBuildProgram(program

	/* End setup */
	size_t input_size = 95;
	size_t neurons = 500;
	
	float *x = (float*)malloc(sizeof(float) * input_size);
	float *w = (float*)malloc(sizeof(float) * neurons * input_size);
	float *y = (float*)malloc(sizeof(float) * neurons);

	for(int i = 0; i < 10; i++){
		for(int j = 0; j < input_size; j++){
			x[j] = uniform(-1, 1);
		}
		for(int j = 0; j < input_size * neurons; j++){
			w[j] = uniform(-1, 1);
		}
		cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_size, x, &status);
		if(status != CL_SUCCESS) error("failed to create clmem from x");
		cl_mem weight_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_size * neurons, w, &status);
		if(status != CL_SUCCESS) error("failed to create clmem from weights");
		cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * neurons, y, &status);
		if(status != CL_SUCCESS) error("failed to create clmem from output buf");

		clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight_buffer);
		clSetKernelArg(kernel, 2, sizeof(int), &input_size);

		printf("about to enqueue\n");
		check_error(clEnqueueNDRangeKernel(queue, output_buffer, 1, NULL, &neurons, 64, 0, NULL, NULL), "couldn't enqueue kernel.");
		exit(1);
	}

	float *input;

	printf("completed.\n");
}
