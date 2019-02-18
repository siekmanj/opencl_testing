#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <time.h>
#include <sys/time.h>
#include <string.h>

/* gpu */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main(){
	int status;
	/*
	cl_uint numPlatforms;
	status = clGetPlatformIDs( 0, NULL, &numPlatforms );
	if( status != CL_SUCCESS )
	fprintf( stderr, "clGetPlatformIDs failed (1)\n" );
	fprintf( stderr, "Number of Platforms = %d\n", numPlatforms );

	cl_platform_id platforms[numPlatforms];
	status = clGetPlatformIDs( numPlatforms, platforms, NULL );
	if( status != CL_SUCCESS )
	fprintf( stderr, "clGetPlatformIDs failed (2)\n" );

	cl_device_id device;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	printf("got gpu: %d\n", status);
	*/
	// find out how many platforms are attached here and get their ids:
	cl_uint numPlatforms;
	status = clGetPlatformIDs( 0, NULL, &numPlatforms );
	if( status != CL_SUCCESS )
	fprintf( stderr, "clGetPlatformIDs failed (1)\n" );
	fprintf( stdout, "Number of Platforms = %d\n", numPlatforms );
	cl_platform_id *platforms = new cl_platform_id[ numPlatforms ];
	status = clGetPlatformIDs( numPlatforms, platforms, NULL );
	if( status != CL_SUCCESS )
	fprintf( stderr, "clGetPlatformIDs failed (2)\n" );
	cl_uint numDevices;
	cl_device_id *devices;
	for( int i = 0; i < (int)numPlatforms; i++ )
	{
		fprintf( stdout, "Platform #%d:\n", i );
		size_t size;
		char *str;
		clGetPlatformInfo( platforms[i], CL_PLATFORM_NAME, 0, NULL, &size );
		str = new char [ size ];
		clGetPlatformInfo( platforms[i], CL_PLATFORM_NAME, size, str, NULL );
		fprintf( stdout, "\tName = '%s'\n", str );
		delete[ ] str;
		clGetPlatformInfo( platforms[i], CL_PLATFORM_VENDOR, 0, NULL, &size );
		str = new char [ size ];
		clGetPlatformInfo( platforms[i], CL_PLATFORM_VENDOR, size, str, NULL );
		fprintf( stdout, "\tVendor = '%s'\n", str );
		delete[ ] str;
		clGetPlatformInfo( platforms[i], CL_PLATFORM_VERSION, 0, NULL, &size );
		str = new char [ size ];
		clGetPlatformInfo( platforms[i], CL_PLATFORM_VERSION, size, str, NULL );
		fprintf( stdout, "\tVersion = '%s'\n", str );
		delete[ ] str;
		clGetPlatformInfo( platforms[i], CL_PLATFORM_PROFILE, 0, NULL, &size );
		str = new char [ size ];
		clGetPlatformInfo( platforms[i], CL_PLATFORM_PROFILE, size, str, NULL );
		fprintf( stdout, "\tProfile = '%s'\n", str );
		delete[ ] str;
		// find out how many devices are attached to each platform and get their ids:
		status = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices );
		if( status != CL_SUCCESS )
		fprintf( stderr, "clGetDeviceIDs failed (2)\n" );
		devices = new cl_device_id[ numDevices ];
		status = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL );
		if( status != CL_SUCCESS )
		fprintf( stderr, "clGetDeviceIDs failed (2)\n" );
		for( int j = 0; j < (int)numDevices; j++ )
		{
		fprintf( stdout, "\tDevice #%d:\n", j );
		size_t size;
		cl_device_type type;
		cl_uint ui;
		size_t sizes[3] = { 0, 0, 0 };
		clGetDeviceInfo( devices[j], CL_DEVICE_TYPE, sizeof(type), &type, NULL );
		fprintf( stdout, "\t\tType = 0x%04x = ", type );
		switch( type )
		{
		case CL_DEVICE_TYPE_CPU:
		fprintf( stdout, "CL_DEVICE_TYPE_CPU\n" );
		break;
		case CL_DEVICE_TYPE_GPU:
		fprintf( stdout, "CL_DEVICE_TYPE_GPU\n" );
		break;
		case CL_DEVICE_TYPE_ACCELERATOR:
		fprintf( stdout, "CL_DEVICE_TYPE_ACCELERATOR\n" );
		break;
		default:
		fprintf( stdout, "Other...\n" );
		break;
		}
		clGetDeviceInfo( devices[j], CL_DEVICE_VENDOR_ID, sizeof(ui), &ui, NULL );
		fprintf( stdout, "\t\tDevice Vendor ID = 0x%04x\n", ui );
		clGetDeviceInfo( devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ui), &ui, NULL );
		fprintf( stdout, "\t\tDevice Maximum Compute Units = %d\n", ui );
		clGetDeviceInfo( devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(ui), &ui, NULL );
		fprintf( stdout, "\t\tDevice Maximum Work Item Dimensions = %d\n", ui );
		clGetDeviceInfo( devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), sizes, NULL );
		fprintf( stdout, "\t\tDevice Maximum Work Item Sizes = %d x %d x %d\n", sizes[0], sizes[1], sizes[2] );
		clGetDeviceInfo( devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size), &size, NULL );
		fprintf( stdout, "\t\tDevice Maximum Work Group Size = %d\n", size );
		clGetDeviceInfo( devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(ui), &ui, NULL );
		fprintf( stdout, "\t\tDevice Maximum Clock Frequency = %d MHz\n", ui );
		}
	}
}
