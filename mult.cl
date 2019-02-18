__kernel void MULT(__global float *x, __global float *y, __global float *weights, int dim){
	const int i = get_global_id(0);
	y[i] = 0;
	int weight_offset = dim * i;

	for(int j = 0; j < dim; j++){
		y[i] += x[j] * weights[j + weight_offset];
	}
}
