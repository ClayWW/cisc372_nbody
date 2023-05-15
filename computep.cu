#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "cuda.h"
#include <cuda_runtime.h>

vector3* vals;
vector3** accels;

__global__ void paccel(vector3* vals, vector3** accels, vector3* d_vel, vector3* d_pos, double* d_mass){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;
    if(i < NUMENTITIES && j < NUMENTITIES){
        if(i == j){
            FILL_VECTOR(accels[i][j],0,0,0);
        }else{
            vector3 distance;
            for (k=0;k<3;k++) distance[k]=d_pos[i][k]-d_pos[j][k];
			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
			FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }
    }

}

__global__ void psum(vector3 *hVel, vector3* hPos, vector3** accels, vector3* accel_sum){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j,k;
    if(i < NUMENTITIES){
        FILL_VECTOR(accel_sum[i],0,0,0);
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++){
				accel_sum[k]+=accels[(i*NUMENTITIES)+j][k];
            }
		}
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[i][k]*INTERVAL;
			hPos[i][k]=hVel[i][k]*INTERVAL;
		}
	}
}


void compute(){
    vector3 *d_hVel;
    vector3 *d_hPos;
    vector3 *d_acc;
    vector3* d_sum;
    double d_mass;
    int blocks = ceilf(NUMENTITIES/16.0f);  //defining our blocks and threads
    int threads = ceilf(NUMENTITIES/(float)blocks);
    dim3 fullgrid(blocks, blocks, 1);
    dim3 blockdim(threads, threads, 1);
    cudaMallocManaged((void**) &d_hPos, sizeof(vector3)*NUMENTITIES); //allocating mem for position, velocity, mass, our acceleration and sum functions
    cudaMallocManaged((void**) &d_hVel, sizeof(vector3)*NUMENTITIES);
    cudaMallocManaged((void**) &d_mass, sizeof(double)*NUMENTITIES);
    cudaMallocManaged((void**) &d_acc, sizeof(vector3)*NUMENTITIES);        
    cudaMallocManaged((void**) &d_sum, sizeof(vector3)*NUMENTITIES);
    cudaMemcpy(d_hPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice); //copying data from host to device memory
    cudaMemcpy(d_hVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);
    paccel<<<fullgrid, blockdim>>>(d_hPos,d_acc,d_mass); //compute accelerations in parallel
    cudaDeviceSynchronization();
    psum<<<fullgrid.x, blockdim.x>>>(d_acc, d_sum, d_hPos, d_hVel); //sum in parallel
    cudaMemcpy(hPos, d_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost); //copy from device to host memory
    cudaMemcpy(hVel, d_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaFree(d_hPos); //free everything that was allocated
    cudaFree(d_hVel);
    cudaFree(d_mass);
    cudaFree(d_acc);
    cudaFree(d_sum);
}
