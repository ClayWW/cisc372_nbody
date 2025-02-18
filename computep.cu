#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "cuda.h"
#include <cuda_runtime.h>

//globals
vector3* values;
vector3** accels;
/*
p_compute: performs the same function as compute.c but in parrallel
Input:
    Globals
    +
    vector3* d_hPos: position on device
    vector3* d_hVel: velocity on device
    double d_mass: mass on device
Output: None, calculates accelereration matrix and then sums the columns for signle acceleration effect for an object
*/
__global__ void p_compute(vector3* values, vector3** accels, vector3* d_hPos, vector3* d_hVel, double *d_mass){
    int currID = blockIdx.x * blockDim.x + threadIdx.x; //current thread block shoutout google
    int i = currID / NUMENTITIES; //locate position once again shoutout google
    int j = currID % NUMENTITIES;
    accels[currID] = &values[currID*NUMENTITIES]; //accels array for all the accel pointers
    if(currID < NUMENTITIES*NUMENTITIES){
        if(i == j){ //imma keep it a full stack(no pun intended) I copied this from the compute.c file
            FILL_VECTOR(accels[i][j],0,0,0);
        }else{
            vector3 distance;
            for (int k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
			FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }
        vector3 accel_sum = {(double) *(accels[currID])[0], (double) *(accels[currID])[1], (double) *(accels[currID])[2]}; //accel array vals
		/* doing this shit manually theres gotta be a better way
        d_hVel[i][0] = d_hVel[i][0]+accel_sum[0]*INTERVAL; //updating the relative acceleration and velocities
		d_hPos[i][0] = d_hVel[i][0]*INTERVAL;
		d_hVel[i][1] = d_hVel[i][1]+accel_sum[1]*INTERVAL;
		d_hPos[i][1] = d_hVel[i][1]*INTERVAL;
		d_hVel[i][2] = d_hVel[i][2]+accel_sum[2]*INTERVAL;
		d_hPos[i][2] = d_hVel[i][2]*INTERVAL;
        */
        for(int z = 0; z < 3; z++){ //seeing if for loop is faster. Update: I am speed.
            d_hVel[i][z] +=accel_sum[z]*INTERVAL;
            d_hPos[i][z] = d_hVel[i][z]*INTERVAL;
        }
    }

}


/* Error that I literally just cannot figure out for my life
__global__ void psum(vector3** accels, vector3* accel_sum, vector3* d_hPos, vector3* d_hVel){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < NUMENTITIES){
        FILL_VECTOR(accel_sum[i],0,0,0);
		for (int j=0;j<NUMENTITIES;j++){
			for (int k=0;k<3;k++){
				accel_sum[i][k] = accel_sum[i][k] + accels[(i * NUMENTITIES) + j][k];
            }
		}
		for (int k=0;k<3;k++){
			d_hVel[i][k]+=accel_sum[i][k]*INTERVAL;
			d_hPos[i][k]=d_hVel[i][k]*INTERVAL;
		}
	}
}
*/

void compute(){
    vector3 *d_hVel;
    vector3 *d_hPos;
    //vector3 *d_acc;
    //vector3 *d_sum;
    double *d_mass;
    int blocksize = 256;
    int totalBlocks = (NUMENTITIES+blocksize-1)/blocksize;
    //int blocks = ceilf(NUMENTITIES/16.0f);  //defining our blocks and threads
    //int threads = ceilf(NUMENTITIES/(float)blocks);
    //dim3 fullgrid(blocks, blocks, 1);
    //dim3 blockdim(threads, threads, 1);
    cudaMallocManaged((void**) &d_hPos, sizeof(vector3) * NUMENTITIES); //allocating mem for position, velocity, mass
    cudaMallocManaged((void**) &d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMallocManaged((void**) &d_mass, sizeof(double) * NUMENTITIES);
    //cudaMallocManaged((void**) &d_acc, sizeof(vector3) * NUMENTITIES);        
    //cudaMallocManaged((void**) &d_sum, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice); //copying data from host to device memory
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
    p_compute<<<totalBlocks, blocksize>>>(values, accels, d_hVel, d_hPos, d_mass); //compute accelerations in parallel
    cudaDeviceSynchronize(); //make sure its all coming together 
    //psum<<<fullgrid.x, blockdim.x>>>(d_acc, d_sum, d_hPos, d_hVel); //sum in parallel
    //cudaDeviceSynchronize();
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost); //copy from device to host memory
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaFree(d_hPos); //free everything that was allocated
    cudaFree(d_hVel);
    cudaFree(d_mass);
    //cudaFree(d_acc);
    //cudaFree(d_sum);
}
