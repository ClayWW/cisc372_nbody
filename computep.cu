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
    if(i,j < NUMENTITIES){
        if(i == j){
            FILL_VECTOR(accels[i][j],0,0,0);
        }else{
            vector3 distance;
            for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			double magnitude=sqrt(magnitude_sq);
			double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
			FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }
    }

}

__global__ void psum(vector3 *hVel, vector3* hPos, vector3** accels, vector3* accel_sum){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k;
    if(i < NUMENTITIES){
        FILL_VECTOR(accel_sum[i],0,0,0);
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[(i*NUMENTITIES)+j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[i][k]*INTERVAL;
			hPos[i][k]=hVel[i][k]*INTERVAL;
		}
	}
}


void compute(){
}
