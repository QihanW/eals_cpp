#include<iostream>
using namespace std;
 
int main()
{
    int deviceCount;
 
    cudaGetDeviceCount(&deviceCount);	//Returns in *deviceCount the number of devices
    cout<<"deviceCount:   "<<deviceCount<<"\n\n";
    if (deviceCount == 0)
    {
        cout<< "error: no devices supporting CUDA.\n";
        exit(EXIT_FAILURE);
    }
 
    int dev = 0;
    cudaSetDevice(dev);	//Sets dev=0 device as the current device for the calling host thread.
 
    cudaDeviceProp devProps;
    cudaGetDeviceProperties(&devProps, dev);
 
    cout<<"name: "<<devProps.name<<"\n";
    cout<<"totalGlobalMem: "<<devProps.totalGlobalMem<<"\n";
    cout<<"regsPerBlock: "<<devProps.regsPerBlock<<"\n";
    cout<<"warpSize: "<<devProps.warpSize<<"\n";
    cout<<"memPitch: "<<devProps.memPitch<<"\n\n";
 
    cout<<"一个线程块中可使用的最大共享内存\n";
    cout<<"devProps.sharedMemPerBlock: "<<devProps.sharedMemPerBlock<<"\n\n";
 
	cout<<"一个线程块中可包含的最大线程数量\n";
    cout<<"maxThreadsPerBlock: "<<devProps.maxThreadsPerBlock<<"\n\n";
 
	cout<<"多维线程块数组中每一维可包含的最大线程数量\n";
    cout<<"maxThreadsDim[0]: "<<devProps.maxThreadsDim[0]<<"\n";
    cout<<"maxThreadsDim[1]: "<<devProps.maxThreadsDim[1]<<"\n";
    cout<<"maxThreadsDim[2]: "<<devProps.maxThreadsDim[2]<<"\n\n";
 
	cout<<"一个线程格中每一维可包含的最大线程块数量\n";
    cout<<"maxGridSize[0]: "<<devProps.maxGridSize[0]<<"\n";
    cout<<"maxGridSize[1]: "<<devProps.maxGridSize[1]<<"\n";
    cout<<"maxGridSize[2]: "<<devProps.maxGridSize[2]<<"\n\n";
 
    cout<<"clockRate: "<<devProps.clockRate<<"\n";
    cout<<"totalConstMem: "<<devProps.totalConstMem<<"\n";
    cout<<"textureAlignment: "<<devProps.textureAlignment<<"\n\n";
 
    cout<<"计算能力："<<devProps.major<< "." <<devProps.minor<<"\n\n";
 
    cout<<"minor: "<<devProps.minor<<"\n";
    cout<<"texturePitchAlignment: "<<devProps.texturePitchAlignment<<"\n";
    cout<<"deviceOverlap: "<<devProps.deviceOverlap<<"\n";
    cout<<"multiProcessorCount: "<<devProps.multiProcessorCount<<"\n";
    cout<<"kernelExecTimeoutEnabled: "<<devProps.kernelExecTimeoutEnabled<<"\n";
    cout<<"integrated: "<<devProps.integrated<<"\n";
    cout<<"canMapHostMemory: "<<devProps.canMapHostMemory<<"\n";
    cout<<"computeMode: "<<devProps.computeMode<<"\n";
    cout<<"maxTexture1D: "<<devProps.maxTexture1D<<"\n";
    cout<<"maxTexture1DMipmap: "<<devProps.maxTexture1DMipmap<<"\n";
    cout<<"maxTexture1DLinear: "<<devProps.maxTexture1DLinear<<"\n";
    cout<<"maxTexture2D: "<<devProps.maxTexture2D<<"\n";
    cout<<"maxTexture2DMipmap: "<<devProps.maxTexture2DMipmap<<"\n";
    cout<<"maxTexture2DLinear: "<<devProps.maxTexture2DLinear<<"\n";
    cout<<"maxTexture2DGather: "<<devProps.maxTexture2DGather<<"\n";
    cout<<"maxTexture3D: "<<devProps.maxTexture3D<<"\n";
    cout<<"maxTexture3DAlt: "<<devProps.maxTexture3DAlt<<"\n";
    cout<<"maxTextureCubemap: "<<devProps.maxTextureCubemap<<"\n";
    cout<<"maxTexture1DLayered: "<<devProps.maxTexture1DLayered<<"\n";
    cout<<"maxTexture2DLayered: "<<devProps.maxTexture2DLayered<<"\n";
    cout<<"maxTextureCubemapLayered: "<<devProps.maxTextureCubemapLayered<<"\n";
    cout<<"maxSurface1D: "<<devProps.maxSurface1D<<"\n";
    cout<<"maxSurface2D: "<<devProps.maxSurface2D<<"\n";
    cout<<"maxSurface3D: "<<devProps.maxSurface3D<<"\n";
    cout<<"maxSurface1DLayered: "<<devProps.maxSurface1DLayered<<"\n";
    cout<<"maxSurface2DLayered: "<<devProps.maxSurface2DLayered<<"\n";
    cout<<"maxSurfaceCubemap: "<<devProps.maxSurfaceCubemap<<"\n";
    cout<<"maxSurfaceCubemapLayered: "<<devProps.maxSurfaceCubemapLayered<<"\n";
    cout<<"surfaceAlignment: "<<devProps.surfaceAlignment<<"\n";
    cout<<"concurrentKernels: "<<devProps.concurrentKernels<<"\n";
    cout<<"ECCEnabled: "<<devProps.ECCEnabled<<"\n";
    cout<<"pciBusID: "<<devProps.pciBusID<<"\n";
    cout<<"pciDeviceID: "<<devProps.pciDeviceID<<"\n";
    cout<<"pciDomainID: "<<devProps.pciDomainID<<"\n";
    cout<<"tccDriver: "<<devProps.tccDriver<<"\n";
    cout<<"asyncEngineCount: "<<devProps.asyncEngineCount<<"\n";
    cout<<"unifiedAddressing: "<<devProps.unifiedAddressing<<"\n";
    cout<<"memoryClockRate: "<<devProps.memoryClockRate<<"\n";
    cout<<"memoryBusWidth: "<<devProps.memoryBusWidth<<"\n";
    cout<<"l2CacheSize: "<<devProps.l2CacheSize<<"\n";
    cout<<"maxThreadsPerMultiProcessor: "<<devProps.maxThreadsPerMultiProcessor<<"\n";
    cout<<"streamPrioritiesSupported: "<<devProps.streamPrioritiesSupported<<"\n";
    cout<<"globalL1CacheSupported: "<<devProps.globalL1CacheSupported<<"\n";
    cout<<"localL1CacheSupported: "<<devProps.localL1CacheSupported<<"\n";
    cout<<"sharedMemPerMultiprocessor: "<<devProps.sharedMemPerMultiprocessor<<"\n";
    cout<<"regsPerMultiprocessor: "<<devProps.regsPerMultiprocessor<<"\n";
    cout<<"isMultiGpuBoard: "<<devProps.isMultiGpuBoard<<"\n";
    cout<<"multiGpuBoardGroupID: "<<devProps.multiGpuBoardGroupID<<"\n";
    cout<<"singleToDoublePrecisionPerfRatio: "<<devProps.singleToDoublePrecisionPerfRatio<<"\n";
    cout<<"pageableMemoryAccess: "<<devProps.pageableMemoryAccess<<"\n";
    cout<<"concurrentManagedAccess: "<<devProps.concurrentManagedAccess<<"\n";
 
 
}

