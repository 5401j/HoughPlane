#include "../inc/HoughPlane.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

using namespace std;
#define PI 3.14159265
#define RAD 0.01745329

#include <CL/opencl.hpp>

auto getPlatform(const std::string& vendorNameFilter) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for(const auto& p: platforms) {
        if(p.getInfo<CL_PLATFORM_VENDOR>().find(vendorNameFilter) != std::string::npos) {
            return p;
        }
    }
    throw cl::Error(CL_INVALID_PLATFORM, "No platform has the given vendorName");
}

auto getDevice(cl::Platform& platform, cl_device_type type, size_t globalMemoryMB) {
    std::vector<cl::Device> devices;
    platform.getDevices(type, &devices);
    globalMemoryMB *= 1024 * 1024; // from MB to bytes
    for(const auto& d: devices) {
        if( d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() >= globalMemoryMB ) return d;
    }
    throw cl::Error(CL_INVALID_DEVICE, "No device has needed global memory size");
}

auto src=R"CLC(
	__kernel void hough(__global const float *x,
						__global const float *y,
						__global const float *z,
						__global int *accumulator,
						uint N,uint sp,uint st,
						const float dt,const float dp,const float dr,const float rho_min){
		uint j = get_global_id(0);
		uint i = get_global_id(1);
		if(j<sp && i<st){
			float t=dt*i*3.1415926/180.0;											
			float p=dp*j*3.1415926/180.0;											
			float cos_sin=cos(t)*sin(p);									
			float sin_sin=sin(t)*sin(p);								
			float cos_=cos(p);												
			for(uint pts=0;pts<N;pts++){								
				float rho = x[pts]*cos_sin+y[pts]*sin_sin+z[pts]*cos_;	
				uint k=floor((rho-rho_min)/dr);				
				atomic_inc(accumulator+k*st*sp+j*st+i);								
			}
		}

	}
)CLC";


bool readPointCloud(const char *filename, mydata& data) {
	fstream inp(filename);
	if (! inp.good()) {
		cerr << "\nError opening point cloud data file: " << filename;
		return false;
	}
	size_t nData;

	inp >> nData;

	char pp = inp.peek(); 
	if (pp == 'D' || pp == 'd') inp >> pp; 

	data.x.resize(nData);
	data.y.resize(nData);
	data.z.resize(nData);
	data.N=nData;data.info.resize(nData);
	// cout<<"check:"<<data.x.size()<<","<<data.N<<endl;
	for (size_t i = 0; i < nData; ++i) {
		float x, y, z;
		string rest;		// some pts files have only intensity, and some have RGB values, so we treat them all as a long string...
		inp >> x >> y >> z;
		// cout<<"check_i:"<<i<<endl;
		// cout<<"check_xyz:"<<x<<","<<y<<","<<z<<endl;
		// cout<<"check_rest:"<<rest<<endl;
		std::getline(inp, rest); 
		// now you have coordinates in x, y, z, and additional information in the string rest.  You need to store them into your data struct...
		data.x[i]=x;
		data.y[i]=y;
		data.z[i]=z;
		data.info[i]=rest;

		// cout<<"check_pts:"<<data.x[i]<<","<<data.y[i]<<","<<data.z[i]<<endl;
		// cout<<"check_info:"<<data.info[i]<<endl;

	}
	inp.close();

	data.platform = getPlatform("NVIDIA");
	data.device=getDevice(data.platform,CL_DEVICE_TYPE_ALL,1024);
	data.ctx=cl::Context(data.device);
	data.queue=cl::CommandQueue(data.ctx,data.device);

	data.prg = cl::Program(data.ctx, src);
	try {
	    data.prg.build();
	} catch (cl::Error &e) {
	    std::cerr << "\n" << data.prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(data.device);
	    throw cl::Error(CL_INVALID_PROGRAM, "Failed to build kernel");
	}


	return true;
}


float centerPointCloudToOrigin(mydata &data) {
	// first, find bounds
	auto [xmin, xmax] = minmax_element(data.x.begin(), data.x.end());//-49.9651,49.9668
	auto [ymin, ymax] = minmax_element(data.y.begin(), data.y.end());//-49.9847,49.9703
	auto [zmin, zmax] = minmax_element(data.z.begin(), data.z.end());//-49.9966,49.9996
	// or
	// auto ymin = min_element(data.y.begin(), data.y.end());
	// auto ymax = max_element(data.y.begin(), data.y.end());
	// cout<<"check_xmin&max:"<<*xmin<<","<<*xmax<<endl;
	// cout<<"check_ymin&max:"<<*ymin<<","<<*ymax<<endl;
	// cout<<"check_zmin&max:"<<*zmin<<","<<*zmax<<endl;

	data.L[0]=0.5*(*xmin+*xmax);//Lx
	data.L[1]=0.5*(*ymin+*ymax);//Ly
	data.L[2]=0.5*(*zmin+*zmax);//Lz
	// cout<<"check_L:"<<data.L[0]<<" "<<data.L[1]<<" "<<data.L[2]<<endl;

	for_each(data.x.begin(), data.x.end(), [&data](float &x){ x-=data.L[0];});
	for_each(data.y.begin(), data.y.end(), [&data](float &y){ y-=data.L[1];});
	for_each(data.z.begin(), data.z.end(), [&data](float &z){ z-=data.L[2];});

	data.device_x=cl::Buffer(data.ctx,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,data.N*sizeof(float),data.x.data());
	data.device_y=cl::Buffer(data.ctx,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,data.N*sizeof(float),data.y.data());
	data.device_z=cl::Buffer(data.ctx,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,data.N*sizeof(float),data.z.data());
	// cout<<sqrt(pow(0.5*(*xmax-*xmin),2)+pow(0.5*(*ymax-*ymin),2)+pow(0.5*(*zmax-*zmin),2))<<endl;

	return sqrt(pow(0.5*(*xmax-*xmin),2)+pow(0.5*(*ymax-*ymin),2)+pow(0.5*(*zmax-*zmin),2));
}

void prepareAccumulator(mydata &votes, const float rho_max, const size_t n_theta, const size_t n_phi, const size_t n_rho) {
	// cout<<"prepareAccumulator"<<endl;

	votes.sr=n_rho+1;	
	votes.sp=n_phi+1;
	votes.st=n_theta;
	votes.rho_min=-rho_max;
	votes.dr=(2.0*rho_max)/n_rho;
	votes.dp=90.0/n_phi;
	votes.dt=180.0/n_theta;
	votes.accumulator_.resize(votes.sr*votes.sp*votes.st,0);
	// cout<<"prepareAccumulator"<<endl;
	votes.dev_accumulator=cl::Buffer(votes.ctx,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,votes.sr*votes.sp*votes.st*sizeof(int),votes.accumulator_.data());
	// votes.queue.enqueueFillBuffer(votes.dev_accumulator,0,0,votes.sr*votes.sp*votes.st*sizeof(cl_int));
	// votes.dev_accumulator=cl::Buffer(votes.ctx,CL_MEM_READ_WRITE,votes.sr*votes.sp*votes.st*sizeof(size_t),votes.accumulator.votes());
	
}

void houghTransform(mydata &data) {
	// cout<<"houghTransform\n";
	// cout<<"0"<<endl;
	cl::KernelFunctor<const cl::Buffer&,const cl::Buffer&,const cl::Buffer&,cl::Buffer&,cl_uint,cl_uint,cl_uint,const cl_float,const cl_float,const cl_float,const cl_float>houghKernel(data.prg,"hough");
	// cout<<"1"<<endl;
	// data.queue.finish();
	auto config =cl::EnqueueArgs(data.queue,{(data.sp+31)/32*32,(data.st+31)/32*32},{32,32});
	// auto config =cl::EnqueueArgs(data.queue,{(data.sp+15)/16*,(data.st+31)/32*32},{32,32});
	// cout<<"2"<<endl;
	// data.queue.finish();
		// __kernal void hough(__global const float *x,
		// 					__global const float *y,
		// 					__global const float *z,
		// 					__global size_t *accumulator,
		// 					uint N,uint sp,uint st,
		// 					const float dt,const float dp,const float dr){
	houghKernel(config,data.device_x,data.device_y,data.device_z,data.dev_accumulator,data.N,data.sp,data.st,data.dt,data.dp,data.dr,data.rho_min);
	// cout<<"3"<<endl;
	// data.queue.finish();
	data.queue.enqueueReadBuffer(data.dev_accumulator, CL_TRUE ,0,data.sr*data.sp*data.st*sizeof(int),data.accumulator_.data());
	// cout<<"4"<<endl;
	data.queue.finish();
	// for(size_t j=0;j<data.sp;j++){
	// 	for(size_t i=0;i<data.st;i++){
	// 		auto t=data.dt*i*RAD;
	// 		auto p=data.dp*j*RAD;
	// 		auto cos_sin=cos(t)*sin(p);
	// 		auto sin_sin=sin(t)*sin(p);
	// 		auto cos_=cos(p);
	// 		for(size_t pts=0;pts<data.N;pts++){
	// 			auto rho = data.x[pts]*cos_sin+
	// 			           data.y[pts]*sin_sin+
	// 					   data.z[pts]*cos_;
	// 			size_t k=int((rho-data.rho_min)/data.dr);
	// 			data.accumulator_[k*data.st*data.sp + j*data.st + i]++;
	// 		}
	// 	}
	// }


}

void identifyPlaneParameters(mydata& data, const float threshold) {
	// cout<<"identifyPlaneParameters\n";
	// cout << "\nYou need to implement something in " << __PRETTY_FUNCTION__ << " @ " << __LINE__ << " of " << __FILE__; 
	data.houghParam.resize(0);
	size_t b=threshold*data.N;
	for(vector<int>::iterator it=data.accumulator_.begin();it<data.accumulator_.end();it++){
    	it = find_if (it,  data.accumulator_.end(), [&b](int &a){return a>=b;});
    
    	// cout << distance(data.accumulator_.begin(),it) << '\n';
    	size_t dis= distance(data.accumulator_.begin(),it);
    	if(dis<data.accumulator_.size()){
    	// cout << i <<". odd value is " << *it << '\n';i++;
		size_t k=dis/(data.st*data.sp);
		size_t j=(dis-k*data.st*data.sp)/data.st;
		size_t i=dis-k*data.st*data.sp-j*data.st;
		// cout<<i<<" "<<j<<" "<<k<<" \n";

		vector<float> xyz;
		xyz.push_back(data.accumulator_[k*data.st*data.sp+j*data.st+i]);
		// xyz.push_back(data.accumulator_[k][j][i]);
		xyz.push_back(i*data.dt);
		xyz.push_back(j*data.dp);
		xyz.push_back(data.rho_min+k*data.dr);
		data.houghParam.push_back(xyz);
      	}
  	}


//V2
	// data.houghParam.resize(0);
	// for(size_t i=0;i<data.st;i++){
	// 	for(size_t j=0;j<data.sp;j++){
	// 		for(size_t k=0;k<data.sr;k++){
	// 			// if(data.accumulator_[k][j][i]>=threshold*data.N){
	// 				if(data.accumulator_[k*data.st*data.sp + j*data.st + i]>=threshold*data.N){
	// 				vector<float> xyz;
	// 				xyz.push_back(data.accumulator_[k*data.st*data.sp+j*data.st+i]);
	// 				// xyz.push_back(data.accumulator_[k][j][i]);
	// 				xyz.push_back(i*data.dt);
	// 				xyz.push_back(j*data.dp);
	// 				xyz.push_back(data.rho_min+k*data.dr);
	// 				data.houghParam.push_back(xyz);
	// 			}
	// 		}
	// 	}
	// }

	/*v1
	data.houghParam.resize(0);

	for(size_t i=0;i<data.theta.size();i++){
		for(size_t j=0;j<data.phi.size();j++){
			for(size_t k=0;k<data.rho.size();k++){
				if(data.accumulator_[k][j][i]>=threshold*data.N){
					vector<float> xyz;
					xyz.push_back(data.accumulator_[k][j][i]);
					xyz.push_back(i);
					xyz.push_back(j);
					xyz.push_back(k);
					data.houghParam.push_back(xyz);
				}
			}
		}
	}*/
	// cout<<"data.houghParam.size:"<<data.houghParam.size()<<endl;

	sort(data.houghParam.begin(), data.houghParam.end(), [](vector<float> &a, vector<float> &b){return a[0] > b[0]; });
	// for(size_t i=0;i<data.houghParam.size();i++){
	// 	cout<<"houghParam:\n";
	// 	cout<<data.houghParam[i][0]<<" votes with (";
	// 	cout<<data.houghParam[i][1]<<",";
	// 	cout<<data.houghParam[i][2]<<",";
	// 	cout<<data.houghParam[i][3]<<"),\n";
	// }
	// v1
	// for(size_t i=0;i<data.houghParam.size();i++){
	// 	cout<<"houghParam:\n";
	// 	cout<<data.houghParam[i][0]<<" votes with (";
	// 	cout<<data.theta[data.houghParam[i][1]]<<",";
	// 	cout<<data.phi[data.houghParam[i][2]]<<",";
	// 	cout<<data.rho[data.houghParam[i][3]]<<"),\n";
	// }
}


bool outputPtxFile(const mydata& data, const char *outputCloudData) {
	// cout<<"outputPtxFile\n";
	
	vector<vector<size_t>> xyz(data.houghParam.size()+1);
	vector<vector<size_t>> rgb(data.houghParam.size()+1,vector<size_t>(3,0));

	for(size_t i=0;i<data.houghParam.size();i++){
		auto theta=	data.houghParam[i][1];
		auto phi=	data.houghParam[i][2];
		rgb[i][0]=abs(255*cos(theta*RAD)*sin(phi*RAD));
		rgb[i][1]=abs(255*sin(theta*RAD)*sin(phi*RAD));
		rgb[i][2]=abs(255*cos(phi*RAD));
	}

	for(size_t pts=0;pts<data.N;pts++){
		for(size_t i=0;i<data.houghParam.size();i++){
			auto theta=	data.houghParam[i][1];
			auto phi=	data.houghParam[i][2];
			auto rho=	data.houghParam[i][3];

			auto num=data.x[pts]*cos(theta*RAD)*sin(phi*RAD)+
					 data.y[pts]*sin(theta*RAD)*sin(phi*RAD)+
					 data.z[pts]*cos(phi*RAD);
			
			if(abs(num-rho)<data.dr){
				xyz[i].push_back(pts);  //x 
				break;
			}
			else if(i==data.houghParam.size()-1){
				xyz[i+1].push_back(pts);//x
			}
		}
	}

	for(size_t i=0;i<data.houghParam.size()+1;i++){	
		xyz[i].insert(xyz[i].begin(),xyz[i].size());
		rgb[i].insert(rgb[i].begin(),xyz[i].size());												
	}
	sort(xyz.begin(), xyz.end()-1, [](vector<size_t> &a, vector<size_t> &b){return a[0] > b[0]; });
	sort(rgb.begin(), rgb.end()-1, [](vector<size_t> &a, vector<size_t> &b){return a[0] > b[0]; });
	ofstream outp(outputCloudData);
	if (!outp) return false; 
	// Output header of PLY file format, which includes the number of points
	outp << "ply\nformat ascii 1.0\nelement vertex " << data.N
		<< "\nproperty float x\nproperty float y\nproperty float z"
		<< "\nproperty uchar red\nproperty uchar green\nproperty uchar blue"
		<< "\nend_header";



	for(size_t i=0;i<xyz.size();i++){
		for(size_t j=1;j<xyz[i].size();j++){
			outp << "\n" << data.x[xyz[i][j]]+data.L[0] << " " << data.y[xyz[i][j]]+data.L[1] << " " << data.z[xyz[i][j]]+data.L[2] << " " << rgb[i][1] << " " << rgb[i][2] << " " << rgb[i][3];
		}
	}
	outp.close(); 

	return true;
}

void release(mydata& data) {
	// mydata zero;
	// data.pts.swap(zero.pts);
	// data.info.swap(zero.info);
}
