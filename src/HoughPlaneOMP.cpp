#include "../inc/HoughPlane.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <array>
#include <atomic>
// #include<execution>
// #include<numeric>

using namespace std;
#define PI 3.14159265

// auto policy=std::execution::par;

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

	return true;
}

float centerPointCloudToOrigin(mydata &data) {
	// cout << "\nYou need to implement something in " << __PRETTY_FUNCTION__ << " @ " << __LINE__ << " of " << __FILE__; 
	// first, find bounds
	auto [xmin, xmax] = minmax_element(data.x.begin(), data.x.end());//-49.9651,49.9668
	auto [ymin, ymax] = minmax_element(data.y.begin(), data.y.end());//-49.9847,49.9703
	auto [zmin, zmax] = minmax_element(data.z.begin(), data.z.end());//-49.9966,49.9996
	// auto xmin=data.x[0],xmax=data.x[0];
	// auto ymin=data.y[0],ymax=data.y[0];
	// auto zmin=data.x[0],zmax=data.z[0];

	// #pragma omp parallel for schedule(static)
	// for(size_t i=1;i<data.N;i++){
	// 	xmin=(xmin>data.x[i])?data.x[i]:xmin;
	// 	xmax=(xmax<data.x[i])?data.x[i]:xmax;
	// 	ymin=(ymin>data.y[i])?data.y[i]:ymin;
	// 	ymax=(ymax<data.y[i])?data.y[i]:ymax;
	// 	zmin=(zmin>data.z[i])?data.z[i]:zmin;
	// 	zmax=(zmax<data.z[i])?data.z[i]:zmax;

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

	// cout<<sqrt(pow(0.5*(*xmax-*xmin),2)+pow(0.5*(*ymax-*ymin),2)+pow(0.5*(*zmax-*zmin),2))<<endl;

	return sqrt(pow(0.5*(*xmax-*xmin),2)+pow(0.5*(*ymax-*ymin),2)+pow(0.5*(*zmax-*zmin),2));
}

void prepareAccumulator(mydata &votes, const float rho_max, const size_t n_theta, const size_t n_phi, const size_t n_rho) {
	// cout<<"prepareAccumulator\n";

	votes.sr=n_rho+1;	
	votes.sp=n_phi+1;
	votes.st=n_theta;

	votes.rho_min=-rho_max;
	votes.dr=(2.0*rho_max)/n_rho;
	votes.dp=90.0/n_phi;
	votes.dt=180.0/n_theta;

	votes.accumulator.resize(votes.sr*votes.sp*votes.st,0);

}

void houghTransform(mydata &data) {
	// cout<<"houghTransform\n";
	
	#pragma omp declare reduction(vec_size_t_plus : std::vector<size_t> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
	vector<size_t> count(data.sr*data.sp*data.st,0);
	#pragma omp parallel reduction(vec_size_t_plus : count)
	{
	#pragma omp for schedule(static) collapse(2)
	for(size_t j=0;j<data.sp;j++){
		for(size_t i=0;i<data.st;i++){
			auto t=data.dt*i*PI/180.0;
			auto p=data.dp*j*PI/180.0;
			auto cos_sin=cos(t)*sin(p);
			auto sin_sin=sin(t)*sin(p);
			auto cos_=cos(p);
			for(size_t pts=0;pts<data.N;pts++){
				auto rho = data.x[pts]*cos_sin+
				           data.y[pts]*sin_sin+
						   data.z[pts]*cos_;
				int k=int((rho-data.rho_min)/data.dr);
				count[k*data.st*data.sp + j*data.st + i]++;
			}
		}
	}
	}
	swap(data.accumulator,count);
	// data.accumulator.assign(count.begin(),count.end());

	/*vector<float> ctheta_sphi(data.st*data.sp);
	vector<float> stheta_sphi(data.st*data.sp);
	vector<float> cphi(data.st*data.sp);

	// #pragma omp for schedule(static) collapse(2)
	for(size_t i=0;i<data.st;i++){
		for(size_t j=0;j<data.sp;j++){
			auto t=data.dt*i*PI/180.0;
			auto p=data.dp*j*PI/180.0;
			ctheta_sphi[j*data.st+i]=cos(t)*sin(p);
			stheta_sphi[j*data.st+i]=sin(t)*sin(p);
			cphi[j*data.st+i]=cos(p);
		}
	}

	// #pragma omp declare reduction(vec_size_t_plus : std::vector<size_t> : \
    //                           std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
    //                 initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
	// vector<size_t> count(data.sr*data.sp*data.st,0);
	// #pragma omp parallel reduction(vec_size_t_plus : count)
	// {
	// #pragma omp for schedule(static) collapse(2)
	for(size_t pts=0;pts<data.N;pts++){
		for(size_t it=0;it<data.st*data.sp;it++){
				auto rho = data.x[pts]*ctheta_sphi[it]+
				           data.y[pts]*stheta_sphi[it]+
						   data.z[pts]*cphi[it];
				int k=int((rho-data.rho_min)/data.dr);
				data.accumulator[k*data.st*data.sp + it]++;
				// count[k*data.st*data.sp + it]++;
		}
	// }
	}*/
	// #pragma omp declare reduction(vec_size_t_plus : std::vector<size_t> : 
    //                           std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) 
    //                 initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
	// vector<size_t> count(data.sr*data.sp*data.st,0);
	// #pragma omp parallel reduction(vec_size_t_plus : count)
	// {
	// #pragma omp for //schedule(static) collapse(2)
	// for(size_t pts=0;pts<data.N;pts++){
	// 	for(size_t i=0;i<data.st;i++){
	// 		for(size_t j=0;j<data.sp;j++){
	// 			auto t=data.dt*i*PI/180.0;
	// 			auto p=data.dp*j*PI/180.0;;
	// 			auto rho = data.x[pts]*cos(t)*sin(p)+
	// 			           data.y[pts]*sin(t)*sin(p)+
	// 					   data.z[pts]*cos(p);
	// 			size_t k=floor((rho-data.rho_min)/data.dr);
	// 			count[k*data.st*data.sp + j*data.st + i]+=1;
	// 		}
	// 	}
	// }
	// }


}

void identifyPlaneParameters(mydata& data, const float threshold) {
	// cout<<"identifyPlaneParameters\n";
	data.houghParam.resize(0);
	size_t b=threshold*data.N;
	
	for(vector<size_t>::iterator it=data.accumulator.begin();it<data.accumulator.end();it++){
    	it = find_if (it,  data.accumulator.end(), [&b](size_t &a){return a>=b;});
    	// cout << distance(data.accumulator.begin(),it) << '\n';
    	size_t dis= distance(data.accumulator.begin(),it);
    	if(dis<data.accumulator.size()){
    	// cout << i <<". odd value is " << *it << '\n';i++;
		size_t k=dis/(data.st*data.sp);
		size_t j=(dis-k*data.st*data.sp)/data.st;
		size_t i=dis-k*data.st*data.sp-j*data.st;
		// cout<<i<<" "<<j<<" "<<k<<" \n";

		vector<float> xyz;
		xyz.push_back(data.accumulator[k*data.st*data.sp+j*data.st+i]);
		// xyz.push_back(data.accumulator[k][j][i]);
		xyz.push_back(i*data.dt);
		xyz.push_back(j*data.dp);
		xyz.push_back(data.rho_min+k*data.dr);
		data.houghParam.push_back(xyz);
      	}
  	}



//V2
	// data.houghParam.resize(0);
	// #pragma omp parallel for ordered
	// for(size_t i=0;i<data.st;i++){
	// 	for(size_t j=0;j<data.sp;j++){
	// 		#pragma omp ordered
	// 		for(size_t k=0;k<data.sr;k++){
	// 			if(data.accumulator[k*data.st*data.sp + j*data.st + i]>=threshold*data.N){
	// 			// if(data.accumulator[k][j][i]>=threshold*data.N){
	// 				vector<float> xyz;
	// 				xyz.push_back(data.accumulator[k*data.st*data.sp+j*data.st+i]);
	// 				// xyz.push_back(data.accumulator[k][j][i]);
	// 				xyz.push_back(i*data.dt);
	// 				xyz.push_back(j*data.dp);
	// 				xyz.push_back(data.rho_min+k*data.dr);
	// 				data.houghParam.push_back(xyz);
	// 			}
	// 		}
	// 	}
	// }
	
	// cout<<"data.houghParam.size"<<data.houghParam.size()<<endl;

	sort(data.houghParam.begin(), data.houghParam.end(), [](vector<float> &a, vector<float> &b){return a[0] > b[0]; });

	// for(size_t i=0;i<data.houghParam.size();i++){
	// 	cout<<"houghParam:\n";
	// 	cout<<data.houghParam[i][0]<<" votes with (";
	// 	cout<<data.houghParam[i][1]<<",";
	// 	cout<<data.houghParam[i][2]<<",";
	// 	cout<<data.houghParam[i][3]<<"),\n";
	// }

}


bool outputPtxFile(const mydata& data, const char *outputCloudData) {
	// cout<<"outputPtxFile\n";
	// cout << "\nYou need to implement something in " << __PRETTY_FUNCTION__ << " @ " << __LINE__ << " of " << __FILE__; 
	vector<vector<size_t>> xyz(data.houghParam.size()+1);
	vector<vector<size_t>> rgb(data.houghParam.size()+1,vector<size_t>(3,0));

	#pragma omp parallel for
	for(size_t i=0;i<data.houghParam.size();i++){
		auto theta=	data.houghParam[i][1];
		auto phi=	data.houghParam[i][2];
		rgb[i][0]=abs(255*cos(theta*PI/180)*sin(phi*PI/180));
		rgb[i][1]=abs(255*sin(theta*PI/180)*sin(phi*PI/180));
		rgb[i][2]=abs(255*cos(phi*PI/180));
	}

	#pragma omp parallel for 
	for(size_t pts=0;pts<data.N;pts++){
		for(size_t i=0;i<data.houghParam.size();i++){
			auto theta=	data.houghParam[i][1];
			auto phi=	data.houghParam[i][2];
			auto rho=	data.houghParam[i][3];

			auto num=data.x[pts]*cos(theta*PI/180)*sin(phi*PI/180)+
					data.y[pts]*sin(theta*PI/180)*sin(phi*PI/180)+
					data.z[pts]*cos(phi*PI/180);
			
			if(abs(num-rho)<data.dr){
				#pragma omp critical
				xyz[i].push_back(pts);  //x 
				break;
			}
			else if(i==data.houghParam.size()-1){
				#pragma omp critical
				xyz[i+1].push_back(pts);//x
			}
		}
	}

	#pragma omp parallel for
	for(size_t i=0;i<data.houghParam.size()+1;i++){	
		xyz[i].insert(xyz[i].begin(),xyz[i].size());
		rgb[i].insert(rgb[i].begin(),xyz[i].size());												
	}
	sort(xyz.begin(), xyz.end()-1, [](vector<size_t> &a, vector<size_t> &b){return a[0] > b[0]; });
	sort(rgb.begin(), rgb.end()-1, [](vector<size_t> &a, vector<size_t> &b){return a[0] > b[0]; });
	ofstream outp(outputCloudData);
	if (!outp) return false; 
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
