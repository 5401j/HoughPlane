#include "../inc/HoughPlane.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <omp.h>
using namespace std;
#define PI 3.14159265
#define RAD 0.01745329


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
		string rest;
		inp >> x >> y >> z;
		// cout<<"check_i:"<<i<<endl;
		// cout<<"check_xyz:"<<x<<","<<y<<","<<z<<endl;
		// cout<<"check_rest:"<<rest<<endl;
		std::getline(inp, rest); 
		
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
	votes.accumulator.resize(votes.sr*votes.sp*votes.st);

	//XXXXX
	// votes.accumulator_.resize(votes.sr*votes.sp*votes.st);
	// votes.accumulator.resize(votes.sr);
	// for (size_t a = 0; a < votes.sr; ++a) {
	// 	votes.accumulator[a].resize(votes.sp);
	// 	for (size_t b = 0; b < votes.sp; ++b)
	// 		votes.accumulator[a][b].resize(votes.st,0);
	// }
	/*V1
	votes.dr=(2*rho_max)/n_rho;
	votes.rho.resize(n_rho+1);
	votes.theta.resize(n_theta);
	votes.phi.resize(n_phi+1);
	// cout<<"rho_max"<< rho_max <<"\n";
	cout<<votes.dr<<"  check_drho\n";
	for(size_t i=0;i<votes.rho.size();i++){
	votes.rho[i]=-rho_max+i*votes.dr;
	cout<<votes.rho[i]<<" ";
	}
	cout<<votes.rho.size()<<"\n";

	cout<<"\ncheck_dtheta\n";
	for(size_t i=0;i<votes.theta.size();i++){
	votes.theta[i]=(180.0/n_theta)*i;
	cout<<votes.theta[i]<<" ";

	}
	cout<<votes.theta.size()<<"\n";
	
	cout<<"\ncheck_dphi\n";
	for(size_t i=0;i<votes.phi.size();i++){
	votes.phi[i]=(90.0/n_phi)*i;
	cout<<votes.phi[i]<<" ";

	}
	cout<<votes.phi.size()<<"\n";

	votes.accumulator.resize(votes.rho.size());
	for (size_t a = 0; a < votes.rho.size(); ++a) {
		votes.accumulator[a].resize(votes.phi.size());
		for (size_t b = 0; b < votes.phi.size(); ++b)
			votes.accumulator[a][b].resize(votes.theta.size());
	}
	*/

}


void houghTransform(mydata &data) {
	// cout<<"houghTransform\n";
	

	auto r=1/data.dr;
	for(size_t j=0;j<data.sp;j++){
		for(size_t i=0;i<data.st;i++){
			auto t=data.dt*i*RAD;
			auto p=data.dp*j*RAD;
			auto cos_sin=cos(t)*sin(p);
			auto sin_sin=sin(t)*sin(p);
			auto cos_=cos(p);
			for(size_t pts=0;pts<data.N;pts++){
				auto rho = data.x[pts]*cos_sin+
				           data.y[pts]*sin_sin+
						   data.z[pts]*cos_;
				int k=int((rho-data.rho_min)*r);
				data.accumulator[k*data.st*data.sp + j*data.st + i]++;
			}
		}
	}


	/* v1
	for(size_t pts=0;pts<data.N;pts++){
		for(size_t i=0;i<data.theta.size();i++){
			for(size_t j=0;j<data.phi.size();j++){
				auto t=data.theta[i]*RAD;
				auto p=data.phi[j]*RAD;
				auto rho = data.x[pts]*cos(t)*sin(p)+
				           data.y[pts]*sin(t)*sin(p)+
						   data.z[pts]*cos(p);
				auto k=floor((rho-data.rho[0])/data.dr);
				data.accumulator[k][j][i]++;
			}
		}
	}*/

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
		// cout<<data.accumulator[k*data.st*data.sp+j*data.st+i]<<" "<<i<<" "<<j<<" "<<k<<" \n";

		vector<float> xyz;
		xyz.push_back(data.accumulator[k*data.st*data.sp+j*data.st+i]);
		// xyz.push_back(data.accumulator[k][j][i]);
		xyz.push_back(i*data.dt);
		xyz.push_back(j*data.dp);
		xyz.push_back(data.rho_min+k*data.dr);
		data.houghParam.push_back(xyz);
      	}
  	}

	sort(data.houghParam.begin(), data.houghParam.end(), [](vector<float> &a, vector<float> &b){return a[0] > b[0]; });

//V2
	// data.houghParam.resize(0);
	// for(size_t i=0;i<data.st;i++){
	// 	for(size_t j=0;j<data.sp;j++){
	// 		for(size_t k=0;k<data.sr;k++){
	// 			// if(data.accumulator[k][j][i]>=threshold*data.N){
	// 				if(data.accumulator[k*data.st*data.sp + j*data.st + i]>=threshold*data.N){
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

	/*v1
	data.houghParam.resize(0);

	for(size_t i=0;i<data.theta.size();i++){
		for(size_t j=0;j<data.phi.size();j++){
			for(size_t k=0;k<data.rho.size();k++){
				if(data.accumulator[k][j][i]>=threshold*data.N){
					vector<float> xyz;
					xyz.push_back(data.accumulator[k][j][i]);
					xyz.push_back(i);
					xyz.push_back(j);
					xyz.push_back(k);
					data.houghParam.push_back(xyz);
				}
			}
		}
	}*/
	// cout<<"çµ„æ•¸:"<<data.houghParam.size()<<endl;

	// for(size_t i=0;i<data.houghParam.size();i++){
	// 	cout<<"houghParam:\n";
	// 	cout<<data.houghParam[i][0]<<" votes with (";
	// 	cout<<data.houghParam[i][1]<<"âˆ˜,";
	// 	cout<<data.houghParam[i][2]<<"âˆ˜,";
	// 	cout<<data.houghParam[i][3]<<"),\n";
	// }
	// v1
	// for(size_t i=0;i<data.houghParam.size();i++){
	// 	cout<<"houghParam:\n";
	// 	cout<<data.houghParam[i][0]<<" votes with (";
	// 	cout<<data.theta[data.houghParam[i][1]]<<"âˆ˜,";
	// 	cout<<data.phi[data.houghParam[i][2]]<<"âˆ˜,";
	// 	cout<<data.rho[data.houghParam[i][3]]<<"),\n";
	// }
}

bool outputPtxFile(const mydata& data, const char *outputCloudData) {
	// cout<<"outputPtxFile\n";
	
	vector<vector<size_t>> xyz(data.houghParam.size()+1);//nå€‹å¹³é¢+ä¸€çµ„noise
	vector<vector<size_t>> rgb(data.houghParam.size()+1,vector<size_t>(3,0));//nå€‹å¹³é¢+ä¸€çµ„noise

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

// task 10 - release all allocated memory 
void release(mydata& data) {
	// mydata zero;
	// data.pts.swap(zero.pts);
	// data.info.swap(zero.info);
}
