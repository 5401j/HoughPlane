#pragma once
#include <cstdint>
#include <iostream>
#include <vector>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS
// #include <CL/cl2.hpp>
#include <CL/opencl.hpp>

using namespace std; 

struct mydata {
    size_t N;
    vector<float> x;
    vector<float> y;
    vector<float> z;
    vector<float> L={0,0,0};
    vector<string> info;

    vector<int> accumulator_;
    vector<size_t> accumulator;
    float rho_min;
    double dr;
    double dt;
    double dp;
    size_t sr;
    size_t st;
    size_t sp;
    vector<vector<float>> houghParam;
    
    cl::Platform platform;
    cl::Device device;
    cl::Context ctx;
    cl::CommandQueue queue;
    cl::Program prg;
    cl::Buffer device_x,device_y,device_z,dev_accumulator;
};


bool readPointCloud(const char *filename, mydata& data);

float centerPointCloudToOrigin(mydata &data);

void prepareAccumulator(mydata& data, const float rho_max, const size_t n_theta, const size_t n_phi, const size_t n_rho);

void houghTransform(mydata &data);

void identifyPlaneParameters(mydata& data, const float threshold);

bool outputPtxFile(const mydata& data, const char *outputCloudData);

void release(mydata& data); 
