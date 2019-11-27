#include "sgm_cl.h"

namespace  sgm_cl{

/**
 * definitions for helper functions
 */
static void HandleError(cl_int err, const std::string& msg){
    if(err != CL_SUCCESS){
        printf("[OpenCL Error] in %s !: %d \n", msg.c_str(), err);
        exit(EXIT_FAILURE);
    }
}

static int GetCLMemFlag(MemFlag mem_flag)
{
    int ret = 0;
    switch(mem_flag){
       case MEM_FLAG_READ_WRITE:
           ret = ret | CL_MEM_READ_WRITE; break;
       case MEM_FLAG_READ_ONLY:
           ret = ret | CL_MEM_READ_ONLY; break;
       case MEM_FLAG_WRITE_ONLY:
           ret = ret | CL_MEM_WRITE_ONLY; break;
       case MEM_FLAG_USE_HOST_PTR:
           ret = ret | CL_MEM_USE_HOST_PTR; break;
       case MEM_FLAG_ALLOC_HOST_PTR:
           ret = ret | CL_MEM_ALLOC_HOST_PTR; break;
       case MEM_FLAG_COPY_HOST_PTR:
           ret = ret | CL_MEM_COPY_HOST_PTR; break;
    }
    return ret;
}

/**
 * definitions for members of CLContext
 */
CLContext::CLContext(int platform_id, int device_id, int num_streams) {
    cl_platform_id p_id;
    cl_int err = 0;
    cl_uint num_platforms, num_divices;
    std::vector<cl_platform_id> p_ids;
    std::vector<cl_device_id> d_ids;

    clGetPlatformIDs(0, nullptr, &num_platforms);
    if(num_platforms > 0){
        p_ids.resize(num_platforms);
        clGetPlatformIDs(num_platforms,p_ids.data(),nullptr);
        if(platform_id < 0 || platform_id >= int(num_platforms)) {
            printf("Incorrect platform id %d!\n",platform_id);
            exit(EXIT_FAILURE);
        }
        p_id = p_ids[platform_id];
    }else {
        printf("Not found any platforms\n");
        exit(EXIT_FAILURE);
    }

    clGetDeviceIDs(p_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_divices);
    if(num_divices > 0){
        d_ids.resize(num_divices);
        clGetDeviceIDs(p_id, CL_DEVICE_TYPE_ALL, num_divices,d_ids.data(),nullptr);
        if(device_id < 0 || device_id >= int(num_divices)){
            printf("Incorrect device id %d!\n",device_id);
            exit(EXIT_FAILURE);
        }
        cl_device_id_ = d_ids[device_id];
    }
    else{
        printf("Not found any devices\n");
        exit(EXIT_FAILURE);
    }

    cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)p_id,0};
    cl_context_ = clCreateContext(prop, 1, &cl_device_id_, nullptr, nullptr, &err);
    HandleError(err, "creating context");
    printf("OpenCL context created! \n");

    cl_command_queues_.resize(num_streams);
    for(int i=0; i < num_streams; i++){
        cl_command_queues_[i] = clCreateCommandQueue(cl_context_, cl_device_id_,
                                                     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        HandleError(err, "creating ClCommandQueue");
    }

    std::ostringstream oss;
    oss<<"Selected platform vendor: "<<GetPlatformInfo(p_id,CL_PLATFORM_VENDOR)<<" "
                                     <<GetPlatformInfo(p_id,CL_PLATFORM_VERSION)<<"\n"
       <<"Selected device name: "<<GetDevInfo(cl_device_id_, CL_DEVICE_NAME)<<"\n"
       <<"Selected device OpenCL device version: "
                                 <<GetDevInfo(cl_device_id_, CL_DEVICE_VERSION)<<"\n"
       <<"Selected device OpenCL C device version: "
                                 <<GetDevInfo(cl_device_id_, CL_DEVICE_OPENCL_C_VERSION)<<"\n";
    cl_info_ = oss.str();
}

CLContext::~CLContext(){
    for(auto& cq : cl_command_queues_)
        clReleaseCommandQueue(cq);
    clReleaseContext(cl_context_);
}

CLContext::CLContext(CLContext&& right) noexcept :cl_command_queues_(right.cl_command_queues_),
    cl_info_(right.cl_info_), cl_context_(right.cl_context_), cl_device_id_(right.cl_device_id_){
    right.cl_command_queues_.clear();
    right.cl_info_ = "";
    right.cl_context_ = nullptr;
    right.cl_device_id_ = nullptr;
}

CLContext& CLContext::operator=(CLContext&& right) noexcept{
    if(this != &right){
        cl_command_queues_ = right.cl_command_queues_;
        cl_info_ = right.cl_info_;
        cl_context_ = right.cl_context_;
        cl_device_id_ = right.cl_device_id_;

        right.cl_command_queues_.clear();
        right.cl_info_ = "";
        right.cl_context_ = nullptr;
        right.cl_device_id_ = nullptr;
    }
    return *this;
}

cl_command_queue CLContext::GetCommandQueue(int id) const{
    return cl_command_queues_.at(id);
}

void CLContext::Finish(int command_queue_id) const{
    cl_int err = clFinish(cl_command_queues_[command_queue_id]);
    HandleError(err, "finishing command queue");
}

std::string CLContext::GetPlatformInfo(cl_platform_id platform_id, int info_name) const{
    size_t info_size = 0;
    clGetPlatformInfo(platform_id, info_name, 0, nullptr, &info_size);
    std::string str;
    str.resize(info_size);
    clGetPlatformInfo(platform_id, info_name, info_size, &str[0], nullptr);
    return std::move(str);
}

std::string CLContext::GetDevInfo(cl_device_id dev_id, int info_name) const{
    size_t info_size = 0;
    clGetDeviceInfo(dev_id, info_name, 0, nullptr, &info_size);
    std::string str;
    str.resize(info_size);
    clGetDeviceInfo(dev_id, info_name, info_size, &str[0], nullptr);
    return std::move(str);
}

/**
 * definitions for members of CLProgram
 */
CLProgram::CLProgram(const std::string& source_path, const CLContext* context,
                     const std::string& compilation_options):context_(context){
    std::ifstream file_in;
    file_in.open(source_path);
    char* data_src = nullptr;
    if(file_in.is_open()){
        std::string str{std::istreambuf_iterator<char>(file_in),
                        std::istreambuf_iterator<char>()};
        size_t num_bytes_src = str.size();
        data_src = new char[num_bytes_src];
        memcpy(data_src, str.data(),num_bytes_src);
        CreateProgram(data_src, num_bytes_src, compilation_options);
        CreateKernels();
        delete []data_src;
    }else {
        printf("Cannot open %s!\n", source_path.c_str());
        exit(EXIT_FAILURE);
    }
    file_in.close();
}

CLProgram::~CLProgram(){
    for(auto& item : kernels_){
        delete item.second;
        item.second = nullptr;
    }
    if(cl_program_ != nullptr){
        cl_int err = clReleaseProgram(cl_program_);
        HandleError(err, "releasing program");
    }
}

bool CLProgram::CreateProgram(const char* source, size_t size_src,
                              const std::string& compilation_options){
    cl_int err;
    cl_program_ = clCreateProgramWithSource(context_->GetCLContext(),1,
                                            &source, &size_src,&err);
    HandleError(err, "creating program with source data");
    cl_device_id dev_id = context_->GetDevId();
    err = clBuildProgram(cl_program_,1,&dev_id,compilation_options.c_str(),nullptr,nullptr);
    HandleError(err, "building program.");
    return true;
}

bool CLProgram::CreateKernels(){
    cl_uint num_kernels = 0;
    cl_int err;
    err = clCreateKernelsInProgram(cl_program_, 0, nullptr, &num_kernels);
    if(num_kernels == 0)
        err = CL_INVALID_BINARY;
    if(err != CL_SUCCESS){
        std::string build_log;
        size_t log_size = 0;
        clGetProgramBuildInfo(cl_program_, context_->GetDevId(),
                              CL_PROGRAM_BUILD_LOG,0,nullptr,&log_size);
        build_log.resize(log_size);
        clGetProgramBuildInfo(cl_program_, context_->GetDevId(),
                              CL_PROGRAM_BUILD_LOG,log_size,&build_log[0],nullptr);
        printf("%s \n",build_log.c_str());
        HandleError(err,"creating kernels");

    }
    return true;
}

CLKernel* CLProgram::GetKernel(const std::string &kernel_name){
    CLKernel* kernel = nullptr;
    auto iter = kernels_.find(kernel_name);
    if(iter != kernels_.end()){
        kernel = iter->second;
    }else{
        kernel = new CLKernel(context_, cl_program_, kernel_name);
        kernels_.insert(std::make_pair(kernel_name,kernel));
    }
    if(kernel == nullptr){
        printf("kernel has been deleted or failed to create!\n");
        exit(EXIT_FAILURE);
    }
    return kernel;
}

/**
 * definitions for members of CLKernel
 */
CLKernel::CLKernel(const CLContext* context, cl_program program,
                   const std::string& kernel_name):context_(context){
    cl_int err = CL_SUCCESS;
    kernel_ = clCreateKernel(program, kernel_name.c_str(),&err);
    HandleError(err, "creating kernel: " + kernel_name);
}

void CLKernel::SetArgs(int num_args, void **argument, size_t *argument_sizes){
    cl_int err = CL_SUCCESS;
    for(int i=0; i<num_args; i++){
        err = clSetKernelArg(kernel_, cl_uint(i), argument_sizes[i], argument[i]);
        HandleError(err, "in setting kernel arguments");
    }
}

void CLKernel::Launch(int queue_id, GridDim gd, BlockDim bd){
    size_t global_w_offset[3] = {0, 0, 0};
    size_t global_w_size[3] = {
                      size_t(gd.x * bd.x),
                      size_t(gd.y * bd.y),
                      size_t(gd.z * bd.z)};
    size_t local_w_size[3]= {size_t(bd.x),size_t(bd.y),size_t(bd.z)};
    cl_int err = clEnqueueNDRangeKernel(context_->GetCommandQueue(queue_id),
                                    kernel_,3,global_w_offset,global_w_size,
                                          local_w_size,0, nullptr, nullptr);
    HandleError(err, "enqueuing kernel");
}

CLKernel::~CLKernel(){
    cl_int err = clReleaseKernel(kernel_);
    HandleError(err, "releasing kernel objects");
}

/**
 * definitions for members of CLBuffer
 */

CLBuffer::CLBuffer(const CLContext * ctx, size_t size, MemFlag flag, void * host_ptr):
                                                context_(ctx),size_(size),flag_(flag){
    cl_int err;
    buffer_ = clCreateBuffer(context_->GetCLContext(), GetCLMemFlag(flag),
                                                    size_, nullptr, &err);
    HandleError(err, "creating buffer");
    if(host_ptr)
        Write(host_ptr, SYNC_MODE_BLOCKING, 0);

}

CLBuffer::~CLBuffer(){
    cl_int err = clReleaseMemObject(buffer_) ;
    HandleError(err, "in releasing buffer");
}

ArgumentPropereties CLBuffer::GetArgumentPropereties() const{
    return ArgumentPropereties(&buffer_, sizeof(buffer_));
}

void CLBuffer::Write(const void * data, SyncMode block_queue,int command_queue){
    Write(data, 0, size_, block_queue, command_queue);
}

void CLBuffer::Write(const void * data, size_t offset, size_t size,
                           SyncMode block_queue,int command_queue){
    cl_bool b_Block = block_queue == SYNC_MODE_BLOCKING ? CL_TRUE : CL_FALSE;
    cl_int err = clEnqueueWriteBuffer(context_->GetCommandQueue(command_queue),
                                       buffer_, b_Block, offset, size, data, 0,
                                                             nullptr, nullptr);
    HandleError(err, "enqueuing writing buffer");
}

void CLBuffer::Read(void *data, SyncMode block_queue, int command_queue) const{
    Read(data, 0, size_, block_queue, command_queue);
}

void CLBuffer::Read(void *data, size_t offset, size_t size, SyncMode block_queue,
                                                       int command_queue) const {
    cl_bool b_Block = block_queue == SYNC_MODE_BLOCKING ? CL_TRUE : CL_FALSE;
    cl_int err = clEnqueueReadBuffer(context_->GetCommandQueue(command_queue),
                                      buffer_, b_Block, offset, size, data, 0,
                                                            nullptr, nullptr);
    HandleError(err, "enqueuing reading buffer");
}

/**
 * definitions for members of StereoSGMCL
 */
StereoSGMCL::StereoSGMCL(int width, int height, int disp_size, const CLContext* ctx):
           width_(width), height_(height), disp_size_(disp_size), context_(nullptr){
    Init(ctx);
}

StereoSGMCL::~StereoSGMCL(){
    delete sgm_prog_;

    delete d_src_left;
    delete d_src_right;
    delete d_left;
    delete d_right;
    delete d_matching_cost;
    delete d_scost;
    delete d_left_disparity;
    delete d_right_disparity;
    delete d_tmp_left_disp;
    delete d_tmp_right_disp;
}

bool StereoSGMCL::Init(const CLContext *ctx) {
    if(!ctx)
        return false;
    context_ = ctx;
    //initialize kernels
    sgm_prog_ = new CLProgram(SGM_SRC_PATH, context_);
    m_census_kernel = sgm_prog_->GetKernel("census_kernel");
    m_matching_cost_kernel_128 = sgm_prog_->GetKernel("matching_cost_kernel_128");
    m_compute_stereo_horizontal_dir_kernel_0 = sgm_prog_->GetKernel("compute_stereo_horizontal_dir_kernel_0");
    m_compute_stereo_horizontal_dir_kernel_4 = sgm_prog_->GetKernel("compute_stereo_horizontal_dir_kernel_4");
    m_compute_stereo_vertical_dir_kernel_2 = sgm_prog_->GetKernel("compute_stereo_vertical_dir_kernel_2");
    m_compute_stereo_vertical_dir_kernel_6 = sgm_prog_->GetKernel("compute_stereo_vertical_dir_kernel_6");
    m_compute_stereo_oblique_dir_kernel_1 = sgm_prog_->GetKernel("compute_stereo_oblique_dir_kernel_1");
    m_compute_stereo_oblique_dir_kernel_3 = sgm_prog_->GetKernel("compute_stereo_oblique_dir_kernel_3");
    m_compute_stereo_oblique_dir_kernel_5 = sgm_prog_->GetKernel("compute_stereo_oblique_dir_kernel_5");
    m_compute_stereo_oblique_dir_kernel_7 = sgm_prog_->GetKernel("compute_stereo_oblique_dir_kernel_7");
    m_winner_takes_all_kernel128 = sgm_prog_->GetKernel("winner_takes_all_kernel128");
    m_check_consistency_left = sgm_prog_->GetKernel("check_consistency_kernel_left");
    m_median_3x3 = sgm_prog_->GetKernel("median3x3");
    m_copy_u8_to_u16 = sgm_prog_->GetKernel("copy_u8_to_u16");
    m_clear_buffer = sgm_prog_->GetKernel("clear_buffer");

    //create buffers
    d_src_left = new CLBuffer(context_,width_ * height_);
    d_src_right = new CLBuffer(context_,width_ * height_);
    d_left = new CLBuffer(context_,sizeof(uint64_t) * width_ * height_);
    d_right = new CLBuffer(context_,sizeof(uint64_t) * width_ * height_);
    d_matching_cost = new CLBuffer(context_,width_ * height_ * disp_size_);
    d_scost = new CLBuffer(context_,sizeof(uint16_t) * width_ * height_ * disp_size_);
    d_left_disparity = new CLBuffer(context_,sizeof(uint16_t) * width_ * height_);
    d_right_disparity = new CLBuffer(context_,sizeof(uint16_t) * width_ * height_);
    d_tmp_left_disp = new CLBuffer(context_,sizeof(uint16_t) * width_ * height_);
    d_tmp_right_disp = new CLBuffer(context_,sizeof(uint16_t) * width_ * height_);

    //setup kernels
    m_census_kernel->SetArgs(d_src_left, d_left, width_, height_);
    m_matching_cost_kernel_128->SetArgs(d_left, d_right, d_matching_cost, width_, height_);
    m_compute_stereo_horizontal_dir_kernel_0->SetArgs(d_matching_cost, d_scost, width_, height_);
    m_compute_stereo_horizontal_dir_kernel_4->SetArgs(d_matching_cost, d_scost, width_, height_);
    m_compute_stereo_vertical_dir_kernel_2->SetArgs(d_matching_cost, d_scost, width_, height_);
    m_compute_stereo_vertical_dir_kernel_6->SetArgs(d_matching_cost, d_scost, width_, height_);
    m_compute_stereo_oblique_dir_kernel_1->SetArgs(d_matching_cost, d_scost, width_, height_);
    m_compute_stereo_oblique_dir_kernel_3->SetArgs(d_matching_cost, d_scost, width_, height_);
    m_compute_stereo_oblique_dir_kernel_5->SetArgs(d_matching_cost, d_scost, width_, height_);
    m_compute_stereo_oblique_dir_kernel_7->SetArgs(d_matching_cost, d_scost, width_, height_);
    m_winner_takes_all_kernel128->SetArgs(d_left_disparity, d_right_disparity, d_scost, width_, height_);
    m_check_consistency_left->SetArgs(d_tmp_left_disp, d_tmp_right_disp, d_src_left, width_, height_);
    m_median_3x3->SetArgs(d_left_disparity, d_tmp_left_disp, width_, height_);
    m_copy_u8_to_u16->SetArgs(d_matching_cost, d_scost);

    return true;
}

void StereoSGMCL::Run(void *left_img, void *right_img, void *output){
    d_src_left->Write(left_img);
    d_src_right->Write(right_img);
    census();
    mem_init();
    matching_cost();
    scan_cost();
    winner_takes_all();
    median();
    context_->Finish(0);
    d_tmp_left_disp->Read(output);
}

void StereoSGMCL::census(){
    m_census_kernel->SetArgs(d_src_left, d_left);
    m_census_kernel->Launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                   BlockDim(16,16));
    context_->Finish(0);
    m_census_kernel->SetArgs(d_src_right,d_right);
    m_census_kernel->Launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                   BlockDim(16,16));
    context_->Finish(0);
}

void StereoSGMCL::mem_init(){
    m_clear_buffer->SetArgs(d_left_disparity);
    m_clear_buffer->Launch(0, GridDim(width_ * height_ * sizeof(uint16_t)/ 32/ 256),
                                                                     BlockDim(256));
    m_clear_buffer->SetArgs(d_right_disparity);
    m_clear_buffer->Launch(0, GridDim(width_ * height_ * sizeof(uint16_t)/ 32/ 256),
                                                                     BlockDim(256));
    m_clear_buffer->SetArgs(d_scost);
    m_clear_buffer->Launch(0, GridDim(width_ * height_ * sizeof(uint16_t) * disp_size_
                                                          / 32/ 256),BlockDim(256));
}

void StereoSGMCL::matching_cost(){
    m_matching_cost_kernel_128->Launch(0, GridDim(height_/2), BlockDim(128,2));
}

void StereoSGMCL::scan_cost(){
    static const int PATHS_IN_BLOCK = 8;
    const int obl_num_paths = width_ + height_ ;

    m_compute_stereo_horizontal_dir_kernel_0->Launch(0,
    GridDim(height_ / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
    m_compute_stereo_horizontal_dir_kernel_4->Launch(0,
    GridDim(height_ / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
    m_compute_stereo_vertical_dir_kernel_2->Launch(0,
    GridDim(width_ / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
    m_compute_stereo_vertical_dir_kernel_6->Launch(0,
    GridDim(width_ / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));

    m_compute_stereo_oblique_dir_kernel_1->Launch(0,
    GridDim(obl_num_paths / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
    m_compute_stereo_oblique_dir_kernel_3->Launch(0,
    GridDim(obl_num_paths / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
    m_compute_stereo_oblique_dir_kernel_5->Launch(0,
    GridDim(obl_num_paths / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
    m_compute_stereo_oblique_dir_kernel_7->Launch(0,
    GridDim(obl_num_paths / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
}

void StereoSGMCL::winner_takes_all(){
    const int WTA_PIXEL_IN_BLOCK = 8;
    m_winner_takes_all_kernel128->Launch(0,
    GridDim(width_ / WTA_PIXEL_IN_BLOCK,1 * height_),
    BlockDim(32, WTA_PIXEL_IN_BLOCK));
}

void StereoSGMCL::median(){
    m_median_3x3->SetArgs(d_left_disparity, d_tmp_left_disp);
    m_median_3x3->Launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                BlockDim(16,16));
    m_median_3x3->SetArgs(d_right_disparity, d_tmp_right_disp);
    m_median_3x3->Launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                BlockDim(16,16));

}

void StereoSGMCL::check_consistency_left(){
    m_check_consistency_left->Launch(0,GridDim((width_ + 16 - 1)/16,
                                              (height_ + 16 - 1)/16),BlockDim(16,16));
}
}
