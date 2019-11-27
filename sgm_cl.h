#ifndef SGM_CL_H
#define SGM_CL_H

#include "sgm_source_path.h"

#include <CL/cl.h>
#include <string.h>

#include <iostream>
#include <cstdio>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>

namespace sgm_cl{

struct BlockDim {
    BlockDim(int _x = 1, int _y = 1, int _z = 1):
                           x(_x), y(_y), z(_z) {}
    int x, y, z;
};

struct GridDim {
    GridDim(int _x = 1, int _y = 1, int _z = 1):
                          x(_x), y(_y), z(_z) {}
    int x, y, z;
};

struct ArgumentPropereties
{
    ArgumentPropereties(void* ptr = nullptr, size_t argsize = 0) :
                               arg_ptr(ptr), sizeof_arg(argsize) {}
    void* arg_ptr;
    size_t sizeof_arg;
};

enum MemFlag
{
    MEM_FLAG_READ_WRITE = 1 << 0,
    MEM_FLAG_WRITE_ONLY = 1 << 1,
    MEM_FLAG_READ_ONLY = 1 << 2,
    MEM_FLAG_USE_HOST_PTR = 1 << 3, // maybe needs to be removed, in cuda not trivial
    MEM_FLAG_ALLOC_HOST_PTR = 1 << 4,
    MEM_FLAG_COPY_HOST_PTR = 1 << 5
};

enum SyncMode
{
    SYNC_MODE_ASYNC = 0,
    SYNC_MODE_BLOCKING = 1
};

class CLContext {
public:
    CLContext(int platform_id = 0, int device_id = 0, int num_streams = 1);
    ~CLContext();
    CLContext(const CLContext& context) = delete;
    CLContext& operator=(const CLContext& context) = delete;
    CLContext(CLContext&& context) noexcept;
    CLContext& operator=(CLContext&& context) noexcept;

    cl_context GetCLContext() const {return cl_context_;}
    cl_device_id GetDevId() const {return cl_device_id_;}
    cl_command_queue GetCommandQueue(int id) const;
    void Finish(int command_queue) const;
    inline const std::string& CLInfo() const {return cl_info_;}

private:
    std::string GetPlatformInfo(cl_platform_id platform_id, int info_name) const;
    std::string GetDevInfo(cl_device_id dev_id, int info_name) const;

    std::vector<cl_command_queue> cl_command_queues_;
    std::string cl_info_;
    cl_context cl_context_;
    cl_device_id cl_device_id_;
};

class CLBuffer {
public:
    CLBuffer(const CLContext* ctx, size_t size, MemFlag flag = MEM_FLAG_READ_WRITE,
             void* host_ptr = nullptr);
    ~CLBuffer();
    void Write(const void* data, SyncMode block_queue = SYNC_MODE_BLOCKING,
               int command_queue = 0);
    void Read(void* data, SyncMode block_queue = SYNC_MODE_BLOCKING,
              int command_queue = 0) const;
    ArgumentPropereties GetArgumentPropereties() const;

private:
    void Write(const void* data, size_t offset, size_t size,
               SyncMode block_queue,int command_queue);
    void Read(void* data, size_t offset, size_t size, SyncMode block_queue,
              int command_queue) const;
    mutable cl_mem buffer_;
    MemFlag flag_;
    size_t size_;
    const CLContext* context_;
};

class CLKernel {
public:
    CLKernel(const CLContext* context, cl_program program, const std::string& kernel_name);
    ~CLKernel();

   void Launch(int queue_id, GridDim gd, BlockDim bd);

   template <typename... Types> void SetArgs(Types&&... args);

private:
   void SetArgs(int num_args, void** argument, size_t* argument_sizes);
   inline void FillArgVector(int arg_idx, void** arg_address, size_t* arg_sizeof);
   template <typename T, typename... Types>
   void FillArgVector(int arg_idx, void** arg_address, size_t* arg_sizeof,
                      T&& arg, Types&&... Fargs);
   template <typename... Types>
   void FillArgVector(int arg_idx, void** arg_address, size_t* arg_sizeof,
                      CLBuffer* arg, Types&&... Fargs);

    cl_kernel kernel_;
    const CLContext* context_;
};

class CLProgram{
public:
    CLProgram(const std::string& source_path="", const CLContext* context=nullptr
                          , const std::string& compilation_options="-I \"./\"");
    ~CLProgram();
    bool CreateProgram(const char* source, size_t size_src,
                       const std::string& compilation_options);
    bool CreateKernels();
    CLKernel* GetKernel(const std::string& kernel_name);
    inline void SetCLContext(const CLContext& context) {context_ = &context;}

private:
    const CLContext* context_;
    cl_program cl_program_;
    std::map<std::string, CLKernel*> kernels_;
};

class StereoSGMCL{
public:
    StereoSGMCL(int width, int height, int disp_size, const CLContext* ctx = nullptr);
    bool Init(const CLContext* ctx);
    void Run(void* left_img, void* right_img, void* output);
    ~StereoSGMCL();

private:
    void initCL();
    void census();
    void mem_init();
    void matching_cost();
    void scan_cost();
    void winner_takes_all();
    void median();
    void check_consistency_left();
private:
    int width_, height_, disp_size_;
    const CLContext* context_;
    CLProgram* sgm_prog_;

//    CLKernel* census_kernel_;
//    CLKernel* matching_cost_kernel_;

//    CLKernel* aggre_cost_top2down_kernel_;
//    CLKernel* aggre_cost_down2top_kernel_;
//    CLKernel* aggre_cost_left2right_kernel_;
//    CLKernel* aggre_cost_right2left_kernel_;
//    CLKernel* aggre_cost_topleft2downright_kernel_;
//    CLKernel* aggre_cost_topright2downleft_kernel_;
//    CLKernel* aggre_cost_downleft2topright_kernel_;
//    CLKernel* aggre_cost_downright2topleft_kernel_;
    CLKernel * m_census_kernel;
    CLKernel * m_matching_cost_kernel_128;

    CLKernel * m_compute_stereo_horizontal_dir_kernel_0;
    CLKernel * m_compute_stereo_horizontal_dir_kernel_4;
    CLKernel * m_compute_stereo_vertical_dir_kernel_2;
    CLKernel * m_compute_stereo_vertical_dir_kernel_6;

    CLKernel * m_compute_stereo_oblique_dir_kernel_1;
    CLKernel * m_compute_stereo_oblique_dir_kernel_3;
    CLKernel * m_compute_stereo_oblique_dir_kernel_5;
    CLKernel * m_compute_stereo_oblique_dir_kernel_7;


    CLKernel * m_winner_takes_all_kernel128;

    CLKernel * m_check_consistency_left;

    CLKernel * m_median_3x3;

    CLKernel * m_copy_u8_to_u16;
    CLKernel * m_clear_buffer;


    CLBuffer * d_src_left,* d_src_right,* d_left, *d_right, *d_matching_cost,
        *d_scost,* d_left_disparity,* d_right_disparity,
        *d_tmp_left_disp, *d_tmp_right_disp;

};

#include "sgm_cl.inl"
}

#endif
