#include <iostream>
#include <chrono>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "sgm_cl.h"

int main(int argc, char const* const* argv)
{
    bool use_default = false;
    if(argc < 3){
        std::cout << "usage: sgm-cl-test <left_image> <right_image> [disp_size]"<<std::endl;
        std::cout << "Invalid arguments, use default input" <<std::endl;
        use_default = true;
    }

    std::string left_path = use_default? DEFAULT_LEFT_PATH : argv[1];
    std::string right_path = use_default? DEFAULT_RIGHT_PATH : argv[2];
    int disp_size = (argc == 4)? atoi(argv[3]) : 128;
    cv::Mat left = cv::imread(left_path,CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat right = cv::imread(right_path,CV_LOAD_IMAGE_GRAYSCALE);

    int width = left.cols;
    int height = left.rows;
    cv::Mat disp(height, width, CV_16U);
    sgm_cl::CLContext* context = new sgm_cl::CLContext;
    std::cout<<context->CLInfo()<<std::endl;
    sgm_cl::StereoSGMCL ssgm(width, height, disp_size,context);

    auto st = std::chrono::steady_clock::now();
    ssgm.Run(left.data,right.data,disp.data);
    auto ed = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(ed - st);
    printf("Processing Time: %lf ms\n",duration.count());
    delete context;

    cv::normalize(disp, disp,0,255,cv::NORM_MINMAX,CV_8U);
    cv::imshow("disparity",disp);
    cv::waitKey();

    return 0;
}
