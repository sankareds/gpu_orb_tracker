#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "settings.h"
#include "Motion.h"
#include <opencv2/cudafeatures2d.hpp>
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

int frame_width = 0;
int frame_number = 0;

using namespace cv;
using namespace cv::cuda;


int main(int argc, char *argv[]) {

    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <filename> <number of points of interest>" << std::endl;
        return 1;
    }



    cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());

    // gather filename
    const std::string filename(argv[1]);


    // declare feature point and description storage
    std::vector<cv::KeyPoint> points;
    cv::Mat cpu_desc, cpu_gray;

    // setup GPU-accelerated ORB detector
    cv::Ptr<cv::ORB> gpu_detector = cv::ORB::create(std::stoi(argv[2]), 2.0f, OCTAVES, 31, 0, 2,
                                                                cv::ORB::HARRIS_SCORE, 31, 50);


    const char* gst =  "rtspsrc location=rtsp://admin:@cam1/ch0_0.264 protocols=tcp ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)RGBA ! nvvidconv ! appsink";
    cv::VideoCapture cap(gst, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
		std::cout<<"Failed to open camera."<<std::endl;
		return (-1);
    }

    // open file on GPU
    //cv::Ptr<cv::cudacodec::VideoReader> gpu_reader = cv::cudacodec::createVideoReader(&cap);

    //introduce motion object
    Motion motion;

    // main loop
    for (;;) {
        if (!cap.read(cpu_desc))
            break;
        if (frame_width == 0) {
            frame_width = cpu_desc.cols;
        }

        //for each frame until end of input:
        // - convert to gray scale


        //cv::cvtColor(cpu_desc, cpu_gray, cv::COLOR_BGRA2GRAY);
        // - detect feature points and compute their descriptors
        gpu_detector->detectAndCompute(cpu_gray, cv::noArray(), points, cpu_desc);
        // - download descriptors to CPU RAM
        //gpu_desc.download(cpu_desc);

        motion.add_frame(points, cpu_desc);

        frame_number++;
    }

    return 0;
}
