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
#include <cuda_runtime.h>

int frame_width = 0;
int frame_number = 0;

using namespace cv;
using namespace cv::cuda;



GpuMat createMat(Size size, int type)
{
    Size size0 = size;

    GpuMat d_m(size0, type);

    if (size0 != size)
        d_m = d_m(Rect((size0.width - size.width) / 2, (size0.height - size.height) / 2, size.width, size.height));

    return d_m;
}

GpuMat loadMat(const Mat& m, cv::cuda::Stream stream )
{
    GpuMat d_m = createMat(m.size(), m.type());
    d_m.upload(m, stream);
    return d_m;
}


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

//    cv::Ptr<cv::cuda::ORB> gpu_detector = cv::cuda::ORB::create(500, 1.2f, 8, 31, 0, 2,
//                                                                   0, 31, 20, true);
    // setup GPU-accelerated ORB detector
    cv::Ptr<cv::cuda::ORB> gpu_detector = cv::cuda::ORB::create(500, 2.0f, OCTAVES, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 50, true);


    const char* gst =  "rtspsrc location=rtsp://admin:@cam1/ch0_0.264 name=r latency=0 protocols=tcp ! application/x-rtp,payload=96,encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM), format=BGRx ! nvvidconv ! videoconvert ! video/x-raw, format=BGR ! appsink";
    cv::VideoCapture cap(gst, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
		std::cout<<"Failed to open camera."<<std::endl;
		return (-1);
    }

    unsigned int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    unsigned int fps    = cap.get(cv::CAP_PROP_FPS);
    unsigned int pixels = width*height;



    void *frame_ptr, *out_pointer, *gpu_desc_pointer;
    unsigned int frameByteSize = pixels * 3;





    cv::Mat cpu_desc;;


    // open file on GPU
    //cv::Ptr<cv::cudacodec::VideoReader> gpu_reader = cv::cudacodec::createVideoReader(&cap);

    //introduce motion object
    Motion motion;

    //cv::namedWindow("MyCameraPreview", cv::WINDOW_AUTOSIZE);
    cv::cuda::Stream m_stream;

    // main loop
    for (;;) {
        if (!cap.read(cpu_desc))
            break;
        if (frame_width == 0) {
            frame_width = cpu_desc.cols;
        }

        //for each frame until end of input:
        // - convert to gray scale

        std::cout << "Before Copy Call" << std::endl;
        //cpu_desc.copyTo(cpu_frame);
        std::cout << "After Copy Call" << std::endl;

        cv::cuda::GpuMat gpu_frame;
        cv::Mat cpu_frame;

        cv::cuda::GpuMat gpu_out;
        cv::Mat cpu_out, cpu_grey;
        cv::cuda::GpuMat gpu_desc;

        cv::cuda::GpuMat gpu_gray;
        cv::cuda::GpuMat gpu_gray2;

        gpu_frame = loadMat(cpu_desc, m_stream);
        cv::cuda::cvtColor(gpu_frame, gpu_gray, cv::COLOR_BGR2GRAY, 0, m_stream);
        // - detect feature points and compute their descriptors
        std::cout << "Before detect" << std::endl;
        //gpu_frame.upload(cpu_frame);
        gpu_gray.download(cpu_frame);
        std::cout << "after upload" << std::endl;
        //cv::imshow("MyCameraPreview",cpu_frame);
		if((char)cv::waitKey(1) == (char)27)
			break;

	    cudaDeviceSynchronize();
	    std::cout << "Before detectasync" << std::endl;
	    std::cout << gpu_gray.cols << std::endl;

	    cv::cuda::GpuMat tmp;

	    //gpu_gray.download(cpu_out, m_stream);

	    cv::cuda::GpuMat out;
	    std::vector<cv::KeyPoint> points;
        gpu_detector->detectAndComputeAsync(gpu_gray, tmp, out, gpu_out, false, m_stream);

        gpu_gray.download(cpu_grey, m_stream);

        // - download descriptors to CPU RAM


        m_stream.waitForCompletion();

        gpu_detector->convert(out, points);


        if(gpu_out.rows > 0){
            gpu_out.download(cpu_out);
        	std::cout << "received data=======================" << std::endl;
        	motion.add_frame(points, cpu_out);
        	//cv::imshow("MyCameraPreview",cpu_out);
        	motion.show(cpu_grey);
        }
        //gpu_out.copyTo(cpu_out);
        std::cout << "After detect" << std::endl;
        //motion.add_frame(points, cpu_out);

        frame_number++;
    }

    cap.release();
    return 0;
}
