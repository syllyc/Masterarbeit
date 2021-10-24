//#pragma once
//#include <openpose/flags.hpp>
//#include <openpose/headers.hpp>
//// Command-line user interface
//#define OPENPOSE_FLAGS_DISABLE_PRODUCER
//#define OPENPOSE_FLAGS_DISABLE_DISPLAY
//
//// Custom OpenPose flags
//// Display
//DEFINE_bool(no_display, false,
//	"Enable to disable the visual display.");
//
////function declaration
//cv::Mat display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr);          //display images with keypoints
//void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr);   //print keypoints
////const auto configureWrapper(op::Wrapper& opWrapper);                                                    //initial and configure OpenPose