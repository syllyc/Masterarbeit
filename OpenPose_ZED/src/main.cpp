/***********************************************************************************************
 ** Use ZED Camera with OpenPose to get the keypoints of wrist.					  	      **
 ** Depth and images are captured with the ZED SDK, converted to OpenCV format and displayed. **
 ***********************************************************************************************/

 // ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Sample includes
#include <SaveDepth.hpp>

// OpenPose includes
#include "openpose.h"

// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <iostream>

// Custom OpenPose flags
// Display
DEFINE_bool(no_display, false,
	"Enable to disable the visual display.");

using namespace sl;

cv::Mat slMat2cvMat(Mat& input);
void printHelp();
void save_3DPostion(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::string fname);

//display images with keypoints
cv::Mat display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
	try
	{
		// User's displaying/saving/other processing here
		// datum.cvOutputData: rendered frame with pose or heatmaps
		// datum.poseKeypoints: Array<float> with the estimated pose
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			// Display image
			const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
			cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
			//cv::waitKey(1);
			return cvMat;
		}
		else
			op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

//print keypoints
void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
	try
	{
		// Example: How to use the pose keypoints
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			op::opLog("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
			//op::opLog("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
			//op::opLog("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString(), op::Priority::High);
			//op::opLog("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString(), op::Priority::High);
		}
		else
			op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

//configure and initial Openpose
const auto configureWrapper(op::Wrapper& opWrapper)
{
	try
	{
		// Configuring OpenPose

		// logging_level
		op::checkBool(
			0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
			__LINE__, __FUNCTION__, __FILE__);
		op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
		op::Profiler::setDefaultX(FLAGS_profile_speed);

		// Applying user defined configuration - GFlags to program variables
		// outputSize
		const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
		// netInputSize
		const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");  //256x176
		// faceNetInputSize
		const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
		// handNetInputSize
		const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
		// poseMode
		const auto poseMode = op::flagsToPoseMode(FLAGS_body);
		// poseModel
		const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose) = "COCO");//BODY_25; COCO 
																							  // JSON saving
		if (!FLAGS_write_keypoint.empty())
			op::opLog(
				"Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
				" instead.", op::Priority::Max);
		// keypointScaleMode
		const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
		// heatmaps to add
		const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
			FLAGS_heatmaps_add_PAFs);
		const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
		// >1 camera view?
		const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
		// Face and hand detectors
		const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
		const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
		// Enabling Google Logging
		const bool enableGoogleLogging = true;

		//std::string model_path = "D:/dahmenjo/Yangle/workspace/openpose/models";

		// Pose configuration (use WrapperStructPose{} for default and recommended configuration)
		const op::WrapperStructPose wrapperStructPose{
			poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
			FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
			poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
			FLAGS_part_to_show, op::String(FLAGS_model_folder) = "D:/Code/OpenPose/openpose/models", heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, //op::String(FLAGS_model_folder)
			(float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
			op::String(FLAGS_prototxt_path), op::String(FLAGS_caffemodel_path),
			(float)FLAGS_upsampling_ratio, enableGoogleLogging };
		opWrapper.configure(wrapperStructPose);
		// Face configuration (use op::WrapperStructFace{} to disable it)
		const op::WrapperStructFace wrapperStructFace{
			FLAGS_face, faceDetector, faceNetInputSize,
			op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
			(float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold };
		opWrapper.configure(wrapperStructFace);
		// Hand configuration (use op::WrapperStructHand{} to disable it)
		const op::WrapperStructHand wrapperStructHand{
			FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
			op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
			(float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold };
		opWrapper.configure(wrapperStructHand);
		// Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
		const op::WrapperStructExtra wrapperStructExtra{
			FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads };
		opWrapper.configure(wrapperStructExtra);
		// Output (comment or use default argument to disable any output)
		const op::WrapperStructOutput wrapperStructOutput{
			FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
			op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
			FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
			op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
			op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
			op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
			op::String(FLAGS_udp_port) };
		opWrapper.configure(wrapperStructOutput);
		// No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
		// Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
		if (FLAGS_disable_multi_thread)
			opWrapper.disableMultiThreading();
		return poseModel;
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

/**
* main loop
**/
int main(int argc, char **argv) {
	// Parameters for saving 3d position
	bool flag_save_postion = false;
	int num_frame_saved = 0;

	// Initializing OpenPose
	FLAGS_model_pose = "COCO";  //COCO MPI
	FLAGS_net_resolution = "320x176";

	op::opLog("Starting OpenPose demo...", op::Priority::High);
	const auto opTimer = op::getTimerInit();

	// Configuring OpenPose
	op::opLog("Configuring OpenPose...", op::Priority::High);
	op::Wrapper opWrapper{ op::ThreadManagerMode::Asynchronous };
	configureWrapper(opWrapper);

	// Starting OpenPose
	op::opLog("Starting thread(s)...", op::Priority::High);
	opWrapper.start();

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters for ZED Camera
	// InitParameters: used to initialize camera before opening the camera
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD720;  //resolution: 1280x720
	init_params.camera_fps = 30;
    init_params.depth_mode = DEPTH_MODE::ULTRA;
    init_params.coordinate_units = UNIT::METER;
	init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;

    if (argc > 1) init_params.input.setFromSVOFile(argv[1]);
        
    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        printf("%s\n", toString(err).c_str());
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Display help in console
    printHelp();

    // Set runtime parameters after opening the camera; 
    // RuntimeParameters:used to change specific parameters during use
    sl::RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE::FILL;
	runtime_parameters.measure3D_reference_frame = sl::REFERENCE_FRAME::CAMERA;

    // Prepare new image size to retrieve half-resolution images
    Resolution image_size = zed.getCameraInformation().camera_resolution;
	std::cout << "camera resolution is: " << image_size.width << "x" << image_size.height << std::endl;
    //int new_width = image_size.width / 2;
    //int new_height = image_size.height / 2;
	int new_width = image_size.width;
	int new_height = image_size.height;

    Resolution new_image_size(new_width, new_height);

    // To share data between sl::Mat and cv::Mat, use slMat2cvMat()
    // Only the headers and pointer to the sl::Mat are copied, not the data itself
    sl::Mat image_zed(new_width, new_height, MAT_TYPE::U8_C4);
    cv::Mat image_ocv = slMat2cvMat(image_zed);
    sl::Mat depth_image_zed(new_width, new_height, MAT_TYPE::U8_C4);
    cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
    sl::Mat point_cloud;
	cv::Mat image_keypoint;

	std::vector<float> x_;     //position of bodyparts in x-axis
	std::vector<float> y_;     //position of bodyparts in y-axis
	std::vector<float> z_;     //position of bodyparts in z-axis
	//float x, y, z;  
	float pixel_x, pixel_y;  // pixel(x,y)

	//std::map<int, sl::float4> keypoints_position;  //3D postion and score of each keypoints
	sl::float4 keypoints_position;  //3D postion and score of each keypoints
	
    // Loop until 'q' is pressed
    char key = ' ';
    while (key != 'q') {

        if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {

            // Retrieve the left image, depth image in half-resolution
            zed.retrieveImage(image_zed, VIEW::LEFT, MEM::CPU, new_image_size);
            zed.retrieveImage(depth_image_zed, VIEW::DEPTH, MEM::CPU, new_image_size);
			
            // Retrieve the RGBA point cloud in half-resolution
            // To learn how to manipulate and display point clouds, see Depth Sensing sample
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::CPU, new_image_size);
			
			//convert 4 channel BGRA to BGR, because the input of openpose should be 3 channel Mat
			cv::cvtColor(image_ocv, image_keypoint, cv::COLOR_BGRA2BGR);
			
			// Process and display image: get and show keypoints
			const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(image_keypoint);
			auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
			if (datumProcessed != nullptr)
			{
				printKeypoints(datumProcessed);
				if (!FLAGS_no_display)
					display(datumProcessed);
			}
			else
				op::opLog("Image could not be processed.", op::Priority::High);

			// Get Keypoints of Wrists: ID of RWrist: 4, ID of LWrist: 7
			//https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/keypoints_pose_18.png
			int part = 4;                 // Index of the bodyparts: 4 represents the body part "RWrist" 

			const auto numberPeopleDetected = datumProcessed->at(0)->poseKeypoints.getSize(0);  //The number of detected person
			const auto numberBodyParts = datumProcessed->at(0)->poseKeypoints.getSize(1);       //The number of dectected bodyparts 

			for (int person = 0; person < numberPeopleDetected; person++)
			{

				// get the specific pixel (keypoint of Wrist)
				pixel_x = round(datumProcessed->at(0)->poseKeypoints[(numberBodyParts * person + part) * 3]);  
				pixel_y = round(datumProcessed->at(0)->poseKeypoints[(numberBodyParts * person + part) * 3 + 1]);
				std::cout << "pixel_x of keypoints wrist " << part << " is: " << pixel_x << std::endl;
				std::cout << "pixel_y of keypoints wrist " << part << " is: " << pixel_y << std::endl;

				// get the 3D postion for the specific pixel (keypoint of Wrist)
				point_cloud.getValue(pixel_x, pixel_y, &keypoints_position, sl::MEM::CPU);
				//std::cout << "posotion x of keypoints wrist " << part << " is: " << keypoints_position.x << std::endl;
				//std::cout << "posotion y of keypoints wrist " << part << " is: " << keypoints_position.y << std::endl;
				//std::cout << "posotion z of keypoints wrist " << part << " is: " << keypoints_position.z << std::endl;

			}


            // Display image and depth using cv:Mat which share sl:Mat data
            //cv::imshow("Image", image_ocv);
            //cv::imshow("Depth", depth_image_ocv);

			// Display the current(actual) FPS
			std::cout << "current FPS: " << zed.getCurrentFPS() << std::endl;

			if (key == 's') // if press 's' start save 3d position of Wrist 
			{
				flag_save_postion = true;
				std::cout << "start record data......" << std::endl;
			}
			if (flag_save_postion)
			{
				x_.push_back(keypoints_position.x);
                y_.push_back(keypoints_position.y);
				z_.push_back(keypoints_position.z);
                std::cout << "the x of keypoints wrist " << part << " is: " << x_.back() << std::endl;
                std::cout << "the y of keypoints wrist " << part << " is: " << y_.back() << std::endl;
				std::cout << "the z of keypoints wrist " << part << " is: " << z_.back() << std::endl;
				num_frame_saved++;
			}

			// Handle key event
			key = cv::waitKey(10);
            //processKeyEvent(zed, key);  // default save function of zed camera
        }
    }

	// save 3d position data into a txt dokument
	std::string fname = "../data/Wrist_position3d4.txt";
	save_3DPostion(x_, y_, z_, fname);

	// close camera
    zed.close();

	// Output Informations
	std::cout << "total number of saved frames is " << num_frame_saved << std::endl;
    return 0;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(sl::Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM::CPU));
}

/**
* This function displays help in console
**/
void printHelp() {
    std::cout << " Press 's' to save Side by side images" << std::endl;
    std::cout << " Press 'p' to save Point Cloud" << std::endl;
    std::cout << " Press 'd' to save Depth image" << std::endl;
    std::cout << " Press 'm' to switch Point Cloud format" << std::endl;
    std::cout << " Press 'n' to switch Depth format" << std::endl;
}

/**
* This function displays help in console
**/
void save_3DPostion(std::vector<float> x, std::vector<float> y, std::vector<float> z, std::string fname)
{
	std::cout << "saving 3D Position... "<< std::endl;
	FILE *f = fopen(fname.c_str(), "w");
	fprintf(f, "x\ty\tz\n");  // titel
	for (int i = 0; i < x.size(); i++)
	{
		//fprintf(f, "%.6f ", "%.6f ", "%.6f", x[i], y[i], z[i]);
		fprintf(f, "%.6f  ", x[i]);
		fprintf(f, "%.6f  ", y[i]);
		fprintf(f, "%.6f\n", z[i]);
	}

	fclose(f);
	std::cout << "save 3D Position under: " << fname << std::endl;
}
