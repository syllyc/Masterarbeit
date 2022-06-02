# Project: Learning from Demonstration 

## Folder1: Openpose_ZED
Detect human body parts and record the motion trajectory of the right wrist
             (Get training trajectory for model training)

## Folder2: DMP_Python 
1. Preprocess the training trajectory.  
2. Automatically get the keypoints of the training trajectory  
3. Use DMP to obtain the model of the demonstrated motion  
4. According to the set start and end position to generate motion trajectories


## Folder3: learn_from_demo 
1. Inverse and Forwards Kinematic of UR10  
2. UR10 execute the generated motion trajectory in ROS Gazebo

