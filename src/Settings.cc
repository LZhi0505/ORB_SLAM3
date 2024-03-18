/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "Settings.h"

#include "CameraModels/Pinhole.h"
#include "CameraModels/KannalaBrandt8.h"

#include "System.h"

#include <opencv2/core/persistence.hpp>
#include <opencv2/core/eigen.hpp>

#include <iostream>

using namespace std;

namespace ORB_SLAM3 {

    template<>
    /**
     *
     * @param fSettings
     * @param name 节点名字
     * @param found
     * @param required  是否为必须存在的参数
     * @return
     */
    float Settings::readParameter<float>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return 0.0f;
            }
        }
        else if(!node.isReal()){
            std::cerr << name << " parameter must be a real number, aborting..." << std::endl;
            exit(-1);
        }
        else{
            found = true;
            return node.real();
        }
    }

    template<>
    int Settings::readParameter<int>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return 0;
            }
        }
        else if(!node.isInt()){
            std::cerr << name << " parameter must be an integer number, aborting..." << std::endl;
            exit(-1);
        }
        else{
            found = true;
            return node.operator int();
        }
    }

    template<>
    string Settings::readParameter<string>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return string();
            }
        }
        else if(!node.isString()){
            std::cerr << name << " parameter must be a string, aborting..." << std::endl;
            exit(-1);
        }
        else{
            found = true;
            return node.string();
        }
    }

    // 将配置文件节点中的数据读取为Mat矩阵
    template<>
    cv::Mat Settings::readParameter<cv::Mat>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];

        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return cv::Mat();
            }
        }
        else{
            found = true;
            return node.mat();
        }
    }

    /**
     * @brief Settings的构造函数
     * @param configFile    配置文件路径
     * @param sensor    传感器类型
     */
    Settings::Settings(const std::string &configFile, const int& sensor) :
        bNeedToUndistort_(false), bNeedToRectify_(false), bNeedToResize1_(false), bNeedToResize2_(false) {
        cout << endl << "[Settings::Settings] 创建Setting" << endl;

        sensor_ = sensor;

        // Open settings file
        // 打开配置文件
        cv::FileStorage fSettings(configFile, cv::FileStorage::READ);

        if (!fSettings.isOpened()) {
            cerr << "[ERROR]: could not open configuration file at: " << configFile << endl;
            cerr << "Aborting..." << endl;

            exit(-1);
        }
        else {
            cout << endl << "Loading settings from " << configFile << " ..." << endl;
        }

        // Read first camera
        // 根据相机1的类型，获取内参，畸变参数。若为单目/单目+IMU 且 有畸变参数，才 单独做畸变校正 bNeedToUndistort_ = true
        readCamera1(fSettings);
        cout << "\t-Loaded camera 1：获取内参，畸变参数。若为单目/单目+IMU 且 有畸变参数，才 单独做畸变校正 bNeedToUndistort_ = true" << endl;

        // Read second camera if stereo (not rectified)
        // 双目，则根据相机2(右目)的类型，获取内参、畸变参数 (未矫正)。且进行双目矫正标志 bNeedToRectify_ = true，获取左目到右目的变换矩阵、基线、bfx、深度阈值系数
        if(sensor_ == System::STEREO || sensor_ == System::IMU_STEREO){
            readCamera2(fSettings);
            cout << "\t-Loaded camera 2：获取内参、畸变参数 (未矫正)。且进行双目矫正标志 bNeedToRectify_ = true，获取左目到右目的变换矩阵、基线、bfx、深度阈值系数" << endl;
        }

        // Read image info
        // 读取想要的图像尺寸并更新标定参数、帧率、颜色空间类型
        readImageInfo(fSettings);
        cout << "\t-Loaded image info" << endl;

        // 如果使用IMU，则读取IMU参数
        if(sensor_ == System::IMU_MONOCULAR || sensor_ == System::IMU_STEREO || sensor_ == System::IMU_RGBD){
            readIMU(fSettings);
            cout << "\t-Loaded IMU calibration" << endl;
        }
        // RGBD，则读取RGBD相机参数
        if(sensor_ == System::RGBD || sensor_ == System::IMU_RGBD){
            readRGBD(fSettings);
            cout << "\t-Loaded RGB-D calibration" << endl;
        }

        // 读取 ORB 参数：nFeatures，scaleFactor，nLevels，iniThFAST，minThFAST
        readORB(fSettings);
        cout << "\t-Loaded ORB settings" << endl;

        // 读取 Viewer 参数
        readViewer(fSettings);
        cout << "\t-Loaded viewer settings" << endl;

        // 读取加载和保存Atlas的参数：System.LoadAtlasFromFile，System.SaveAtlasToFile
        readLoadAndSave(fSettings);
        cout << "\t-Loaded Atlas settings" << endl;

        // 读取参数：System.thFarPoints
        readOtherParameters(fSettings);
        cout << "\t-Loaded misc parameters" << endl;

        // 默认为false；相机模型为PinHole且为双目，则置true，需进行双目校正
        if(bNeedToRectify_){
            cout << "\t计算极线矫正映射图" << endl;
            precomputeRectificationMaps();
            cout << "\t-Computed rectification maps" << endl;
        }

        cout << "----------------------------------" << endl;
    }

    // 获取相机1的内参、畸变参数
    void Settings::readCamera1(cv::FileStorage &fSettings) {
        bool found;
        // Read camera model
        // 获取相机类型
        string cameraModel = readParameter<string>(fSettings, "Camera.type", found);    // 从fSettings中读取一个名为"Camera.type"的参数，并返回一个字符串类型

        vector<float> vCalibration;
        // Camera类型为 PinHole
        if (cameraModel == "PinHole") {
            cameraType_ = PinHole;

            // Read intrinsic parameters
            // 读取内参
            float fx = readParameter<float>(fSettings,"Camera1.fx",found);
            float fy = readParameter<float>(fSettings,"Camera1.fy",found);
            float cx = readParameter<float>(fSettings,"Camera1.cx",found);
            float cy = readParameter<float>(fSettings,"Camera1.cy",found);

            vCalibration = {fx, fy, cx, cy};

            calibration1_ = new Pinhole(vCalibration);
            originalCalib1_ = new Pinhole(vCalibration);

            // Check if it is a distorted PinHole
            // 检查配置文件中是否有畸变参数
            readParameter<float>(fSettings,"Camera1.k1",found,false);   // required: 是否为必须存在的参数
            // 如果有畸变参数，则加载
            if(found){
                // 检查是否有 k3
                readParameter<float>(fSettings,"Camera1.k3",found,false);
                // 有k3则加载
                if(found){
                    vPinHoleDistorsion1_.resize(5);
                    vPinHoleDistorsion1_[4] = readParameter<float>(fSettings,"Camera1.k3",found);
                }
                else{
                    vPinHoleDistorsion1_.resize(4);
                }
                vPinHoleDistorsion1_[0] = readParameter<float>(fSettings,"Camera1.k1",found);
                vPinHoleDistorsion1_[1] = readParameter<float>(fSettings,"Camera1.k2",found);
                vPinHoleDistorsion1_[2] = readParameter<float>(fSettings,"Camera1.p1",found);
                vPinHoleDistorsion1_[3] = readParameter<float>(fSettings,"Camera1.p2",found);
            }

            // Check if we need to correct distortion from the images
            // 单目/单目+IMU，且有畸变参数，则需 对特征点进行畸变矫正
            if((sensor_ == System::MONOCULAR || sensor_ == System::IMU_MONOCULAR) && vPinHoleDistorsion1_.size() != 0){
                bNeedToUndistort_ = true;
            }
        }
        // Camera类型为 Rectified ()，则只读取内参
        else if(cameraModel == "Rectified"){
            cameraType_ = Rectified;

            //Read intrinsic parameters
            float fx = readParameter<float>(fSettings,"Camera1.fx",found);
            float fy = readParameter<float>(fSettings,"Camera1.fy",found);
            float cx = readParameter<float>(fSettings,"Camera1.cx",found);
            float cy = readParameter<float>(fSettings,"Camera1.cy",found);

            vCalibration = {fx, fy, cx, cy};

            calibration1_ = new Pinhole(vCalibration);
            originalCalib1_ = new Pinhole(vCalibration);

            //Rectified images are assumed to be ideal PinHole images (no distortion)
        }
        else if(cameraModel == "KannalaBrandt8"){
            cameraType_ = KannalaBrandt;

            //Read intrinsic parameters
            float fx = readParameter<float>(fSettings,"Camera1.fx",found);
            float fy = readParameter<float>(fSettings,"Camera1.fy",found);
            float cx = readParameter<float>(fSettings,"Camera1.cx",found);
            float cy = readParameter<float>(fSettings,"Camera1.cy",found);

            float k0 = readParameter<float>(fSettings,"Camera1.k1",found);
            float k1 = readParameter<float>(fSettings,"Camera1.k2",found);
            float k2 = readParameter<float>(fSettings,"Camera1.k3",found);
            float k3 = readParameter<float>(fSettings,"Camera1.k4",found);

            vCalibration = {fx,fy,cx,cy,k0,k1,k2,k3};

            calibration1_ = new KannalaBrandt8(vCalibration);
            originalCalib1_ = new KannalaBrandt8(vCalibration);

            if(sensor_ == System::STEREO || sensor_ == System::IMU_STEREO){
                int colBegin = readParameter<int>(fSettings,"Camera1.overlappingBegin",found);
                int colEnd = readParameter<int>(fSettings,"Camera1.overlappingEnd",found);
                vector<int> vOverlapping = {colBegin, colEnd};

                static_cast<KannalaBrandt8*>(calibration1_)->mvLappingArea = vOverlapping;
            }
        }
        else{
            cerr << "Error: " << cameraModel << " not known" << endl;
            exit(-1);
        }
    }

    // 获取相机2的内参、畸变参数。如果是双目会被调用
    void Settings::readCamera2(cv::FileStorage &fSettings) {
        bool found;
        vector<float> vCalibration;
        // 相机类型为PinHole，则读取相机2的内参、畸变参数，且设置双目校正标志位为true
        if (cameraType_ == PinHole) {
            bNeedToRectify_ = true;

            //Read intrinsic parameters
            float fx = readParameter<float>(fSettings,"Camera2.fx",found);
            float fy = readParameter<float>(fSettings,"Camera2.fy",found);
            float cx = readParameter<float>(fSettings,"Camera2.cx",found);
            float cy = readParameter<float>(fSettings,"Camera2.cy",found);


            vCalibration = {fx, fy, cx, cy};

            calibration2_ = new Pinhole(vCalibration);
            originalCalib2_ = new Pinhole(vCalibration);

            //Check if it is a distorted PinHole
            readParameter<float>(fSettings,"Camera2.k1",found,false);
            if(found){
                readParameter<float>(fSettings,"Camera2.k3",found,false);
                if(found){
                    vPinHoleDistorsion2_.resize(5);
                    vPinHoleDistorsion2_[4] = readParameter<float>(fSettings,"Camera2.k3",found);
                }
                else{
                    vPinHoleDistorsion2_.resize(4);
                }
                vPinHoleDistorsion2_[0] = readParameter<float>(fSettings,"Camera2.k1",found);
                vPinHoleDistorsion2_[1] = readParameter<float>(fSettings,"Camera2.k2",found);
                vPinHoleDistorsion2_[2] = readParameter<float>(fSettings,"Camera2.p1",found);
                vPinHoleDistorsion2_[3] = readParameter<float>(fSettings,"Camera2.p2",found);
            }
        }
        // KannalaBrandt相机模型，不进行双目校正
        else if(cameraType_ == KannalaBrandt){
            //Read intrinsic parameters
            float fx = readParameter<float>(fSettings,"Camera2.fx",found);
            float fy = readParameter<float>(fSettings,"Camera2.fy",found);
            float cx = readParameter<float>(fSettings,"Camera2.cx",found);
            float cy = readParameter<float>(fSettings,"Camera2.cy",found);

            float k0 = readParameter<float>(fSettings,"Camera1.k1",found);
            float k1 = readParameter<float>(fSettings,"Camera1.k2",found);
            float k2 = readParameter<float>(fSettings,"Camera1.k3",found);
            float k3 = readParameter<float>(fSettings,"Camera1.k4",found);


            vCalibration = {fx,fy,cx,cy,k0,k1,k2,k3};

            calibration2_ = new KannalaBrandt8(vCalibration);
            originalCalib2_ = new KannalaBrandt8(vCalibration);

            int colBegin = readParameter<int>(fSettings,"Camera2.overlappingBegin",found);
            int colEnd = readParameter<int>(fSettings,"Camera2.overlappingEnd",found);
            vector<int> vOverlapping = {colBegin, colEnd};

            static_cast<KannalaBrandt8*>(calibration2_)->mvLappingArea = vOverlapping;
        }

        // Load stereo extrinsic calibration
        // 若相机模型为Rectified，不进行双目矫正，则读取 基线、bfx
        if(cameraType_ == Rectified){
            b_ = readParameter<float>(fSettings,"Stereo.b",found);
            bf_ = b_ * calibration1_->getParameter(0);
        }
        // 若相机模型为Pinhole、KannalaBrandt，则读取双目矫正参数：右目到左目的变换矩阵 (其中t表示右目在左目坐标系下的坐标)、基线(平移向量的模长)、bfx
        else{
            cv::Mat cvTlr = readParameter<cv::Mat>(fSettings,"Stereo.T_c1_c2",found);
            Tlr_ = Converter::toSophus(cvTlr);  // Mat 转成 Sophus::SE3<float>

            //TODO: also search for Trl and invert if necessary

            b_ = Tlr_.translation().norm();
            bf_ = b_ * calibration1_->getParameter(0);
        }
        // 读取 深度阈值系数
        thDepth_ = readParameter<float>(fSettings,"Stereo.ThDepth",found);
    }

    // 读取想要的图像尺寸并更新标定参数，帧率，颜色空间类型
    void Settings::readImageInfo(cv::FileStorage &fSettings) {
        bool found;
        // Read original and desired image dimensions
        // 读取相机原始的和想要的图像尺寸
        int originalRows = readParameter<int>(fSettings,"Camera.height",found);
        int originalCols = readParameter<int>(fSettings,"Camera.width",found);
        originalImSize_.width = originalCols;
        originalImSize_.height = originalRows;

        newImSize_ = originalImSize_;
        // 如果配置文件中有新的图像尺寸，则需对图片大小进行调整
        // 对Height作调整
        int newHeigh = readParameter<int>(fSettings,"Camera.newHeight",found,false);
        if(found){
            bNeedToResize1_ = true;
            newImSize_.height = newHeigh;

            // 若不进行极线矫正
            if(!bNeedToRectify_){
                // Update calibration
                // 更新标定参数
                float scaleRowFactor = (float)newImSize_.height / (float)originalImSize_.height;

                calibration1_->setParameter(calibration1_->getParameter(1) * scaleRowFactor, 1);    // 更新fy
                calibration1_->setParameter(calibration1_->getParameter(3) * scaleRowFactor, 3);    // 更新cy

                // 若双目 且 非Rectified (即KannalaBrandt双目)，也更新相机2的fy, cy
                if((sensor_ == System::STEREO || sensor_ == System::IMU_STEREO) && cameraType_ != Rectified){
                    calibration2_->setParameter(calibration2_->getParameter(1) * scaleRowFactor, 1);
                    calibration2_->setParameter(calibration2_->getParameter(3) * scaleRowFactor, 3);
                }
            }
        }

        // 对Width作调整
        int newWidth = readParameter<int>(fSettings,"Camera.newWidth",found,false);
        if(found){
            bNeedToResize1_ = true;
            newImSize_.width = newWidth;

            if(!bNeedToRectify_){
                //Update calibration
                float scaleColFactor = (float)newImSize_.width / (float) originalImSize_.width;
                calibration1_->setParameter(calibration1_->getParameter(0) * scaleColFactor, 0);    // 更新fx
                calibration1_->setParameter(calibration1_->getParameter(2) * scaleColFactor, 2);    // 更新cx

                // 更新相机2的fx, cx
                if((sensor_ == System::STEREO || sensor_ == System::IMU_STEREO) && cameraType_ != Rectified){
                    calibration2_->setParameter(calibration2_->getParameter(0) * scaleColFactor, 0);
                    calibration2_->setParameter(calibration2_->getParameter(2) * scaleColFactor, 2);

                    // KannalaBrandt再对重叠区域的大小进行调整
                    if(cameraType_ == KannalaBrandt){
                        static_cast<KannalaBrandt8*>(calibration1_)->mvLappingArea[0] *= scaleColFactor;
                        static_cast<KannalaBrandt8*>(calibration1_)->mvLappingArea[1] *= scaleColFactor;

                        static_cast<KannalaBrandt8*>(calibration2_)->mvLappingArea[0] *= scaleColFactor;
                        static_cast<KannalaBrandt8*>(calibration2_)->mvLappingArea[1] *= scaleColFactor;
                    }
                }
            }
        }

        fps_ = readParameter<int>(fSettings,"Camera.fps",found);
        bRGB_ = (bool) readParameter<int>(fSettings,"Camera.RGB",found);
    }

    // 读取IMU参数
    void Settings::readIMU(cv::FileStorage &fSettings) {
        bool found;
        noiseGyro_ = readParameter<float>(fSettings,"IMU.NoiseGyro",found); // 角速度白噪声
        noiseAcc_ = readParameter<float>(fSettings,"IMU.NoiseAcc",found);   // 加速度白噪声
        gyroWalk_ = readParameter<float>(fSettings,"IMU.GyroWalk",found);   // 角速度 walk
        accWalk_ = readParameter<float>(fSettings,"IMU.AccWalk",found);     // 加速度 walk
        imuFrequency_ = readParameter<float>(fSettings,"IMU.Frequency",found);

        cv::Mat cvTbc = readParameter<cv::Mat>(fSettings,"IMU.T_b_c1",found);
        Tbc_ = Converter::toSophus(cvTbc);

        readParameter<int>(fSettings,"IMU.InsertKFsWhenLost",found,false);
        if(found){
            insertKFsWhenLost_ = (bool) readParameter<int>(fSettings,"IMU.InsertKFsWhenLost",found,false);
        }
        else{
            insertKFsWhenLost_ = true;
        }
    }

    void Settings::readRGBD(cv::FileStorage& fSettings) {
        bool found;

        depthMapFactor_ = readParameter<float>(fSettings,"RGBD.DepthMapFactor",found);
        thDepth_ = readParameter<float>(fSettings,"Stereo.ThDepth",found);
        b_ = readParameter<float>(fSettings,"Stereo.b",found);
        bf_ = b_ * calibration1_->getParameter(0);
    }

    void Settings::readORB(cv::FileStorage &fSettings) {
        bool found;

        nFeatures_ = readParameter<int>(fSettings,"ORBextractor.nFeatures",found);
        scaleFactor_ = readParameter<float>(fSettings,"ORBextractor.scaleFactor",found);
        nLevels_ = readParameter<int>(fSettings,"ORBextractor.nLevels",found);
        initThFAST_ = readParameter<int>(fSettings,"ORBextractor.iniThFAST",found);
        minThFAST_ = readParameter<int>(fSettings,"ORBextractor.minThFAST",found);
    }

    void Settings::readViewer(cv::FileStorage &fSettings) {
        bool found;

        keyFrameSize_ = readParameter<float>(fSettings,"Viewer.KeyFrameSize",found);
        keyFrameLineWidth_ = readParameter<float>(fSettings,"Viewer.KeyFrameLineWidth",found);
        graphLineWidth_ = readParameter<float>(fSettings,"Viewer.GraphLineWidth",found);
        pointSize_ = readParameter<float>(fSettings,"Viewer.PointSize",found);
        cameraSize_ = readParameter<float>(fSettings,"Viewer.CameraSize",found);
        cameraLineWidth_ = readParameter<float>(fSettings,"Viewer.CameraLineWidth",found);
        viewPointX_ = readParameter<float>(fSettings,"Viewer.ViewpointX",found);
        viewPointY_ = readParameter<float>(fSettings,"Viewer.ViewpointY",found);
        viewPointZ_ = readParameter<float>(fSettings,"Viewer.ViewpointZ",found);
        viewPointF_ = readParameter<float>(fSettings,"Viewer.ViewpointF",found);
        imageViewerScale_ = readParameter<float>(fSettings,"Viewer.imageViewScale",found,false);

        // 如果没提供 Viewer.imageViewScale，则赋值为 1.0f
         if(!found)
            imageViewerScale_ = 1.0f;
    }

    void Settings::readLoadAndSave(cv::FileStorage &fSettings) {
        bool found;

        sLoadFrom_ = readParameter<string>(fSettings,"System.LoadAtlasFromFile",found,false);
        sSaveto_ = readParameter<string>(fSettings,"System.SaveAtlasToFile",found,false);
    }

    void Settings::readOtherParameters(cv::FileStorage& fSettings) {
        bool found;

        thFarPoints_ = readParameter<float>(fSettings,"System.thFarPoints",found,false);
    }

    /**
     * @brief 预先计算双目校正图像映射M1l_，M2l_，M1r_，M2r_
     */
    void Settings::precomputeRectificationMaps() {
        // Precompute rectification maps, new calibrations, ...
        // 左目内参矩阵
        cv::Mat K1 = static_cast<Pinhole*>(calibration1_)->toK();
        K1.convertTo(K1,CV_64F);
        // 右目内参矩阵
        cv::Mat K2 = static_cast<Pinhole*>(calibration2_)->toK();
        K2.convertTo(K2,CV_64F);

//        std::cout << std::endl << "单目畸变矫正后参数: " << std::endl;
//        std::cout << "Tlr_: " << Tlr_.matrix() << std::endl << std::endl;
        // 得到左目到右目的变换矩阵
        cv::Mat cvTlr;
        cv::eigen2cv(Tlr_.inverse().matrix3x4(),cvTlr);
//        std::cout << "Tlr: " << cvTlr << std::endl << std::endl;
        // 左目到右目的旋转矩阵
        cv::Mat R12 = cvTlr.rowRange(0,3).colRange(0,3);
        R12.convertTo(R12,CV_64F);
        // 左目到右目的平移向量 (为负数，实际表示为 左目相机在右目相机坐标系下的坐标)
        cv::Mat t12 = cvTlr.rowRange(0,3).col(3);
        t12.convertTo(t12,CV_64F);

//        std::cout << "K1: " << K1 << std::endl << std::endl;
//        std::cout << "D1: " << camera1DistortionCoef() << std::endl << std::endl;
//        std::cout << "K2: " << K2 << std::endl << std::endl;
//        std::cout << "D2: " << camera2DistortionCoef() << std::endl << std::endl;
//        std::cout << "R12: " << R12 << std::endl << std::endl;
//        std::cout << "t12: " << t12 << std::endl << std::endl;


        std::cout << "双目校正计算：得到变换矩阵、投影矩阵" << std::endl;
        cv::Mat R_r1_u1, R_r2_u2;
        cv::Mat P1, P2, Q;
        // 得到双目矫正所需的变换矩阵、投影矩阵
        cv::stereoRectify(K1,                   // 输入：左目内参矩阵
                          camera1DistortionCoef(), // 输入：左目畸变参数, 4/5x1
                          K2,                   // 输入：右目内参矩阵
                          camera2DistortionCoef(), // 输入：右目畸变参数, 4/5x1
                          newImSize_,               // 输入：输入图像大小
                          R12, t12,                     // 输入：左目相机坐标系到右目相机坐标系的旋转矩阵、平移向量
                          R_r1_u1, R_r2_u2,           // 输出：R_{左 矫正相机坐标系}{左 未矫正相机坐标系}、R_{右 矫正相机坐标系}{右 未矫正相机坐标系}
                          P1, P2,                             // 输出：3x4: 左矫正坐标系到左图像坐标系的透视投影矩阵、右矫正坐标系到左图像坐标系的透视投影矩阵
                          Q,                                  // 输出：4x4: 视差深度映射矩阵
                          cv::CALIB_ZERO_DISPARITY,
                          -1,
                          newImSize_);  // 矫正后图像大小
//        std::cout << "R_r1_u1: " << R_r1_u1 << std::endl << std::endl;
//        std::cout << "R_r2_u2: " << R_r2_u2 << std::endl << std::endl;
//        std::cout << "P1: " << P1 << std::endl << std::endl;
//        std::cout << "P2: " << P2 << std::endl << std::endl;

//        std::cout << "畸变映射计算: " << std::endl;
        // 计算畸变映射
        cv::initUndistortRectifyMap(K1, camera1DistortionCoef(), R_r1_u1, P1.rowRange(0, 3).colRange(0, 3),
                                    newImSize_, CV_32F, M1l_, M2l_);
        cv::initUndistortRectifyMap(K2, camera2DistortionCoef(), R_r2_u2, P2.rowRange(0, 3).colRange(0, 3),
                                    newImSize_, CV_32F, M1r_, M2r_);

//        std::cout << "M1l_: " << M1l_ << std::endl << std::endl;
//        std::cout << "M2l_: " << M2l_ << std::endl << std::endl;
//        std::cout << "M1r_: " << M1r_ << std::endl << std::endl;
//        std::cout << "M2r_: " << M2r_ << std::endl << std::endl;

        // Update calibration
        // 更新矫正后的内参矩阵
        calibration1_->setParameter(P1.at<double>(0,0), 0); // fx
        calibration1_->setParameter(P1.at<double>(1,1), 1); // fy
        calibration1_->setParameter(P1.at<double>(0,2), 2); // cx
        calibration1_->setParameter(P1.at<double>(1,2), 3); // cy

//        std::cout << "双目矫正后的内参矩阵: " << std::endl;
//        std::cout << "fx: " << calibration1_->getParameter(0) << std::endl;
//        std::cout << "fy: " << calibration1_->getParameter(1) << std::endl;
//        std::cout << "cx: " << calibration1_->getParameter(2) << std::endl;
//        std::cout << "cy: " << calibration1_->getParameter(3) << std::endl;

        // Update bf
        // 更新 bfx
        bf_ = b_ * P1.at<double>(0,0);
//        std::cout << "b_: "<< setprecision(17) << b_ << std::endl << std::endl;
//        std::cout << "双目矫正后的bfx: "<< bf_ << std::endl << std::endl;

        // Update relative pose between camera 1 and IMU if necessary
        // IMU模式，则更新左目到IMU (body) 的相对位姿
        if(sensor_ == System::IMU_STEREO){
            Eigen::Matrix3f eigenR_r1_u1;
            cv::cv2eigen(R_r1_u1,eigenR_r1_u1);
            Sophus::SE3f T_r1_u1(eigenR_r1_u1,Eigen::Vector3f::Zero());
//            std::cout << "双目矫正前的tbc: "<< Tbc_.matrix() << std::endl << std::endl;
            Tbc_ = Tbc_ * T_r1_u1.inverse();
//            std::cout << "双目矫正后的tbc: "<< Tbc_.matrix() << std::endl << std::endl;
        }
    }

    ostream &operator<<(std::ostream& output, const Settings& settings){
        output << "SLAM settings (极线矫正后): " << endl;

        // 打印相机1内参
        output << "\t-Camera 1 parameters (";
        if(settings.cameraType_ == Settings::PinHole || settings.cameraType_ ==  Settings::Rectified){
            output << "Pinhole: " << settings.cameraType_;
        }
        else{
            output << "Kannala-Brandt";
        }
        output << ")" << ": [";
        for(size_t i = 0; i < settings.originalCalib1_->size(); i++) {
            output << " " << settings.originalCalib1_->getParameter(i);
        }
        output << " ]" << endl;

        // 打印相机1 畸变系数
        if(!settings.vPinHoleDistorsion1_.empty()){
            output << "\t-Camera 1 distortion parameters: [ ";
            for(float d : settings.vPinHoleDistorsion1_){
                output << " " << d;
            }
            output << " ]" << endl;
        }

        if(settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO){
            // 打印相机2 内参
            output << "\t-Camera 2 parameters (";
            if(settings.cameraType_ == Settings::PinHole || settings.cameraType_ ==  Settings::Rectified){
                output << "Pinhole: " << settings.cameraType_;
            }
            else{
                output << "Kannala-Brandt";
            }
            output << ")" << ": [";
            for(size_t i = 0; i < settings.originalCalib2_->size(); i++){
                output << " " << settings.originalCalib2_->getParameter(i);
            }
            output << " ]" << endl;
            // 打印相机2 畸变系数
            if(!settings.vPinHoleDistorsion2_.empty()){
                output << "\t-Camera 1 distortion parameters: [";
                for(float d : settings.vPinHoleDistorsion2_){
                    output << " " << d;
                }
                output << " ]" << endl;
            }
        }

        output << "\t-Original image size: [ " << settings.originalImSize_.width << ", " << settings.originalImSize_.height << " ]" << endl;
        output << "\t-Current image size: [ " << settings.newImSize_.width << ", " << settings.newImSize_.height << " ]" << endl;

        // 若需进行极线校正，则输出极线校正后的内参
        if(settings.bNeedToRectify_){
            output << "\t-Camera 1 parameters after rectification: [";
            for(size_t i = 0; i < settings.calibration1_->size(); i++){
                output << " " << settings.calibration1_->getParameter(i);
            }
            output << " ]" << endl;
        }
        // 若需调整图片大小
        else if(settings.bNeedToResize1_){
            output << "\t-Camera 1 parameters after resize: [";
            for(size_t i = 0; i < settings.calibration1_->size(); i++){
                output << " " << settings.calibration1_->getParameter(i);
            }
            output << " ]" << endl;

            if((settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO) &&
                settings.cameraType_ == Settings::KannalaBrandt){
                output << "\t-Camera 2 parameters after resize: [";
                for(size_t i = 0; i < settings.calibration2_->size(); i++){
                    output << " " << settings.calibration2_->getParameter(i);
                }
                output << " ]" << endl;
            }
        }

        output << "\t-Sequence FPS: " << settings.fps_ << endl;

        //Stereo stuff
        if(settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO){
            output << "\t-Stereo baseline: " << settings.b_ << endl;
            output << "\t-Stereo baseline * fx: " << settings.bf_ << endl;
            output << "\t-Stereo depth threshold : " << settings.thDepth_ << endl;

            if(settings.cameraType_ == Settings::KannalaBrandt){
                auto vOverlapping1 = static_cast<KannalaBrandt8*>(settings.calibration1_)->mvLappingArea;
                auto vOverlapping2 = static_cast<KannalaBrandt8*>(settings.calibration2_)->mvLappingArea;
                output << "\t-Camera 1 overlapping area: [ " << vOverlapping1[0] << " , " << vOverlapping1[1] << " ]" << endl;
                output << "\t-Camera 2 overlapping area: [ " << vOverlapping2[0] << " , " << vOverlapping2[1] << " ]" << endl;
            }
        }

        // IMU模式
        if(settings.sensor_ == System::IMU_MONOCULAR || settings.sensor_ == System::IMU_STEREO || settings.sensor_ == System::IMU_RGBD) {
            output << "\t-Gyro noise: " << settings.noiseGyro_ << endl;
            output << "\t-Accelerometer noise: " << settings.noiseAcc_ << endl;
            output << "\t-Gyro walk: " << settings.gyroWalk_ << endl;
            output << "\t-Accelerometer walk: " << settings.accWalk_ << endl;
            output << "\t-IMU frequency: " << settings.imuFrequency_ << endl;
        }
        // RGBD / IMU+RGBD
        if(settings.sensor_ == System::RGBD || settings.sensor_ == System::IMU_RGBD){
            output << "\t-RGB-D depth map factor: " << settings.depthMapFactor_ << endl;
        }

        output << "\t-Features per image: " << settings.nFeatures_ << endl;
        output << "\t-ORB scale factor: " << settings.scaleFactor_ << endl;
        output << "\t-ORB number of scales: " << settings.nLevels_ << endl;
        output << "\t-Initial FAST threshold: " << settings.initThFAST_ << endl;
        output << "\t-Min FAST threshold: " << settings.minThFAST_ << endl;

        return output;
    }
};
