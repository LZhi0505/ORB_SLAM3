/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez
 * Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Tracking.h"

#include <chrono>
#include <filesystem> // liuzhi加
#include <iostream>
#include <mutex>

#include "Converter.h"
#include "FrameDrawer.h"
#include "G2oTypes.h"
#include "GeometricTools.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Pinhole.h"

using namespace std;

namespace ORB_SLAM3 {

/**
 * @brief 跟踪线程 构造函数
 * @param pSys 系统类指针
 * @param pVoc 词袋
 * @param pFrameDrawer 画图像的
 * @param pMapDrawer 画地图的
 * @param pAtlas 地图集
 * @param pKFDB 关键帧数据库
 * @param strSettingPath 参数文件路径
 * @param sensor 传感器类型
 * @param settings 参数类
 * @param _strSeqName 序列名字，没用到
 */
Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor,
                   Settings *settings, const string &_nameSeq)
    : mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false), mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
      mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0),
      time_recently_lost(5.0), mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame *>(NULL)) {
    // Load camera parameters from settings file
    // 加载相机参数：如果从System中传入settings_，则加载(一般是)；否则从配置文件路径中加载
    if (settings) {
        std::cout << "[Tracking::Tracking] 从settings_类中加载设置" << std::endl;
        newParameterLoader(settings);
    } else {
        std::cout << "[Tracking::Tracking] 从配置文件 " << strSettingPath << " 加载设置" << std::endl;
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if (!b_parse_cam) {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if (!b_parse_orb) {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;
        if (sensor == System::IMU_MONOCULAR || sensor == System::IMU_STEREO || sensor == System::IMU_RGBD) {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if (!b_parse_imu) {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if (!b_parse_cam || !b_parse_orb || !b_parse_imu) {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try {
                throw -1;
            } catch (exception &e) {
            }
        }
    }

    initID = 0;
    lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    // 遍历地图中的相机，然后打印出来
    vector<GeometricCamera *> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for (GeometricCamera *pCam : vpCams) {
        std::cout << "Camera " << pCam->GetId();
        if (pCam->GetType() == GeometricCamera::CAM_PINHOLE) {
            std::cout << " is pinhole" << std::endl;
        } else if (pCam->GetType() == GeometricCamera::CAM_FISHEYE) {
            std::cout << " is fisheye" << std::endl;
        } else {
            std::cout << " is unknown" << std::endl;
        }
    }

    ofs_frame.open("/home/liuzhi/下载/ORB_SLAM3/Examples/f_stereo.txt", std::ios::out);

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times) {
    double accum = 0;
    for (double value : v_times) {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average) {
    double accum = 0;
    for (double value : v_times) {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values) {
    double accum = 0;
    int total = 0;
    for (double value : v_values) {
        if (value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average) {
    double accum = 0;
    int total = 0;
    for (double value : v_values) {
        if (value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File() {
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF "
         "culling[ms], Total[ms]"
      << endl;
    for (int i = 0; i < mpLocalMapper->vdLMTotal_ms.size(); ++i) {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << "," << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] << "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for (int i = 0; i < mpLocalMapper->vdLBASync_ms.size(); ++i) {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << "," << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }

    f.close();
}

void Tracking::TrackStats2File() {
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU "
         "preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]"
      << endl;

    for (int i = 0; i < vdTrackTotal_ms.size(); ++i) {
        double stereo_rect = 0.0;
        if (!vdRectStereo_ms.empty()) {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if (!vdResizeImage_ms.empty()) {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if (!vdStereoMatch_ms.empty()) {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if (!vdIMUInteg_ms.empty()) {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << "," << vdPosePred_ms[i] << "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i]
          << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats() {
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();

    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    // Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if (!vdRectStereo_ms.empty()) {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if (!vdResizeImage_ms.empty()) {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if (!vdStereoMatch_ms.empty()) {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if (!vdIMUInteg_ms.empty()) {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map *> vpMaps = mpAtlas->GetAllMaps();
    Map *pBestMap = vpMaps[0];
    for (int i = 1; i < vpMaps.size(); ++i) {
        if (pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size()) {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();
}

#endif

Tracking::~Tracking() {
    // f_track_stats.close();
}

/**
 * @brief 从已构建的settings_类读取参数
 * @param settings 参数类
 */
void Tracking::newParameterLoader(Settings *settings) {
    std::cout << "[Tracking::newParameterLoader] 读取settings_各参数" << std::endl;
    // Step 1: 读取相机参数
    // 读取settings_的相机1 (左目)为 mpCamera，并加入到地图集中
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    // 如果settings_设置需要进行畸变校正 (PinHole单目为true，其余为false)，则加载左目的畸变参数；否则将畸变参数置为0
    if (settings->needToUndistort()) {
        std::cout << "\t需进行 特征点去畸变，读取左目的畸变参数" << std::endl;
        mDistCoef = settings->camera1DistortionCoef();
    } else {
        std::cout << "\t不需要 特征点畸变校正，畸变参数设为0" << std::endl;
        mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
    }

    // TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    // 加载左目的内参矩阵
    mK = cv::Mat::eye(3, 3, CV_32F);
    mK.at<float>(0, 0) = mpCamera->getParameter(0);
    mK.at<float>(1, 1) = mpCamera->getParameter(1);
    mK.at<float>(0, 2) = mpCamera->getParameter(2);
    mK.at<float>(1, 2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0, 0) = mpCamera->getParameter(0);
    mK_(1, 1) = mpCamera->getParameter(1);
    mK_(0, 2) = mpCamera->getParameter(2);
    mK_(1, 2) = mpCamera->getParameter(3);

    // KB鱼眼相机的 双目 或 IMU+双目 或 IMU+RGBD，则读取 mpCamera2
    if ((mSensor == System::STEREO || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && settings->cameraType() == Settings::KannalaBrandt) {
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr = settings->Tlr();

        mpFrameDrawer->both = true;
    }

    // 双目 / RGBD / IMU+双目 / IMU+RGBD，读取bfx, 计算深度阈值：b * 深度阈值系数
    if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
        mbf = settings->bf();                           // bfx
        mThDepth = settings->b() * settings->thDepth(); // 基线 * 深度阈值系数(可能为35 / 40 / 60)
        std::cout << "mbf: " << mbf << ", Depth Threshold (Close/Far Points): " << mThDepth << std::endl;
        //        Verbose::PrintMess("mbf: " + std::to_string(mbf) + ", Depth
        //        Threshold (Close/Far Points): " + std::to_string(mThDepth),
        //        Verbose::VERBOSITY_DEBUG);
    }
    // 读取RGBD
    if (mSensor == System::RGBD || mSensor == System::IMU_RGBD) {
        mDepthMapFactor = settings->depthMapFactor();
        if (fabs(mDepthMapFactor) < 1e-5)
            mDepthMapFactor = 1;
        else
            mDepthMapFactor = 1.0f / mDepthMapFactor;
    }

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    // ORB parameters
    // Step 2: 读取特征点参数
    int nFeatures = settings->nFeatures();   // 指定要提取的特征点数目 1000
    int nLevels = settings->nLevels();       // 指定图像金字塔的缩放系数 1.2
    int fIniThFAST = settings->initThFAST(); // 指定图像金字塔的层数 8
    int fMinThFAST = settings->minThFAST();  // 指定初始的FAST特征点提取参数，可以提取出最明显的角点 20
    float fScaleFactor = settings->scaleFactor(); // 如果因为图像纹理不丰富提取出的特征点不多，为了达到想要的特征点数目，就使用这个参数提取出不是那么明显的角点 7

    // 创建ORB特征提取器
    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    // IMU parameters
    // Step 3: 读取IMU参数
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    mImuPer = 0.001; // 1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);                                        // 缩放因子，将连续时间的噪声参数转换成离散时间的噪声参数
    mpImuCalib = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf); // 存储IMU标定参数

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib); // 存储IMU的预积分量，包括IMU的零偏与IMU标定参数
}

/**
 * @brief 根据文件读取相机参数
 * @param fSettings 配置文件
 */
bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings) {
    mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if (sCameraName == "PinHole") {
        float fx, fy, cx, cy; // 内参矩阵 K
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if (!node.empty() && node.isReal()) {
            fx = node.real();
        } else {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if (!node.empty() && node.isReal()) {
            fy = node.real();
        } else {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if (!node.empty() && node.isReal()) {
            cx = node.real();
        } else {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if (!node.empty() && node.isReal()) {
            cy = node.real();
        } else {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters 畸变参数，存储到 mDistCoef 内
        node = fSettings["Camera.k1"];
        if (!node.empty() && node.isReal()) {
            mDistCoef.at<float>(0) = node.real();
        } else {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if (!node.empty() && node.isReal()) {
            mDistCoef.at<float>(1) = node.real();
        } else {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if (!node.empty() && node.isReal()) {
            mDistCoef.at<float>(2) = node.real();
        } else {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if (!node.empty() && node.isReal()) {
            mDistCoef.at<float>(3) = node.real();
        } else {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if (!node.empty() && node.isReal()) {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        // 如果配置文件中指定了 ImageScale，则使用配置文件里的；否则默认为 1.f
        node = fSettings["Camera.imageScale"];
        if (!node.empty() && node.isReal()) {
            mImageScale = node.real();
        }

        if (b_miss_params) {
            return false;
        }
        // 不为 1.f，则更新内参矩阵
        if (mImageScale != 1.f) {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx, fy, cx, cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- Image scale: " << mImageScale << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;

        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if (mDistCoef.rows == 5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        // 内参矩阵 存储到 mK
        mK = cv::Mat::eye(3, 3, CV_32F);
        mK.at<float>(0, 0) = fx;
        mK.at<float>(1, 1) = fy;
        mK.at<float>(0, 2) = cx;
        mK.at<float>(1, 2) = cy;

        mK_.setIdentity();
        mK_(0, 0) = fx;
        mK_(1, 1) = fy;
        mK_(0, 2) = cx;
        mK_(1, 2) = cy;
    } else if (sCameraName == "KannalaBrandt8") {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if (!node.empty() && node.isReal()) {
            fx = node.real();
        } else {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if (!node.empty() && node.isReal()) {
            fy = node.real();
        } else {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if (!node.empty() && node.isReal()) {
            cx = node.real();
        } else {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if (!node.empty() && node.isReal()) {
            cy = node.real();
        } else {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if (!node.empty() && node.isReal()) {
            k1 = node.real();
        } else {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if (!node.empty() && node.isReal()) {
            k2 = node.real();
        } else {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if (!node.empty() && node.isReal()) {
            k3 = node.real();
        } else {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if (!node.empty() && node.isReal()) {
            k4 = node.real();
        } else {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if (!node.empty() && node.isReal()) {
            mImageScale = node.real();
        }

        if (!b_miss_params) {
            if (mImageScale != 1.f) {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx, fy, cx, cy, k1, k2, k3, k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3, 3, CV_32F);
            mK.at<float>(0, 0) = fx;
            mK.at<float>(1, 1) = fy;
            mK.at<float>(0, 2) = cx;
            mK.at<float>(1, 2) = cy;

            mK_.setIdentity();
            mK_(0, 0) = fx;
            mK_(1, 1) = fy;
            mK_(0, 2) = cx;
            mK_(1, 2) = cy;
        }

        if (mSensor == System::STEREO || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if (!node.empty() && node.isReal()) {
                fx = node.real();
            } else {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if (!node.empty() && node.isReal()) {
                fy = node.real();
            } else {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if (!node.empty() && node.isReal()) {
                cx = node.real();
            } else {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if (!node.empty() && node.isReal()) {
                cy = node.real();
            } else {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if (!node.empty() && node.isReal()) {
                k1 = node.real();
            } else {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if (!node.empty() && node.isReal()) {
                k2 = node.real();
            } else {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if (!node.empty() && node.isReal()) {
                k3 = node.real();
            } else {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if (!node.empty() && node.isReal()) {
                k4 = node.real();
            } else {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if (!node.empty() && node.isInt()) {
                leftLappingBegin = node.operator int();
            } else {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if (!node.empty() && node.isInt()) {
                leftLappingEnd = node.operator int();
            } else {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if (!node.empty() && node.isInt()) {
                rightLappingBegin = node.operator int();
            } else {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if (!node.empty() && node.isInt()) {
                rightLappingEnd = node.operator int();
            } else {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if (!node.empty()) {
                cvTlr = node.mat();
                if (cvTlr.rows != 3 || cvTlr.cols != 4) {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            } else {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if (!b_miss_params) {
                if (mImageScale != 1.f) {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx, fy, cx, cy, k1, k2, k3, k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << cvTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if (b_miss_params) {
            return false;
        }

    } else {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
        cv::FileNode node = fSettings["Camera.bf"];
        if (!node.empty() && node.isReal()) {
            mbf = node.real();
            if (mImageScale != 1.f) {
                mbf *= mImageScale;
            }
        } else {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
    }

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if (!node.empty() && node.isReal()) {
            mThDepth = node.real();
            mThDepth = mbf * mThDepth / fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        } else {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
    }

    if (mSensor == System::RGBD || mSensor == System::IMU_RGBD) {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if (!node.empty() && node.isReal()) {
            mDepthMapFactor = node.real();
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        } else {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
    }

    if (b_miss_params) {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings) {
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if (!node.empty() && node.isInt()) {
        nFeatures = node.operator int();
    } else {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an "
                     "integer*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if (!node.empty() && node.isReal()) {
        fScaleFactor = node.real();
    } else {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not "
                     "a real number*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if (!node.empty() && node.isInt()) {
        nLevels = node.operator int();
    } else {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if (!node.empty() && node.isInt()) {
        fIniThFAST = node.operator int();
    } else {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an "
                     "integer*"
                  << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if (!node.empty() && node.isInt()) {
        fMinThFAST = node.operator int();
    } else {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an "
                     "integer*"
                  << std::endl;
        b_miss_params = true;
    }

    if (b_miss_params) {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings) {
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if (!node.empty()) {
        cvTbc = node.mat();
        if (cvTbc.rows != 4 || cvTbc.cols != 4) {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    } else {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    cout << endl;
    cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if (!node.empty() && node.isInt()) {
        mInsertKFsLost = (bool)node.operator int();
    }

    if (!mInsertKFsLost)
        cout << "Do not insert keyframes when lost visual tracking " << endl;

    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if (!node.empty() && node.isInt()) {
        mImuFreq = node.operator int();
        mImuPer = 0.001; // 1.0 / (double) mImuFreq;
    } else {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if (!node.empty() && node.isReal()) {
        Ng = node.real();
    } else {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if (!node.empty() && node.isReal()) {
        Na = node.real();
    } else {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if (!node.empty() && node.isReal()) {
        Ngw = node.real();
    } else {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if (!node.empty() && node.isReal()) {
        Naw = node.real();
    } else {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if (!node.empty()) {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if (mFastInit)
        cout << "Fast IMU initialization. Acceleration is not checked \n";

    if (b_miss_params) {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    cout << endl;
    cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);

    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) { mpLocalMapper = pLocalMapper; }

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) { mpLoopClosing = pLoopClosing; }

void Tracking::SetViewer(Viewer *pViewer) { mpViewer = pViewer; }

void Tracking::SetStepByStep(bool bSet) { bStepByStep = bSet; }

bool Tracking::GetStepByStep() { return bStepByStep; }

/**
 * 双目跟踪
 * @param imRectLeft    左目图像（极线矫正、调整大小后的）
 * @param imRectRight   右目图像（极线矫正、调整大小后的）
 * @param timestamp     图片时间戳
 * @param filename      左目图片路径（IMU模式下为空）
 * @return
 */
Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename) {
    // cout << "GrabImageStereo" << endl;

    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if (mImGray.channels() == 3) {
        // 配置文件中指定图像格式为RGB, 则以RGB格式读取图像
        if (mbRGB) {
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGB2GRAY);
        } else {
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGR2GRAY);
        }
    } else if (mImGray.channels() == 4) {
        if (mbRGB) {
            cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGBA2GRAY);
        } else {
            cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGRA2GRAY);
        }
    }

    // 构建一帧
    // PinHole 双目 (未提供Camera2)
    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
    // KannalaBrandt8 鱼眼双目 (提供Camera2, 传入mTlr)
    else if (mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr);
    // PinHole 双目+IMU (未提供Camera2)
    else if (mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);
    // KannalaBrandt8 鱼眼双目+IMU (提供Camera2, 传入mTlr)
    else if (mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame =
            Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr, &mLastFrame, *mpImuCalib);

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    Verbose::PrintMess("Frame " + std::to_string(mCurrentFrame.mnId) + " 特征点个数：" + std::to_string(mCurrentFrame.N), Verbose::VERBOSITY_VERBOSE);

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    Track();

    std::cout << "\t\tcur_id: " << mCurrentFrame.mnId << ", 地图库关键帧个数: " << mpAtlas->GetAllKeyFrames().size() << ", 地图点个数: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    // 保存每帧位姿
    if (mState == OK) {
        trackOK_frame_num++;

        Sophus::SE3f Twb = mCurrentFrame.GetPose().inverse();
        Eigen::Quaternionf q = Twb.unit_quaternion();
        Eigen::Vector3f twb = Twb.translation();
        ofs_frame << setprecision(19) << 1e9 * (mCurrentFrame.mTimeStamp) << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " "
                  << q.w() << endl;
    }
    total_frame_num++;
    std::cout << "\tvalid frame: " << trackOK_frame_num << ", total frame: " << total_frame_num << endl;

    return mCurrentFrame.GetPose();
}

Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename) {
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if (mImGray.channels() == 3) {
        if (mbRGB)
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
    } else if (mImGray.channels() == 4) {
        if (mbRGB)
            cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
    }

    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

    if (mSensor == System::RGBD)
        mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
    else if (mSensor == System::IMU_RGBD)
        mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    Track();

    return mCurrentFrame.GetPose();
}

/**
 * @brief 输入左目RGB或RGBA图像，输出世界坐标系到该帧相机坐标系的变换矩阵 Tcw
 *
 * @param im 图像
 * @param timestamp 时间戳
 * @param filename 文件名字，貌似调试用的
 *
 * Step 1 ：将彩色图像转为灰度图像
 * Step 2 ：构造Frame
 * Step 3 ：跟踪
 */
Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename) {
    mImGray = im;
    // Step 1：将彩色图像转为灰度图像
    // 若图片是3、4通道的彩色图，还需要转化成单通道灰度图
    if (mImGray.channels() == 3) {
        if (mbRGB)
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
    } else if (mImGray.channels() == 4) {
        if (mbRGB)
            cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
    }

    // Step 2：构造Frame类，创建一个新的帧对象：会提取该帧的关键点 mvKeys 和描述子 mDescriptors，并对关键点进行去畸变 mvKeysUn，再将它们分配到网格中
    // 单目
    if (mSensor == System::MONOCULAR) {
        // 如果 系统尚未初始化 或 还没有接收到图像帧 或 最新接收的图像帧ID与初始化帧ID之间的差 < mMaxFrames
        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET || (lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
        else
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
    }
    // 单目 + IMU
    else if (mSensor == System::IMU_MONOCULAR) {
        // 判断该帧是不是初始化
        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET) // 没有成功初始化的前一个状态就是NO_IMAGES_YET
        {
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
        } else
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
    }

    // t0 存储未初始化时的第 1 帧图像时间戳
    if (mState == NO_IMAGES_YET)
        t0 = timestamp;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId;
    Verbose::PrintMess("Frame " + std::to_string(mCurrentFrame.mnId) + " 特征点个数：" + std::to_string(mCurrentFrame.N), Verbose::VERBOSITY_VERBOSE);

    // Step 3：跟踪
    Track();

    std::cout << "\t\tcur_id: " << mCurrentFrame.mnId << ", 地图库关键帧个数: " << mpAtlas->GetAllKeyFrames().size() << ", 地图点个数: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    // cout << "Tracking end" << endl;
    // ------- liuzhi加 -------
    std::ofstream of_frm("/home/liuzhi/下载/ORB_SLAM3/Examples/Monocular/f_mono.txt", std::ios::app);
    Eigen::Quaternionf q;
    Eigen::Vector3f twb;
    if (mState == OK) {
        trackOK_frame_num++;
        Sophus::SE3f Twb = mCurrentFrame.GetPose().inverse();
        q = Twb.unit_quaternion();
        twb = Twb.translation();
        // 正常的数据集保存时以ns为单位
        //        of_frm << setprecision(15) << 1e9*(mCurrentFrame.mTimeStamp) << "
        //        " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2)
        //        << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() <<
        //        endl;
    } else {
        q = Eigen::Quaternionf(1, 0, 0, 0);
        twb = Eigen::Vector3f(0, 0, 0);
    }
    // shangtang 数据集GT的时间戳, 保存时以s为单位
    of_frm << setprecision(15) << (mCurrentFrame.mTimeStamp) << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
           << endl;
    total_frame_num++;
    std::cout << "\tvalid frame: " << trackOK_frame_num << ", total frame: " << total_frame_num << endl;
    // -----------------------

    // 返回当前帧的位姿
    return mCurrentFrame.GetPose();
}

/**
 * @brief 将两帧间的IMU测量数据存放在list链表类型的 mlQueueImuData 里
 * @param[in] imuMeasurement IMU测量数据
 */
void Tracking::GrabImuData(const IMU::Point &imuMeasurement) {
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

/**
 * @brief 对上一帧到当前帧的IMU数据 进行预积分（有两种预积分，一种是相对于上一帧，一种是相对于上一个关键帧）
 */
void Tracking::PreintegrateIMU() {
    // Step 1: 获取两帧之间的IMU数据，组成一个集合
    // 上一帧不存在, 说明两帧之间没有IMU数据，不进行预积分，将IMU预积分状态置为true，返回
    if (!mCurrentFrame.mpPrevFrame) {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated(); // 设置当前帧已做完预积分 mbImuPreintegrated = true;
        return;
    }
    // 上一帧存在

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());

    // 没有IMU数据，不进行预积分，将IMU预积分状态置为true
    if (mlQueueImuData.size() == 0) {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }
    // 上一帧存在，且mlQueueImuData中有IMU数据

    while (true) {
        // 数据还没有时, 会等待一段时间, 直到mlQueueImuData中有IMU数据。一开始不需要等待
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            // 队列中有IMU数据
            if (!mlQueueImuData.empty()) {
                // 拿到第一个IMU数据作为起始数据
                IMU::Point *m = &mlQueueImuData.front();
                cout.precision(17);
                // IMU起始数据会比 当前帧的前一帧时间戳早, 但如果相差0.001s (或1/IMU频率)，则舍弃这个数据
                if (m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - mImuPer) {
                    mlQueueImuData.pop_front();
                }
                // 当前帧时间戳 - 最后一个IMU数据的时间戳 也不能 > 0.001
                else if (m->t < mCurrentFrame.mTimeStamp - mImuPer) {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                // 将两帧间的IMU数据放入 mvImuFromLastFrame 中, 得到后面预积分的处理数据
                else {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            // 没有IMU数据，则退出
            else {
                break;
                bSleep = true;
            }
        }
        if (bSleep)
            usleep(500);
    }

    // Step 2: 对两帧之间进行中值积分处理
    // n个IMU组数据会有 n-1 个预积分量
    const int n = mvImuFromLastFrame.size() - 1;
    if (n == 0) {
        // 只有一个IMU数据，则返回
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    // 构造IMU预处理器,并初始化标定数据
    IMU::Preintegrated *pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias, mCurrentFrame.mImuCalib);

    // 针对预积分位置的不同做不同中值积分的处理
    /**
     *  根据上面IMU帧的筛选，IMU与图像帧的时序如下：
     *  Frame---IMU0---IMU1---IMU2---IMU3---IMU4---------------IMUx---Frame---IMUx+1
     *  T_------T0-----T1-----T2-----T3-----T4-----------------Tx-----_T------Tx+1
     *  A_------A0-----A1-----A2-----A3-----A4-----------------Ax-----_T------Ax+1
     *  W_------W0-----W1-----W2-----W3-----W4-----------------Wx-----_T------Wx+1
     *  T_和_T分别表示上一图像帧和当前图像帧的时间戳，A(加速度数据)，W(陀螺仪数据)，同理
     */
    // 进行n-1个预积分量的处理
    // 遍历第一个 到 倒数第二个IMU数据
    for (int i = 0; i < n; i++) {
        float tstep;                 // 当前时刻 到 下一时刻的 时间间隔
        Eigen::Vector3f acc, angVel; // 当前时刻 到 下一时刻的 加速度、角速度
        // 第一个时刻数据 且 不是最后两个时刻的, 即IMU数据总个数需>2
        if ((i == 0) && (i < (n - 1))) {
            float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;            // 当前时刻IMU 到 下一时刻IMU的 时间间隔
            float tini = mvImuFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp; // 当前帧的上一帧 到 当前时刻IMU的 时间间隔
            // 设当前时刻IMU的加速度为 a0，下一时刻加速度a1，时间间隔tab为 t10，tini为 t0p
            // 正常情况下是为了求 上一帧 到 当前时刻IMU的一个平均加速度，但是IMU时间不会正好落在上一帧的时刻，需要做补偿；
            // 先求得 上一帧 到 a0的 加速度变化量 (a1 - a0) * (tini / tab);
            // a0 - (a1 - a0) * (tini / tab) 则为上一帧的加速度 (tini可正可负, 表示时间上的先后);
            // 其加上a1，再除以2 就为这段时间的平均加速度
            acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a - (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) * 0.5f;
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w - (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) * 0.5f;
            // 当前帧的上一帧 到 a1 的时间间隔
            tstep = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        // 中间的数据不存在帧的干扰，正常计算
        else if (i < (n - 1)) {
            acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a) * 0.5f;    // 当前时刻IMU 到 下一时刻IMU 的平均加速度
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w) * 0.5f; // 当前时刻IMU 到 下一时刻IMU 的平均角速度
            tstep = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;           // 时间间隔
        }
        // 倒数第二个IMU时刻，计算过程跟第一时刻类似，都需要考虑帧与IMU时刻的关系
        else if ((i > 0) && (i == (n - 1))) {
            float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;   // 倒数第二个时刻IMU (当前时刻) 到 最后一个时刻IMU的 时间间隔
            float tini = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;     // 倒数第二个时刻IMU 到 当前帧的 时间间隔
            float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp; // 当前帧 到  最后一个时刻IMU的 时间间隔
            acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a - (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) * 0.5f;
            angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w - (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) * 0.5f;
            tstep = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
        }
        // 第一个数据 且 是最后一个数据 (即只有一个数据), 使用其对应时刻的，这种情况应该没有吧，回头应该试试看
        else if ((i == 0) && (i == (n - 1))) {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp; // 上一帧 到 当前帧的 时间间隔
        }

        // Step 3：进行预积分计算
        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        // 相对上一关键帧的预积分计算
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);
        // 相对上一帧的预积分计算
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
    } // n-1个预积分量处理完毕

    // 记录当前预积分的图像帧 与 关键帧
    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;
    // 将当前帧的状态 mbImuPreintegrated 置为 true，表示已做完预积分
    mCurrentFrame.setIntegrated();

    // Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}

// 用 IMU 估计当前状态量：当前时刻的状态量用上一时刻状态量加上 IMU 预积分量转到 world 系下来估计
/**
 * @brief 跟踪不成功的时候，用初始化好的imu数据做跟踪处理，通过IMU预测状态
 * 两个地方用到：
 * 1. 恒速运动模型计算速度, 但并没有给当前帧位姿赋值；
 * 2. 跟踪丢失时不直接判定丢失，通过这个函数预测 当前帧位姿 看看能不能拽回来，代替纯视觉中的重定位
 *
 * @return true
 * @return false
 */
bool Tracking::PredictStateIMU() {
    if (!mCurrentFrame.mpPrevFrame) {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    // 总结下都在什么时候地图更新，也就是 mbMapUpdated 为 true
    // 1. 回环或融合
    // 2. 局部地图BA LocalBundleAdjustment
    // 3. IMU三阶段的初始化
    // 下面的代码流程一模一样，只不过计算时相对的帧不同:
    // 地图更新 会更新关键帧与MP，所以关键帧相对更准 ==> 相对于上一关键帧做
    // 而没更新的话，距离上一帧更近，计算起来误差更小 ==> 相对于上一帧做

    // 地图已更新 且 上一关键帧存在，相对于 上一关键帧 的信息
    if (mbMapUpdated && mpLastKeyFrame) {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;

        // 计算当前帧的世界位姿 Twc, 原理都是用预积分的位姿（预积分的值不会变化）与上一关键帧的位姿（会迭代变化）进行更新
        // 计算当前帧IMU的世界位姿 Twb (15-15)
        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        // 设置当前帧的位姿 Tcw = Tcb * Tbw
        mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

        std::cout << "\t\tlast_key_frame velocity: " << Vwb1.transpose() << std::endl;
        std::cout << "\t\tcur_frame velocity: " << mCurrentFrame.GetVelocity().transpose() << std::endl;

        // 记录bias
        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    // 地图未更新时，用上一普通帧的信息
    else if (!mbMapUpdated) {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        // mpImuPreintegratedFrame 是当前帧的上一帧，不一定是关键帧
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

        std::cout << "\t\tlast_frame velocity: " << Vwb1.transpose() << std::endl;
        std::cout << "\t\tcur_frame velocity: " << mCurrentFrame.GetVelocity().transpose() << std::endl;

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    } else
        cout << "not IMU prediction!!" << endl;

    return false;
}

void Tracking::ResetFrameIMU() {
    // TODO To implement...
}

/**
 * @brief 跟踪过程，包括恒速模型跟踪、参考关键帧跟踪、局部地图跟踪
 * track包含两部分：估计运动、跟踪局部地图
 *
 * Step 1：初始化
 * Step 2：跟踪
 * Step 3：记录位姿信息，用于轨迹复现
 */
void Tracking::Track() {
    // 如果处于“逐步”模式，即需要逐步执行
    if (bStepByStep) {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        while (!mbStep && bStepByStep)
            usleep(500);
        mbStep = false;
    }

    // Step 1：检查Local Mapping线程认为IMU有问题，则重置当前活跃地图，然后返回
    if (mpLocalMapper->mbBadImu) {
        cout << "TRACK: Reset map because local mapper set the bad IMU flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }

    // 获取当前活跃地图 pCurrentMap
    Map *pCurrentMap = mpAtlas->GetCurrentMap();
    // 检查该活跃地图是否存在
    if (!pCurrentMap) {
        cout << "ERROR: There is not an active map in the atlas" << endl;
    }

    // Step 2：处理时间戳异常的情况
    // 如果系统不是刚刚开始的状态，则进行一系列 时间戳 和 帧ID 的检查，以处理时间戳错误和丢失的情况，最后返回，中断当前帧的处理
    if (mState != NO_IMAGES_YET) {
        // 当前时间戳 < 上一帧时间戳，出现错误
        if (mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp) {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear(); // 清空IMU数据队列
            CreateMapInAtlas();     // 创建新的子地图
            return;
        }
        // 当前时间戳 > 上一帧时间戳 + 1秒，时间戳跳变，重置地图或新建子地图，返回
        else if (mCurrentFrame.mTimeStamp > mLastFrame.mTimeStamp + 1.0) {
            // IMU模式
            if (mpAtlas->isInertial()) {

                // IMU已初始化，说明时间戳跳变可能是由于数据丢失或错误引起的
                if (mpAtlas->isImuInitialized()) {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    // IMU未完成第3次初始化，则重置地图
                    if (!pCurrentMap->GetIniertialBA2()) {
                        mpSystem->ResetActiveMap();
                    }
                    // 若已完成，则创建新的子地图，保存当前地图
                    else {
                        CreateMapInAtlas();
                    }
                }
                // IMU未初始化，时间戳跳变是在IMU初始化之前发生的，则重置当前活动地图，返回
                else {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap();
                }

                return;
            }
        }
    }

    // Step 3：IMU模式 且 上一关键帧 存在，则将上一关键帧的IMU零偏 赋予 当前帧
    // IMU零偏：IMU传感器测量数据中的固有误差，如加速度计和陀螺仪的零偏。由于这些误差可能会对IMU数据的精度和稳定性产生影响，因此在SLAM系统中，通常会使用关键帧的IMU零偏 来校准和纠正
    // 当前帧的IMU数据，从而提高SLAM系统的性能和稳定性
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame) {
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());
    }

    // 如果系统刚刚启动，置状态为 未初始化，在后续操作中进行初始化
    if (mState == NO_IMAGES_YET) {
        mState = NOT_INITIALIZED;
    }

    // 更新系统上一帧状态。后续可以比较当前帧和上一帧的状态，判断系统状态是否发生了变化或执行特定的逻辑
    mLastProcessedState = mState;

    // Step 4：IMU预积分（IMU模式 且 未创建地图时）
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap) {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        // 对两帧间的IMU数据进行预积分。IMU数据的预积分是将IMU传感器的测量数据转化为相对于时间的姿态变化和速度变化
        PreintegrateIMU();

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu);
#endif
    }

    // 地图尚未被创建或初始化
    mbCreatedMap = false;

    // 使用互斥锁（mutex）来保护对地图的并发访问，确保同一时间只有一个线程可以访问地图。保证地图不会发生变化
    // 疑问:这样子会不会影响地图的实时更新? 回答：主要耗时在构造帧中特征点的提取和匹配部分,在那个时候地图是没有被上锁的,有足够的时间更新地图
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    // 地图未被更新
    mbMapUpdated = false;

    // 判断地图 ID 是否更新了
    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex(); // 获取当前地图的变化索引(Change Index)，用于跟踪地图是否已经发生了变化
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();     // 获取上次地图的变化索引
    // 如果当前地图的变化索引 > 上次地图的变化索引，则表明 地图发生变化，置 mbMapUpdated 为真
    if (nCurMapChangeIndex > nMapChangeIndex) {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex); // 更新上次地图变化的索引 为 当前地图的变化索引，以便在下次更新时比较
        mbMapUpdated = true;                               // 检测到地图更新了
    }

    // Step 5：初始化。若初始化成功，则状态 mState = OK
    if (mState == NOT_INITIALIZED) {
        // 双目/双目+IMU、RGB-D/RGB-D+IMU，只需1帧
        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
            Verbose::PrintMess("INITIALIZE: Frame " + std::to_string(mCurrentFrame.mnId) + "，双目或RGB-D初始化...", Verbose::VERBOSITY_DEBUG);
            StereoInitialization();
        }
        // 单目，需要至少2帧，且初始化的两张图像必须有一定程度的平移，之后的轨迹都以此平移为单位。通常选择让相机进行左右平移进行初始化，不能纯旋转
        else {
            Verbose::PrintMess("INITIALIZE: Frame " + std::to_string(mCurrentFrame.mnId) + "，单目初始化...", Verbose::VERBOSITY_DEBUG);
            MonocularInitialization();
        }
        // 初始化后的状态：-1: SYSTEM_NOT_READY, 0: NO_IMAGES_YET, 1: NOT_INITIALIZED, 2: OK, 3: RECENTLY_LOST, 4: LOST

        // mpFrameDrawer->Update(this);

        // 如果初始化失败，则将当前帧置为上一帧，直接返回
        if (mState != OK) {
            Verbose::PrintMess("\t初始化失败，下一帧继续初始化", Verbose::VERBOSITY_DEBUG);
            mLastFrame = Frame(mCurrentFrame); // 下一次迭代中继续初始化
            return;
        }

        Verbose::PrintMess("\t初始化成功，下一帧开始跟踪", Verbose::VERBOSITY_DEBUG);

        // 如果当前地图是第一个地图，记录当前帧 ID 为第一帧 mnFirstFrameId
        if (mpAtlas->GetAllMaps().size() == 1) {
            mnFirstFrameId = mCurrentFrame.mnId; // 在跟踪过程中，当需要判断是否需要创建新的关键帧时，可以将当前帧的ID与 mnFirstFrameId 进行比较，以确定是否需要创建新的关键帧。
        }
    }
    // Step 6：系统已完成初始化，则跟踪当前帧
    else {
        //----------------开始第一步跟踪，帧匹配，初步估计相机的位姿---------------------
        bool bOK; // 当前帧跟踪状态

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now(); // 位姿估计开始时间
#endif

        // mbOnlyTracking = false 表示SLAM模式（定位 + 建图），mbOnlyTracking = true表示纯定位模式
        // tracking 类构造时默认为 false。在 viewer 中有个开关 ActivateLocalizationMode，可以控制是否开启 mbOnlyTracking
        if (!mbOnlyTracking) {
            Verbose::PrintMess("   SLAM模式", Verbose::VERBOSITY_VERBOSE);

            // 跟踪状态正常
            if (mState == OK) {
                // Step 6.1 局部建图线程可能会对 上一帧跟踪到的地图点更改，需检查并更新
                CheckReplacedInLastFrame();

                int num_landmark = std::count_if(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), [](MapPoint *pMp) { return pMp != static_cast<MapPoint *>(NULL); });
                Verbose::PrintMess("   [vio] cur_frm landmark size: " + std::to_string(num_landmark), Verbose::VERBOSITY_DEBUG);

                // Step 6.2 速度无效 且 IMU未初始化 或 重定位后的1帧，则参考关键帧跟踪；否则恒速运行模型跟踪
                // 条件1：速度无效 且 IMU未初始化，说明是时第一帧的跟踪，或前面已经跟丢了；
                // 条件2：当前帧是重定位后的1帧，则用重定位成功的那帧来恢复位姿
                if ((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId < mnLastRelocFrameId + 2) { // mnLastRelocFrameId 上一次重定位的那一帧
                    Verbose::PrintMess("\t参考关键帧跟踪...", Verbose::VERBOSITY_VERBOSE);
                    bOK = TrackReferenceKeyFrame();
                }
                // 有速度 或 IMU已初始化，则先使用 恒速运动模型进行跟踪；如果跟踪失败，再跟踪参考关键帧
                else {
                    Verbose::PrintMess("\t恒速运动模型跟踪...", Verbose::VERBOSITY_VERBOSE);
                    // 使用恒速模型和上一帧进行匹配。所谓的恒速就是假设上上帧到上一帧的位姿 = 上一帧的位姿到当前帧位姿
                    // 根据恒速模型设定当前帧的初始位姿，通过投影的方式在上一帧中找到当前帧特征点的匹配点，优化每个特征点所对应3D点的投影误差即可得到优化后的位姿
                    bOK = TrackWithMotionModel();

                    if (!bOK) {
                        Verbose::PrintMess("\t恒速运动模型跟踪失败，开始参考关键帧跟踪...", Verbose::VERBOSITY_VERBOSE);
                        bOK = TrackReferenceKeyFrame();
                    }
                }

                // 第一阶段跟踪失败：踪参考关键帧、恒速模型跟踪都失败，根据情况标记为 RECENTLY_LOST 或 LOST
                if (!bOK) {
                    // IMU模式 且 当前帧 距离 上次重定位成功 <= 1s，标记为LOST
                    if (mCurrentFrame.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)) {
                        mState = LOST;
                    }
                    // (非IMU模式 或 当前帧 距离上次重定位帧 > 1s) 且 当前地图中关键帧个数 > 10，则状态标记为 RECENTLY_LOST，表明在短时间内跟丢，后面会结合IMU预测的位姿看看能不能拽回来
                    else if (pCurrentMap->KeyFramesInMap() > 10) {
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp; // 记录丢失时间
                    } else {
                        mState = LOST;
                    }
                    Verbose::PrintMess("\tFailed: 帧匹配失败，mState = " + std::to_string(mState) + ", bOK = " + std::to_string(bOK), Verbose::VERBOSITY_VERBOSE);
                }
                // 第一阶段跟踪成功
                else {
                    Verbose::PrintMess("\tSucceed: 帧匹配成功，mState = " + std::to_string(mState) + ", bOK = " + std::to_string(bOK), Verbose::VERBOSITY_DEBUG);
                }
            }
            //  跟踪状态不正常，mState != OK
            else {
                // RECENTLY_LOST状态，再挣扎一下，用IMU估计位姿 或 使用重定位
                if (mState == RECENTLY_LOST) {
                    Verbose::PrintMess("\t上一帧短时间内丢失，RECENTLY_LOST", Verbose::VERBOSITY_VERBOSE);
                    bOK = true; // 先置为true

                    // IMU模式，可以用IMU估计位姿，看能否拽回来
                    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)) {
                        Verbose::PrintMess("\t处于IMU模式", Verbose::VERBOSITY_VERBOSE);
                        // Step 6.4 IMU已初始化，则估计
                        if (pCurrentMap->isImuInitialized()) {
                            Verbose::PrintMess("\t\tIMU已初始化，使用IMU数据预测相机状态...", Verbose::VERBOSITY_VERBOSE);
                            PredictStateIMU();
                        }
                        // IMU未初始化，则不能估计，置为false
                        else {
                            Verbose::PrintMess("\t\tIMU未初始化，bOK = " + std::to_string(bOK), Verbose::VERBOSITY_VERBOSE);
                            bOK = false;
                        }

                        // IMU模式 且 当前帧 距离 跟丢帧>5s，还没有找回，则放弃，状态 mState = LOST，跟踪失败 bOK = false
                        if (mCurrentFrame.mTimeStamp - mTimeStampLost > time_recently_lost) {
                            mState = LOST;
                            bOK = false;
                            Verbose::PrintMess("\tIMU模式下，当前帧 距离 跟丢帧超过5s，还没有找回，跟踪丢失，Track Lost... mState = " + std::to_string(mState) + ", bOK = " + std::to_string(bOK),
                                               Verbose::VERBOSITY_NORMAL);
                        }
                    }
                    // Step 6.5 纯视觉，则重定位。主要是BOW搜索，EPnP求解位姿
                    else {
                        Verbose::PrintMess("\t\t纯视觉模式，进行重定位...", Verbose::VERBOSITY_VERBOSE);
                        bOK = Relocalization();
                        // std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        // std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        Verbose::PrintMess("\t\t重定位后状态，mState = " + std::to_string(mState) + ", bOK = " + std::to_string(bOK), Verbose::VERBOSITY_NORMAL);

                        // 如果重定位失败 且 当前帧时间戳 - 丢失时间 > 3s，则状态 mState = LOST，跟踪失败 bOK = false
                        if (!bOK && mCurrentFrame.mTimeStamp - mTimeStampLost > 3.0f) {
                            mState = LOST;
                            Verbose::PrintMess("\t\tFailed: 重定位一直失败，且跟踪丢失超过 3s，Track Lost... mState = " + std::to_string(mState) + ", bOK = " + std::to_string(bOK),
                                               Verbose::VERBOSITY_NORMAL);
                            bOK = false;
                        }
                    }
                }
                // Step 6.6 LOST状态，说明上面的操作失败，且长时间丢失，则重置活跃地图或创建新的子地图，并清空最后一个关键帧，然后返回
                else if (mState == LOST) {
                    Verbose::PrintMess("\t上一帧长时间跟踪丢失，LOST", Verbose::VERBOSITY_VERBOSE);
                    Verbose::PrintMess("\t\t创建一个新地图，A new map is started...", Verbose::VERBOSITY_NORMAL);
                    // 当前地图中关键帧个数 < 10，则重置当前地图
                    if (pCurrentMap->KeyFramesInMap() < 10) {
                        mpSystem->ResetActiveMap();
                        Verbose::PrintMess("\t\t地图中关键帧数量: " + std::to_string(pCurrentMap->KeyFramesInMap()) + " < 10，重置当前地图，Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    }
                    // 地图中关键帧数量 >= 10，创建一个新地图
                    else {
                        CreateMapInAtlas();
                        Verbose::PrintMess("\t\t地图中关键帧数量: " + std::to_string(pCurrentMap->KeyFramesInMap()) + " >= 10，创建新地图...", Verbose::VERBOSITY_NORMAL);
                    }

                    // 如果存在最后一个关键帧，则将其置NULL
                    if (mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

                    Verbose::PrintMess("\tdone", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            }
        }
        // 纯定位模式，只进行跟踪，不建图
        else {
            Verbose::PrintMess("   纯定位模式", Verbose::VERBOSITY_VERBOSE);
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            // Step 6.1 LOST状态，则进行重定位
            if (mState == LOST) {
                // 如果使用IMU，打印 IMU状态下丢失
                if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);

                bOK = Relocalization();
            }
            // 正常跟踪 OK 或 短时间内丢失 RECENTLY_LOST
            else {
                // mbVO 是 mbOnlyTracking = true时的才有的一个变量
                // mbVO = false 表示此帧匹配了很多的MapPoints，跟踪很正常 (注意有点反直觉)
                // mbVO = true  表明此帧匹配了很少的MapPoints，少于10个，要跟丢
                // Step 6.2 如果跟踪状态正常，使用恒速模型 或 参考关键帧跟踪
                if (!mbVO) {
                    // In last frame we tracked enough MapPoints in the map
                    // 运动模型有效，优先使用恒速模型跟踪
                    if (mbVelocity) {
                        bOK = TrackWithMotionModel();
                    }
                    // 运动模型不可用，那只能够通过参考关键帧来跟踪
                    else {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                // 当前帧匹配了很少（小于10）的地图点，可能要跟丢，既做跟踪 又做重定位
                else {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // 计算两个相机姿态：一个来自 运动模型，另一个来自 重定位
                    // If relocalization is sucessfull we choose that solution, otherwise we retain the "visual odometry" solution.
                    // 如果重定位成功，则选择重新定位的解决方案，否则保留视觉里程计的解决方案

                    bool bOKMM = false;         // 运动模型跟踪是否成功
                    bool bOKReloc = false;      // 重定位是否成功
                    vector<MapPoint *> vpMPsMM; // 运动模型中构造的 地图点
                    vector<bool> vbOutMM;       // 在追踪运动模型后发现的 外点
                    Sophus::SE3f TcwMM;         // 运动模型得到的 位姿

                    // Step 6.3 当运动模型有效时,根据运动模型计算位姿
                    if (mbVelocity) {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.GetPose();
                    }
                    // Step 6.4 使用重定位的方法来得到当前帧的位姿
                    bOKReloc = Relocalization();

                    // Step 6.5 根据恒速模型、重定位结果来更新状态
                    // 重定位失败 且 运动模型跟踪成功，则将运动模型计算得到的相机位姿 赋给 当前帧，保存地图点和离群点。
                    if (bOKMM && !bOKReloc) {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        // 如果是 VO 视觉里程计模式下，对不被标记为离群点的特征点增加它们被观测的次数，以提高它们在后续的操作中的重要性，有助于保持在VO模式下对这些特征点的跟踪
                        // 如果当前帧匹配的3D点很少，增加当前可视地图点的被观测次数
                        if (mbVO) {
                            // 遍历当前帧中的每个特征点
                            for (int i = 0; i < mCurrentFrame.N; i++) {
                                // 如果当前特征点存在 且 不是离群点
                                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound(); // 增加当前特征点的“被观测次数”，目的是增加特征点的权重，以提高它们在后续的操作中的重要性
                                }
                            }
                        }
                    }
                    // 如果重定位成功，VO模式标志设为false
                    // 只要重定位成功，整个跟踪过程正常进行（重定位与跟踪，更相信重定位）
                    else if (bOKReloc) {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM; // 更新跟踪成功与否
                }
            }
        } // 帧匹配完毕

        // 将最新的关键帧作为当前帧的参考关键帧
        // mpReferenceKF 先是上一时刻的参考关键帧，如果当前为新关键帧则变成当前关键帧，如果不是新的关键帧则先为上一帧的参考关键帧，而后经过更新局部关键帧重新确定
        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now(); // 位姿估计结束时间

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndPosePred - time_StartPosePred).count(); // 位姿估计耗时
        vdPosePred_ms.push_back(timePosePred);
#endif

        // --------------------开始第二步，与局部地图匹配---------------------
        // Step 7: 在 帧匹配 得到当前帧初始姿态后，现在进行局部地图跟踪得到更多的匹配，并优化当前位姿
        // 前面主要是两两跟踪（恒速模型跟踪上一帧 / 跟踪参考帧）跟踪一帧得到初始位姿，这里搜索局部关键帧后搜集所有 局部地图点，然后将局部 MapPoints 和 当前帧进行投影匹配，得到更多匹配的
        // MapPoints后进行 Pose优化 局部的地图: 当前帧、当前帧的 MapPoints、当前关键帧与其它关键帧共视关系
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now(); // 局部地图跟踪开始时间
#endif
        // 非 纯定位模式
        if (!mbOnlyTracking) {
            int num_landmark = std::count_if(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), [](MapPoint *pMp) { return pMp != static_cast<MapPoint *>(NULL); });
            Verbose::PrintMess("   [vio] after track_current_frame landmark size: " + std::to_string(num_landmark), Verbose::VERBOSITY_DEBUG);
            // 帧匹配成功，则继续 局部地图匹配
            if (bOK) {
                Verbose::PrintMess("\t局部地图跟踪...", Verbose::VERBOSITY_DEBUG);

                bOK = TrackLocalMap();

                if (!bOK) {
                    Verbose::PrintMess("\tFailed: 局部地图跟踪失败！！ bOK = " + std::to_string(bOK), Verbose::VERBOSITY_VERBOSE);
                } else {
                    Verbose::PrintMess("\tSucceed: 局部地图跟踪成功！", Verbose::VERBOSITY_VERBOSE);
                }
            } else {
                Verbose::PrintMess("\tFailed: 帧匹配失败，不进行局部地图跟踪", Verbose::VERBOSITY_VERBOSE);
            }
        }
        // 纯定位模式
        else {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            // 第一步跟踪成功 且 不是纯视觉里程计模式，则跟踪局部地图中的特征点
            if (bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        // 到此为止跟踪确定位姿阶段结束，下面开始做收尾工作和为下一帧做准备

        // 查看到此为止时的两个状态变化
        // bOK的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---true                     -->OK   1 跟踪局部地图成功
        //          \               \              \---局部地图跟踪失败---false
        //           \               \---当前帧跟踪失败---false
        //            \---上一帧跟踪失败---重定位成功---局部地图跟踪成功---true                       -->OK  2 重定位
        //                          \           \---局部地图跟踪失败---false
        //                           \---重定位失败---false

        //
        // mState的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---OK                  -->OK  1 跟踪局部地图成功
        //            \               \              \---局部地图跟踪失败---OK                  -->OK  3 正常跟踪
        //             \               \---当前帧跟踪失败---非OK
        //              \---上一帧跟踪失败---重定位成功---局部地图跟踪成功---非OK
        //                            \           \---局部地图跟踪失败---非OK
        //                             \---重定位失败---非OK（传不到这里，因为直接return了）
        // 由上图可知当前帧的状态OK的条件是跟踪局部地图成功，重定位或正常跟踪都可

        // Step 8：根据上面的操作来判断是否追踪成功
        // 如果两步都跟踪成功，则状态 mState = OK
        if (bOK) {
            Verbose::PrintMess("\t跟踪成功！状态置为 OK", Verbose::VERBOSITY_NORMAL);
            mState = OK;
        }
        // 如果第一步帧匹配成功 但 第二步局部地图跟踪失败，则将状态置为 RECENTLY_LOST
        else if (mState == OK) {
            Verbose::PrintMess("\t帧跟踪成功 但 局部地图跟踪失败，状态置为 RECENTLY_LOST", Verbose::VERBOSITY_NORMAL);
            // IMU模式下
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
                // IMU未初始化 或 没有完成惯性BA优化，则重置活跃地图
                if (!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2()) {
                    cout << "\t\tIMU模式下IMU未初始化 或 没有完成惯性BA优化，则重置当前活跃地图..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState = RECENTLY_LOST;
            }
            // 如果未使用IMU，状态直接置为 RECENTLY_LOST
            else {
                mState = RECENTLY_LOST; // visual to lost
            }

            /*if(mCurrentFrame.mnId > mnLastRelocFrameId + mMaxFrames)
            {*/
            mTimeStampLost = mCurrentFrame.mTimeStamp; // 记录丢失时间戳
            //}
        }
        Verbose::PrintMess("\t更新状态后，mState = " + std::to_string(mState) + ", bOK = " + std::to_string(bOK), Verbose::VERBOSITY_DEBUG);

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it should be once mCurrFrame is completely modified)
        // 如果进行了重定位，将当前帧保存起来，用于之后的IMU重置
        // IMU重置时，可能会用到之前的帧信息。所以保存一分当前帧的副本，确保之后对当前帧的修改，不影响到IMU的重置
        // 当前帧 距离 最近的一次重定位 在1s内   且 当前帧的ID > 20帧 且 IMU模式 且 IMU已初始化
        if ((mCurrentFrame.mnId < (mnLastRelocFrameId + mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
            (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized()) {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame *pF = new Frame(mCurrentFrame);    // 创建一个当前帧的副本
            pF->mpPrevFrame = new Frame(mLastFrame); // 创建上一帧的副本，并设置为 当前帧副本的 前一帧

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame); // 创建一个当前帧的IMU预积分对象的副本
        }

        // 如果当前地图的IMU已初始化，且跟踪成功，则重置IMU或者更新偏置
        if (pCurrentMap->isImuInitialized()) {
            // 跟踪成功
            if (bOK) {
                // 当前帧 距离 上次重定位帧 刚好等于1s，重置
                // 如果 当前帧ID = 重定位帧ID + IMU重置帧数阈值
                if (mCurrentFrame.mnId == (mnLastRelocFrameId + mnFramesToResetIMU)) {
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU(); // 重置帧的IMU信息
                }
                // 如果 当前帧ID > 重定位帧ID + 30
                else if (mCurrentFrame.mnId > (mnLastRelocFrameId + 30)) {
                    mLastBias = mCurrentFrame.mImuBias; // 更新mLastBias为 当前帧的IMU偏置
                }
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now(); // 局部地图跟踪结束时间

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndLMTrack - time_StartLMTrack).count(); // 局部地图跟踪耗时
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        // --------------------更新绘制器、状态，执行关键帧插入的判定---------------------
        // 更新显示线程中的图像、特征点、地图点等信息
        // 更新帧绘制器，将当前帧的信息传递给帧绘制器
        mpFrameDrawer->Update(this);
        // 更新地图绘制器：如果当前帧位姿已设置，则将其传递给地图绘制器
        if (mCurrentFrame.isSet())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        // 查看到此为止时的两个状态变化
        // bOK的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---true
        //          \               \              \---局部地图跟踪失败---false
        //           \               \---当前帧跟踪失败---false
        //            \---上一帧跟踪失败---重定位成功---局部地图跟踪成功---true
        //                          \           \---局部地图跟踪失败---false
        //                           \---重定位失败---false

        // mState的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---OK
        //            \               \              \---局部地图跟踪失败---非OK（IMU时为RECENTLY_LOST）
        //             \               \---当前帧跟踪失败---非OK(地图超过10个关键帧时 RECENTLY_LOST)
        //              \---上一帧跟踪失败(RECENTLY_LOST)---重定位成功---局部地图跟踪成功---OK
        //               \                           \           \---局部地图跟踪失败---LOST
        //                \                           \---重定位失败---LOST（传不到这里，因为直接return了）
        //                 \--上一帧跟踪失败(LOST)--LOST（传不到这里，因为直接return了）

        // Step 9：如果跟踪成功 或 最近刚刚跟丢，更新速度，清除无效地图点，按需创建关键帧
        if (bOK || mState == RECENTLY_LOST) {
            // Update motion model
            // Step 9.1：更新恒速运动模型 TrackWithMotionModel 中的 mVelocity
            // 如果上一帧和当前帧的位姿都已设置，则计算速度并置速度标志为真；否则为假
            if (mLastFrame.isSet() && mCurrentFrame.isSet()) {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse(); // 上一帧位姿的 逆变换
                // mVelocity = Tcl = Tcw * Twl
                mVelocity = mCurrentFrame.GetPose() * LastTwc; // 上一帧到当前帧的位姿变换，即速度
                mbVelocity = true;                             // 速度标志设置为真
            }
            // 否则没有速度
            else {
                mbVelocity = false;
            }

            // 使用IMU积分的位姿显示
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose()); // 将当前帧的相机位姿信息传递给地图绘制器

            // Clean VO matches 清理视觉里程计（VO）匹配
            // Step 9.2：清除观测不到的地图点
            for (int i = 0; i < mCurrentFrame.N; i++) // 遍历当前帧中的每个特征点
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i]; // 当前帧的第i个特征点对应的 地图点
                if (pMP)
                    if (pMP->Observations() < 1) // 特征点存在且观测次数小于1
                    {
                        mCurrentFrame.mvbOutlier[i] = false;                           // 标记为不是异常点
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL); // 将该特征点从当前帧中移除
                    }
            }

            // Delete temporal MapPoints 清理临时地图点
            // Step 9.3：清除恒速模型跟踪中 UpdateLastFrame中为当前帧临时添加的MapPoints（仅双目和rgbd）
            // 上个步骤中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
            // 临时地图点仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
            for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(),
                                            lend = mlpTemporalPoints.end();
                 lit != lend; lit++) // 对于临时地图点列表中的每个地图点，删除该地图点
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
            // 不能够直接执行这个是因为其中存储的都是指针,之前的操作都是为了避免内存泄露
            mlpTemporalPoints.clear(); // 清空临时地图点列表

            //----------------是否要插入关键帧---------------------
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now(); // 插入关键帧开始时间
#endif
            // 检查是否需要插入新的关键帧
            bool bNeedKF = NeedNewKeyFrame();

            // Check if we need to insert a new keyframe
            // if(bNeedKF && bOK)
            // Step 9.4：根据条件来判断是否插入关键帧
            // 需要同时满足下面条件1和2
            // 条件1：bNeedKF = true，需要插入新关键帧
            // 条件2：bOK = true 跟踪成功 或 (IMU模式下的 RECENTLY_LOST状态 且 mInsertKFsLost = true)
            if (bNeedKF && (bOK || (mInsertKFsLost && mState == RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)))) {
                CreateNewKeyFrame(); // 创建关键帧，对于双目或RGB-D会产生新的地图点
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now(); // 插入关键帧结束时间

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndNewKF - time_StartNewKF).count(); // 插入关键帧耗时
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position with those points so we discard them in the frame. Only has effect if lastframe is tracked
            // 作者这里说允许在BA中被Huber核函数判断为外点的传入新的关键帧中，让后续的BA来审判他们是不是真正的外点
            // 但是估计下一帧位姿的时候我们不想用这些外点，所以删掉

            // Step 9.5 删除那些在BA中检测为外点的地图点
            for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i]) // 如果特征点存在且被标记为异常点，该特征点从当前帧中移除
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // Step 10：如果跟踪失败，跟踪状态为LOST，则重置地图并新建地图
        if (mState == LOST) {
            // 当前地图中的关键帧数量 <= 10，则重置当前地图，清除地图中的数据，然后返回，退出当前跟踪
            if (pCurrentMap->KeyFramesInMap() <= 10) {
                Verbose::PrintMess("\t长时间跟踪失败，当前地图中的关键帧数量 = " + std::to_string(pCurrentMap->KeyFramesInMap()) + " <= 10，开始重置当前地图...", Verbose::VERBOSITY_NORMAL);
                mpSystem->ResetActiveMap();
                Verbose::PrintMess("\t退出当前跟踪", Verbose::VERBOSITY_NORMAL);
                return;
            }

            // 如果是IMU模式
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
                // 如果IMU未初始化，则重置活动地图，然后返回
                if (!pCurrentMap->isImuInitialized()) {
                    Verbose::PrintMess("\tIMU模式 且 未进行IMU初始化，重置当前地图，Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    Verbose::PrintMess("\t退出当前跟踪", Verbose::VERBOSITY_NORMAL);
                    return;
                }
            }

            // 如果地图中关键帧数 > 10 且 纯视觉模式(没使用IMU) 或 虽是IMU模式但已完成IMU初始化，则在Atlas中保存当前地图，创建新地图，所有跟状态相关的变量全部重置(包括mState =
            // NO_IMAGES_YET，这使得系统会重新初始化)
            Verbose::PrintMess("\t地图中关键帧数 KeyFramesInMap = " + std::to_string(pCurrentMap->KeyFramesInMap()) +
                                   " > 10 且 纯视觉模式 或 虽是IMU模式但已完成IMU初始化，保存当前地图，创建新的地图...",
                               Verbose::VERBOSITY_QUIET);
            CreateMapInAtlas();
            Verbose::PrintMess("\t退出当前跟踪", Verbose::VERBOSITY_NORMAL);
            return;
        }

        // 确保已经设置了参考关键帧
        // 如果当前帧没有参考关键帧，则将 预先存储的参考关键帧指针 作为 当前帧的参考关键帧
        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame); // 将当前帧更新为上一帧，以便后续跟踪
    }                                      // 所有跟踪流程结束

    // 查看到此为止
    // mState的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---OK
    //            \               \              \---局部地图跟踪失败---非OK（IMU时为RECENTLY_LOST）
    //             \               \---当前帧跟踪失败---非OK(地图超过10个关键帧时 RECENTLY_LOST)
    //              \---上一帧跟踪失败(RECENTLY_LOST)---重定位成功---局部地图跟踪成功---OK
    //               \                           \           \---局部地图跟踪失败---LOST
    //                \                           \---重定位失败---LOST（传不到这里，因为直接return了）
    //                 \--上一帧跟踪失败(LOST)--LOST（传不到这里，因为直接return了）

    // ------------------Step 11：记录位姿信息，用于最后保存所有的轨迹------------------
    // 如果跟踪正常 或 最近跟踪丢失，则存储帧的位姿，以便在后续可以检索完整的相机轨迹
    if (mState == OK || mState == RECENTLY_LOST) {
        // 如果当前帧位姿已设置（在参考关键帧跟踪特征点匹配数>15，恒速运动模型，重定位mlpnp求解位姿后 会设置当前帧位姿）
        if (mCurrentFrame.isSet()) {
            // 计算 当前帧相对于参考关键帧的 相对位姿 = 当前帧位姿 * 参考关键帧的逆位姿：Tcr = Tcw * Twr。    其中 Twr = Trw^-1
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF); // 当前帧的参考关键帧
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST); // 将 当前帧是否丢失 的布尔值添加到状态丢失列表中
            Verbose::PrintMess("\t\tmState = " + std::to_string(mState) + "，当前帧位姿已设置，存储当前帧位姿", Verbose::VERBOSITY_DEBUG);
        }
        // 跟踪失败，则存储为上一帧的相对位姿
        else {
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back()); // 存储 上一帧的相对位姿
            mlpReferences.push_back(mlpReferences.back());               // 存储 上一帧的参考关键帧
            mlFrameTimes.push_back(mlFrameTimes.back());                 // 存储 上一帧的时间戳
            mlbLost.push_back(mState == LOST);                           // 存储 上一帧的丢失状态
            Verbose::PrintMess("\t\tmState = " + std::to_string(mState) + "，当前帧位姿未设置，存储为上一帧的位姿", Verbose::VERBOSITY_DEBUG);
        }
    }

    // 在注册循环的情况下进行控制和同步
#ifdef REGISTER_LOOP
    if (Stop()) {
        // Safe area to stop
        while (isStopped()) {
            usleep(3000); // 3000 微秒 = 3 毫秒
        }
    }
#endif
}

/**
 * @brief 双目 和 RGBD 的地图初始化，比单目简单很多
 *
 * 由于具有深度信息，直接生成 MapPoints
 */
void Tracking::StereoInitialization() {
    // 初始化要求当前帧的特征点 > 500
    if (mCurrentFrame.N > 500) {
        // IMU模式
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated) {
                cout << "not IMU meas" << endl;
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA - mLastFrame.mpImuPreintegratedFrame->avgA).norm() < 0.5) {
                cout << "not enough acceleration" << endl;
                return;
            }

            if (mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        // IMU模式下设置的是相机位姿
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        // 非IMU模式，设置初始位姿为单位旋转，0平移
        else
            mCurrentFrame.SetPose(Sophus::SE3f());

        // Create KeyFrame
        // 将当前帧构造为 初始关键帧
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        // Insert KeyFrame in the map
        // 在地图中添加该初始关键帧
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        // 配置文件中没有 mpCamera2（相机类型为PinHole，双目立体匹配）
        if (!mpCamera2) {
            // 为每个特征点构造MapPoint
            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i]; // 特征点的深度
                // 只有具有正深度的点才会被构造地图点
                if (z > 0) {
                    // 通过反投影得到该特征点的世界坐标系下3D坐标
                    Eigen::Vector3f x3D;
                    // TODO
                    mCurrentFrame.UnprojectStereo(i, x3D); // 根据该特征点的深度（特征点在左右目图像中匹配后才会计算出深度），恢复出地图点在世界坐标系下的坐标
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    // 为该MapPoint添加属性：
                    // a.观测到该MapPoint的关键帧
                    // b.该MapPoint的描述子
                    // c.该MapPoint的平均观测方向和深度范围
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
            }
        }
        // 配置文件中有 mpCamera2（相机类型为KannalaBrandt8）
        else {
            // 遍历左目鱼眼的特征点
            for (int i = 0; i < mCurrentFrame.Nleft; i++) {
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if (rightIndex != -1) {
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini, i);
                    pNewMP->AddObservation(pKFini, rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP, i);
                    pKFini->AddMapPoint(pNewMP, rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft] = pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        // cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;
        // 在局部地图中添加该初始关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);

        // 更新当前帧为上一帧
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        // mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);            // 局部关键帧中添加 当前帧
        mvpLocalMapPoints = mpAtlas->GetAllMapPoints(); // 局部地图点中添加 当前帧的地图点
        mpReferenceKF = pKFini;                         // 设置当前帧为 参考关键帧
        mCurrentFrame.mpReferenceKF = pKFini;           // 设置当前帧为 当前帧的参考关键帧

        // 把当前（最新的）局部MapPoints 作为 ReferenceMapPoints
        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        // 追踪成功
        mState = OK;
    }
}

/**
 * @brief 单目模式的地图初始化
 *
 * 并行地计算基础矩阵 E 和单应性矩阵 H，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始 MapPoints
 *
 * Step 1：（未创建）得到用于初始化的第一帧，初始化需要两帧
 * Step 2：（已创建）如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
 * Step 3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
 * Step 4：如果初始化的两帧之间的匹配点太少，重新初始化
 * Step 5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
 * Step 6：删除那些无法进行三角化的匹配点
 * Step 7：将三角化得到的3D点包装成MapPoints
 */
void Tracking::MonocularInitialization() {
    // Step 1：如果单目初始器还没有被创建，则进行创建。后面如果重新初始化时会清掉这个标志，会置false
    if (!mbReadyToInitializate) {
        // Set Reference Frame
        // 初始帧的特征点数必须 > 100
        if (mCurrentFrame.mvKeys.size() > 100) {
            Verbose::PrintMess("\t\t\t\t当前帧的特征点数 > 100，可作为初始参考帧，开始构造初始化器，进行初始化配置相关操作", Verbose::VERBOSITY_DEBUG);
            // 初始化需要两帧，分别是 mInitialFrame，mCurrentFrame
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame); // 用当前帧更新上一帧

            // mvbPrevMatched：初始化第一帧的 去畸变后的特征点的 坐标
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            // 初始化为-1 表示没有任何匹配。这里面存储的是 匹配的点的ID
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            // 如果 IMU 模式，进行初始化预积分
            if (mSensor == System::IMU_MONOCULAR) {
                if (mpImuPreintegratedFromLastKF) {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            }

            // 下一帧准备做单目初始化了
            mbReadyToInitializate = true;
            Verbose::PrintMess("\t\t\t\t构造完毕，下一帧将与当前帧匹配，进行初始化", Verbose::VERBOSITY_DEBUG);
            return;
        } else
            Verbose::PrintMess("\t\t\t\t当前帧的特征点数 <= 100，不可作为初始参考帧，下一帧继续判断", Verbose::VERBOSITY_DEBUG);
    }
    // 第二帧来了
    else {
        // Step 2：如果当前帧特征点数太少(<= 100），则重新构造初始器
        if (((int)mCurrentFrame.mvKeys.size() <= 100) || ((mSensor == System::IMU_MONOCULAR) && (mLastFrame.mTimeStamp - mInitialFrame.mTimeStamp > 1.0))) {
            mbReadyToInitializate = false;
            Verbose::PrintMess("\t\t\t\t当前帧的特征点数 = " + std::to_string((int)mCurrentFrame.mvKeys.size()) + " <= 100，太少，删除初始化器，下一帧重新构造", Verbose::VERBOSITY_DEBUG);
            return;
        }

        Verbose::PrintMess("\t\t\t\t已创建初始化器，且当前帧的特征点数 = " + std::to_string((int)mCurrentFrame.mvKeys.size()) + " > 100，当前帧 与 初始参考帧ID：" +
                               std::to_string(mInitialFrame.mnId) + "进行特征匹配",
                           Verbose::VERBOSITY_DEBUG);
        // Find correspondences
        // Step 3. ----------- 特征点匹配：在 mInitialFrame 与 mCurrentFrame 中找匹配的 特征点对 -------------
        // Step 3.1 构建匹配器
        ORBmatcher matcher(0.9, true); // 0.9 表示最佳的和次佳特征点评分的比值阈值，这里是比较宽松的，跟踪时一般是0.7。 true 表示检查特征点的方向

        // Step 3.2 特征点匹配，返回匹配成功点的个数 nmatches
        int nmatches = matcher.SearchForInitialization(
            mInitialFrame, mCurrentFrame,
            mvbPrevMatched, // 初始化前 存储的是 参考帧mInitialFrame(初始化第一帧)的特征点坐标，匹配后 存储的是 与参考帧匹配好的 当前帧mCurrentFrame的 特征点坐标
            mvIniMatches, // 参考帧 mInitialFrame 中特征点与 mCurrentFrame中特征点的匹配关系。index：是 mInitialFrame 对应特征点ID，value：匹配上则为 mCurrentFrame 的特征点ID；未匹配上则为 -1
            100);

        // Check if there are enough correspondences
        // Step 4：验证匹配结果，如果初始化的两帧之间的匹配点太少，重新初始化
        if (nmatches < 100) {
            mbReadyToInitializate = false;
            Verbose::PrintMess("\t\t\t\t当前帧与初始参考帧 匹配的特征点数 nmatches = " + std::to_string(nmatches) + " < 100，太少，删除初始化器，下一帧重新构造", Verbose::VERBOSITY_DEBUG);
            return;
        }

        Verbose::PrintMess("\t\t\t\t当前帧与初始参考帧 匹配的特征点数 nmatches = " + std::to_string(nmatches) + " >= 100，初始参考帧 " + std::to_string(mInitialFrame.mnId) +
                               " 作为世界坐标系，其变换矩阵为单位矩阵。开始求解当前帧的姿态",
                           Verbose::VERBOSITY_DEBUG);

        //        Verbose::PrintMess("\t\t\t\t当前帧与上一帧 匹配的特征点数 nmatches = " + std::to_string(nmatches) + " >= 100，通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始 MapPoints...",
        //        Verbose::VERBOSITY_DEBUG);
        // Step 5. ------------- 求解当前帧位姿，且三角化：通过H或F矩阵，得到两帧间相对运动。通过两帧的三角化，生成初始地图点 MapPoints ---------------
        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)，存储 匹配点 能否被三角化

        // 根据相机类型，进入不同的重建三维点的方法。当前mpCamera 为 Pinhole，因此进入 Pinhole::ReconstructWithTwoViews()函数
        // mbMonoInitail 是否初始化成功
        bool mbMonoInitail = mpCamera->ReconstructWithTwoViews(
            mInitialFrame.mvKeysUn, // 输入：初始参考帧的特征点
            mCurrentFrame.mvKeysUn, // 输入：当前帧的特征点
            mvIniMatches, // 输入：两帧之间特征点的匹配关系，index 保存的是 参考帧mInitialFrame 中特征点索引；如果匹配上value 保存的是匹配好的 当前帧mCurrentFrame 的特征点索引，未匹配上则为-1
            Tcw,          // 输出：当前相机位姿的变换矩阵
            mvIniP3D,     // 输出：恢复出的三维点的三维坐标
            vbTriangulated); // 输出：匹配点是否可以被三角化成功
        if (mbMonoInitail) {
            // Step 6. 删除无法进行三角化的匹配点
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }
            Verbose::PrintMess("\t\t\t\t单目初始化，三角化成功！获取到当前相机的位姿、三维空间点，删除无法进行三角化后的匹配点数 nmatches = " + std::to_string(nmatches) + "，开始创建初始地图点",
                               Verbose::VERBOSITY_DEBUG);

            // Set Frame Poses
            // Step 7. 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(Sophus::SE3f());
            // 由 Rcw 和 tcw 构造 Tcw，并赋值给 mTcw，mTcw为 初始化第一帧到第二帧的变换矩阵，即世界坐标系到相机坐标系的变换矩阵
            mCurrentFrame.SetPose(Tcw);

            // Step 8：创建初始化地图点 MapPoints
            // CreateInitialMapMonocular 将3D点包装成 MapPoint类型 存入 KeyFrame 和 Map 中
            CreateInitialMapMonocular(); // 这里面会将 mState 置 OK
        } else
            Verbose::PrintMess("\t\t\t\t单目初始化 失败！！", Verbose::VERBOSITY_DEBUG);
    }
}

/**
 * @brief 单目相机成功初始化后，用三角化得到的三维点 生成地图点 MapPoints(3D点与地图点存在一定比例关系)
 *
 * 1 将初始关键帧,当前关键帧的描述子转为BoW
 * 2 将关键帧插入到地图
 * 3 用初始化得到的3D点来生成地图点MapPoints
 * 4 全局BA优化，同时优化所有位姿和三维点
 * 5 取场景的中值深度，用于尺度归一化
 * 6 将两帧之间的变换归一化到平均深度1的尺度下
 * 7 把3D点的尺度也归一化到1
 * 8 将关键帧插入局部地图，更新归一化后的位姿、局部地图点
 */
void Tracking::CreateInitialMapMonocular() {
    // Create KeyFrames 认为单目初始化时候的参考帧和当前帧都是关键帧
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
    Verbose::PrintMess("\t\t\t\t将当前帧 " + std::to_string(mCurrentFrame.mnId) + " 和初始参考帧 " + std::to_string(mInitialFrame.mnId) + " 都 作为 关键帧", Verbose::VERBOSITY_DEBUG);

    if (mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated *)(NULL);

    // Step 1. 将初始关键帧，当前关键帧的描述子转为词袋BoW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    Verbose::PrintMess("\t\t\t\t插入关键帧 " + std::to_string(pKFini->mnId) + ", " + std::to_string(pKFcur->mnId), Verbose::VERBOSITY_DEBUG);
    // Insert KFs in the map
    // Step 2. 将关键帧插入到地图
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    // Step 3. 用初始化得到的3D点来生成地图点 MapPoints
    //  mvIniMatches[i]：表示初始化两帧特征点匹配关系。具体解释：i表示帧1中关键点的索引值，值为帧2的关键点索引值,没有匹配关系的话，值为 -1
    // 遍历 初始化第一帧的特征点（匹配成功）
    for (size_t i = 0; i < mvIniMatches.size(); i++) {
        // 跳过没有匹配关系的特征点
        if (mvIniMatches[i] < 0)
            continue;

        // Create MapPoint.
        // 用三角化点的坐标 初始化为 地图点的世界坐标
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        // Step 3.1 用3D点构造地图点
        MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpAtlas->GetCurrentMap());

        // Step 3.2 为该地图点 MapPoint添加属性：
        // a.观测到该MapPoint的关键帧
        // b.该MapPoint的描述子
        // c.该MapPoint的平均观测方向和深度范围

        // 绑定关键帧的 特征点索引 和对应的 地图点 之间的对应关系
        pKFini->AddMapPoint(pMP, i); // mvpMapPoints[i] = pMP;
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        // a.表示该地图点 MapPoint可以 被哪个KeyFrame观测到，i为该地图点在该关键帧中的索引，即被该关键帧的哪个特征点观测到
        pMP->AddObservation(pKFini, i); // mObservations[pKF] = indexes;
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        // b.从众多观测到该MapPoint的特征点中 挑选最有代表性的描述子
        // 同一地图点会被多个关键帧观测到，一个地图点在不同关键帧中对应不同的特征点和描述子。其特征描述子mDescriptor是其在所有观测关键帧中描述子的中位数(准确地说,该描述子与其他所有描述子的中值距离最小)
        pMP->ComputeDistinctiveDescriptors();
        // c.更新该MapPoint 平均观测方向 以及 观测距离的范围
        pMP->UpdateNormalAndDepth();

        // Fill Current Frame structure
        // mvIniMatches：下标i表示在初始化参考帧中的特征点的索引
        // mvIniMatches[i]：是初始化当前帧中的特征点的索引
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        // Add to Map
        mpAtlas->AddMapPoint(pMP);
    } // 生成初始化地图点结束
    Verbose::PrintMess("\t\t\t\t用三角化得到的三维点生成初始地图点结束，共生成 " + std::to_string(mpAtlas->MapPointsInMap()) + " 个地图点", Verbose::VERBOSITY_DEBUG);

    // Update Connections
    // Step 3.3 更新关键帧间的连接关系
    // 每个边有一个权重，边的权重是该关键帧与当前关键帧共视地图点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint *> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    // Step 4 全局BA优化，同时优化所有位姿和三维点
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(), 20);

    // 为什么是 pKFini 而不是 pKCur ? 答：都可以的，内部做了位姿变换了
    float medianDepth = pKFini->ComputeSceneMedianDepth(2); // 对当前关键帧下所有地图点的深度进行从小到大排序,返回距离头部其中1/q处的深度值作为当前场景的平均深度
    float invMedianDepth;
    if (mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f / medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f / medianDepth;

    // 两个条件,一个是平均深度要大于0,另外一个是在当前帧中被观测到的地图点的数目应该大于50
    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("当前关键帧场景的平均深度 < 0，或当前关键帧被观测到的地图点数 < 50，初始化失败！！Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap();
        return;
    }

    // Step 6 将两帧之间的变换归一化到平均深度1的尺度下
    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    // x/z y/z 将z归一化到1
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w); // 归一化当前关键帧的位姿

    // Scale points
    // Step 7 把三维地图点的尺度也归一化到1
    // 为什么是pKFini? 是不是就算是使用 pKFcur 得到的结果也是相同的? 答：是的，因为是同样的三维点
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
        if (vpAllMapPoints[iMP]) {
            MapPoint *pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR) {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(), pKFcur->mImuCalib);
    }

    // Step 8：将关键帧插入局部地图，更新归一化后的位姿、局部地图点
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs = pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    // mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    // ------------单目初始化之后，得到的初始地图中的所有地图点都是 局部地图点------------
    mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame *> vKFs = mpAtlas->GetAllKeyFrames();

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux = (mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) / (mCurrentFrame.mTimeStamp - mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    // 初始化成功，至此，初始化过程完成
    mState = OK;

    initID = pKFcur->mnId;
}

/**
 * @brief 在Atlas中保存当前地图，创建新地图，所有跟状态相关的变量全部重置
 * 1. 前后两帧对应的时间戳反了
 * 2. imu模式下前后帧超过1s
 * 3. 上一帧为最近丢失且重定位失败时
 * 4. 重定位成功，局部地图跟踪失败
 */
void Tracking::CreateMapInAtlas() {
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mbSetInit = false;

    mnInitialFrameId = mCurrentFrame.mnId + 1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
    // mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId + 1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR) {
        mbReadyToInitializate = false;
    }

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF) {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
    }

    if (mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

    if (mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame *>(NULL);

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame() {
    for (int i = 0; i < mLastFrame.N; i++) {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];

        if (pMP) {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep) {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/**
 * @brief 用 参考关键帧的地图点 来对 当前普通帧进行跟踪
 *
 * Step 1：将当前普通帧的描述子转化为BoW向量
 * Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
 * Step 3: 将上一帧的位姿态作为当前帧位姿的初始值
 * Step 4: 通过PnP优化3D-2D的重投影误差来优化位姿
 * Step 5：剔除优化后的匹配地图点中的外点
 * @return 如果匹配数超10，返回true
 */
bool Tracking::TrackReferenceKeyFrame() {
    // Compute Bag of Words vector
    // Step 1：将当前帧的 描述子 转化为 BoW词袋向量
    mCurrentFrame.ComputeBoW(); // 计算当前帧的词袋向量，用于进行特征匹配

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7, true); // 创建一个ORB特征匹配器，其中0.7是匹配阈值，用于确定匹配的特征点，true表示要使用双向匹配。最小距离 < 0.7 * 次小距离 匹配成功，检查旋转
    vector<MapPoint *> vpMapPointMatches; // 存储 当前帧特征点 匹配到的 关键帧地图点，index：当前帧特征点索引，value：如果匹配上为 匹配的关键帧特征点对应的 地图点，未匹配上则为 空

    // Step 2：使用 词袋 BoW 加速 当前帧特征点 与 参考帧地图点 之间的匹配，返回 匹配的特征点数
    //    Verbose::PrintMess("\t\t计算属于同一词袋节点的当前帧特征点与参考关键帧特征点对应描述子之间的距离，将最佳距离小于阈值的参考关键帧特征点对应的地图点设为 当前帧匹配特征点的地图点，当前帧的ID："
    //    + to_string(mCurrentFrame.mnId) + "，其参考关键帧的相对ID：" + std::to_string(mpReferenceKF->mnId) + "，绝对ID：" + std::to_string(mpReferenceKF->mnFrameId), Verbose::VERBOSITY_DEBUG);
    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame,
                                       vpMapPointMatches); // 输出：当前帧特征点 匹配到的 关键帧地图点。
                                                           // index：当前帧特征点索引，value：与当前帧的特征点匹配的 关键帧特征点对应的 地图点

    // 匹配的特征点数 < 15，跟踪失败
    if (nmatches < 15) {
        //        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        Verbose::PrintMess("\t\t匹配的个数 nmatches = " + std::to_string(nmatches) + " < 15，跟踪失败！！", Verbose::VERBOSITY_VERBOSE);

        // -------------liuzhi加----------
        // 保存当前帧和参考关键帧的图像
        SaveSearchByBowFailed(mCurrentFrame, mpReferenceKF, vpMapPointMatches, 1); // 1表示参考关键帧跟踪 词袋匹配时丢失
        // -------------------------------

        return false;
    }

    Verbose::PrintMess("\t\t匹配的个数 nmatches = " + std::to_string(nmatches) + " >= 15，进行位姿优化", Verbose::VERBOSITY_VERBOSE);
    // 匹配的特征点数 >= 15
    // Step 3：将上一帧的位姿 作为 当前帧位姿 的初始值
    mCurrentFrame.mvpMapPoints = vpMapPointMatches; // 当前帧的地图点：当前帧特征点 与参考关键帧匹配的特征点对应的 地图点
    mCurrentFrame.SetPose(mLastFrame.GetPose());    // 将上一帧的位姿 赋给 当前帧，作为初始位姿估计，上一帧的Tcw，在 PoseOptimization 可以收敛快一些

    // mCurrentFrame.PrintPointDistribution();

    //    Verbose::PrintMess("\t\t将上一帧的位姿作为当前帧的初始位姿，使用PnP优化当前帧位姿", Verbose::VERBOSITY_VERBOSE);
    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    // Step 4：通过PnP优化 3D-2D 的重投影误差来优化当前位姿
    Optimizer::PoseOptimization(&mCurrentFrame); // 通过解决PnP问题来对当前帧的位姿进行优化

    // Discard outliers
    // Step 5：剔除优化后的匹配特征点中的外点。之所以在优化之后才剔除外点，是因为在优化的过程中就有了对这些外点的标记
    int nmatchesMap = 0; // 匹配到的地图点数

    // 遍历当前帧中的 所有特征点
    for (int i = 0; i < mCurrentFrame.N; i++) {
        // if(i >= mCurrentFrame.Nleft) break;
        // 如果该特征点对应 有地图点
        if (mCurrentFrame.mvpMapPoints[i]) {
            // 如果该特征点是外点，则清除它在当前帧中存在过的痕迹
            if (mCurrentFrame.mvbOutlier[i]) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i]; // 取出特征点 在参考关键帧中匹配的 地图点

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL); // 将该特征点对应的 地图点置NULL
                mCurrentFrame.mvbOutlier[i] = false;                           // 取消标记该特征点为外点

                // 对于左右相机中的特征点，将其标记为不在视野中
                if (i < mCurrentFrame.Nleft) {
                    pMP->mbTrackInView = false;
                } else {
                    pMP->mbTrackInViewR = false;
                }

                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; // 更新该地图点的最后一次观察帧为当前帧
                nmatches--;                                // 匹配的特征点数 - 1
            }
            // 如果 该特征点 在参考关键帧中匹配的 地图点的 被相机观测次数 > 0，则表示该特征点与地图点匹配成功，说明该特征点是一个内点
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++; // 当前帧匹配到的地图点数 + 1
        }
    }
    // IMU模式，直接判定跟踪成功，返回
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
        Verbose::PrintMess("\t\t使用IMU，跟踪成功！", Verbose::VERBOSITY_DEBUG);
        return true;
    }
    // 非IMU，得判定匹配到的地图点数 >= 10，则跟踪成功；否则跟踪失败
    else {
        if (nmatchesMap >= 10) {
            Verbose::PrintMess("\t\t剔除位姿优化后是外点的特征点，则当前帧在参考关键帧匹配到的地图点数 nmatchesMap = " + std::to_string(nmatchesMap) + " >= 10，跟踪成功！", Verbose::VERBOSITY_DEBUG);
            return true;
        } else {
            Verbose::PrintMess("\t\t剔除位姿优化后是外点的特征点，则当前帧在参考关键帧中匹配到的地图点数 nmatchesMap = " + std::to_string(nmatchesMap) + " < 10，跟踪失败！！",
                               Verbose::VERBOSITY_VERBOSE);
            // -------------liuzhi加----------
            SaveSearchByBowFailed(mCurrentFrame, mpReferenceKF, mCurrentFrame.mvpMapPoints, 2); // 2表示参考关键帧跟踪 位姿优化后丢失
            // -------------------------------
            return false;
        }
    }
}

/**
 * @brief 更新上一帧的位姿；当纯定位模式时，会添加一些临时的地图点
 * 当前帧生成新的地图点，一般只会只会用于更新局部地图，只有当前帧为关键帧时，进行关键帧创建才会生成新的全局地图点
 * 单目情况：只计算了上一帧的世界坐标系位姿
 * 双目和rgbd情况：选取有有深度值的并且没有被选为地图点的点生成新的临时地图点，提高跟踪鲁棒性
 */
// 补充地图点是之对于 RGB-D 和 双目的情况才这么做，因为特征点的深度值在构建 Frame 的时候就求出来了，而单目模式没有特征点的深度信息
void Tracking::UpdateLastFrame() {
    // Update pose according to reference keyframe
    // Step 1：利用参考关键帧更新上一帧在世界坐标系下的位姿
    // 上一帧的参考关键帧，注意这里用的是参考关键帧（位姿准）而不是上上一帧的普通帧
    KeyFrame *pRef = mLastFrame.mpReferenceKF;

    Sophus::SE3f Tlr = mlRelativeFramePoses.back(); // ref_keyframe 到 lastframe 的位姿变换

    // 将上一帧的世界坐标系下的位姿计算出来
    // l:last, r:reference, w:world
    // Tlw = Tlr * Trw  上一帧的位姿 = 参考关键帧到上一帧的相对位姿 * 参考关键帧的位姿
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    // 如果上一帧为关键帧，或单目/单目 +IMU，或SLAM模式，则退出
    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR || !mbOnlyTracking) {
        Verbose::PrintMess("\t\t[UpdateLastFrame] 只更新了上一帧的位姿", Verbose::VERBOSITY_DEBUG);
        return;
    }

    // Step 2：对于双目或rgbd相机，为上一帧生成新的临时地图点
    // 注意这些地图点只是用来跟踪，不加入到地图中，跟踪完后会删除
    // Step 2.1：得到上一帧中具有有效深度值的特征点（不一定是地图点）
    vector<pair<float, int>> vDepthIdx;
    const int Nfeat = mLastFrame.Nleft == -1 ? mLastFrame.N : mLastFrame.Nleft; // mLastFrame.Nleft == -1 为立体匹配双目、RGBD
    vDepthIdx.reserve(Nfeat);
    for (int i = 0; i < Nfeat; i++) {
        float z = mLastFrame.mvDepth[i];
        if (z > 0) {
            vDepthIdx.push_back(make_pair(z, i)); // vDepthIdx第一个元素是某个点的深度,第二个元素是对应的特征点id
        }
    }

    // 如果上一帧中没有有效深度的点,那么就直接退出
    if (vDepthIdx.empty())
        return;

    // 按照深度从小到大排序
    sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // Step 2.2：从中找出不是地图点的部分
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second; // 特征点索引

        bool bCreateNew = false;

        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        // 如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到, 那么就生成一个新的临时的地图点
        if (!pMP)
            bCreateNew = true;
        else if (pMP->Observations() < 1) // 地图点被创建后就没有被观测，认为不靠谱，也需要重新创建
            bCreateNew = true;

        // Step 2.3：如果需要创建，则为临时地图点。只是为了提高双目和RGBD的跟踪成功率，并没有添加复杂属性，因为后面会扔掉
        if (bCreateNew) {
            // 反投影到世界坐标系中
            Eigen::Vector3f x3D;

            // 立体匹配双目、RGBD，生成地图点的世界坐标
            if (mLastFrame.Nleft == -1) {
                mLastFrame.UnprojectStereo(i, x3D);
            } else {
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }
            // 加入上一帧的地图点中
            MapPoint *pNewMP = new MapPoint(x3D, mpAtlas->GetCurrentMap(), &mLastFrame, i);
            mLastFrame.mvpMapPoints[i] = pNewMP;

            // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++; // 累加 新建的地图点
        } else {
            // 累加 已有的有效地图点有效
            nPoints++;
        }

        // Step 2.4：如果地图点质量不好，停止创建地图点
        // 停止新增临时地图点必须同时满足以下条件：
        // 1、当前点的深度 > 设定的深度阈值（40倍基线）
        // 2、nPoints已经超过100个点，因为从近到远排序，说明后面的点距离比较远了，可能不准确，停掉退出
        if (vDepthIdx[j].first > mThDepth && nPoints > 100) {
            Verbose::PrintMess("\t\tUpdateLastFrame中，上一帧至少有 " + std::to_string(nPoints) + " 个地图点", Verbose::VERBOSITY_DEBUG);
            break;
        }
    }
}

/**
 * @brief 根据 恒速运动模型 使用 上一帧地图点 来 对当前帧进行跟踪
 * Step 1：更新上一帧的位姿：纯定位模式下，对于双目或RGB-D相机，还会根据深度值生成临时地图点
 * Step 2：根据上一帧特征点对应地图点进行投影匹配
 * Step 3：优化当前帧位姿
 * Step 4：剔除地图点中外点
 * @return 如果匹配数大于10，认为跟踪成功，返回true
 */
// 用 2D-3D 求位姿，与 上一帧 的3D地图点进行匹配。还有对于 IMU 的情况，直接用 IMU 积分得到的位姿 赋给 当前帧，因此速度会快一些
bool Tracking::TrackWithMotionModel() {
    //    Verbose::PrintMess("\t\t\t\t开始运动模型跟踪", Verbose::VERBOSITY_DEBUG);
    ORBmatcher matcher(0.9, true); // 创建一个ORB特征匹配器，最小距离 < 0.9 * 次小距离 匹配成功，检查旋转

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // Step 1：(1)
    // 使用参考关键帧更新上一帧的位姿。单目仅更新上一帧的位姿；为什么要更新上一帧的位姿，主要是在ORB_SLAM中优化的是参考关键帧的位姿，对于普通帧，虽然在开始设置了位姿，但是没有参与优化，因此在下一次跟踪时，需要用优化后的参考关键帧的位姿更新上一帧的位姿；
    //         (2) 如果是 双目 或 RGB-D 相机 且 纯定位模式时，当上一帧的地图点深度<阈值 或 从近到远的点数<100时，还会根据深度值生成临时地图点，存放于mLastFrame.mvpMapPoints，并且标记为临时地图点
    //         mLastFrame.mlpTemporalPoints 中。一般只会只会用于更新局部地图，只有当前帧为关键帧时，进行关键帧创建才会生成新的全局地图点；
    UpdateLastFrame();

    // Step 2：IMU模式 或 恒速运动模型 得到当前帧的初始位姿
    // 如果 IMU完成初始化 且 距离重定位很久(已 > 1s)，则不需要重置 IMU，用 IMU 来估计位姿，并直接返回
    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU)) {
        Verbose::PrintMess("\t\tIMU完成初始化, 且 当前帧id" + std::to_string(mCurrentFrame.mnId) + " > 上一重定位帧" + std::to_string(mnLastRelocFrameId) + "+" + std::to_string(mnFramesToResetIMU) +
                               ", 用IMU来估计位姿",
                           Verbose::VERBOSITY_DEBUG);

        PredictStateIMU();
        return true;
    }
    // 根据之前估计的速度 和 上一帧的位姿，用 恒速运动模型 得到 当前帧的 初始位姿
    else {
        Verbose::PrintMess("\t\tIMU未完成初始化 或 当前帧id" + std::to_string(mCurrentFrame.mnId) + " <= 上一重定位帧" + std::to_string(mnLastRelocFrameId) + "+" + std::to_string(mnFramesToResetIMU) +
                               ", 用上一帧位姿来估计位姿",
                           Verbose::VERBOSITY_DEBUG);
        // 速度 V：当跟踪成功或者刚刚跟丢，会更新该速度，该速度表示上一帧到当前帧的变换。V = Tcl = Tcw * Twl = 当前帧位姿 * 上一帧位姿的逆变换
        // 则当前帧的初始位姿 Tcw = Tcl * Tlw
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
        //        Verbose::PrintMess("\t\t使用恒速运动模型，由速度和上一帧位姿得到当前帧的初始位姿" ,Verbose::VERBOSITY_VERY_VERBOSE);
    }

    std::cout << "visual last pose: " << mLastFrame.GetPose().matrix() << std::endl;
    std::cout << "visual cur pose: " << mCurrentFrame.GetPose().matrix() << std::endl;

    // 清空当前帧的地图点
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

    // Project points seen in previous frame
    // 设置特征匹配过程中的搜索半径
    int th;
    if (mSensor == System::STEREO)
        th = 7; // 仅双目
    else
        th = 15; // 单目、RGB-D、IMU+单目、IMU+双目、IMU+RGB-D
    std::cout << "\t\tmargin: " << th << std::endl;

    // Step 3：用上一帧地图点投影到当前帧，进行匹配，返回匹配成功的特征点的数量
    //    Verbose::PrintMess("\t\t将上一帧的有效地图点投影到当前帧，在地图点投影像素坐标附近寻找候选特征点，计算上一帧地图点的描述子和当前帧候选特征点的距离，将最佳距离小于阈值的上一帧地图点设为当前帧匹配特征点的地图点。当前帧的ID："
    //    + to_string(mCurrentFrame.mnId) + "，上一帧的ID：" + to_string(mLastFrame.mnId), Verbose::VERBOSITY_DEBUG);
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);

    Verbose::PrintMess("\t\t第一次匹配的个数 nmatches = " + to_string(nmatches), Verbose::VERBOSITY_DEBUG);

    // 如果匹配点不够，则扩大搜索半径再来一次
    if (nmatches < 20) {
        Verbose::PrintMess("\t\t匹配的个数 nmatches = " + std::to_string(nmatches) + " < 20，扩大范围一倍匹配半径重新匹配", Verbose::VERBOSITY_VERY_VERBOSE);

        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));                                  // 清空当前帧的地图点
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR); // 扩大一倍搜索半径，重新匹配
    }
    // 如果还是 < 20
    if (nmatches < 20) {
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
            Verbose::PrintMess("\t\t重新匹配的个数 nmatches = " + to_string(nmatches) + " 仍然 < 20，但 使用了IMU，匹配成功！", Verbose::VERBOSITY_VERBOSE);
            return true;
        } else {
            Verbose::PrintMess("\t\t重新匹配的个数 nmatches = " + to_string(nmatches) + " 仍然 < 20，且 没有使用IMU，匹配失败！！", Verbose::VERBOSITY_VERBOSE);
            return false;
        }
    }
    Verbose::PrintMess("\t\t最终匹配的个数 nmatches = " + to_string(nmatches) + " >= 20，进行位姿优化", Verbose::VERBOSITY_VERY_VERBOSE);

    //    Verbose::PrintMess("\t\t使用PnP优化当前帧位姿" ,Verbose::VERBOSITY_VERY_VERBOSE);
    // Optimize frame pose with all matches
    // Step 4：利用 3D-2D 投影关系，优化当前帧位姿
    std::cout << "恒速运动模型跟踪 优化前位姿:\n" << mCurrentFrame.GetPose().matrix() << std::endl;
    Optimizer::PoseOptimization(&mCurrentFrame);
    std::cout << "恒速运动模型跟踪 优化后位姿:\n" << mCurrentFrame.GetPose().matrix() << std::endl;

    // Discard outliers
    // Step 5：剔除特征点中的外点
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i]) // 特征点对应有 地图点
        {
            // 如果该特征点是外点，清除它的所有关系
            if (mCurrentFrame.mvbOutlier[i]) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i]; // 特征点对应的 地图点

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                if (i < mCurrentFrame.Nleft) {
                    pMP->mbTrackInView = false;
                } else {
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            // 如果 该特征点对应的地图点 的被相机观测次数 > 0，则表示该特征点与地图点匹配成功，说明该特征点是一个内点
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++; // 当前帧匹配到的地图点数 + 1
        }
    }

    // 纯定位模式下：如果成功追踪的地图点非常少，那么这里的 mbVO 标志就会置位
    if (mbOnlyTracking) {
        mbVO = nmatchesMap < 10;

        if (nmatches > 20) {
            Verbose::PrintMess("\t\t纯定位模式，匹配的特征点数 nmatches > 20，匹配成功！", Verbose::VERBOSITY_DEBUG);
            return true;
        } else {
            Verbose::PrintMess("\t\t纯定位模式，匹配的特征点数 nmatches <= 20，匹配失败！！", Verbose::VERBOSITY_VERBOSE);
            return false;
        }
    }

    // 非纯定位模式
    // IMU模式
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
        Verbose::PrintMess("\t\t恒速运动模型跟踪，位姿优化后，由于为IMU模式，直接判定跟踪成功", Verbose::VERBOSITY_DEBUG);
        return true;
    }
    // 纯视觉模式
    else {
        if (nmatchesMap >= 10) {
            Verbose::PrintMess("\t\t纯视觉，剔除位姿优化后是外点的特征点，则当前帧在上一帧匹配到的地图点数 matchesMap = " + std::to_string(nmatchesMap) + " >= 10，匹配成功！",
                               Verbose::VERBOSITY_DEBUG);
            return true;
        } else {
            Verbose::PrintMess("\t\t纯视觉，剔除位姿优化后是外点的特征点，则当前帧在上一帧匹配到的地图点数 nmatchesMap = " + std::to_string(nmatchesMap) + " < 10，匹配失败！！",
                               Verbose::VERBOSITY_VERBOSE);
            return false;
        }
    }
}

/**
 * @brief 局部地图跟踪，进一步优化位姿
 *
 * 1. 更新局部地图，包括局部关键帧和地图点
 * 2. 对局部 MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return true if success
 *
 * Step 1：更新局部关键帧 mvpLocalKeyFrames 和 局部地图点 mvpLocalMapPoints
 * Step 2：在局部地图中查找与当前帧匹配的 MapPoints, 其实也就是 对局部地图点进行跟踪
 * Step 3：更新 局部所有MapPoints后 对位姿再次优化
 * Step 4：更新当前帧的 MapPoints 的被观测程度，并统计跟踪局部地图的效果
 * Step 5：决定是否跟踪成功
 */
bool Tracking::TrackLocalMap() {
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;

    // Step 1：获取当前帧的局部关键帧集合 mvpLocalKeyFrames，其中匹配程度最高的一级共视关键帧 作为 当前帧的 参考关键帧；更新局部地图点 mvpLocalMapPoints
    UpdateLocalMap();
    Verbose::PrintMess("\t\t局部关键帧个数：" + std::to_string(mvpLocalKeyFrames.size()) + " 和 局部地图点个数：" + std::to_string(mvpLocalMapPoints.size()), Verbose::VERBOSITY_DEBUG);

    // Step 2：筛选出局部地图中新增的在视野范围内的地图点，投影到当前帧进行投影匹配，得到更多的 当前帧特征点 与 局部地图点的匹配关系
    //    Verbose::PrintMess("\t\t筛选出局部地图中新增的在视野范围内的地图点，投影到当前帧进行投影匹配，得到更多的 当前帧特征点 与 局部地图点的匹配关系", Verbose::VERBOSITY_DEBUG);
    SearchLocalPoints();

    // TOO check outliers before PO
    int aux1 = 0; // 有对应地图点的 特征点数
    int aux2 = 0; // 外点数
    // 遍历 当前帧的 特征点
    for (int i = 0; i < mCurrentFrame.N; i++) {
        // 该特征点对应有 地图点
        if (mCurrentFrame.mvpMapPoints[i]) {
            aux1++;
            // 该特征点被判定为外点
            if (mCurrentFrame.mvbOutlier[i])
                aux2++;
        }
    }
    //    Verbose::PrintMess("\t\t投影匹配后，当前帧中特征点对应有地图点的个数 aux1 = " + std::to_string(aux1) + "，其中特征点被判定为外点的个数 aux2 = " + std::to_string(aux2),
    //    Verbose::VERBOSITY_DEBUG);

    // 在这个函数之前，在 Relocalization、TrackReferenceKeyFrame、TrackWithMotionModel 中都有位姿优化
    // Step 3：前面新增了更多的 特征点与地图点的匹配关系，进行BA优化，得到更准确的位姿
    int inliers;
    std::cout << "局部地图跟踪 优化前位姿:\n" << mCurrentFrame.GetPose().matrix() << std::endl;

    // IMU未初始化，仅优化位姿
    if (!mpAtlas->isImuInitialized()) {
        Verbose::PrintMess("\t\tIMU未完成初始化, 仅优化当前帧位姿", Verbose::VERBOSITY_DEBUG);
        Optimizer::PoseOptimization(&mCurrentFrame);
    }
    // IMU已初始化
    else {
        // 初始化、重定位、重新开启一个地图都会使 mnLastRelocFrameId 变化
        // 如果当前帧 距离 上一次重定位 1s 内（刚刚重定位），仅位姿优化
        // 注：mnFramesToResetIMU 在euroc imu时为0
        if (mCurrentFrame.mnId <= mnLastRelocFrameId + mnFramesToResetIMU) {
            Verbose::PrintMess("\t\tIMU已初始化, 但当前帧 <= 上一次重定位帧id " + std::to_string(mnLastRelocFrameId) + "+" + std::to_string(mnFramesToResetIMU) + ", 仅优化当前帧位姿",
                               Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        // 距离重定位比较久，即积累的IMU数据量比较多，考虑使用IMU数据优化
        else {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers > 30))
            // mbMapUpdated变化见 Tracking::PredictStateIMU()
            // 未更新地图，使用上一普通帧 以及 当前帧 的视觉信息 和 IMU信息，联合优化当前帧位姿、速度和IMU零偏
            if (!mbMapUpdated) //  && (mnMatchesInliers>30))
            {
                Verbose::PrintMess("\t\tIMU已初始化, 且 当前帧 > 上一次重定位帧id " + std::to_string(mnLastRelocFrameId) + "+" + std::to_string(mnFramesToResetIMU) +
                                       ", 且 未更新地图，使用上一帧和IMU联合优化位姿",
                                   Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            // 地图已更新，关键帧更准确，所以使用上一关键帧的视觉信息 和 IMU信息，联合优化当前帧位姿、速度和IMU零偏
            else {
                Verbose::PrintMess("\t\tIMU已初始化, 且 当前帧 > 上一次重定位帧id " + std::to_string(mnLastRelocFrameId) + "+" + std::to_string(mnFramesToResetIMU) +
                                       ", 且 已更新地图, 使用上一关键帧和IMU联合优化位姿",
                                   Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }
    std::cout << "局部地图跟踪 优化后位姿:\n" << mCurrentFrame.GetPose().matrix() << std::endl;
    // 查看内外点数目，调试用
    aux1 = 0, aux2 = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
        if (mCurrentFrame.mvpMapPoints[i]) {
            aux1++;
            if (mCurrentFrame.mvbOutlier[i])
                aux2++;
        }
    //    Verbose::PrintMess("\t\t位姿优化后，当前帧中特征点对应有地图点的个数: " + std::to_string(aux1) + "，被判定为外点的个数: " + std::to_string(aux2), Verbose::VERBOSITY_DEBUG);

    //    Verbose::PrintMess("\t\t当前帧的特征点对应有地图点，且其不是外点，且其地图点被观测数 > 0，则表示该特征点与局部地图点匹配成功", Verbose::VERBOSITY_DEBUG);
    mnMatchesInliers = 0; // 统计当前帧中 与地图点匹配成功的特征点数

    // Update MapPoints Statistics
    // Step 4：更新当前帧中 特征点对应地图点 的被观测程度；并统计 跟踪局部地图后，与地图点匹配成功的特征点数
    // 遍历当前帧所有特征点
    for (int i = 0; i < mCurrentFrame.N; i++) {
        // != NULL 该特征点有对应的地图点
        if (mCurrentFrame.mvpMapPoints[i]) {
            // 该特征点不是外点
            if (!mCurrentFrame.mvbOutlier[i]) {
                // 由于当前帧的地图点可以被当前帧观测到，其被观测统计量加1
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound(); // 找到该点的帧数mnFound 加 1

                // 非纯定位模式
                if (!mbOnlyTracking) {
                    // 如果该特征点对应的地图点 被相机观测数目 nObs > 0，则表示该特征点与地图点匹配成功，说明该特征点是一个内点
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        mnMatchesInliers++; // 与地图点匹配成功的特征点数。单目 + 1，双目或RGB-D则 + 2
                }
                // 纯定位模式
                else
                    mnMatchesInliers++;
            }
            // 是外点 且 当前相机时双目，则删除这个点。原因分析：因为双目本身可以左右互匹配，删掉无所谓
            else if (mSensor == System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // 当前帧跟踪局部地图后，匹配到的内点个数
    mpLocalMapper->mnMatchesInliers = mnMatchesInliers;

    // Step 5：根据跟踪匹配数目 及 重定位情况决定是否跟踪成功
    // 如果初始化或重定位后20帧内，那么至少成功匹配50个点才认为是成功跟踪
    std::cout << "mMaxFrames: " << mMaxFrames << ", mnFramesToResetIMU: " << mnFramesToResetIMU << std::endl;
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50) {
        Verbose::PrintMess("\t\t当前帧id " + std::to_string(mCurrentFrame.mnId) + " < 重定位帧id " + std::to_string(mnLastRelocFrameId) + "+" + std::to_string(mMaxFrames) +
                               "，且成功匹配点的个数 = " + std::to_string(mnMatchesInliers) + " < 50，局部地图跟踪失败！！",
                           Verbose::VERBOSITY_DEBUG);
        return false;
    }

    // RECENTLY_LOST状态下，至少成功跟踪10个才算成功
    if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST)) {
        Verbose::PrintMess("\t\tRECENTLY_LOST状态下，成功匹配点的个数 = " + std::to_string(mnMatchesInliers) + " > 10，局部地图跟踪成功！", Verbose::VERBOSITY_DEBUG);
        return true;
    }

    // 单目 + IMU模式下，已初始化至少成功跟踪15个；未初始化至少需要50个
    if (mSensor == System::IMU_MONOCULAR) {
        if ((mnMatchesInliers < 15 && mpAtlas->isImuInitialized()) || (mnMatchesInliers < 50 && !mpAtlas->isImuInitialized())) {
            if (mpAtlas->isImuInitialized())
                Verbose::PrintMess("\t\tIMU+单目，IMU已初始化，成功匹配点的个数 = " + std::to_string(mnMatchesInliers) + " < 15，局部地图跟踪失败！！", Verbose::VERBOSITY_DEBUG);
            else if (!mpAtlas->isImuInitialized())
                Verbose::PrintMess("\t\tIMU+单目，IMU未初始化，成功匹配点的个数 = " + std::to_string(mnMatchesInliers) + " < 50，局部地图跟踪失败！！", Verbose::VERBOSITY_DEBUG);

            return false;
        } else {
            Verbose::PrintMess("\t\tIMU+单目，成功匹配点的个数 = " + std::to_string(mnMatchesInliers) + "，局部地图跟踪成功！", Verbose::VERBOSITY_DEBUG);
            return true;
        }

    }
    // 双目/RGBD + IMU模式
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
        if (mnMatchesInliers < 15) {
            Verbose::PrintMess("\t\tIMU+双目/RGBD，成功匹配点的个数 = " + std::to_string(mnMatchesInliers) + " < 15，局部地图跟踪失败！！", Verbose::VERBOSITY_DEBUG);
            return false;
        } else {
            Verbose::PrintMess("\t\tIMU+双目/RGBD，成功匹配点的个数 = " + std::to_string(mnMatchesInliers) + " >= 15，局部地图跟踪成功！", Verbose::VERBOSITY_DEBUG);
            return true;
        }
    }
    // 以上情况都不满足，只要跟踪的地图点大于30个就认为成功了
    else {
        if (mnMatchesInliers < 30) {
            Verbose::PrintMess("\t\t纯视觉模式，成功匹配点的个数 = " + std::to_string(mnMatchesInliers) + " < 30，局部地图跟踪失败！！", Verbose::VERBOSITY_DEBUG);
            return false;
        } else {
            Verbose::PrintMess("\t\t纯视觉模式，成功匹配点的个数 = " + std::to_string(mnMatchesInliers) + " >= 30，局部地图跟踪成功！", Verbose::VERBOSITY_DEBUG);
            return true;
        }
    }
}

/**
 * @brief 判断当前帧是否需要插入关键帧
 *
 * Step 1：纯VO模式下不插入关键帧，如果局部地图被闭环检测使用，则不插入关键帧
 * Step 2：如果距离上一次重定位比较近，或者关键帧数目超出最大限制，不插入关键帧
 * Step 3：得到参考关键帧跟踪到的地图点数量
 * Step 4：查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
 * Step 5：对于双目或RGBD摄像头，统计可以添加的有效地图点总数 和 跟踪到的地图点数量
 * Step 6：决策是否需要插入关键帧
 * @return true         需要
 * @return false        不需要
 */
bool Tracking::NeedNewKeyFrame() {
    // IMU模式 且 IMU未初始化，若距离上一关键帧5帧，则插入一个关键帧；<5帧，则不插入
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized()) {
        // 单目+IMU，且 当前帧距离上一关键帧时间戳超过0.25s，则需要插入关键帧
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25) {
            Verbose::PrintMess("\t\tIMU+单目, 且 IMU未初始化, 且当前帧距离上一关键帧时间戳>=0.25s, 需插入一个关键帧", Verbose::VERBOSITY_DEBUG);
            return true;
        }
        // 双目+IMU 或 RGBD+IMU，且 当前帧距离上一关键帧时间戳超过0.25s，则需要插入关键帧
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25) {
            Verbose::PrintMess("\t\tIMU+双目/RGBD, 且 IMU未初始化, 且当前帧距离上一关键帧时间戳>=0.25s, 需要一个插入关键帧", Verbose::VERBOSITY_DEBUG);
            return true;
        } else {
            Verbose::PrintMess("\t\tIMU模式 且 IMU未初始化, 但当前帧距离上一关键帧时间戳<0.25s, 不需要插入关键帧", Verbose::VERBOSITY_DEBUG);
            return false;
        }
    }
    Verbose::PrintMess("\t\t非IMU模式 或 IMU模式，且IMU已初始化, 需进行后续判断", Verbose::VERBOSITY_DEBUG);

    // Step 1：纯定位模式下不插入关键帧
    if (mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // Step 2：如果LocalMapping线程被闭环检测使用，则不插入关键帧
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false;
    }

    // 获取当前地图中的关键帧数目
    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // mCurrentFrame.mnId：当前帧的ID
    // mnLastRelocFrameId：最近一次重定位帧的ID
    // mMaxFrames：图像输入的帧率
    // Step 3：如果距离初始化 或 上一次重定位20帧内，且关键帧个数>20，不插入关键帧
    std::cout << "[needNewKeyFrame: " << mMaxFrames << std::endl;
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames) {
        Verbose::PrintMess("\t\t关键帧个数" + std::to_string(nKFs) + " > " + std::to_string(mMaxFrames) + " 且 当前帧id" + std::to_string(mCurrentFrame.mnId) + " < 上一次重定位id " +
                               std::to_string(mnLastRelocFrameId) + " + " + std::to_string(mMaxFrames) + ", 不需要插入关键帧",
                           Verbose::VERBOSITY_DEBUG);
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    // Step 4：得到参考关键帧跟踪到的地图点数量
    // UpdateLocalKeyFrames 函数中会将与当前帧共视程度最高的关键帧 设定为当前帧的 参考关键帧

    // 地图点的最小观测次数阈值
    int nMinObs = 3;
    // 关键帧个数<=2时，阈值设低
    if (nKFs <= 2)
        nMinObs = 2;
    // 参考关键帧的地图点中，被其他关键帧观测的次数 >= nMinObs 的地图点数
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    Verbose::PrintMess("\t\t参考关键帧中，被观测次数 >=" + to_string(nMinObs) + " 的地图点个数: " + to_string(nRefMatches), Verbose::VERBOSITY_DEBUG);

    // Local Mapping accept keyframes?
    // Step 5：查询局部地图线程是否繁忙，当前能否接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    // Step 6：对于双目或RGBD摄像头，统计成功跟踪的近点的数量，如果跟踪到的近点太少，没有跟踪到的近点较多，可以插入关键帧
    int nNonTrackedClose = 0; // 双目或RGB-D中 近点中内点个数
    int nTrackedClose = 0;    // 双目或RGB-D中 近点中外点个数
    // 不是单目、单目+IMU，即双目或RGB-D
    if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for (int i = 0; i < N; i++) {
            // 深度值在有效范围内 (0, mThDepth) 基线*系数(40)
            if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
                // 近地图点存在 且 是内点
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                // 近地图点不存在 或 是外点
                else
                    nNonTrackedClose++;
            }
        }
    }

    // 双目或RGBD情况下：跟踪到的地图点中近点太少 同时 没有跟踪到的地图点太多，可以插入关键帧了
    // 单目时，为false
    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);
    Verbose::PrintMess("\t\t近地图点是内点的个数: " + to_string(nTrackedClose) + " 需 < 100; 近地图点不存在或是外点的个数: " + to_string(nNonTrackedClose) +
                           " 需 > 70。是否需插入关键帧: " + std::to_string(bNeedToInsertClose),
                       Verbose::VERBOSITY_DEBUG);

    // Step 7：决策是否需要插入关键帧
    // Step 7.1：设定比例阈值，当前帧和参考关键帧跟踪到点的比例，比例越大，越倾向于增加关键帧
    float thRefRatio = 0.75f;
    // 如果地图中的关键帧只有一帧，那么插入关键帧的阈值设置的低一点，插入频率较低
    if (nKFs < 2)
        thRefRatio = 0.4f;

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    // 单目：插入关键帧的频率很高
    if (mSensor == System::MONOCULAR)
        thRefRatio = 0.9f;

    if (mpCamera2)
        thRefRatio = 0.75f; // 双目非立体匹配

    // 单目+IMU：如果匹配内点数目超过350，插入关键帧的频率可以适当降低
    if (mSensor == System::IMU_MONOCULAR) {
        if (mnMatchesInliers > 350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // c1a、c1b、c1c 三个满足一个即可
    // Step 7.2：很长时间没有插入关键帧，可以插入
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Step 7.3：满足插入关键帧的最小间隔 且 局部建图localMapper处于空闲状态，可以插入
    const bool c1b = ((mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames) && bLocalMappingIdle); // mpLocalMapper->KeyframesInQueue() < 2);
    // Step 7.4：双目/RGB-D+非IMU，当前帧跟踪到的地图点比参考关键帧的0.25倍还少 或 满足bNeedToInsertClose
    const bool c1c = mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR && mSensor != System::IMU_STEREO && mSensor != System::IMU_RGBD &&
                     (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);

    // Step 7.5：c2两个条件必须同时满足
    // 条件1：和参考帧相比，当前跟踪到的地图点少(比例阈值越高，插帧越频繁) 或 满足bNeedToInsertClose，跟踪到的点也不能太多，否则信息冗余过多
    // 条件2：跟踪到的内点 > 15，内点数必须超过设定的最小阈值
    const bool c2 = (((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose)) && mnMatchesInliers > 15);
    Verbose::PrintMess("\t\t当前帧跟踪地图点数 " + to_string(mnMatchesInliers) + ", 需 < 参考关键帧地图点阈值 " + std::to_string(nRefMatches) + " * " + std::to_string(thRefRatio) + " = " +
                           to_string(nRefMatches * thRefRatio),
                       Verbose::VERBOSITY_DEBUG);

    // std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    // 新增的条件c3：单目/双目+IMU模式下，并且IMU完成初始化（隐藏条件），当前帧和上一关键帧之间时间超过0.5秒 (10帧)，则c3=true
    bool c3 = false;
    // 存在上一关键帧
    if (mpLastKeyFrame) {
        // 单目+IMU
        if (mSensor == System::IMU_MONOCULAR) {
            if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5) {
                Verbose::PrintMess("\t\tIMU+单目，已完成IMU初始化, 且当前帧距离上一关键帧时间戳>=0.5s, 需插入关键帧", Verbose::VERBOSITY_DEBUG);
                c3 = true;
            }
        }
        // 双目+IMU、RGBD+IMU
        else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
            if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5) {
                Verbose::PrintMess("\t\tIMU+双目/RGBD，已完成IMU初始化, 当前帧距离上一关键帧时间戳>=0.5s, 需插入关键帧", Verbose::VERBOSITY_DEBUG);
                c3 = true;
            }
        }
    }

    // 新增的条件c4：单目+IMU，当前帧匹配内点数在15~75之间 或 是RECENTLY_LOST状态，c4=true
    bool c4 = false;
    if ((((mnMatchesInliers < 75) && (mnMatchesInliers > 15)) || mState == RECENTLY_LOST) &&
        (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4 = true;
    else
        c4 = false;

    Verbose::PrintMess("\t\tA1:" + to_string(c1a) + " A2:" + to_string(c1b) + " A3:" + to_string(c1c) + " B:" + to_string(c2) + " C:" + to_string(c3), Verbose::VERBOSITY_DEBUG);

    // 相比ORB-SLAM2多了c3,c4
    // 非IMU模式，c2必须满足，c1a、c1b、c1c 满足一个即可
    // IMU模式，c3、c4 满足一个即可。或者c3、c4不满足，但c2满足，且c1a、c1b、c1c 满足一个
    if (((c1a || c1b || c1c) && c2) || c3 || c4) {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        // Step 7.6：LocalMapping空闲时 或 正在进行IMU初始化，可以直接插入，不空闲的时候要根据情况插入
        if (bLocalMappingIdle || mpLocalMapper->IsInitializing()) {
            Verbose::PrintMess("\t\t局部建图空闲 或 正在IMU初始化，可以直接插入关键帧", Verbose::VERBOSITY_DEBUG);
            return true;
        }
        // (隐藏条件) LocalMapping在忙 且 未在进行IMU初始化
        else {
            mpLocalMapper->InterruptBA();
            // 双目、双目+IMU、RGB-D、RGB-D+IMU时，如队列里没有太多关键帧，可以插入；太多，则不能插入
            // Tracking插入的关键帧先缓存到 mlNewKeyFrames中，然后LocalMapper再逐个pop出来插入到地图中(mspKeyFrames中)
            if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) {
                // 队列中的关键帧个数 < 3, 可以插入
                if (mpLocalMapper->KeyframesInQueue() < 3) {
                    Verbose::PrintMess("\t\t双目、双目+IMU、RGB-D、RGB-D+IMU，队列中的关键帧个数 < 3，可以插入关键帧", Verbose::VERBOSITY_DEBUG);
                    return true;
                }
                // 队列中缓冲的关键帧>=3, 数目太多, 暂时不能插入
                else {
                    Verbose::PrintMess("\t\t双目、双目+IMU、RGB-D、RGB-D+IMU，队列中的关键帧个数 >= 3，目前不能插入关键帧", Verbose::VERBOSITY_DEBUG);
                    return false;
                }
            }
            // 单目、单目+IMU，就直接无法插入关键帧了
            else {
                // std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                // ? 为什么这里对单目情况的处理不一样? 回答：可能是单目关键帧相对比较密集
                Verbose::PrintMess("\t\t单目、单目+IMU，目前不能插入关键帧", Verbose::VERBOSITY_DEBUG);
                return false;
            }
        }
    }
    // 不满足上面的条件, 自然不能插入关键帧
    else {
        Verbose::PrintMess("\t\t条件不满足，不需要插入关键帧", Verbose::VERBOSITY_DEBUG);
        return false;
    }
}

/**
 * @brief 创建新的关键帧
 * 对于非单目的情况，同时创建新的地图点 MapPoints
 *
 * Step 1：将当前帧构造成关键帧
 * Step 2：将当前关键帧设置为当前帧的参考关键帧
 * Step 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
 */
void Tracking::CreateNewKeyFrame() {
    // 如果局部建图线程正在初始化且没做完或关闭了,无法插入关键帧
    if (mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if (!mpLocalMapper->SetNotStop(true))
        return;

    // Step 1：将当前帧构造成 关键帧
    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

    if (mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);

    // Step 2：把创建出来的关键帧赋值为追踪器的参考关键帧，同时赋值给当前帧的参考关键帧
    // 但它会在UpdateLocalKeyFrames函数中被更新， 会被替换成与当前帧共视程度最高的关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if (mpLastKeyFrame) {
        pKF->mPrevKF = mpLastKeyFrame; // 记录当前关键帧的 上一关键帧
        mpLastKeyFrame->mNextKF = pKF;
    } else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    // IMU模式下
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(), pKF->mImuCalib);
    }

    // 这段代码和 Tracking::UpdateLastFrame 中的那一部分代码功能相同
    // Step 3：双目、双目+IMU、RGBD、RGBD+IMU，为当前帧生成新的地图点；单目无操作
    // TODO check if include imu_stereo
    if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) {
        // 根据 Tcw计算mRcw、mtcw 和 mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        // Step 3.1：得到当前帧有深度值的特征点（不一定是地图点）
        vector<pair<float, int>> vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        // 遍历当前帧（新关键帧）深度值 > 0 的特征点
        for (int i = 0; i < N; i++) {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0) {
                // (特征点深度, 特征点的id)
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty()) {
            // Step 3.2：按照深度从小到大排序。地图点从近到远
            sort(vDepthIdx.begin(), vDepthIdx.end());

            Verbose::PrintMess("\t\ttrue_depth_thr_: " + std::to_string(mThDepth) + ", 深度 > 0 的点个数: " + std::to_string(vDepthIdx.size()), Verbose::VERBOSITY_DEBUG);

            // Step 3.3：从中找出不是地图点的生成临时地图点
            // 处理的近点的个数
            int nPoints = 0;
            int newPoints = 0;
            int oldPoints = 0;
            // 遍历每个有深度的点（深度从小到大排列）
            for (size_t j = 0; j < vDepthIdx.size(); j++) {
                int i = vDepthIdx[j].second; // 特征点索引

                bool bCreateNew = false;

                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                // 如果这个特征点对应没有地图点, 就新建一个地图点
                if (!pMP) {
                    //                    std::cout << "j: " << j << ", depth: " << vDepthIdx[j].first << ", newPoints: " << newPoints << ", oldPoints: " << oldPoints << std::endl;
                    bCreateNew = true;
                }
                // 如果对应有地图点(隐藏条件) 但 创建后就没有被观测到, 先将该地图点置空，再标记为需新建一个地图点
                else if (pMP->Observations() < 1) {
                    bCreateNew = true;
                    //                    std::cout << "j: " << j << ", depth: " << vDepthIdx[j].first << ", newPoints: " << newPoints << ", oldPoints: " << oldPoints << ", lm.obs: " <<
                    //                    pMP->Observations() << std::endl;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                // 如果需要就新建地图点，这里的地图点不是临时的，是全局地图中新建地图点，用于跟踪
                if (bCreateNew) {
                    Eigen::Vector3f x3D;

                    if (mCurrentFrame.Nleft == -1) {
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    } else {
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpAtlas->GetCurrentMap());
                    // 添加 新地图点 对 当前KF的观测
                    pNewMP->AddObservation(pKF, i);

                    // Check if it is a stereo observation in order to not
                    // duplicate mappoints
                    // 非立体匹配（鱼眼）双目
                    if (mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0) {
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]] = pNewMP;
                        pNewMP->AddObservation(pKF, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    // 添加 当前KF 对 新地图点 的观测
                    pKF->AddMapPoint(pNewMP, i);             // 该地图点被 当前关键帧pkF的 特征点i看到
                    pNewMP->ComputeDistinctiveDescriptors(); // 更新地图点的描述子：所有观测到该地图点的关键帧中 描述子的中位数
                    pNewMP->UpdateNormalAndDepth();          // 更新该地图点的 平均观测方向 以及 观测距离的范围
                    mpAtlas->AddMapPoint(pNewMP);            // 加入到地图集中

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++; // 累加新建的地图点个数
                    newPoints++;
                } else {
                    nPoints++; // 累加已有的有效地图点个数
                    oldPoints++;
                    //                    std::cout << "j: " << j << ", depth: " << vDepthIdx[j].first << ", newPoints: " << newPoints << ", oldPoints: " << oldPoints << ", lm.obs: " <<
                    //                    pMP->Observations() << std::endl;
                }

                // Step 3.4：停止新建地图点必须同时满足以下条件：
                // 1、当前的地图点的深度 > 深度阈值（挑战数据集40倍基线，Euroc 60倍）
                // 2、nPoints > 100个点，说明距离比较远了，可能不准确，停掉退出
                if (vDepthIdx[j].first > mThDepth && nPoints > maxPoint) {
                    std::cout << "[退出] j: " << j << ", depth: " << vDepthIdx[j].first << ", newPoints: " << newPoints << ", oldPoints: " << oldPoints << std::endl;
                    break;
                }
            } // 每个特征点遍历完
            Verbose::PrintMess("\t插入新关键帧 id: " + std::to_string(pKF->mnId) + " (" + std::to_string(pKF->mnFrameId) + ")，新建了 " + std::to_string(newPoints) + " 个全局地图点",
                               Verbose::VERBOSITY_DEBUG);
        }
    }

    // Step 4：插入关键帧
    // 关键帧插入到列表 mlNewKeyFrames中，等待local mapping线程临幸
    mpLocalMapper->InsertKeyFrame(pKF);

    // 插入好了，允许局部建图停止
    mpLocalMapper->SetNotStop(false);

    // 当前帧成为新的关键帧，更新
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

/**
 * @brief 筛选局部地图中新增的在当前帧视野范围内的地图点，投影到当前帧,进行搜索匹配，得到更多的匹配关系
 *
 * 注意：局部地图点中已经是当前帧地图点的不需要再投影，只需要将除此之外的、并且在当前帧视野范围内的局部地图点 和 当前帧进行投影匹配
 */
void Tracking::SearchLocalPoints() {
    // Do not search map points already matched
    // Step 1：遍历当前帧的地图点，标记这些地图点不参与之后的投影搜索匹配
    for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++) {
        MapPoint *pMP = *vit;
        if (pMP) {
            if (pMP->isBad()) {
                *vit = static_cast<MapPoint *>(NULL);
            }
            // 当前帧的地图点是好的
            else {
                pMP->IncreaseVisible();                    // 更新 能观测到该地图点的帧数加1(被当前帧观测了)
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; // 该地图点最后被当前帧观测到
                // 标记 该点在后面搜索匹配时不被投影，因为已经有匹配了
                pMP->mbTrackInView = false; // 表示局部地图中，除当前帧能够看到的地图点外 的地图点 是否在当前帧的视野范围内，
                                            // 所以当前帧的地图点、TrackWithMotionModel、TrackReferenceKeyFrame中优化后的外点 的mbTrackInView 标记为false
                pMP->mbTrackInViewR = false;
            }
        }
    }

    // 准备进行投影匹配的 局部地图点的数目
    int nToMatch = 0;

    // Project points in frame and check its visibility
    // Step 2：判断所有局部地图点中 除当前帧的地图点外 的点，是否在当前帧的视野范围内
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++) {
        MapPoint *pMP = *vit; // 局部地图点

        // 已经被当前帧观测到的局部地图点肯定在视野范围内，跳过
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        // 跳过坏点
        if (pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        // 判断局部地图点是否在在当前帧视野内，在的话 pMP->mbTrackInView置为True
        if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
            pMP->IncreaseVisible(); // 观测到该点的帧数加1
            nToMatch++;             // 只有在视野范围内的局部地图点才参与之后的投影匹配
        }
        if (pMP->mbTrackInView) {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    Verbose::PrintMess("\t\t在视野范围内，需要进行投影匹配的 局部地图点的个数：" + std::to_string(nToMatch), Verbose::VERBOSITY_NORMAL);

    // Step 3：如果需要进行投影匹配的局部地图点的数目 > 0，则将其投影到当前帧，增加更多的匹配关系
    if (nToMatch > 0) {
        ORBmatcher matcher(0.8);
        int th = 1;
        // RGBD相机输入的时候,搜索的阈值会变得稍微大一些
        if (mSensor == System::RGBD || mSensor == System::IMU_RGBD)
            th = 3;
        // 如果IMU已初始化
        if (mpAtlas->isImuInitialized()) {
            if (mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th = 2;
            else
                th = 6; // 0.4版本这里是3
        }
        // IMU模式 且 IMU未初始化
        else if (!mpAtlas->isImuInitialized() && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)) {
            th = 10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            th = 5;

        // 跟踪丢失或刚刚丢失
        if (mState == LOST || mState == RECENTLY_LOST) // Lost for less than 1 second
            th = 15;                                   // 15
        std::cout << "\t\t[局部地图跟踪中的 投影匹配] margin: " << th << std::endl;
        //        Verbose::PrintMess("\t\tth = "+std::to_string(th)+", mbFarPoints = "+std::to_string(mpLocalMapper->mbFarPoints)+", mThFarPoints = "+std::to_string(mpLocalMapper->mThFarPoints),
        //        Verbose::VERBOSITY_DEBUG);
        // 投影匹配得到更多的匹配关系
        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

/**
 * @brief 更新局部地图 LocalMap
 * 包括：
 * 1、K1个关键帧、K2个临近关键帧和参考关键帧
 * 2、由这些关键帧观测到的地图点 MapPoints
 */
void Tracking::UpdateLocalMap() {
    // This is for visualization
    // 设置参考地图点用于绘图显示 局部地图点（红色）
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    // 用共视图来更新 局部关键帧 和 局部地图点
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/**
 * @brief 更新局部地图点。将局部关键帧 mvpLocalKeyFrames 中关键帧的地图点添加到局部地图点 mvpLocalMapPoints中
 */
void Tracking::UpdateLocalPoints() {
    // Step 1：清空局部地图点
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    // Step 2：遍历局部关键帧
    for (vector<KeyFrame *>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(), itEndKF = mvpLocalKeyFrames.rend(); itKF != itEndKF; ++itKF) {
        KeyFrame *pKF = *itKF;
        // 获取 该关键帧的 mvpMapPoints。key：2D特征点索引，value：3D地图点
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        // step 3：将该局部关键帧观测到的地图点添加到 mvpLocalMapPoints
        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++) {
            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            // 用该地图点的成员变量mnTrackReferenceForFrame 记录当前帧的id
            // 表示它已经是当前帧的局部地图点了，可以防止重复添加局部地图点
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad()) {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

/**
 * @brief 更新局部关键帧，并将与当前帧共视程度最高的关键帧作为其参考关键帧
 *
 * 遍历当前帧或上一帧的地图点，将观测到这些地图点的关键帧和相邻的关键帧及其父子关键帧，作为 mvpLocalKeyFrames
 * Step 1：遍历当前帧或上一帧的地图点，记录所有能观测到当前帧地图点的关键帧
 * Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧包括以下3种类型
 *      类型1：能观测到当前帧地图点的关键帧，也称一级共视关键帧
 *      类型2：一级共视关键帧的共视关键帧，称为二级共视关键帧
 *      类型3：一级共视关键帧的子关键帧、父关键帧
 * Step 3：更新当前帧的参考关键帧，与自己共视程度最高的一级共视关键帧作为参考关键帧
 */
void Tracking::UpdateLocalKeyFrames() {
    // Each map point vote for the keyframes in which it has been observed
    // Step 1：遍历当前帧地图点，记录所有能观测到当前帧地图点的关键帧

    // key: 能观测到当前帧或上一帧地图点的所有关键帧，value:该关键帧看到了多少当前帧的地图点，也就是共视程度
    map<KeyFrame *, int> keyframeCounter;
    // IMU未完成初始化 或 刚刚完成重定位(2帧内)，使用当前帧的地图点
    if (!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2)) {
        std::cout << "\t\t使用当前帧地图点" << std::endl;
        // 遍历当前帧特征点
        for (int i = 0; i < mCurrentFrame.N; i++) {
            // 对应地图点
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

            if (pMP) {
                if (!pMP->isBad()) {
                    // 获取该地图点的观测情况
                    // key: 观测到该地图点的关键帧, value: 该地图点在该关键帧中的 索引，默认为<-1,-1>；如果是单目或PinHole双目，则为<idx,-1>；如果是KB鱼眼双目且在右目中，则为<-1, idx>
                    const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();

                    // 遍历每一个观测，得到观测到当前帧地图点的关键帧 看到地图点的个数
                    for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                        // 这里的操作非常精彩！
                        // map[key] = value，当要插入的键存在时，会覆盖键对应的原来的值。如果键不存在，则添加一组键值对
                        // it->first 是观测到该地图点的关键帧。++则为累加同一个关键帧看到当前帧地图点的数目
                        keyframeCounter[it->first]++;
                } else {
                    mCurrentFrame.mvpMapPoints[i] = NULL; // 删除坏的地图点
                }
            }
        } // 循环遍历当前帧地图点结束
    }
    // IMU已初始化（恒速运动未匹配地图点） 且 跟踪不错：使用上一帧的地图点
    else {
        std::cout << "\t\t使用上一帧地图点" << std::endl;
        // 遍历上一帧的地图点
        for (int i = 0; i < mLastFrame.N; i++) {
            // Using lastframe since current frame has not matches yet
            if (mLastFrame.mvpMapPoints[i]) {
                MapPoint *pMP = mLastFrame.mvpMapPoints[i];
                if (!pMP)
                    continue;
                if (!pMP->isBad()) {
                    const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                    for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                        keyframeCounter[it->first]++;
                } else {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i] = NULL;
                }
            }
        }
    } // 循环结束，找到所有能观测到当前帧或上一帧地图点的关键帧

    // 获取 最大观测次数，和对应的关键帧
    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    // Step 2：添加局部关键帧（mvpLocalKeyFrames），添加局部关键帧有3种类型
    // 先清空局部关键帧
    mvpLocalKeyFrames.clear();
    // 先申请3倍内存，不够后面再加
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    // Step 2.1 类型1：一级共视关键帧：所有能观测到当前帧地图点的关键帧作为局部关键帧 （将邻居们拉拢入伙）
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++) {
        KeyFrame *pKF = it->first;

        // 如果该关键帧要删除，则跳过它
        if (pKF->isBad())
            continue;

        std::cout << "\t\t   一级共视关键帧id：" << pKF->mnId << " (" << pKF->mnFrameId << "), weight: " << it->second << std::endl;

        // 更新 最大观测地图点个数 和 共视程度最高的一级共视关键帧
        if (it->second > max) {
            max = it->second;
            pKFmax = pKF;
        }

        // 添加为 局部关键帧
        mvpLocalKeyFrames.push_back(pKF);

        // 用该关键帧的成员变量mnTrackReferenceForFrame 记录当前帧的id
        // 表示它已经是当前帧的局部关键帧了，可以防止重复添加局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }
    Verbose::PrintMess("\t\t一级共视关键个数: " + std::to_string(mvpLocalKeyFrames.size()), Verbose::VERBOSITY_DEBUG);

    // Step 2.2 遍历一级共视关键帧，添加更多的局部关键帧
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++) {
        // 处理的局部关键帧不超过80帧，若超过则直接退出，不再添加
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame *pKF = *itKF; // 取出 一级共视关键帧

        // 类型2: 二级共视关键帧：遍历该一级共视关键帧的 前10个共视关键帧，只找到1个就好（将邻居的一个邻居拉拢入伙）
        // 如果共视关键帧不足10帧,那么就返回所有具有共视关系的关键帧
        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        // vNeighs 是按照共视程度从大到小排列
        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++) {
            KeyFrame *pNeighKF = *itNeighKF; // 一级共视关键帧的 共视关键帧
            if (!pNeighKF->isBad()) {
                // mnTrackReferenceForFrame防止重复添加局部关键帧
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break; // 只会跳出最内层循环，即找到当前一级共视关键帧的 一个 二级共视关键帧
                }
            }
        }

        // 类型3:将该一级共视关键帧的一个子关键帧作为局部关键帧（将邻居们的一个孩子拉拢入伙）
        const set<KeyFrame *> spChilds = pKF->GetChilds();
        for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
            KeyFrame *pChildKF = *sit;
            if (!pChildKF->isBad()) {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break; // 只会跳出最内层循环，即找到当前一级共视关键帧的 一个 子关键帧
                }
            }
        }

        // 类型3:将该一级共视关键帧的一个父关键帧（将邻居们的一个父关键帧拉拢入伙）
        KeyFrame *pParent = pKF->GetParent();
        if (pParent) {
            // mnTrackReferenceForFrame防止重复添加局部关键帧
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                //! 感觉是个bug，找到一个父关键帧就跳出整个for循环，跳出查找所有一级共视关键帧的循环
                //                break;
                //                continue;
            }
        }
    } // 遍历所有一级共视关键帧结束，每个一级共视关键帧找到一个二级共视关键帧，一个其父、子关键帧。？如果找到某个一级共视关键帧的父关键帧，则跳出循环

    // Add 10 last temporal KFs (mainly for IMU)
    // IMU模式 且 局部关键帧个数<80，则添加当前帧最近的20帧临时关键帧
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mvpLocalKeyFrames.size() < 80) {
        KeyFrame *tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for (int i = 0; i < Nd; i++) {
            if (!tempKeyFrame)
                break;
            if (tempKeyFrame->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                tempKeyFrame = tempKeyFrame->mPrevKF;
            }
        }
    }

    // Step 3：更新当前帧的参考关键帧，将与当前帧共视程度最高的一级共视关键帧作为其参考关键帧
    if (pKFmax) {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
    Verbose::PrintMess("\t\t局部关键帧的个数: " + std::to_string(mvpLocalKeyFrames.size()) + ", 当前帧的参考关键帧id: " + std::to_string(mpReferenceKF->mnId) + "(" +
                           std::to_string(mpReferenceKF->mnFrameId) + ")",
                       Verbose::VERBOSITY_DEBUG);
}

/**
 * @details 重定位
 * @return true
 * @return false
 *
 * Step 1：计算当前帧特征点的词袋向量
 * Step 2：找到与当前帧相似的候选关键帧
 * Step 3：通过 BoW进行匹配
 * Step 4：通过 EPnP算法估计姿态
 * Step 5：通过 PoseOptimization对姿态进行优化求解
 * Step 6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
 */
bool Tracking::Relocalization() {
    Verbose::PrintMess("\t\t开始重定位...", Verbose::VERBOSITY_DEBUG);
    // Compute Bag of Words Vector
    // Step 1: 将当前帧特征点的描述子 转换成 词袋向量，得到词袋向量 mBowVec 以及 特征向量 mFeatVec
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // Step 2：找到与当前帧匹配度高的 候选关键帧组 (用于重定位，系统将使用这些关键帧来尝试估计相机的位姿)。
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if (vpCandidateKFs.empty()) {
        Verbose::PrintMess("\t\tFailed: 没有找到与当前帧匹配度高的候选关键帧，重定位失败！！", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();
    Verbose::PrintMess("\t\t总共找到 nKFs = " + std::to_string(nKFs) + " 个候选关键帧，下面开始匹配遍历...", Verbose::VERBOSITY_NORMAL);

    // We perform first an ORB matching with each candidate. If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true); // 创建一个ORB特征匹配器，最小距离 < 0.75 * 次小距离 匹配成功，true表示要使用双向匹配。

    // 存储 每个关键帧的 解算器
    vector<MLPnPsolver *> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    // 存储 当前帧特征点 与每个关键帧 匹配到的 地图点
    vector<vector<MapPoint *>> vvpMapPointMatches; // 第一维的key：每个关键帧的序号。第二维的key：当前帧特征点索引，value：与当前帧的特征点匹配的 关键帧特征点对应的 地图点
    vvpMapPointMatches.resize(nKFs);

    // 存储放弃某个关键帧的标记
    vector<bool> vbDiscarded; // 如果 vpCandidateKFs 的关键帧与当前帧没有匹配上，则vbDiscarded对应的位置会被赋值为true
    vbDiscarded.resize(nKFs);

    int nCandidates = 0; // 有效的候选关键帧数

    // Step 3：遍历所有的候选关键帧，通过 BoW进行快速匹配，并用匹配结果初始化 PnP Solver
    for (int i = 0; i < nKFs; i++) {
        KeyFrame *pKF = vpCandidateKFs[i];

        // 去除坏的候选关键帧
        if (pKF->isBad()) {
            vbDiscarded[i] = true;
            Verbose::PrintMess("\t\t\t\t\t第 " + std::to_string(i) + " 个候选关键帧是坏的，该关键帧 被放弃", Verbose::VERBOSITY_NORMAL);
        }
        // 该候选关键帧有效
        else {
            // 第一次匹配：当前帧 和 候选关键帧 利用词袋进行快速匹配（与参考关键帧的匹配相同），当前帧特征点 与关键帧i 匹配到的 地图点匹配存储在 vvpMapPointMatches[i] 中。并返回 匹配的数目 nmatches
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            // 如果候选关键帧 和当前帧的匹配数 < 15，那么只能放弃这个关键帧
            if (nmatches < 15) {
                vbDiscarded[i] = true;
                Verbose::PrintMess("\t\t\t\t\t第 " + std::to_string(i) + " 个候选关键帧（相对ID:" + std::to_string(pKF->mnId) + "，绝对ID:" + std::to_string(pKF->mnFrameId) +
                                       "）和 当前帧 词袋匹配的特征点数 nmatches = " + std::to_string(nmatches) + " < 15，该关键帧 被放弃",
                                   Verbose::VERBOSITY_NORMAL);
                SaveSearchByBowFailed(mCurrentFrame, pKF, vvpMapPointMatches[i], 3); // 3表示重定位中的词袋匹配时丢失
                continue;
            } else {
                Verbose::PrintMess("\t\t\t\t\t第 " + std::to_string(i) + " 个候选关键帧（相对ID:" + std::to_string(pKF->mnId) + "，绝对ID:" + std::to_string(pKF->mnFrameId) +
                                       "）和 当前帧 匹配的特征点数 nmatches = " + std::to_string(nmatches) + " >= 15，且当前帧匹配成功的特征点都已在该关键帧中匹配到地图点，该关键帧 被保留",
                                   Verbose::VERBOSITY_NORMAL);
                // 如果匹配数够用( >= 15)，用匹配结果 vvpMapPointMatches[i] 初始化 该关键帧对应的 PnP求解器 MLPnPsolver
                // ? 为什么用MLPnP? 因为考虑了鱼眼相机模型，解耦某些关系？参考论文《MLPNP-A REAL-TIME MAXIMUM LIKELIHOOD SOLUTION TO THE PERSPECTIVE-N-POINT PROBLEM》
                MLPnPsolver *pSolver = new MLPnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                // 构造函数调用了一遍，这里重新设置参数
                pSolver->SetRansacParameters(
                    0.99, // 用于计算RANSAC迭代次数理论值的概率，默认 0.9
                    10,   // 内点的最小阈值，但是要注意在程序中实际上是min(给定最小内点数,最小集,内点数理论值),不一定使用这个，默认 8
                    300,  // 最大迭代次数，默认 300
                    6, // 最小集(求解这个问题在一次采样中所需要采样的最少的点的个数,对于Sim3是3,EPnP是4)，每次采样六个点，即最小集应该设置为6，论文里面写着 I > 5
                    0.5, // 理论最少内点个数，这里是按照总数的比例计算(最小内点数/样本总数)，所以 epsilon 是比例，默认是 0.4，实际上的RANSAC正常退出的时候所需要的最小内点数其实是根据这个量来计算得到的
                    5.991); // 自由度为2的卡方检验的阈值,程序中还会根据特征点所在的图层对这个阈值进行缩放

                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }
    Verbose::PrintMess("\t\t\t\t总共筛选到 nCandidates = " + std::to_string(nCandidates) + " 个有效的候选关键帧", Verbose::VERBOSITY_NORMAL);

    // Alternatively perform some iterations of P4P RANSAC. Until we found a camera pose supported by enough inliers
    // 足够的内点才能匹配使用PNP算法，MLPnP需要至少 6 个点
    bool bMatch = false; // 是否已经找到相匹配的关键帧的标志

    ORBmatcher matcher2(0.9, true); // 创建一个ORB特征匹配器，最小距离 < 0.9 * 次小距离 匹配成功，true表示要使用双向匹配。

    // Step 4: 通过一系列操作，直到找到能够匹配上的关键帧，并通过MLPnP算法估计位姿
    // 为什么搞这么复杂？答：是担心误闭环
    while (nCandidates > 0 && !bMatch) {
        Verbose::PrintMess("\t\t\t\t开始遍历每个候选关键帧，估计相机位姿...", Verbose::VERBOSITY_NORMAL);
        // 遍历当前所有 候选关键帧
        for (int i = 0; i < nKFs; i++) {
            // 忽略已经放弃的
            if (vbDiscarded[i])
                continue;

            vector<bool> vbInliers; // 存储 内点的标记
            int nInliers;           // 存储 内点数
            bool bNoMore;           // = true 表示RANSAC已经没有更多的迭代次数可用 -- 也就是说数据不够好，RANSAC也已经尽力了。。。

            MLPnPsolver *pSolver = vpMLPnPsolvers[i]; // 取出候选关键帧i的PnP求解器
            Eigen::Matrix4f eigTcw;                   // 存储 估计的相机位姿
            Verbose::PrintMess("\t\t\t\t\t第 " + std::to_string(i) + " 个候选关键帧有效，通过 MLPnP算法 估计相机位姿，迭代 5 次 Ransac", Verbose::VERBOSITY_NORMAL);
            // Step 4.1：通过 MLPnP算法 估计相机位姿，迭代 5 次 Ransac
            bool bTcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            // 如果 bNoMore = true，表示已经超过了RANSAC最大迭代次数，就放弃当前关键帧
            if (bNoMore) {
                Verbose::PrintMess("\t\t\t\t\t已经超过了RANSAC最大迭代次数 5 ，当前关键帧 被放弃", Verbose::VERBOSITY_NORMAL);
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            // Step 4.2：如果 MLPnP 计算出了位姿，则对内点进行BA优化
            if (bTcw) {
                Verbose::PrintMess("\t\t\t\t\t\t已经计算出相机的位姿，现进行第一次BA优化", Verbose::VERBOSITY_NORMAL);

                Sophus::SE3f Tcw(eigTcw);   // 创建了一个名为 Tcw 的 Sophus 三维欧氏变换，并用 eigTcw 矩阵初始化它
                mCurrentFrame.SetPose(Tcw); // 将估计的相机位姿 Tcw 应用于当前帧
                // Tcw.copyTo(mCurrentFrame.mTcw);

                // 存储 MLPnP 里RANSAC后的地图点的集合
                set<MapPoint *> sFound;

                const int np = vbInliers.size();

                // 遍历所有内点
                for (int j = 0; j < np; j++) {
                    // 该内点有效
                    if (vbInliers[j]) {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j]; // 将当前帧特征点j 在 第i个关键帧的匹配到的特征点对应的地图点 作为 当前帧的特征点j的地图点
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    // 无效则置空
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                // 第一次BA优化：只优化位姿，不优化地图点的坐标，返回 内点的数量 nGood
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                // 如果优化之后的内点数目不多，则跳过 当前候选关键帧，但是却没有放弃当前帧的重定位
                if (nGood < 10) {
                    Verbose::PrintMess("\t\t\t\t\t\t第一次BA优化之后的内点数 nGood = " + std::to_string(nGood) + " < 10，跳过当前候选关键帧", Verbose::VERBOSITY_NORMAL);
                    continue;
                }

                // 删除外点对应的地图点，这里直接设为空指针
                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // Step 4.3：如果内点较少( < 50)，则通过投影的方式对之前 未匹配的点 进行匹配，再进行优化求解
                // 前面的匹配关系是用词袋匹配过程得到的
                if (nGood < 50) {
                    Verbose::PrintMess("\t\t\t\t\t\t第一次BA优化之后的内点数 10 <= nGood = " + std::to_string(nGood) + " < 50，可通过投影的方式对之前 未匹配的点 再次进行匹配，抢救一下",
                                       Verbose::VERBOSITY_NORMAL);
                    // 第二次匹配：通过投影的方式将 候选关键帧i中 未匹配的地图点 投影到当前帧中, 生成新的匹配，返回 新匹配的地图点数目 nadditional （与恒速运动模型跟踪中的相同）
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    // 如果通过投影过程新增了比较多的匹配特征点对
                    if (nadditional + nGood >= 50) {
                        Verbose::PrintMess("\t\t\t\t\t\t\t通过投影过程新增的匹配特征点数 nadditional = " + std::to_string(nadditional) + " + 第一次优化之后的内点数 nGood " + std::to_string(nGood) +
                                               " = " + std::to_string(nadditional + nGood) + " >= 50，进行第二次BA优化位姿",
                                           Verbose::VERBOSITY_NORMAL);
                        // 第二次BA优化：根据第一次投影匹配的结果，再次采用 3D-2D pnp BA优化位姿
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        // Step 4.4：如果 BA 优化后内点数还是比较少 (< 50) 但 还不至于太少(> 30)，可以挽救一下, 最后垂死挣扎
                        // 重新执行上一步 4.3的过程，只不过使用更小的搜索窗口。这里的位姿已经使用了更多的点进行了优化，应该更准，所以使用更小的窗口搜索
                        if (nGood > 30 && nGood < 50) {
                            Verbose::PrintMess("\t\t\t\t\t\t\t第二次BA优化之后的内点数 30 < nGood =  " + std::to_string(nGood) + " < 50，可再次通过投影的方式对之前 未匹配的点 进行匹配，再抢救一下",
                                               Verbose::VERBOSITY_NORMAL);

                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            // 第三次匹配：再次进行投影匹配。用更小窗口、更严格的描述子阈值
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            // 如果成功挽救回来，匹配数目达到要求，最后BA优化一下
                            if (nGood + nadditional >= 50) {
                                Verbose::PrintMess("\t\t\t\t\t\t\t再次通过投影过程新增的匹配特征点数 nadditional = " + std::to_string(nadditional) + " + 第二次优化之后的内点数 nGood " +
                                                       std::to_string(nGood) + " = " + std::to_string(nadditional + nGood) + " >= 50，进行第三次BA优化位姿",
                                                   Verbose::VERBOSITY_NORMAL);
                                // 第三次BA优化
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                // 更新地图点。删除外点对应的地图点，这里直接设为空指针
                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                            // 如果还是不能够满足就放弃了 第二次BA优化内点个数 nGood + 第三次匹配新增个数 nadditional < 50
                        }
                        // 第二次BA优化内点个数 nGood <= 30 || nGood >= 50
                    }
                    // 第二次匹配新增个数 nadditional + 第一次BA优化内点个数nGood < 50
                }
                // nGood >= 50
                // If the pose is supported by enough inliers stop ransacs and continue
                // 如果对于当前的候选关键帧已经有足够的内点( > 50个)，那么就认为重定位成功
                if (nGood >= 50) {
                    Verbose::PrintMess("\t\t\t\t\t\t最终优化之后的内点数 nGood = " + std::to_string(nGood) + " >= 50，当前候选关键帧已经有足够的内点", Verbose::VERBOSITY_NORMAL);
                    bMatch = true;
                    // 只要有一个候选关键帧重定位成功，就退出循环，不考虑其他候选关键帧了
                    break;
                } else
                    Verbose::PrintMess("\t\t\t\t\t\t最终优化之后的内点数 nGood = " + std::to_string(nGood) + " < 50，当前候选关键帧没有足够的内点", Verbose::VERBOSITY_NORMAL);
            }
            // 未计算出位姿
        } // 所有关键帧遍历结束
    }

    // 折腾了这么久还是没有匹配上，重定位失败
    if (!bMatch) {
        Verbose::PrintMess("\t\t\t\tFailed: 未找到和当前帧匹配的关键帧，重定位失败！！", Verbose::VERBOSITY_NORMAL);
        return false;
    }
    // 如果匹配上了，说明当前帧重定位成功了(当前帧已经有了自己的位姿)
    else {
        // 记录成功重定位帧的id，防止短时间多次重定位
        mnLastRelocFrameId = mCurrentFrame.mnId;
        Verbose::PrintMess("\t\t\t\tsucceed: 已找到和当前帧匹配的关键帧，且已计算出当前帧的位姿，重定位成功！", Verbose::VERBOSITY_NORMAL);
        return true;
    }
}

void Tracking::Reset(bool bLocMap) {
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if (mpViewer) {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap) {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit = false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame *>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
    mvIniMatches.clear();

    if (mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap) {
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if (mpViewer) {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            usleep(3000);
    }

    Map *pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap) {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();

    // KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    // Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    // mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; // NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for (Map *pMap : mpAtlas->GetAllMaps()) {
        if (pMap->GetAllKeyFrames().size() > 0) {
            if (index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    // cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for (list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++) {
        if (index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame *>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if (mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint *> Tracking::GetLocalMapMPS() { return mvpLocalMapPoints; }

void Tracking::ChangeCalibration(const string &strSettingPath) {
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0, 0) = fx;
    mK_(1, 1) = fy;
    mK_(0, 2) = cx;
    mK_(1, 2) = cy;

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0) {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag) { mbOnlyTracking = flag; }

/**
 * @brief IMU初始化后更新了关键帧的位姿，因此需要更新每一普通帧的位姿。LocalMapping中IMU初始化中调用
 * Step 1: 尺度对齐 相对位姿中的平移部分
 * Step 2: 更新当前帧与上一帧的IMU预积分结果
 * 最开始 平均速度定义于IMU第一阶段初始化时，每个关键帧的速度都根据 前后关键帧之间的IMU位移/时间得到，经过非线性优化保存于KF中.
 * 上一帧与当前帧 分别与它们的上一关键帧做速度叠加得到，后面新的帧的速度由上一个帧的速度决定，如果使用匀速模型（大多数情况下），通过IMU积分更新速度
 * @param [in] s 更新后的 尺度
 * @param [in] b 更新后的 初始关键帧的 陀螺仪和加速度计零偏
 * @param [in] pCurrentKeyFrame 当前关键帧
 */
void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame) {
    Map *pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId; // 第一帧的ID

    // 存储每一普通帧的 参考关键帧 的初始迭代器
    list<ORB_SLAM3::KeyFrame *>::iterator lRit = mlpReferences.begin();
    // 存储每一普通帧是否 跟踪丢失 的初始迭代器
    list<bool>::iterator lbL = mlbLost.begin();

    // Step 1: 更新参考关键帧 到 各普通帧 的相对位姿的 尺度
    // mlRelativeFramePoses 存放参考关键帧 到 当前帧的 相对位姿 Tcr。在IMU初始化之前里面的数据没有尺度，所以要更新下尺度
    for (auto lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lbL++) {
        // 该普通帧丢失，则跳过
        if (*lbL)
            continue;

        // 该普通帧的 参考关键帧
        KeyFrame *pKF = *lRit;
        while (pKF->isBad()) {
            pKF = pKF->GetParent(); // 坏的，则用其父关键帧代替
        }

        //　更新该普通帧的相对位姿的 平移部分
        if (pKF->GetMap() == pMap) {
            (*lit).translation() *= s; // Tcr的平移部分 * 尺度s
        }
    }

    // 设置上一零偏 为 优化后的初始关键帧的零偏
    mLastBias = b;

    // 设置上一关键帧 为 当前关键帧
    // 如果说mpLastKeyFrame已经是经过添加的新的kf，而pCurrentKeyFrame还是上一个kf，mpLastKeyFrame直接指向之前的kf
    mpLastKeyFrame = pCurrentKeyFrame;

    // 更新上一帧与当前帧的零偏 为  优化后的初始关键帧的零偏
    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    // 若当前帧还未IMU预积分，则等待其预积分完毕 (这段函数是在LocalMapping里调用的)
    while (!mCurrentFrame.imuIsPreintegrated()) {
        usleep(500);
    }

    // TODO 如果上一帧正好是上一帧的上一关键帧（mLastFrame.mpLastKeyFrame与mLastFrame不可能是一个，可以验证一下）
    // 上一帧 是 上一帧的上一关键帧，则设置上一帧的IMU旋转、位置和速度 = 上一帧的上一关键帧的IMU旋转、位置、速度
    if (mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId) {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(), mLastFrame.mpLastKeyFrame->GetImuPosition(), mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    // 不是，则重新计算 上一帧的IMU预积分结果 (包括旋转、位置、速度，根据上一帧的上一个关键帧的IMU信息 和 上一帧的IMU预积分结果)
    else {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);                      // 重力方向
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition(); // 上一帧的上一关键帧的IMU位置向量
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation(); // 上一帧的上一关键帧的IMU旋转矩阵
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();    // 上一帧的上一关键帧的速度向量
        float t12 = mLastFrame.mpImuPreintegrated->dT;                            // 上一帧的IMU预积分的时间间隔

        // mpImuPreintegrated表示从上一关键帧积分到当前普通帧一段的预积分
        // mpImuPreintegrated->GetUpdatedDeltaRotation()取到了这段时间的 Delta R
        // 也就是上一段时间的旋转R_wb1（上一个关键帧中IMU坐标系在世界坐标系下的坐标） * 这段时间的预积分Delta R（这段时间内R的变换值）得到（这个帧IMU坐标系在世界坐标系下的坐标），其他变量更新同理

        // 根据上一帧的上一个关键帧的IMU信息（此时IMU已经初始化了，所以关键帧的信息都是校正后的）和 上一帧的IMU预积分结果，重新计算 上一帧的IMU位姿和速度
        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz * t12 + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    // 当前帧已完成预积分，则重新计算 当前帧的IMU预积分结果 (包括旋转、位置、速度，根据当前帧的上一关键帧的IMU信息 与 当前帧的预积分结果)
    if (mCurrentFrame.mpImuPreintegrated) {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        // 根据 当前帧的上一关键帧的IMU信息 与 当前帧的预积分结果，重新计算 当前帧的IMU旋转、位置、速度
        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                         twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                         Vwb1 + Gz * t12 + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }
    // 设置第一个IMU更新帧 为 当前帧
    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset() { mnNumDataset++; }

int Tracking::GetNumberDataset() { return mnNumDataset; }

int Tracking::GetMatchesInliers() { return mnMatchesInliers; }

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder) {
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    // mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map *pMap) {
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if (!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale() { return mImageScale; }

#ifdef REGISTER_LOOP
void Tracking::RequestStop() {
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop() {
    unique_lock<mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop) {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested() {
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped() {
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release() {
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

// -----------liuzhi加------------------------
/**
 *@brief 保存参考关键帧跟踪丢失时的 特征点对匹配关系
 * @param F
 * @param KF
 * @param vMapPointMatches
 * @param img_idx 1表示参考关键帧跟踪 词袋匹配时丢失
 *                2表示参考关键帧跟踪 位姿优化后丢失
 *                3表示重定位中 词袋匹配时丢失
 */
void Tracking::SaveSearchByBowFailed(Frame &F, KeyFrame *&KF, std::vector<MapPoint *> &vMapPointMatches, const int img_idx) {
    // 获取当前帧和参考关键帧的特征点
    vector<cv::KeyPoint> vCurrentKeyPoints = F.mvKeysUn;
    vector<cv::KeyPoint> vReferenceKeyPoints = KF->mvKeysUn;
    // 获取当前帧和参考关键帧的匹配关系
    vector<cv::DMatch> matches;
    // 遍历当前帧每个特征点
    for (int i = 0; i < vMapPointMatches.size(); i++) {
        if (vMapPointMatches[i]) // 存在匹配的地图点
        {
            cv::DMatch m{i, 0, 256};
            tuple<int, int> indexes = vMapPointMatches[i]->GetObservations()[KF]; // 该地图点 被 参考关键帧KF 的哪个特征点观测到
            int idx = get<0>(indexes);                                            // 参考帧中观测到该地图点的 特征点索引
            m.trainIdx = idx;
            matches.push_back(m);
        }
    }
    // 创建一个用于存储匹配结果图像的矩阵
    cv::Mat imgMatches;
    cv::drawMatches(F.imgLeft, vCurrentKeyPoints, KF->imgLeft, vReferenceKeyPoints, matches, imgMatches);

    string pathMatches = "./Images/";
    if (img_idx == 3) {
        pathMatches = "./Images/Relocalization/SBB_Match_F_" + std::to_string(F.mnId) + "_KF_" + std::to_string(KF->mnFrameId) + "_fp_" + std::to_string(F.mvKeysUn.size()) + "_kfp_" +
                      std::to_string(KF->mvKeysUn.size()) + "_matches_" + std::to_string(matches.size()) + "_" + std::to_string(img_idx) + ".png";
    } else
        pathMatches = "./Images/SBB_Match_F_" + std::to_string(F.mnId) + "_KF_" + std::to_string(KF->mnFrameId) + "_fp_" + std::to_string(F.mvKeysUn.size()) + "_kfp_" +
                      std::to_string(KF->mvKeysUn.size()) + "_matches_" + std::to_string(matches.size()) + "_" + std::to_string(img_idx) + ".png";
    cv::imwrite(pathMatches, imgMatches);
}
// void Tracking::SaveSearchByProjectionFailed(Frame &F, KeyFrame* &KF)
//{
//
//}

// --------------------------------------------------

} // namespace ORB_SLAM3
