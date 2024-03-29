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

#include "System.h"
#include "Converter.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <iomanip>
#include <openssl/md5.h>
#include <pangolin/pangolin.h>
#include <thread>

namespace ORB_SLAM3 {

Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NORMAL;

/**
 * @brief System的构造函数，将会启动其他的线程
 * @param strVocFile    词袋模型路径
 * @param strSettingsFile 配置文件路径
 * @param sensor        传感器类型
 * @param bUseViewer    是否使用可视化界面
 * @param initFr        初始化帧ID，开始设置为 0
 * @param strSequence   序列名，在跟踪线程和局部建图线程用得到
 */
// mpViewer：使用 Pangolin 画图和相机位姿，这里先将其强制转换成乘以Viewer空指针
System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer, const int initFr, const string &strSequence)
    : mSensor(sensor), mpViewer(static_cast<Viewer *>(NULL)), mbReset(false), mbResetActiveMap(false), mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false), mbShutDown(false) {
    cout << endl << "[System::System] 创建System" << endl;
    // Output welcome message
    cout << endl
         << "ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl
         << "ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl
         << "This program comes with ABSOLUTELY NO WARRANTY;" << endl
         << "This is free software, and you are welcome to redistribute it" << endl
         << "under certain conditions. See LICENSE.txt." << endl
         << endl;

    cout << "Input sensor was set to: ";

    if (mSensor == MONOCULAR)
        cout << "Monocular" << endl;
    else if (mSensor == STEREO)
        cout << "Stereo" << endl;
    else if (mSensor == RGBD)
        cout << "RGB-D" << endl;
    else if (mSensor == IMU_MONOCULAR)
        cout << "Monocular-Inertial" << endl;
    else if (mSensor == IMU_STEREO)
        cout << "Stereo-Inertial" << endl;
    else if (mSensor == IMU_RGBD)
        cout << "RGB-D-Inertial" << endl;

    std::cout << "打开配置文件 strSettingsFile: " << strSettingsFile.c_str() << std::endl;
    // Check settings file
    // 打开配置文件
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ); // c_str() 为将文件名转换成字符串
    if (!fsSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }

    // 查看配置文件版本，不同版本有不同处理方法
    cv::FileNode node = fsSettings["File.version"];
    // 配置文件版本为 1.0，则新建配置对象，加载相应参数
    if (!node.empty() && node.isString() && node.string() == "1.0") {
        // 创建一个 Settings 对象，并将指向该对象的指针赋值给 settings_
        settings_ = new Settings(strSettingsFile, mSensor);
        // 加载和保存地图的名字
        mStrLoadAtlasFromFile = settings_->atlasLoadFile(); // 单目下为空
        mStrSaveAtlasToFile = settings_->atlasSaveFile();   // 单目下为空

        cout << (*settings_) << endl; // * 解引用 settings_ 指针，输出 settings 指向的对象
    } else {
        settings_ = nullptr;
        cv::FileNode node = fsSettings["System.LoadAtlasFromFile"];
        if (!node.empty() && node.isString()) {
            mStrLoadAtlasFromFile = (string)node;
        }

        node = fsSettings["System.SaveAtlasToFile"];
        if (!node.empty() && node.isString()) {
            mStrSaveAtlasToFile = (string)node;
        }
    }

    // 是否激活回环：没有相应节点则默认为true；有节点则按节点来
    node = fsSettings["loopClosing"];
    bool activeLC = true;

    // loopClosing节点不空，则将其转换为整型并判断是否为0。不为0 则activeLC置为true；否则置为false
    if (!node.empty()) {
        activeLC = static_cast<int>(fsSettings["loopClosing"]) != 0;
    }

    mStrVocabularyFilePath = strVocFile; // 词袋模型路径

    bool loadedAtlas = false; // 加载Atlas标识符：ORBSLAM3新加的多地图管理功能

    // 加载 ORB 词袋模型
    if (mStrLoadAtlasFromFile.empty()) {
        // 加载 ORB 词袋模型，用于识别地图点和特征匹配
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORBVocabulary(); // 新建一个词袋 ORBVocabulary

        // 加载预训练好的ORB词袋模型
        bool bVocLoad = false;
        if (strVocFile.find(".txt") != std::string::npos) {
            cout << endl << "\t加载 txt 格式的词袋..." << endl;
            bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        } else if (strVocFile.find(".bin") != std::string::npos) {
            cout << endl << "\t加载 bin 格式的词袋..." << endl;
            bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
        } else
            cout << endl << "\t没有找到 txt 或 bin 格式的词袋" << endl;

        if (!bVocLoad) {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        // Create KeyFrame Database
        // 创建关键帧数据库
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        // Create the Atlas
        // 创建地图集 Atlas，初始化关键帧id为0
        cout << "Initialization of Atlas from scratch " << endl;
        mpAtlas = new Atlas(0);
    } else {
        // Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if (!bVocLoad) {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        // Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        cout << "Load File" << endl;

        // Load the file with an earlier session
        // clock_t start = clock();
        cout << "Initialization of Atlas from file: " << mStrLoadAtlasFromFile << endl;
        bool isRead = LoadAtlas(FileType::BINARY_FILE);

        if (!isRead) {
            cout << "Error to load the file, please try with other session file or vocabulary file" << endl;
            exit(-1);
        }
        // mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        // cout << "KF in DB: " << mpKeyFrameDatabase->mnNumKFs << "; words: " << mpKeyFrameDatabase->mnNumWords << endl;

        loadedAtlas = true;

        mpAtlas->CreateNewMap();

        // clock_t timeElapsed = clock() - start;
        // unsigned msElapsed = timeElapsed / (CLOCKS_PER_SEC / 1000);
        // cout << "Binary file read in " << msElapsed << " ms" << endl;

        // usleep(10*1000*1000);
    }

    // IMU模式，则需设置惯性传感器，即将Map中的 mbIsInertial 置为true。以后的跟踪和预积分将和这个标志有关
    if (mSensor == IMU_STEREO || mSensor == IMU_MONOCULAR || mSensor == IMU_RGBD)
        mpAtlas->SetInertialSensor();

    // 创建跟踪、局部建图、闭环、显示线程

    // 创建用于显示帧和地图的类，由Viewer调用
    mpFrameDrawer = new FrameDrawer(mpAtlas);
    mpMapDrawer = new MapDrawer(mpAtlas, strSettingsFile, settings_);

    //! 创建 Tracking 线程，不会立刻开启，会在对图像和IMU预处理后在 main 函数中执行。
    // 在创建时，会加载配置设置
    cout << "Seq. Name: " << strSequence << endl << "创建跟踪线程" << endl;
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer, mpAtlas, mpKeyFrameDatabase, strSettingsFile, mSensor, settings_, strSequence);

    //! 创建并开启 局部建图 线程，线程函数名为 LocalMapping::Run()
    mpLocalMapper = new LocalMapping(this, mpAtlas, mSensor == MONOCULAR || mSensor == IMU_MONOCULAR, mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD, strSequence);
    mptLocalMapping = new thread(&ORB_SLAM3::LocalMapping::Run, mpLocalMapper);
    // 初始化帧的id，代码中设置为0
    mpLocalMapper->mInitFr = initFr;
    // 设置区分远点和近点的阈值，即最远3D地图点的深度值，如果超过阈值，说明可能三角化不太准确，丢弃
    if (settings_)
        mpLocalMapper->mThFarPoints = settings_->thFarPoints();
    else
        mpLocalMapper->mThFarPoints = fsSettings["thFarPoints"];

    if (mpLocalMapper->mThFarPoints != 0) {
        cout << "Discard points further than " << mpLocalMapper->mThFarPoints << " m from current camera" << endl;
        mpLocalMapper->mbFarPoints = true;
    } else
        mpLocalMapper->mbFarPoints = false;

    //! 创建并开启 回环检测 线程
    mpLoopCloser = new LoopClosing(mpAtlas, mpKeyFrameDatabase, mpVocabulary, mSensor != MONOCULAR, activeLC); // mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM3::LoopClosing::Run, mpLoopCloser);

    //! 将各线程通过指针建立起联系
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    // usleep(10*1000*1000);

    //! 创建并开启 显示 线程
    if (bUseViewer)
    // if(false) // TODO
    {
        mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile, settings_);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
        mpLoopCloser->mpViewer = mpViewer;
        mpViewer->both = mpFrameDrawer->both;
    }

    // 打印输出中间的信息，设置为安静模式
    // 设置打印信息的等级，只有小于这个等级的信息才会被打印
    //    Verbose::SetTh(Verbose::VERBOSITY_QUIET);   // QUIET 表示不会打印
    //    freopen("log_run.txt", "w", stdout);
    //    freopen("log_err.txt", "w", stderr);
    Verbose::SetTh(Verbose::VERBOSITY_DEBUG); // 调整需要打印的信息的类别
    std::cout << "[System创建结束]" << std::endl;
}

/**
 * @brief 双目 / 双目+IMU跟踪
 * @param imLeft    左目图像
 * @param imRight   右目图像
 * @param timestamp 当前帧的图片时间戳 (s)
 * @param vImuMeas  上一帧至当前帧的 IMU测量数据 (非IMU模式下为空)
 * @param filename  左目图片路径 (IMU模式下为空)
 * @return
 */
Sophus::SE3f System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point> &vImuMeas, string filename) {
    if (mSensor != STEREO && mSensor != IMU_STEREO) {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to Stereo nor Stereo-Inertial." << endl;
        exit(-1);
    }

    cv::Mat imLeftToFeed, imRightToFeed;
    // 若需极线矫正，则加载矫正映射参数，对输入图像remap
    if (settings_ && settings_->needToRectify()) {
        std::cout << "[System::TrackStereo] 需进行极线矫正，加载畸变映射参数对输入图像remap" << std::endl;
        cv::Mat M1l = settings_->M1l();
        cv::Mat M2l = settings_->M2l();
        cv::Mat M1r = settings_->M1r();
        cv::Mat M2r = settings_->M2r();

        cv::remap(imLeft, imLeftToFeed, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(imRight, imRightToFeed, M1r, M2r, cv::INTER_LINEAR);
    }
    // 若需调整图像大小
    else if (settings_ && settings_->needToResize()) {
        std::cout << "[System::TrackStereo] 需调整图像大小" << std::endl;
        cv::resize(imLeft, imLeftToFeed, settings_->newImSize());
        cv::resize(imRight, imRightToFeed, settings_->newImSize());
    } else {
        std::cout << "[System::TrackStereo] 无需调整图像，直接传输" << std::endl;
        imLeftToFeed = imLeft.clone();
        imRightToFeed = imRight.clone();
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if (mbActivateLocalizationMode) {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while (!mpLocalMapper->isStopped()) {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }

        if (mbDeactivateLocalizationMode) {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbReset) {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        } else if (mbResetActiveMap) {
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    // 双目+IMU
    if (mSensor == System::IMU_STEREO) {
        // 将上一帧到当前帧的IMU数据 ，传递给Tracking中的 mlQueueImuData 链表里
        for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++) {
            mpTracker->GrabImuData(vImuMeas[i_imu]);
        }
    }

    // 进行双目跟踪
    Sophus::SE3f Tcw = mpTracker->GrabImageStereo(imLeftToFeed, imRightToFeed, timestamp, filename);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

Sophus::SE3f System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point> &vImuMeas, string filename) {
    if (mSensor != RGBD && mSensor != IMU_RGBD) {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }

    cv::Mat imToFeed = im.clone();
    cv::Mat imDepthToFeed = depthmap.clone();
    if (settings_ && settings_->needToResize()) {
        cv::Mat resizedIm;
        cv::resize(im, resizedIm, settings_->newImSize());
        imToFeed = resizedIm;

        cv::resize(depthmap, imDepthToFeed, settings_->newImSize());
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if (mbActivateLocalizationMode) {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while (!mpLocalMapper->isStopped()) {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if (mbDeactivateLocalizationMode) {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbReset) {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        } else if (mbResetActiveMap) {
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    if (mSensor == System::IMU_RGBD)
        for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    Sophus::SE3f Tcw = mpTracker->GrabImageRGBD(imToFeed, imDepthToFeed, timestamp, filename);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

// 单目跟踪：获取 单目或单目+IMU 模式下的 相机位姿。(in&out 图像Mat矩阵；in&out 对应的时间戳)
Sophus::SE3f System::TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point> &vImuMeas, string filename) {

    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbShutDown)
            return Sophus::SE3f();
    }

    // 检查传感器类型，不为单目或单目IMU，则报错
    if (mSensor != MONOCULAR && mSensor != IMU_MONOCULAR) {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular nor Monocular-Inertial." << endl;
        exit(-1);
    }
    // 如果需要改变图像尺寸，则返回重定义尺寸的灰度图
    cv::Mat imToFeed = im.clone();
    if (settings_ && settings_->needToResize()) {
        cv::Mat resizedIm;
        cv::resize(im, resizedIm, settings_->newImSize());
        imToFeed = resizedIm;
    }

    // Check mode change
    // 检查跟踪的模式
    {
        unique_lock<mutex> lock(mMutexMode); // 独占锁，防止 mbActivateLocalizationMode 和 mbDeactivateLocalizationMode 变量不会被其他线程调用发生混乱

        // 如果是 纯定位模式，局部建图线程会关闭
        if (mbActivateLocalizationMode) {
            mpLocalMapper->RequestStop(); // 请求停止局部建图线程

            // Wait until Local Mapping has effectively stopped
            while (!mpLocalMapper->isStopped()) {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true); // 告诉Tracking线程，现在只进行跟踪，只计算相机位姿，不进行局部地图的更新
            mbActivateLocalizationMode = false;  // 设置为false，避免重复进行以上操作
        }

        // 如果需要 关闭纯定位，则 重新打开局部建图线程
        if (mbDeactivateLocalizationMode) {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    // 检查是否需要重置 整个跟踪器还是仅活动地图
    {
        unique_lock<mutex> lock(mMutexReset);
        if (mbReset) // 重置整个跟踪器
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        } else if (mbResetActiveMap) // 仅重置活动地图
        {
            cout << "SYSTEM-> Reseting active map in monocular case" << endl;
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    // 如果是 单目+IMU模式，把IMU数据存储到 mlQueueImuData 中
    if (mSensor == System::IMU_MONOCULAR)
        for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    // ------------开始跟踪，计算相机位姿：[更新尺寸后的图像；时间戳]--------------
    Sophus::SE3f Tcw = mpTracker->GrabImageMonocular(imToFeed, timestamp, filename);

    // 更新跟踪状态和参数
    unique_lock<mutex> lock2(mMutexState);

    mTrackingState = mpTracker->mState;                        // 记录跟踪状态
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints; // 当前帧跟踪到的地图点
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;   // 当前帧的未畸变关键点

    return Tcw;
}

void System::ActivateLocalizationMode() {
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode() {
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged() {
    static int n = 0;
    int curn = mpAtlas->GetLastBigChangeIdx();
    if (n < curn) {
        n = curn;
        return true;
    } else
        return false;
}

void System::Reset() {
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::ResetActiveMap() {
    unique_lock<mutex> lock(mMutexReset);
    mbResetActiveMap = true;
}

void System::Shutdown() {
    {
        unique_lock<mutex> lock(mMutexReset);
        mbShutDown = true;
    }

    cout << "Shutdown" << endl;

    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    /*if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }*/

    // Wait until all thread have effectively stopped
    /*while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        if(!mpLocalMapper->isFinished())
            cout << "mpLocalMapper is not finished" << endl;*/
    /*if(!mpLoopCloser->isFinished())
        cout << "mpLoopCloser is not finished" << endl;
    if(mpLoopCloser->isRunningGBA()){
        cout << "mpLoopCloser is running GBA" << endl;
        cout << "break anyway..." << endl;
        break;
    }*/
    /*usleep(5000);
}*/

    if (!mStrSaveAtlasToFile.empty()) {
        Verbose::PrintMess("Atlas saving to file " + mStrSaveAtlasToFile, Verbose::VERBOSITY_NORMAL);
        SaveAtlas(FileType::BINARY_FILE);
    }

    /*if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");*/

#ifdef REGISTER_TIMES
    mpTracker->PrintTimeStats();
#endif
}

bool System::isShutDown() {
    unique_lock<mutex> lock(mMutexReset);
    return mbShutDown;
}

void System::SaveTrajectoryTUM(const string &filename) {
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if (mSensor == MONOCULAR) {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for (list<Sophus::SE3f>::iterator lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++) {
        if (*lbL)
            continue;

        KeyFrame *pKF = *lRit;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while (pKF->isBad()) {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Two;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();

        Eigen::Vector3f twc = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();

        f << setprecision(6) << *lT << " " << setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    f.close();
    // cout << endl << "trajectory saved!" << endl;
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename) {
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];

        // pKF->SetPose(pKF->GetPose()*Two);

        if (pKF->isBad())
            continue;

        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        Eigen::Vector3f t = Twc.translation();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }

    f.close();
}

/**
 * 保存相机位姿：以 Tum 格式保存所有帧的位姿
 * @param filename
 */
void System::SaveTrajectoryEuRoC(const string &filename) {
    cout << endl << "Saving trajectory to " << filename << " ..." << endl;
    /*if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }*/
    // 1. 从地图集中获取所有的地图，存储在名为 vpMaps 的vector<Map*>中
    vector<Map *> vpMaps = mpAtlas->GetAllMaps();
    int numMaxKFs = 0;
    Map *pBiggerMap;
    std::cout << "There are " << std::to_string(vpMaps.size()) << " maps in the atlas" << std::endl;

    // 寻找 拥有最多关键帧的 地图，并获取该地图中的 最大关键帧数
    for (Map *pMap : vpMaps) {
        std::cout << "  Map " << std::to_string(pMap->GetId()) << " has " << std::to_string(pMap->GetAllKeyFrames().size()) << " KFs" << std::endl;
        if (pMap->GetAllKeyFrames().size() > numMaxKFs) {
            numMaxKFs = pMap->GetAllKeyFrames().size(); // 当前地图所有关键帧的个数
            pBiggerMap = pMap;                          // 将当前地图设为 关键帧更多的地图
        }
    }
    // 2. 从拥有最多关键帧地图中的 获取其所有关键帧
    vector<KeyFrame *> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId); // 按ID排序

    // Transform all keyframes so that the first keyframe is at the origin. 转换所有的关键帧，使得第一个关键帧在原点
    // After a loop closure the first keyframe might not be at the origin. 在闭环之后，第一个关键帧可能不在原点

    // 3. 初始位姿 Twb
    Sophus::SE3f Twb; // Can be world to cam0 or world to b depending on IMU or not.
    // IMU 模式，Twb 为第一个关键帧的IMU位姿
    if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose(); // 如果使用IMU，
    // 视觉模式（非IMU），Twb 为第一个关键帧的逆位姿。第一个关键帧的位姿为世界坐标原点。
    else
        Twb = vpKFs[0]->GetPoseInverse();

    // 4. 打开位姿存储文件
    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // 5. 保存位姿数据
    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // 帧位姿的存储是相对于它的参考关键帧（由BA和位姿图进行优化），Tcr
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // ? 我们需要获取第一个关键帧位姿并将它与 相对变换 拼接
    // Frames not localized (tracking failure) are not saved.
    // 未定位（跟踪失败）的帧不会被保存

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag which is true when tracking failed (lbL).
    // 每一帧都有一个参考关键帧，它相对于参考关键帧有一个相对位姿变换，所以每一帧的位姿是按相对参考关键帧的位姿保存的
    // 这样保存的目的在于：关键帧位姿在 建图localmapping 中是不断调整的，所以认为更加准确，如果直接保存帧的位姿，那就没那么准确

    list<ORB_SLAM3::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin(); // 参考关键帧列表 迭代器
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();                   // 时间戳列表 迭代器
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();                         // 跟踪失败标志位 迭代器

    // cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    // cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    // cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    // cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;

    // 对于每一帧，遍历 当前帧相对其参考关键帧的位姿 mlRelativeFramePoses
    for (auto lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++) {
        // 如果当前帧跟踪失败，没有位姿，则跳过
        if (*lbL)
            continue;

        // 获取当前帧的 参考关键帧
        KeyFrame *pKF = *lRit;
        // cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw; // 当前帧好的参考关键位姿

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        // 如果当前帧没有参考关键帧，则跳过不保存
        if (!pKF)
            continue;

        // 如果当前帧的参考关键帧被标记为 坏的(可能是检查冗余时被剔除了)，则迭代找到一个有效的父关键帧，并计算累积的父关键帧到当前参考关键帧的 相对位姿
        while (pKF->isBad()) {
            Trw = Trw * pKF->mTcp;  // 累积 坏的当前关键帧 相对 好的父关键帧 的位姿
            pKF = pKF->GetParent(); // 更新为好的父关键帧
            // cout << "--Parent KF: " << pKF->mnId << endl;
        }
        // 如果找不到合适的参考关键帧 或 找到的参考关键帧不在最多关键帧的地图中，则跳过不保存
        if (!pKF || pKF->GetMap() != pBiggerMap) {
            // cout << "--Parent KF is from another map" << endl;
            continue;
        }

        // 计算旧的参考关键帧 -> 世界坐标系 的位姿
        // Trw：             (1)当前关键帧是好的，为 单位阵；              (2)当前关键帧是坏的，则为 坏的当前关键帧 相对 其好的父关键帧 的累积位姿 Trp
        // pKF->GetPose()：                       当前关键帧位姿 Trw；                         好的父关键帧位姿 Tpw
        // Trw * pKF->GetPose()：                 当前关键帧位姿 Trw;                          坏的当前关键帧位姿 Trw = Trp * Tpw
        // Trw * pKF->GetPose() * Twb：当前关键帧相对于第一个关键帧（世界坐标原点）的位姿 Trw
        Trw = Trw * pKF->GetPose() * Twb;

        // 获取当前帧位姿的 旋转 和 平移，并将其保存到文件中
        // IMU模式，计算Twb（IMU坐标系到当前帧位姿变换），并提取旋转和平移
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD) {
            // 计算 当前帧相对于IMU坐标系的位姿
            Sophus::SE3f Twb =
                (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse(); // IMU到相机坐标系的变换(pKF->mImuCalib.mTbc) * 当前帧相对于参考关键帧的相对位姿(*lit) * 之前计算的相对于世界坐标系的位姿(Trw)
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            // 将时间戳、平移和四元数写入输出文件：1e9 * (*lT) 将时间戳转换为纳秒；twb(0)、twb(1) 和 twb(2) 是平移向量的X、Y 和 Z 分量；q.x()、q.y()、q.z() 和 q.w() 是四元数的分量
            f << 1e9 * (*lT) << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        // 纯视觉（非IMU），计算当前帧位姿 Twc，并提取旋转和平移，然后将其保存到文件中
        else {
            Sophus::SE3f Twc = ((*lit) * Trw).inverse();  // Twc = (Tcr * Trw)^-1
            Eigen::Quaternionf q = Twc.unit_quaternion(); // 单位四元数表示的 旋转
            Eigen::Vector3f twc = Twc.translation();      // 平移
            // TartanAir数据集
            //            f <<  int(*lT * 20) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
            // Euroc数据集，其前几十帧没有真值
            f << 1e9 * (*lT) << " " << setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
    }
    // cout << "end saving trajectory" << endl;
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}

void System::SaveTrajectoryEuRoC(const string &filename, Map *pMap) {

    cout << endl << "Saving trajectory of map " << pMap->GetId() << " to " << filename << " ..." << endl;
    /*if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }*/

    int numMaxKFs = 0;

    vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
    if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose();
    else
        Twb = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    // cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    // cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    // cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    // cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;

    for (auto lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++) {
        // cout << "1" << endl;
        if (*lbL)
            continue;

        KeyFrame *pKF = *lRit;
        // cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        // cout << "2.5" << endl;

        while (pKF->isBad()) {
            // cout << " 2.bad" << endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
            // cout << "--Parent KF: " << pKF->mnId << endl;
        }

        if (!pKF || pKF->GetMap() != pMap) {
            // cout << "--Parent KF is from another map" << endl;
            continue;
        }

        // cout << "3" << endl;

        Trw = Trw * pKF->GetPose() * Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;

        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD) {
            Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        } else {
            Sophus::SE3f Twc = ((*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f twc = Twc.translation();
            f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    // cout << "end saving trajectory" << endl;
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}

/*void System::SaveTrajectoryEuRoC(const string &filename)
{

    cout << endl << "Saving trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose_();
    else
        Twb = vpKFs[0]->GetPoseInverse_();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //cout << "1" << endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //cout << "2.5" << endl;

        while(pKF->isBad())
        {
            //cout << " 2.bad" << endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
            //cout << "--Parent KF: " << pKF->mnId << endl;
        }

        if(!pKF || pKF->GetMap() != pBiggerMap)
        {
            //cout << "--Parent KF is from another map" << endl;
            continue;
        }

        //cout << "3" << endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;


        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Tbw = pKF->mImuCalib.Tbc_ * (*lit) * Trw;
            Sophus::SE3f Twb = Tbw.inverse();

            Eigen::Vector3f twb = Twb.translation();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f Tcw = (*lit) * Trw;
            Sophus::SE3f Twc = Tcw.inverse();

            Eigen::Vector3f twc = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    //cout << "end saving trajectory" << endl;
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}*/

/*void System::SaveKeyFrameTrajectoryEuRoC_old(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            cv::Mat R = pKF->GetImuRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat twb = pKF->GetImuPosition();
            f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb.at<float>(0) << " " << twb.at<float>(1) << " " << twb.at<float>(2) << " " << q[0] << " " << q[1] << " " <<
q[2] << " " << q[3] << endl;

        }
        else
        {
            cv::Mat R = pKF->GetRotation();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << "
" << q[3] << endl;
        }
    }
    f.close();
}*/

/**
 * @brief 保存关键帧分支
 * @param filename
 */
void System::SaveKeyFrameTrajectoryEuRoC(const string &filename) {
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<Map *> vpMaps = mpAtlas->GetAllMaps();
    Map *pBiggerMap;
    int numMaxKFs = 0;
    // 遍历地图集中的所有子地图，找到拥有关键帧个数最多的哪个子地图，并记录最大关键帧个数
    for (Map *pMap : vpMaps) {
        if (pMap && pMap->GetAllKeyFrames().size() > numMaxKFs) {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    if (!pBiggerMap) {
        std::cout << "There is not a map!!" << std::endl;
        return;
    }

    // 获取拥有最多关键帧的子地图中的所有关键帧
    vector<KeyFrame *> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;
    // 遍历所有关键帧
    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];

        // pKF->SetPose(pKF->GetPose()*Two);

        if (!pKF || pKF->isBad())
            continue;
        // IMU 模式
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD) {
            Sophus::SE3f Twb = pKF->GetImuPose();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9 * pKF->mTimeStamp << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

        }
        // 纯视觉模式（非IMU）
        else {
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << pKF->mnFrameId << " " << setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
            //            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
            //            << endl;
        }
    }
    f.close();
}

void System::SaveKeyFrameTrajectoryEuRoC(const string &filename, Map *pMap) {
    cout << endl << "Saving keyframe trajectory of map " << pMap->GetId() << " to " << filename << " ..." << endl;

    vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for (size_t i = 0; i < vpKFs.size(); i++) {
        KeyFrame *pKF = vpKFs[i];

        if (!pKF || pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor == IMU_RGBD) {
            Sophus::SE3f Twb = pKF->GetImuPose();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9 * pKF->mTimeStamp << " " << setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

        } else {
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << setprecision(6) << 1e9 * pKF->mTimeStamp << " " << setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
    }
    f.close();
}

/*void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM3::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
            Trw = Trw * Converter::toCvMat(pKF->mTcp.matrix());
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPoseCv() * Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
}*/

void System::SaveTrajectoryKITTI(const string &filename) {
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if (mSensor == MONOCULAR) {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Tow = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for (list<Sophus::SE3f>::iterator lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++) {
        ORB_SLAM3::KeyFrame *pKF = *lRit;

        Sophus::SE3f Trw;

        if (!pKF)
            continue;

        while (pKF->isBad()) {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Tow;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Matrix3f Rwc = Twc.rotationMatrix();
        Eigen::Vector3f twc = Twc.translation();

        f << setprecision(9) << Rwc(0, 0) << " " << Rwc(0, 1) << " " << Rwc(0, 2) << " " << twc(0) << " " << Rwc(1, 0) << " " << Rwc(1, 1) << " " << Rwc(1, 2) << " " << twc(1) << " " << Rwc(2, 0)
          << " " << Rwc(2, 1) << " " << Rwc(2, 2) << " " << twc(2) << endl;
    }
    f.close();
}

void System::SaveDebugData(const int &initIdx) {
    // 0. Save initialization trajectory
    SaveTrajectoryEuRoC("init_FrameTrajectoy_" + to_string(mpLocalMapper->mInitSect) + "_" + to_string(initIdx) + ".txt");

    // 1. Save scale
    ofstream f;
    f.open("init_Scale_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mScale << endl;
    f.close();

    // 2. Save gravity direction
    f.open("init_GDir_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mRwg(0, 0) << "," << mpLocalMapper->mRwg(0, 1) << "," << mpLocalMapper->mRwg(0, 2) << endl;
    f << mpLocalMapper->mRwg(1, 0) << "," << mpLocalMapper->mRwg(1, 1) << "," << mpLocalMapper->mRwg(1, 2) << endl;
    f << mpLocalMapper->mRwg(2, 0) << "," << mpLocalMapper->mRwg(2, 1) << "," << mpLocalMapper->mRwg(2, 2) << endl;
    f.close();

    // 3. Save computational cost
    f.open("init_CompCost_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mCostTime << endl;
    f.close();

    // 4. Save biases
    f.open("init_Biases_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mbg(0) << "," << mpLocalMapper->mbg(1) << "," << mpLocalMapper->mbg(2) << endl;
    f << mpLocalMapper->mba(0) << "," << mpLocalMapper->mba(1) << "," << mpLocalMapper->mba(2) << endl;
    f.close();

    // 5. Save covariance matrix
    f.open("init_CovMatrix_" + to_string(mpLocalMapper->mInitSect) + "_" + to_string(initIdx) + ".txt", ios_base::app);
    f << fixed;
    for (int i = 0; i < mpLocalMapper->mcovInertial.rows(); i++) {
        for (int j = 0; j < mpLocalMapper->mcovInertial.cols(); j++) {
            if (j != 0)
                f << ",";
            f << setprecision(15) << mpLocalMapper->mcovInertial(i, j);
        }
        f << endl;
    }
    f.close();

    // 6. Save initialization time
    f.open("init_Time_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mInitTime << endl;
    f.close();
}

int System::GetTrackingState() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint *> System::GetTrackedMapPoints() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn() {
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

double System::GetTimeFromIMUInit() {
    double aux = mpLocalMapper->GetCurrKFTime() - mpLocalMapper->mFirstTs;
    if ((aux > 0.) && mpAtlas->isImuInitialized())
        return mpLocalMapper->GetCurrKFTime() - mpLocalMapper->mFirstTs;
    else
        return 0.f;
}

bool System::isLost() {
    if (!mpAtlas->isImuInitialized())
        return false;
    else {
        if ((mpTracker->mState == Tracking::LOST)) //||(mpTracker->mState==Tracking::RECENTLY_LOST))
            return true;
        else
            return false;
    }
}

bool System::isFinished() { return (GetTimeFromIMUInit() > 0.1); }

void System::ChangeDataset() {
    if (mpAtlas->GetCurrentMap()->KeyFramesInMap() < 12) {
        mpTracker->ResetActiveMap();
    } else {
        mpTracker->CreateMapInAtlas();
    }

    mpTracker->NewDataset();
}

float System::GetImageScale() { return mpTracker->GetImageScale(); }

#ifdef REGISTER_TIMES
void System::InsertRectTime(double &time) { mpTracker->vdRectStereo_ms.push_back(time); }

void System::InsertResizeTime(double &time) { mpTracker->vdResizeImage_ms.push_back(time); }

void System::InsertTrackTime(double &time) { mpTracker->vdTrackTotal_ms.push_back(time); }
#endif

void System::SaveAtlas(int type) {
    if (!mStrSaveAtlasToFile.empty()) {
        // clock_t start = clock();

        // Save the current session
        mpAtlas->PreSave();

        string pathSaveFileName = "./";
        pathSaveFileName = pathSaveFileName.append(mStrSaveAtlasToFile);
        pathSaveFileName = pathSaveFileName.append(".osa");

        string strVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath, TEXT_FILE);
        std::size_t found = mStrVocabularyFilePath.find_last_of("/\\");
        string strVocabularyName = mStrVocabularyFilePath.substr(found + 1);

        if (type == TEXT_FILE) // File text
        {
            cout << "Starting to write the save text file " << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::text_oarchive oa(ofs);

            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            cout << "End to write the save text file" << endl;
        } else if (type == BINARY_FILE) // File binary
        {
            cout << "Starting to write the save binary file" << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::binary_oarchive oa(ofs);
            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            cout << "End to write save binary file" << endl;
        }
    }
}

bool System::LoadAtlas(int type) {
    string strFileVoc, strVocChecksum;
    bool isRead = false;

    string pathLoadFileName = "./";
    pathLoadFileName = pathLoadFileName.append(mStrLoadAtlasFromFile);
    pathLoadFileName = pathLoadFileName.append(".osa");

    if (type == TEXT_FILE) // File text
    {
        cout << "Starting to read the save text file " << endl;
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if (!ifs.good()) {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::text_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        cout << "End to load the save text file " << endl;
        isRead = true;
    } else if (type == BINARY_FILE) // File binary
    {
        cout << "Starting to read the save binary file" << endl;
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if (!ifs.good()) {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::binary_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        cout << "End to load the save binary file" << endl;
        isRead = true;
    }

    if (isRead) {
        // Check if the vocabulary is the same
        string strInputVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath, TEXT_FILE);

        if (strInputVocabularyChecksum.compare(strVocChecksum) != 0) {
            cout << "The vocabulary load isn't the same which the load session was created " << endl;
            cout << "-Vocabulary name: " << strFileVoc << endl;
            return false; // Both are differents
        }

        mpAtlas->SetKeyFrameDababase(mpKeyFrameDatabase);
        mpAtlas->SetORBVocabulary(mpVocabulary);
        mpAtlas->PostLoad();

        return true;
    }
    return false;
}

string System::CalculateCheckSum(string filename, int type) {
    string checksum = "";

    unsigned char c[MD5_DIGEST_LENGTH];

    std::ios_base::openmode flags = std::ios::in;
    if (type == BINARY_FILE) // Binary file
        flags = std::ios::in | std::ios::binary;

    ifstream f(filename.c_str(), flags);
    if (!f.is_open()) {
        cout << "[E] Unable to open the in file " << filename << " for Md5 hash." << endl;
        return checksum;
    }

    MD5_CTX md5Context;
    char buffer[1024];

    MD5_Init(&md5Context);
    while (int count = f.readsome(buffer, sizeof(buffer))) {
        MD5_Update(&md5Context, buffer, count);
    }

    f.close();

    MD5_Final(c, &md5Context);

    for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
        char aux[10];
        sprintf(aux, "%02x", c[i]);
        checksum = checksum + aux;
    }

    return checksum;
}

} // namespace ORB_SLAM3
