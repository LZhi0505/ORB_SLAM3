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

#include "Frame.h"

#include <include/CameraModels/KannalaBrandt8.h>
#include <include/CameraModels/Pinhole.h>

#include <thread>

#include "Converter.h"
#include "G2oTypes.h"
#include "GeometricCamera.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Tracking.h"

namespace ORB_SLAM3 {
// 下一个生成的帧的ID,这里是初始化类的静态成员变量
long unsigned int Frame::nNextId = 0;

// 是否要进行初始化操作的标志
// 这里给这个标志置位的操作是在最初系统开始加载到内存的时候进行的，下一帧就是整个系统的第一帧，所以这个标志要置位
bool Frame::mbInitialComputations = true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

// For stereo fisheye matching，用于双目鱼眼的立体匹配
// 第一个参数：normType：它是用来指定要使用的距离测试类型,默认值为cv2.Norm_L2,这很适合SIFT和SURF等（c2.NORM_L1也可）。
// 对于使用二进制描述符的ORB、BRIEF和BRISK算法等，要使用cv2.NORM_HAMMING，这样就会返回两个测试对象之间的汉明距离。
// 如果ORB算法的参数设置为WTA_K==3或4，normType就应该设置成cv2.NORM_HAMMING2。
// 第二个参数：crossCheck：默认值为False。如果设置为True，匹配条件就会更加严格，只有到A中的第i个特征点与B中的第j个特征点距离最近，并且B中的第j个特征点到A中的第i个特征点也是最近时才会返回最佳匹配(i,j)，即这两个特征点要互相匹配才行。
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame()
    : mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
      mbHasPose(false), mbHasVelocity(false) {
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}

// Copy Constructor
Frame::Frame(const Frame &frame)
    : mpcpi(frame.mpcpi), mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight), mTimeStamp(frame.mTimeStamp),
      mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()), mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
      mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight), mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
      mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()), mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib),
      mnCloseMPs(frame.mnCloseMPs), mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias), mnId(frame.mnId),
      mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor), mvScaleFactors(frame.mvScaleFactors),
      mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset), mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
      mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame), mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
      mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright), monoLeft(frame.monoLeft), monoRight(frame.monoRight),
      mvLeftToRightMatch(frame.mvLeftToRightMatch), mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints), mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr),
      mTrl(frame.mTrl), mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false) {
    for (int i = 0; i < FRAME_GRID_COLS; i++)
        for (int j = 0; j < FRAME_GRID_ROWS; j++) {
            mGrid[i][j] = frame.mGrid[i][j];
            if (frame.Nleft > 0) {
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if (frame.mbHasPose)
        SetPose(frame.GetPose());

    if (frame.HasVelocity()) {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}

/**
 * @brief PinHole相机 双目（未提供Camera2）
 * @param imLeft
 * @param imRight
 * @param timeStamp
 * @param extractorLeft
 * @param extractorRight
 * @param voc
 * @param K
 * @param distCoef
 * @param bf
 * @param thDepth
 * @param pCamera
 * @param pPrevF
 * @param ImuCalib
 */
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef,
             const float &bf, const float &thDepth, GeometricCamera *pCamera, Frame *pPrevF, const IMU::Calib &ImuCalib)
    : mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),
      mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL),
      mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false) {
    std::cout << std::endl << "构建帧（双目 立体匹配模式，未提供Camera2，相机模型为PinHole）" << endl;
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    // Step 1：帧ID的自增
    mnId = nNextId++;

    // Scale Level Info
    // Step 2：计算图像金字塔的参数
    mnScaleLevels = mpORBextractorLeft->GetLevels();                      // 获取图像金字塔的层数
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();                 // 获得层与层之间的缩放比
    mfLogScaleFactor = log(mfScaleFactor);                                // 计算上面缩放比的对数
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();               // 获取每层图像的缩放因子
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();     // 同样获取每层图像缩放因子的倒数
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();           // 高斯模糊的时候，使用的方差
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares(); // 获取sigma^2的倒数

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    // Step 3：对左目右目图像提取ORB特征点, 第一个参数0-左图， 1-右图。为加速计算，同时开了两个线程计算
    //    thread threadLeft(&Frame::ExtractORB, this, 0, imLeft, 0, 0);    // 对左目图像提取orb特征
    //    thread threadRight(&Frame::ExtractORB, this, 1, imRight, 0, 0);  // 对右目图像提取orb特征
    //    // 等待两张图像特征点提取过程完成
    //    threadLeft.join();
    //    threadRight.join();
    // liuzhi改为顺序执行
    ExtractORB(0, imLeft, 0, 0);
    ExtractORB(1, imRight, 0, 0);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB).count();
#endif

    // mvKeys 保存的是左目的特征点，N 获取左目特征点的个数
    N = mvKeys.size();
    if (mvKeys.empty())
        return;

    // Step 4：特征点去畸变：PinHole双目有畸变参数，但特征点去畸变标志为false，实际不进行特征点去畸变，mvKeysUn = mvKeys，且mK也未改变
    UndistortKeyPoints();
    //    for (int i = 0; i < N; ++i) {
    //        auto keypt = mvKeys[i];
    //        auto keyptUn = mvKeysUn[i];
    //        Verbose::PrintMess("\t ("+std::to_string(keypt.pt.x)+", "+std::to_string(keypt.pt.y)+"), 去畸变后: ("+std::to_string(keyptUn.pt.x)+", "+std::to_string(keyptUn.pt.y)+")",
    //        Verbose::VERBOSITY_DEBUG);
    //    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    // 引擎中初始有值，因此在此处提前赋值了
    fx = K.at<float>(0, 0);
    // 双目相机基线长度
    mb = mbf / fx;

    // Step 5：双目立体匹配
    // 只有左图特征点 在右图中 找到了最佳匹配点，才会计算其深度，并存储在 mvDepth 中。key:左图特征点索引，value:其深度值 mvuRight 存储的是左图特征点 在 右图匹配到的点的u坐标
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    // 初始化当前帧的地图点为 NULL，外点标志为 false
    mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
    mvbOutlier = vector<bool>(N, false);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    // This is done only for the first Frame (or after a change in the calibration)
    //  Step 5：计算去畸变后图像边界 (第一帧或者是相机标定参数发生变化之后进行)
    if (mbInitialComputations) {
        // 计算去畸变后图像的边界
        ComputeImageBounds(imLeft);

        // 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        // 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        // 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        // 特殊的初始化过程完成，标志复位
        mbInitialComputations = false;
    }

    // 双目相机基线长度
    mb = mbf / fx;

    // IMU模式会在这里设置IMU当前帧的速度 = 上一帧的速度
    // 上一帧存在 且 有速度
    if (pPrevF) {
        if (pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    } else {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    // Set no stereo fisheye information
    // 因为不是鱼眼相机，所以不设置与鱼眼相机相关的参数
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    // Step 6：将特征点分配到图像网格中
    AssignFeaturesToGrid();
}

/**
 * @brief RGB-D
 * @param imGray
 * @param imDepth
 * @param timeStamp
 * @param extractor
 * @param voc
 * @param K
 * @param distCoef
 * @param bf
 * @param thDepth
 * @param pCamera
 * @param pPrevF
 * @param ImuCalib
 */
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,
             GeometricCamera *pCamera, Frame *pPrevF, const IMU::Calib &ImuCalib)
    : mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),
      mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL),
      mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false) {
    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0, imGray, 0, 0);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size();

    if (mvKeys.empty())
        return;

    // 若有畸变参数
    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N, false);

    // This is done only for the first Frame (or after a change in the calibration)
    if (mbInitialComputations) {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    mb = mbf / fx;

    if (pPrevF) {
        if (pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    } else {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    // Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

/**
 * @brief 单目，构建图像帧 Frame
 * @param imGray    灰度图像
 * @param timeStamp 时间戳
 * @param extractor ORB 特征提取器
 * @param voc       词袋
 * @param pCamera   相机
 * @param distCoef
 * @param bf
 * @param thDepth
 * @param pPrevF
 * @param ImuCalib
 * step 1. 提取该帧的 关键点和描述子，
 * step 2. 对关键点去畸变，
 * step 3. 再将它们分配到网格中
 */
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, ORBVocabulary *voc, GeometricCamera *pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth,
             Frame *pPrevF, const IMU::Calib &ImuCalib)
    : mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)), mTimeStamp(timeStamp), mK(static_cast<Pinhole *>(pCamera)->toK()),
      mK_(static_cast<Pinhole *>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),
      mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(nullptr), mbHasPose(false),
      mbHasVelocity(false) {
    imgLeft = imGray.clone(); // liuzhi 加
    // Frame ID
    // Step 1：帧的ID 自增
    mnId = nNextId++;

    // Step 2：计算图像金字塔的参数
    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();                      // 图像金字塔的层数
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();                 // 每层的缩放因子
    mfLogScaleFactor = log(mfScaleFactor);                                // 每层缩放因子的 自然对数
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();               // 各层图像的缩放因子
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();     // 各层图像的缩放因子 的倒数
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();           // sigma^2
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares(); // sigma^2的倒数

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    // Step 3：-------- ORB特征提取 ----------，第一个参数flag：0-表示当前图是左图， 1-右图。
    // 得到该帧的特征点：关键点 和 描述子
    ExtractORB(0, imGray, 0, 1000);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size(); // 关键点的个数

    // 如果没有能够成功提取出关键点，则直接返回了
    if (mvKeys.empty())
        return;

    // Step 4：特征点去畸变：若有畸变参数，用OpenCV的去畸变函数，对特征点 去畸变，结果保存在 mvKeysUn 中，且会更新内参矩阵 mK
    UndistortKeyPoints();

    // Set no stereo information
    // 由于单目相机无法直接获得立体信息，所以这里要给 右图像对应点 和 深度赋值-1，表示没有相关信息
    mvuRight = vector<float>(N, -1);
    mvDepth = vector<float>(N, -1);
    mnCloseMPs = 0;

    // 初始化本帧的 地图点为 NULL
    mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));

    mmProjectPoints.clear(); // = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    // 记录特征点是否为外点，初始化为 false，表明不是外点
    mvbOutlier = vector<bool>(N, false);

    // This is done only for the first Frame (or after a change in the calibration)
    // Step 5：计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if (mbInitialComputations) {
        // 计算去畸变后图像的边界
        ComputeImageBounds(imGray);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);  // 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY); // 表示一个图像像素相当于多少个图像网格行（高

        fx = static_cast<Pinhole *>(mpCamera)->toK().at<float>(0, 0);
        fy = static_cast<Pinhole *>(mpCamera)->toK().at<float>(1, 1);
        cx = static_cast<Pinhole *>(mpCamera)->toK().at<float>(0, 2);
        cy = static_cast<Pinhole *>(mpCamera)->toK().at<float>(1, 2);
        // 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    // 计算 基线 basline
    mb = mbf / fx;

    // Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    // 将特征点分配到图像网格中
    AssignFeaturesToGrid();

    if (pPrevF) {
        if (pPrevF->HasVelocity()) {
            SetVelocity(pPrevF->GetVelocity());
        }
    } else {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();
}

/**
 * @brief 特征分网格
 */
void Frame::AssignFeaturesToGrid() {
    // Fill matrix with points
    // Step 1：给存储特征点的网格数组 Frame::mGrid 预分配空间
    const int nCells = FRAME_GRID_COLS * FRAME_GRID_ROWS;

    int nReserve = 0.5f * N / (nCells);

    // 开始对 mGrid 这个二维数组中的每一个 vector 元素遍历 并预分配空间
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++) {
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++) {
            mGrid[i][j].reserve(nReserve);
            if (Nleft != -1) {
                mGridRight[i][j].reserve(nReserve);
            }
        }
    }

    // Step 2：遍历每个特征点，将每个特征点在 mvKeysUn 中的索引值放到对应的网格 mGrid 中
    for (int i = 0; i < N; i++) {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i] : (i < Nleft) ? mvKeys[i] : mvKeysRight[i - Nleft];
        // 存储某个特征点所在网格的网格坐标，nGridPosX范围：[0,FRAME_GRID_COLS], nGridPosY范围：[0,FRAME_GRID_ROWS]
        int nGridPosX, nGridPosY;
        // 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
        if (PosInGrid(kp, nGridPosX, nGridPosY)) {
            if (Nleft == -1 || i < Nleft)
                // 如果找到特征点所在网格坐标，将这个特征点的索引添加到对应网格的数组mGrid中
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

/**
 * @brief 特征点提取
 * @param flag 左右标志位
 * @param im 图片
 * @param x0 界限 0
 * @param x1 界限 1000
 */
void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1) {
    vector<int> vLapping = {x0, x1};

    // 判断是左图还是右图。0-左图， 1-右图
    if (flag == 0)
        // 左图的话就套使用左图指定的特征点提取器，并将提取结果保存到对应的变量中。
        // 下面的语句会进入 ORBextractor::operator() 中。获取该帧的ORB关键点 mvKeys 和 描述子 mDescriptors
        // 返回左目不在共视范围内的特征点的数量 (PinHole这个数值=所有特征点的数量，鱼眼相机会有在共视范围内的特征点，这个数值作为索引对应的特征点是在共视范围内的第一个特征点)
        monoLeft = (*mpORBextractorLeft)(im,        // 待提取特征点的图像
                                         cv::Mat(), // 掩摸图像, 实际没有用到
                                         mvKeys, mDescriptors, vLapping);
    else
        // 右图的话就需要使用右图指定的特征点提取器，并将提取结果保存到对应的变量中
        // 返回右目不在共视范围内的特征点的数量 (PinHole这个数值=所有特征点的数量，鱼眼相机会有在共视范围内的特征点，这个数值作为索引对应的特征点是在共视范围内的第一个特征点)
        monoRight = (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight, vLapping);
}

bool Frame::isSet() const { return mbIsSet; }

/**
 * @brief 赋值位姿
 * @param Tcw 位姿
 */
void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

/**
 * @brief 赋值新的偏置
 * @param b 偏置
 */
void Frame::SetNewBias(const IMU::Bias &b) {
    mImuBias = b;
    if (mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

/**
 * @brief 赋值新的速度
 * @param Vwb 速度
 */
void Frame::SetVelocity(Eigen::Vector3f Vwb) {
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const { return mVw; }

/**
 * @brief 赋值位姿与速度
 */
void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb) {
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

/**
 * mTcw -> 相机位姿：世界坐标系到相机坐标坐标系的变换矩阵, 是我们常规理解中的相机位姿
 * 根据 mTcw可以计算出 mOw(当前相机光心在世界坐标系下坐标)，
 *                  mRcw(世界坐标系到相机坐标系的旋转矩阵)，
 *                  mtcw(世界坐标系到相机坐标系的平移向量)，
 *                  mRwc(相机坐标系到世界坐标系的旋转矩阵)。
 */
void Frame::UpdatePoseMatrices() {
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

/**
 * @brief 获取IMU的位置 Pwb
 */
Eigen::Matrix<float, 3, 1> Frame::GetImuPosition() const { return mRwc * mImuCalib.mTcb.translation() + mOw; }

/**
 * @brief 获取IMU的旋转 Rwb
 */
Eigen::Matrix<float, 3, 3> Frame::GetImuRotation() { return mRwc * mImuCalib.mTcb.rotationMatrix(); }

/**
 * @brief 获取IMU的位姿  Twb
 */
Sophus::SE3<float> Frame::GetImuPose() { return mTcw.inverse() * mImuCalib.mTcb; }

/**
 * @brief 获得左右目的相对位姿
 */
Sophus::SE3f Frame::GetRelativePoseTrl() { return mTrl; }

Sophus::SE3f Frame::GetRelativePoseTlr() { return mTlr; }

/**
 * @brief 获得左右目的相对旋转
 */
Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation() { return mTlr.rotationMatrix(); }

Eigen::Vector3f Frame::GetRelativePoseTlr_translation() { return mTlr.translation(); }

/**
 * @brief 判断地图点是否在视野中
 * 步骤
 * Step 1 获得这个地图点的世界坐标
 * Step 2 关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，返回false
 * Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
 * Step 4 关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
 * Step 5 关卡四：计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值, 若小于设定阈值，返回false
 * Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
 * Step 7 记录计算得到的一些参数
 * @param[in] pMP                       当前地图点
 * @param[in] viewingCosLimit           夹角余弦，用于限制地图点和光心连线 和 法线的夹角
 * @return true                         地图点合格，且在视野内
 * @return false                        地图点不合格，抛弃
 */
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
    // 单目，PinHole双目，RGBD
    if (Nleft == -1) {
        // mbTrackInView是决定一个地图点是否进行重投影的标志
        // 这个标志的确定要经过多个函数的确定，isInFrustum()只是其中的一个验证关卡。这里默认设置为否
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        // Step 1：获得这个地图点的世界坐标
        Eigen::Matrix<float, 3, 1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        // 根据当前帧(粗糙)位姿转化到当前相机坐标系下的三维点Pc
        const Eigen::Matrix<float, 3, 1> Pc = mRcw * P + mtcw;
        const float Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f / PcZ;
        // Step 2：关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，直接返回false
        if (PcZ < 0.0f)
            return false;

        const Eigen::Vector2f uv = mpCamera->project(Pc); // 地图点的像素坐标

        // Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
        // 判断是否在图像边界中，只要不在那么就说明无法在当前帧下进行重投影
        if (uv(0) < mnMinX || uv(0) > mnMaxX)
            return false;
        if (uv(1) < mnMinY || uv(1) > mnMaxY)
            return false;

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        // Step 4：关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
        // 得到认为的可靠距离范围:[0.8f*mfMinDistance, 1.2f*mfMaxDistance]
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        // 得到当前地图点距离当前帧相机光心的距离,注意P，mOw都是在同一坐标系下才可以
        //  mOw：当前相机光心在世界坐标系下坐标
        const Eigen::Vector3f PO = P - mOw;
        // 取模就得到了距离
        const float dist = PO.norm();

        // 如果不在允许的尺度变化范围内，认为重投影不可靠
        if (dist < minDistance || dist > maxDistance)
            return false;

        // Check viewing angle
        // Step 5：关卡四：
        // 计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值,
        // 若小于cos(viewingCosLimit), 即夹角大于viewingCosLimit弧度则返回
        Eigen::Vector3f Pn = pMP->GetNormal();

        // 计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值，注意平均观测方向为单位向量
        const float viewCos = PO.dot(Pn) / dist;

        // 如果大于给定的阈值 cos(60°)=0.5，即夹角 > 60°认为这个点方向太偏了，重投影不可靠，返回false
        if (viewCos < viewingCosLimit)
            return false;

        // Predict scale in the image
        // Step 6：根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Step 7：记录计算得到的一些参数
        // Data used by the tracking
        // 通过置位标记 MapPoint::mbTrackInView 来表示这个地图点要被投影
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = uv(0);               // 该地图点投影在当前图像（一般是左图）的像素横坐标
        pMP->mTrackProjXR = uv(0) - mbf * invz; // bf/z其实是视差，相减得到右图（如有）中对应点的横坐标

        pMP->mTrackDepth = Pc_dist;

        pMP->mTrackProjY = uv(1);                 // 该地图点投影在当前图像（一般是左图）的像素纵坐标
        pMP->mnTrackScaleLevel = nPredictedLevel; // 根据地图点到光心距离，预测的该地图点的尺度层级
        pMP->mTrackViewCos = viewCos;             // 保存当前视角和法线夹角的余弦值

        return true; // 执行到这里说明这个地图点在相机的视野中并且进行重投影是可靠的，返回true
    }
    // 鱼眼双目，左右目时分别验证
    else {
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP->mnTrackScaleLevel = -1;
        pMP->mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP, viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP, viewingCosLimit, true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint *pMP, cv::Point2f &kp, float &u, float &v) {
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY = Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if (PcZ < 0.0f) {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    u = fx * PcX * invz + cx;
    v = fy * PcY * invz + cy;

    if (u < mnMinX || u > mnMaxX)
        return false;
    if (v < mnMinY || v > mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if (mDistCoef.total() == 5) {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw) { return mRcw * pCw + mtcw; }

/**
 * @brief 在给定的坐标(x, y)，返回圆形区域r内所有特征点索引
 * @param x 给定点的x
 * @param y 给定点的y
 * @param r 搜索半径 windowSize
 * @param minLevel 金字塔下边界，为0
 * @param maxLevel 金字塔上边界，为0
 * @param bRight 是否是右相机
 */
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel, const bool bRight) const {
    vector<size_t> vIndices; // 在范围r内，所有候选特征点索引
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    // Step 1: 计算半径为r圆左右上下边界所在的网格列和行的id
    // 查找半径为r的圆左侧边界所在网格列坐标。这个地方有点绕，慢慢理解下：
    // (mnMaxX-mnMinX) / FRAME_GRID_COLS：列方向每个网格可以平均分得几个像素（肯定大于1）
    // mfGridElementWidthInv = FRAME_GRID_COLS / (mnMaxX - mnMinX) 是上面倒数，表示每个像素可以均分几个网格列（肯定小于1）
    // (x - mnMinX - r)，从图像的左边界mnMinX 到 半径r的圆的左边界区域 占的像素列数
    // 两者相乘，就是求出那个半径为r的圆的左侧边界在哪个网格列中
    // 保证nMinCellX 结果 >= 0
    const int nMinCellX = max(0, (int)floor((x - mnMinX - factorX) * mfGridElementWidthInv));
    // 如果最终求得的圆的左边界所在的网格列超过了设定了上限，那么就说明计算出错，找不到符合要求的特征点，返回空vector
    if (nMinCellX >= FRAME_GRID_COLS) {
        return vIndices;
    }

    // 计算圆所在的右边界网格列索引
    const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + factorX) * mfGridElementWidthInv));
    if (nMaxCellX < 0) {
        return vIndices;
    }

    // 计算出这个圆上边界所在的网格行的id
    const int nMinCellY = max(0, (int)floor((y - mnMinY - factorY) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS) {
        return vIndices;
    }

    // 计算出这个圆下边界所在的网格行的id
    const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + factorY) * mfGridElementHeightInv));
    if (nMaxCellY < 0) {
        return vIndices;
    }

    // 检查需要搜索的图像金字塔层数范围是否符合要求
    // ? 疑似bug。(minLevel>0) 后面条件 (maxLevel>=0)肯定成立
    // ? 改为 const bool bCheckLevels = (minLevel>=0) || (maxLevel>=0);
    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

    // Step 2： 遍历圆形区域内的所有网格，寻找满足条件的候选特征点，并将其index放到输出里
    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
            // 获取这个网格内的所有特征点在 Frame::mvKeysUn 中的索引，存储在 vCell中
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            // 如果这个网格中没有特征点，那么跳过这个网格继续下一个
            if (vCell.empty())
                continue;

            // 如果这个网格中有特征点，那么遍历这个图像网格中所有的特征点
            for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                // 根据索引先读取这个 候选特征点
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]] : (!bRight) ? mvKeys[vCell[j]] : mvKeysRight[vCell[j]];
                if (bCheckLevels) {
                    // cv::KeyPoint::octave 表示的是从金字塔的哪一层提取的数据
                    // 保证特征点是在金字塔层级 minLevel 和 maxLevel 之间，不是的话跳过
                    if (kpUn.octave < minLevel)
                        continue;
                    if (maxLevel >= 0)
                        if (kpUn.octave > maxLevel)
                            continue;
                }

                // 通过检查，计算 候选特征点 到圆中心的距离，查看是否是在这个圆形区域之内
                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if (fabs(distx) < factorX && fabs(disty) < factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

    // Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}

// 将 当前帧的描述子 转换成 词袋向量
void Frame::ComputeBoW() {
    // 如果词袋向量为空 未计算过，则计算；已计算过，则跳过
    if (mBowVec.empty()) {
        // 将描述子 mDescriptors 转换为 DBoW 要求的输入格式
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // 将 特征点的描述子 转换成 词袋向量 mBowVec 以及 特征向量 mFeatVec。
        mpORBvocabulary->transform(
            vCurrentDesc, // 输入：当前描述子vector
            mBowVec,      // 输出，词袋向量 mBowVec，记录 单词的ID（词汇树中距离最近的叶子节点的id） 及其 对应权重
            mFeatVec, // 输出，特征向量 mFeatVec，记录当前帧包含的 node id（距离叶子节点深度为level up对应的node的Id）及其 对应的图像feature的IDs（该节点下所有叶子节点对应的feature的id）
            4);       // 表示从叶节点向前数的层。如果 level up越大，那么featureVec的size越大，搜索的范围越广，速度越慢；
    }
}

/**
 * @brief 用畸变参数 和 内参矩阵 对特征点 去畸变，结果保存在 mvKeysUn 中，且会更新内参矩阵 mK
 */
void Frame::UndistortKeyPoints() {
    // Step 1：如果第一个畸变参数为0 (PinHole单目不为0, 其余为0)，不需要矫正。第一个畸变参数k1是最重要的，一般不为0，为0的话，说明畸变参数都是0
    // 变量mDistCoef中存储了opencv指定格式的去畸变参数，格式为：(k1,k2,p1,p2,k3)
    if (mDistCoef.at<float>(0) == 0.0) {
        mvKeysUn = mvKeys;
        std::cout << "\t畸变参数为0，不进行特征点畸变矫正，mvKeysUn = mvKeys" << std::endl;
        return;
    }
    std::cout << "\t畸变参数不为0，进行特征点畸变矫正" << std::endl;

    // Step 2：如果畸变参数不为0，用OpenCV函数进行畸变矫正
    // Fill matrix with points
    cv::Mat mat(N, 2, CV_32F); // N 为提取的特征点数量，为满足OpenCV函数输入要求，将N个特征点保存在 N*2 的矩阵中

    // 遍历每个特征点，并将它们的坐标保存到mat中
    for (int i = 0; i < N; i++) {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    // 函数reshape(int cn, int rows=0)：cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
    // 为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）
    // cv::undistortPoints 最后一个矩阵为空矩阵时，得到的点为归一化坐标点
    mat = mat.reshape(2);

    cv::undistortPoints(mat,                                     // 输入的特征点坐标
                        mat,                                     // 输出校正后的特征点坐标
                        static_cast<Pinhole *>(mpCamera)->toK(), // 相机内参矩阵
                        mDistCoef,                               // 相机畸变参数矩阵
                        cv::Mat(),                               // 空矩阵
                        mK);                                     // 新内参矩阵
    // 调整回只有一个通道，回归我们正常的处理方式
    mat = mat.reshape(1);

    // Step 3：将坐标矫正后的特征点存入mvKeysUn.
    mvKeysUn.resize(N);
    for (int i = 0; i < N; i++) {
        // 取出原特征点，再改变其坐标。注意之所以这样做而不是直接重新声明一个特征点对象的目的是，能够得到源特征点对象的其他属性
        cv::KeyPoint kp = mvKeys[i];
        // 校正后的坐标 覆盖 老坐标
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
    if (mDistCoef.at<float>(0) != 0.0) {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = imLeft.cols;
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;
        mat.at<float>(2, 1) = imLeft.rows;
        mat.at<float>(3, 0) = imLeft.cols;
        mat.at<float>(3, 1) = imLeft.rows;

        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, static_cast<Pinhole *>(mpCamera)->toK(), mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
        mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
        mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
    } else {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/**
 * @brief 双目立体匹配
 *
 * 为左图的每一个特征点在右图中找到匹配点
 * 根据基线(有冗余范围)上描述子距离找到匹配, 再进行SAD精确定位
 * 这里所说的SAD是一种双目立体视觉匹配算法，可参考[https://blog.csdn.net/u012507022/article/details/51446891]
 * 最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对，然后利用抛物线拟合得到亚像素精度的匹配
 * 这里所谓的亚像素精度，就是使用这个拟合得到一个小于一个单位像素的修正量，这样可以取得更好的估计结果，计算出来的点的深度也就越准确
 * 匹配成功后会更新 mvuRight(ur) 和 mvDepth(Z)
 *
 * 两帧图像稀疏立体匹配（即：ORB特征点匹配，非逐像素的密集匹配，但依然满足行对齐）
 * 输入：两帧立体矫正后的图像 img_left 和 img_right 对应的orb特征点集
 * 过程：
 *     1. 行特征点统计. 统计 img_right 每一行上的 ORB特征点集，便于使用立体匹配思路(行搜索/极线搜索）进行同名点搜索, 避免逐像素的判断.
 *     2. 粗匹配. 根据步骤1的结果，对 img_left 第 i 行的 orb特征点pi，在 img_right 的第 i 行上的 orb特征点集中搜索相似 orb特征点, 得到 qi
 *     3. 精确匹配. 以点 qi 为中心，半径为 r 的范围内，进行块匹配（归一化SAD），进一步优化匹配结果
 *     4. 亚像素精度优化. 步骤3得到的视差为 uchar/int 类型精度，并不一定是真实视差，通过亚像素差值（抛物线插值)获取 float精度的真实视差
 *     5. 最优视差值 / 深度选择. 通过胜者为王算法（ WTA ）获取最佳匹配点。
 *     6. 删除离缺点(outliers). 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是正确匹配，比如光照变化、弱纹理等会造成误匹配
 * 输出：稀疏特征点视差图 / 深度图（亚像素精度）mvDepth 匹配结果 mvuRight
 */
void Frame::ComputeStereoMatches() {
    //    for (int i = 0; i < mvKeys.size(); ++i) {
    //        std::cout << "左目坐标: (" << mvKeys[i].pt.x << ", " << mvKeys[i].pt.y << ")" << std::endl;
    //        std::cout << "\t描述子: ";
    //        for (int j = 0; j < 32; j++) {
    //            uchar descriptor = mDescriptors.at<uchar>((int)i, j);
    //            std::cout << static_cast<int>(descriptor);
    //            if (j < 31) std::cout << " ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    for (int i = 0; i < mvKeysRight.size(); ++i) {
    //        std::cout << "右目坐标: (" << mvKeysRight[i].pt.x << ", " << mvKeysRight[i].pt.y << ")" << std::endl;
    //        std::cout << "\t描述子: ";
    //        for (int j = 0; j < 32; j++) {
    //            uchar descriptor = mDescriptorsRight.at<uchar>((int)i, j);
    //            std::cout << static_cast<int>(descriptor);
    //            if (j < 31) std::cout << " ";
    //        }
    //        std::cout << std::endl;
    //    }
    // 为匹配结果预先分配内存，数据类型为float型
    mvuRight = vector<float>(N, -1.0f); // 存储右图匹配点的 u
    mvDepth = vector<float>(N, -1.0f);  // 存储特征点的深度

    // orb特征相似度阈值  -> mean ～= (max  + min) / 2
    const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

    // nRows：图像高度，即金字塔顶层（0层）的高度
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    // Assign keypoints to row table
    // vRowIndices 每一行存储的是右目特征点的索引。即这个二维向量表示的是右目特征点索引可能出现的行号
    // 为什么是vector，因为每一行的特征点有可能不一样，例如：
    // vRowIndices[0] = [1, 2, 5, 8, 11]   第0行有5个右目特征点, 它们的索引分别是 1,2,5,8,11
    vector<vector<size_t>> vRowIndices(nRows, vector<size_t>());

    for (int i = 0; i < nRows; i++)
        vRowIndices[i].reserve(200);

    // 右图特征点数量，N表示数量 r表示右图，且不能被修改
    const int Nr = mvKeysRight.size();

    // Step 1：行特征点统计. 考虑到尺度金字塔特征，一个特征点可能存在于多行，而非唯一的一行
    // 遍历右目特征点
    for (int iR = 0; iR < Nr; iR++) {
        const cv::KeyPoint &kp = mvKeysRight[iR]; // 右目特征点iR
        const float &kpY = kp.pt.y;               // iR的y坐标 (行号)
        // 计算特征点iR在行方向上，可能的偏移范围 r，即可能的行号为[kpY + r, kpY - r]
        // 2：在全尺寸(scale = 1)的情况下，假设有2个像素的偏移，随着尺度变化，r也跟着变化
        const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
        // 可能存在的行号区间 [minr, maxr]
        const int maxr = ceil(kpY + r);
        const int minr = floor(kpY - r);
        //        Verbose::PrintMess("right_idx: "+std::to_string(iR)+", y_right: "+std::to_string(kpY)+", r: "+std::to_string(r)+", max_r: "+std::to_string(maxr)+", min_r: "+std::to_string(minr),
        //        Verbose::VERBOSITY_DEBUG);

        // 将右目特征点索引iR 保存 在可能的行号中
        for (int yi = minr; yi <= maxr; yi++)
            vRowIndices[yi].push_back(iR);
    }
    std::stringstream ss;
    for (int i = 0; i < vRowIndices.size(); i++) {
        ss << "row " << i << ": ";
        for (int j = 0; j < vRowIndices[i].size(); j++) {
            ss << vRowIndices[i][j] << " ";
        }
        ss << std::endl;
    }
    std::cout << ss.str().c_str() << std::endl;

    // Step 2 -> 3：粗匹配 + 精匹配
    // 对于立体矫正后的两张图，在列方向(x 横坐标)存在最大视差 maxd 和 最小视差 mind
    // 也即是左图中任何一点 p，在右图上的匹配点的范围为应该是[p - maxd, p - mind], 而不需要遍历每一行所有的像素
    // maxd = baseline * length_focal / minZ
    // mind = baseline * length_focal / maxZ
    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf / minZ; // = fx = 320

    // For each left keypoint search a match in the right image
    // 保存SAD块匹配相似度 和 左图特征点索引
    vector<pair<int, int>> vDistIdx; // key: 最小SAD值, value: 左目特征点索引
    vDistIdx.reserve(N);

    int num_cu = 0, num_jing = 0;

    // 遍历左图每一个特征点，为左图每一个特征点iL，在右图搜索最相似的特征点iR
    for (int iL = 0; iL < N; iL++) {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y; // 行号
        const float &uL = kpL.pt.x; // 列号

        // 获取可能与该左图特征点iL 在相同行 的右图特征点的索引集合
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if (vCandidates.empty())
            continue;

        // 计算理论上的最佳搜索范围为: [minU, maxU] = [u - maxD, u], 左图特征点横坐标左侧
        const float minU = uL - maxD;
        const float maxU = uL - minD;

        // 最大搜索范围 < 0，说明无匹配点
        if (maxU < 0)
            continue;

        // 初始化最佳匹配距离，以及最佳匹配索引
        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        // 左图特征点iL 的 描述子 dL
        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        // Step2：粗配准。左图特征点iL 与 在相同行的 右图中的可能的匹配点 进行逐个比较，得到最相似匹配点的相似度 和 索引
        // 遍历右图中的候选特征点
        for (size_t iC = 0; iC < vCandidates.size(); iC++) {
            const size_t iR = vCandidates[iC]; // 右图候选特征点索引
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            // 左图特征点iL 与 待匹配点iR的 金字塔层级差2，则放弃
            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                continue;

            // 获取右图候选匹配点的x坐标，以进行匹配
            const float &uR = kpR.pt.x;
            // 如果右图特征点横坐标超出理论搜索范围[minU, maxU]，可能是误匹配，放弃
            if (uR >= minU && uR <= maxU) // 在范围内
            {
                // 计算左图特征点iL和 右图候选特征点iR的描述子的距离
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL, dR);

                // 更新最佳距离 及 最佳匹配索引索引
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        } // 一个左图特征点iL在右图中找到最佳匹配特征点 bestIdxR
        //        Verbose::PrintMess("左图特征点横坐标: "+std::to_string(uL)+", 在右图的搜索范围: ("+std::to_string(minU)+", "+std::to_string(maxU)+"), 最佳匹配右目特征点坐标:
        //        ("+std::to_string(mvKeysRight[bestIdxR].pt.y)+", "+std::to_string(mvKeysRight[bestIdxR].pt.x)+"), 最佳匹配距离: "+std::to_string(bestDist), Verbose::VERBOSITY_DEBUG);

        // Subpixel match by correlation
        // Step 3：精匹配：如果刚才匹配点对的最佳匹配距离 < 阈值，则进行精匹配
        if (bestDist < thOrbDist) {
            num_cu++;
            // coordinates in image pyramid at keypoint scale
            // 在左目特征点iL所在的金字塔层进行块匹配，特征点坐标也需缩小
            // 获取右图最佳匹配特征点的x坐标 和 左图特征点iL所在层的金字塔尺度的倒数
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            // 尺度缩放后的左图特征点iL的 x,y坐标
            const float scaleduL = round(kpL.pt.x * scaleFactor);
            const float scaledvL = round(kpL.pt.y * scaleFactor);
            // 右图匹配点bestIdxR的 x坐标
            const float scaleduR0 = round(uR0 * scaleFactor);

            // sliding window search
            // 滑动窗口搜索, 类似模版卷积或滤波
            // w 表示 SAD (灰度值差的绝对和) 相似度的窗口半径
            const int w = 5;
            // 提取左图中，以尺度缩放后特征点(scaleduL, scaledvL)为中心, 半径为 w 的图像 块patch
            // 左图的图像块是固定的
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);

            // 初始化最佳相似度
            int bestDist = INT_MAX;
            // 通过滑动窗口搜索优化，得到的列坐标 (横坐标)偏移量
            int bestincR = 0;
            // 滑动窗口的滑动范围为（-L, L）
            const int L = 5;
            // 存储滑动块在每个偏移位置的SAD值
            vector<float> vDists;
            vDists.resize(2 * L + 1);

            // 计算右图匹配特征点的 滑动窗口 滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
            // 列方向起点 iniu = r0 + 窗口滑动起始偏移量 - 图像块尺寸
            // 列方向终点 eniu = r0 + 窗口滑动终止偏移量 + 图像块尺寸 + 1
            // 此次 + 1 和下面的提取图像块是列坐标+1是一样的，保证提取的图像块的宽是2 * w + 1
            const float iniu = scaleduR0 - L - w;     // scaleduR0
            const float endu = scaleduR0 + L + w + 1; // scaleduR0 + 11
            // iniu越界，则跳过这对匹配
            if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            // 右图的图像块在搜索范围内从左到右滑动，并计算图像块得相似度 (SAD值)
            for (int incR = -L; incR <= +L; incR++) {
                // 提取右图中 匹配的特征点 (scaleduR0, scaledvL) 为中心, 半径为w的图像快patch
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);

                // SAD 计算
                float dist = cv::norm(IL, IR, cv::NORM_L1);
                // 统计最小SAD 和 对应的窗口偏移量incR
                if (dist < bestDist) {
                    bestDist = dist;
                    bestincR = incR;
                }

                // 存储滑动块在每个偏移位置[0, 2L)的SAD值
                vDists[L + incR] = dist;
            }

            // 搜索窗口刚好越界判断
            if (bestincR == -L || bestincR == L)
                continue;

            // Step 4：亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线
            // 使用3点拟合抛物线的方式，用极小值代替之前计算的最优差值
            //    \                 / <- 由视差为14，15，16的相似度拟合的抛物线
            //      .             .(16)
            //         .14     .(15) <- int/uchar最佳视差值
            //              .
            //           （14.5）<- 真实的视差值
            //   deltaR = 15.5 - 16 = -0.5
            // 公式参考opencv sgbm源码中的亚像素插值公式
            // 或论文<<On Building an Accurate Stereo Matching System on Graphics Hardware>> 公式7
            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L + bestincR - 1]; // 最佳匹配块左侧位置的 SAD 值
            const float dist2 = vDists[L + bestincR];     // 最佳匹配块的 SAD 值
            const float dist3 = vDists[L + bestincR + 1]; // 最佳匹配块右侧位置的 SAD 值

            const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2)); // 拟合抛物线计算出的最小SAD值的 亚像素横坐标修正偏移量

            // 亚像素精度的修正量应该是在[-1, 1]之间，否则就是误匹配
            if (deltaR < -1 || deltaR > 1)
                continue;

            // Re-scaled coordinate
            // 确定右目最佳匹配特征点的 x (横)坐标： 右目特征点原始x坐标 + 计算的最小SAD对应图像块的偏移量 + 亚像素精度偏移量delta
            // 再将这个坐标放大到第0图层上
            float bestuR = mvScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

            // 视差 (左目特征点横坐标 - 右目匹配特征点亚像素横坐标)
            float disparity = (uL - bestuR);

            // 视差满足需 [0, fx) fx单位为像素
            if (disparity >= minD && disparity < maxD) {
                // 如果存在负视差，则约束为0.01
                if (disparity <= 0) {
                    disparity = 0.01;
                    bestuR = uL - 0.01;
                }
                // 根据视差值计算深度信息
                mvDepth[iL] = mbf / disparity;
                // 保存右目最相似点的x (横)坐标
                mvuRight[iL] = bestuR;
                // 保存归一化最小相似度 SAD
                // Step 5：最优视差值 / 深度选择.
                vDistIdx.push_back(pair<int, int>(bestDist, iL)); // key: 最小SAD值, value: 左目特征点索引
            }
            num_jing++;
            // 一个左图的特征点在右图找到了匹配的像素点
        }
    } // 遍历完每个左图特征点，有的在右图中找到了最佳匹配点。如果找到了，则记录左目特征点iL的深度值 mvDepth[iL] 和 其在右图中的匹配点索引 mvuRight[iL]

    // Step 6：删除离缺点(outliers)
    // 块匹配相似度阈值判断，归一化SAD最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
    // 误匹配: SAD值 > 1.5 * 1.4 * median
    sort(vDistIdx.begin(), vDistIdx.end()); // 所有匹配点对的SAD值从小到大排序
    const float median = vDistIdx[vDistIdx.size() / 2].first;
    const float thDist = 1.5f * 1.4f * median;
    int num_del = 0;
    // 从SAD值最大处遍历
    for (int i = vDistIdx.size() - 1; i >= 0; i--) {
        // SAD值 < 阈值，则退出循环，后面的都满足条件
        if (vDistIdx[i].first < thDist)
            break;
        // SAD值 >= 阈值，视为误匹配，删除
        else {
            num_del++;
            // 误匹配点置为-1，和初始化时保持一致，作为error code
            mvuRight[vDistIdx[i].second] = -1;
            mvDepth[vDistIdx[i].second] = -1;
        }
    }
    Verbose::PrintMess("粗匹配个数: " + std::to_string(num_cu) + ", 精匹配个数: " + std::to_string(num_jing) + ", correlation_thr: " + std::to_string(thDist) +
                           ", num_del: " + std::to_string(num_del) + ", num: " + std::to_string(num_jing - num_del),
                       Verbose::VERBOSITY_DEBUG);
}

void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
    mvuRight = vector<float>(N, -1);
    mvDepth = vector<float>(N, -1);

    for (int i = 0; i < N; i++) {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v, u);

        if (d > 0) {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x - mbf / d;
        }
    }
}

/**
 * @brief 当某个特征点的深度信息或者双目特征点深度值>0时，计算其在相机坐标系下的三维坐标，并将它反投影到三维世界坐标系中
 */
bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D) {
    const float z = mvDepth[i];
    if (z > 0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mRwc * x3Dc + mOw; // 转换到世界坐标系下的三维坐标
        //        std::cout << "cur_id: " << this->mnId << "   (x, y): (" << u << ", " << v << "),   (unproj_x, umproj_y, depth): (" << x << ", " << y << ", " << z << ")" << std::endl;
        //        std::cout << "cx: " << cx << ", cy: " << cy << ", fx: " << fx << ", fy: " << fy << std::endl;
        //        std::cout << "rot_wc_: " << mRwc << std::endl;
        //        std::cout << "cam_center_: " << mOw << std::endl;
        return true;
    } else
        return false;
}

/**
 * @brief 是否做完预积分
 */
bool Frame::imuIsPreintegrated() {
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

/**
 * @brief 设置 当前帧已做完预积分
 */
void Frame::setIntegrated() {
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

/**
 * @brief KannalaBrandt鱼眼相机 双目（提供Camera2，即右目的）
 * @param imLeft
 * @param imRight
 * @param timeStamp
 * @param extractorLeft
 * @param extractorRight
 * @param voc
 * @param K
 * @param distCoef
 * @param bf
 * @param thDepth
 * @param pCamera
 * @param pCamera2
 * @param Tlr
 * @param pPrevF
 * @param ImuCalib
 */
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor *extractorLeft, ORBextractor *extractorRight, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef,
             const float &bf, const float &thDepth, GeometricCamera *pCamera, GeometricCamera *pCamera2, Sophus::SE3f &Tlr, Frame *pPrevF, const IMU::Calib &ImuCalib)
    : mpcpi(NULL), mpORBvocabulary(voc), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),
      mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL),
      mpReferenceKF(static_cast<KeyFrame *>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2), mbHasPose(false), mbHasVelocity(false) {
    std::cout << "相机模型为 KannalaBrandt，构建双目鱼眼帧（双目，提供Camera2）" << endl;
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB, this, 0, imLeft, static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[0], static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB, this, 1, imRight, static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[0], static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB).count();
#endif

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if (N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if (mbInitialComputations) {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    mb = mbf / fx;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    // 鱼眼双目立体匹配
    ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    // Put all descriptors in the same matrix
    cv::vconcat(mDescriptors, mDescriptorsRight, mDescriptors);

    mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(nullptr));
    mvbOutlier = vector<bool>(N, false);

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    // 特征点去畸变：鱼眼相机有畸变参数，但特征点去畸变标志为false，实际不进行特征点去畸变，mvKeysUn = mvKeys，且mK也未改变
    UndistortKeyPoints();
}

/**
 * 鱼眼相机的双目立体匹配
 */
void Frame::ComputeStereoFishEyeMatches() {
    // Step 1: 分别只取出左、右目在共视区域的特征点，来加速
    // Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    // Step 2: 分别取出上述特征带你对应的描述子
    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    // 一些在当前模式用不到的变量给他填一下
    mvLeftToRightMatch = vector<int>(Nleft, -1);
    mvRightToLeftMatch = vector<int>(Nright, -1);
    mvDepth = vector<float>(Nleft, -1.0f);
    mvuRight = vector<float>(Nleft, -1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    // Perform a brute force between Keypoint in the left and right image
    // Step 3: 使用OpenCV的BFmatcher (Brute Force)进行暴力匹配

    // 存储匹配结果，但其中包含 非对应特征点检测为匹配。
    // 对于消除上述误匹配可采用：
    // (1) 创建cv::BFMatcher时的第二个参数设为true，进行交叉验证
    // (2) 使用KNN-matching算法，令K=2，则每个match得到两个最接近的descriptor，然后计算最接近距离和次接近距离之间的比值，当比值大于既定值时，才作为最终match。(SLAM3使用的这种方法)
    // (3) 使用RANSAC算法，对匹配结果进行筛选。
    vector<vector<cv::DMatch>> matches;
    // 使用knnMatch方法为每个关键点返回k个最佳匹配，这里k=2。输入的是左右目特征点的描述子
    BFmatcher.knnMatch(stereoDescLeft, stereoDescRight, matches, 2);

    int nMatches = 0;
    int descMatches = 0;

    // Check matches using Lowe's ratio
    // 使用一个比例阈值来判断是否为good match，来删除误匹配
    for (vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it) {
        // 对于每一对候选匹配，最小距离比次小距离的0.7倍还小，则认为是好的匹配
        if ((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7) {
            // 对于好的匹配，做三角化，且深度值有效的放入结果。并检查视差与重投影误差，以再筛查误匹配
            // For every good match, check parallax and reprojection error to discard spurious matches
            Eigen::Vector3f p3D;
            descMatches++; // 好的匹配个数 + 1
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            // 三角化
            float depth = static_cast<KannalaBrandt8 *>(mpCamera)->TriangulateMatches(mpCamera2, mvKeys[(*it)[0].queryIdx + monoLeft], mvKeysRight[(*it)[0].trainIdx + monoRight], mRlr, mtlr, sigma1,
                                                                                      sigma2, p3D);
            // 填充数据
            if (depth > 0.0001f) {
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

/**
 * @brief 鱼眼相机两个相机模式下的 单相机验证，判断路标点是否在视野中
 * @param pMP
 * @param viewingCosLimit
 * @param bRight
 * @return
 */
bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    // Step 1 获得这个地图点的世界坐标
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if (bRight) {
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.translation() + mOw;
    } else {
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    // 根据当前帧(粗糙)位姿转化到当前相机坐标系下的三维点Pc
    Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    // Step 2 关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，直接返回false
    if (PcZ < 0.0f)
        return false;

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    if (bRight)
        uv = mpCamera2->project(Pc);
    else
        uv = mpCamera->project(Pc);

    // Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
    // 判断是否在图像边界中，只要不在那么就说明无法在当前帧下进行重投影
    if (uv(0) < mnMinX || uv(0) > mnMaxX)
        return false;
    if (uv(1) < mnMinY || uv(1) > mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    // Step 4 关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
    // 得到认为的可靠距离范围:[0.8f*mfMinDistance, 1.2f*mfMaxDistance]
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    // 得到当前地图点距离当前帧相机光心的距离,注意P，mOw都是在同一坐标系下才可以
    // mOw：当前相机光心在世界坐标系下坐标
    const Eigen::Vector3f PO = P - twc;
    // 取模就得到了距离
    const float dist = PO.norm();

    // 如果不在允许的尺度变化范围内，认为重投影不可靠
    if (dist < minDistance || dist > maxDistance)
        return false;

    // Check viewing angle
    // Step 5 关卡四：计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值, 若小于cos(viewingCosLimit), 即夹角大于viewingCosLimit弧度则返回
    Eigen::Vector3f Pn = pMP->GetNormal();

    // 计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值，注意平均观测方向为单位向量
    const float viewCos = PO.dot(Pn) / dist;

    // 如果大于给定的阈值 cos(60°)=0.5，认为这个点方向太偏了，重投影不可靠，返回false
    if (viewCos < viewingCosLimit)
        return false;

    // Predict scale in the image
    // Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
    const int nPredictedLevel = pMP->PredictScale(dist, this);

    // Step 7 记录计算得到的一些参数
    if (bRight) {
        pMP->mTrackProjXR = uv(0);
        pMP->mTrackProjYR = uv(1);
        pMP->mnTrackScaleLevelR = nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    } else {
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

/**
 * @brief 鱼眼相机，根据位姿将第i个点投到世界坐标系下
 */
Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i) { return mRwc * mvStereo3Dpoints[i] + mOw; }

} // namespace ORB_SLAM3
