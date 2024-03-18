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


#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "GeometricTools.h"

#include<mutex>
#include<chrono>

namespace ORB_SLAM3
{


/**
 * @brief 局部地图线程构造函数
 * @param pSys 系统类指针
 * @param pAtlas atlas
 * @param bMonocular 是否是单目 (bug)用float赋值了
 * @param bInertial 是否是IMU模式
 * @param _strSeqName 序列名字，没用到
 */
LocalMapping::LocalMapping(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
    mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false), mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), bInitializing(false),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    mIdxInit(0), mScale(1.0), mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0), infoInertial(Eigen::MatrixXd::Zero(9,9))
{
    /*
    * mbStopRequested:    外部线程调用，为true，表示外部线程请求停止 local mapping
    * mbStopped:          为true表示可以并终止localmapping 线程
    * mbNotStop:          为true，表示不要停止 localmapping 线程，因为要插入关键帧了。需要和 mbStopped 结合使用
    * mbAcceptKeyFrames:  为true，允许接受关键帧。tracking 和local mapping 之间的关键帧调度
    * mbAbortBA:          是否流产BA优化的标志位
    * mbFinishRequested:  请求终止当前线程的标志。注意只是请求，不一定终止。终止要看 mbFinished
    * mbResetRequested:   请求当前线程复位的标志。true，表示一直请求复位，但复位还未完成；表示复位完成为false
    * mbFinished:         判断最终LocalMapping::Run() 是否完成的标志。
    */
    mnMatchesInliers = 0;

    mbBadImu = false;

    mTinit = 0.f;

    mNumLM = 0;
    mNumKFCulling=0;

#ifdef REGISTER_TIMES
    nLBA_exec = 0;
    nLBA_abort = 0;
#endif

}

/**
 * @brief 设置回环类指针
 * @param pLoopCloser 回环类指针
 */
void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

/**
 * @brief 设置跟踪类指针
 * @param pLoopCloser 回环类指针
 */
void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

/**
 * @brief 局部建图线程 主函数
 */
void LocalMapping::Run()
{
    // 标记状态，表示当前run函数正在运行，尚未结束
    mbFinished = false;

    // 主循环
    while(1)
    {
        // Step 1：告诉Tracking，LocalMapping正处于繁忙状态，不接收关键帧
        // LocalMapping线程处理的关键帧都是Tracking线程发过来的
        SetAcceptKeyFrames(false);

        // 列表 mlNewKeyFrames 有待处理的关键帧 且 IMU正常，则进行处理
        if(CheckNewKeyFrames() && !mbBadImu)
        {
#ifdef REGISTER_TIMES
            double timeLBA_ms = 0;
            double timeKFCulling_ms = 0;

            std::chrono::steady_clock::time_point time_StartProcessKF = std::chrono::steady_clock::now();
#endif
            // Step 2：处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
            ProcessNewKeyFrame();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndProcessKF = std::chrono::steady_clock::now();

            double timeProcessKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndProcessKF - time_StartProcessKF).count();
            vdKFInsert_ms.push_back(timeProcessKF);
#endif
            // Step 3：剔除质量不好的地图点，根据地图点的观测情况
            MapPointCulling();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCulling = std::chrono::steady_clock::now();

            double timeMPCulling = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCulling - time_EndProcessKF).count();
            vdMPCulling_ms.push_back(timeMPCulling);
#endif
            // Triangulate new MapPoints
            // Step 4：当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
            CreateNewMapPoints();

            // 注意orbslam2中放在了函数SearchInNeighbors（用到了mbAbortBA）后面，应该放这里更合适
            mbAbortBA = false;

            // 已经处理完队列中的最后的一个关键帧，列表为空
            if(!CheckNewKeyFrames())
            {
                // Step 5：检查并融合当前关键帧与相邻关键帧（两级相邻）中重复的地图点
                // 先完成相邻关键帧与当前关键帧的地图点的融合（在相邻关键帧中查找当前关键帧的地图点），再完成当前关键帧与相邻关键帧的地图点的融合（在当前关键帧中查找当前相邻关键帧的地图点）
                SearchInNeighbors();
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCreation = std::chrono::steady_clock::now();

            double timeMPCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCreation - time_EndMPCulling).count();
            vdMPCreation_ms.push_back(timeMPCreation);
#endif
            bool b_doneLBA = false;
            int num_FixedKF_BA = 0;
            int num_OptKF_BA = 0;
            int num_MPs_BA = 0;
            int num_edges_BA = 0;

            // Step 6：已经处理完队列中的最后的一个关键帧 且 闭环检测没有请求停止LocalMapping，则进行局部BA
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // 当前地图中关键帧数目 > 2，才进行BA优化
                if(mpAtlas->KeyFramesInMap() > 2)
                {
                    // Step 6.1：IMU模式 且 IMU已完成第一阶段初始化，进行局部地图+IMU BA
                    if(mbInertial && mpCurrentKeyFrame->GetMap()->isImuInitialized())
                    {
                        // 计算 上一关键帧到当前关键帧相机光心的距离 + 上上关键帧到上一关键帧相机光心的距离
                        float dist = (mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()).norm() +
                                (mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter()).norm();

                        // 即上上一KF到当前KF的距离>5cm，则累计 上一KF到当前KF的时间差
                        if(dist > 0.05)
                            mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;

                        // IMU 未完成第三阶段初始化
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            // 累计时间差<10s 且 上上一KF到当前KF的距离<2cm，认为运动幅度太小，不足以初始化IMU，将 mbBadImu 置为true
                            if((mTinit < 10.f) && (dist < 0.02))
                            {
                                cout << "Not enough motion for initializing. Reseting..." << endl;
                                unique_lock<mutex> lock(mMutexReset);
                                mbResetRequestedActiveMap = true;
                                mpMapToReset = mpCurrentKeyFrame->GetMap();
                                mbBadImu = true;    // 在Tracking线程里会重置当前活跃地图
                            }
                        }

                        // 判断成功跟踪匹配的点数是否足够多
                        // 1.单目 且 跟踪成功的内点数目>75 ------ 或 ------2.非单目 且 跟踪成功的内点数目>100
                        bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>100)&&!mbMonocular);

                        // 局部地图+IMU BA，优化关键帧位姿、地图点、IMU参数
                        Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());

                        Verbose::PrintMess("\t\t[optimize] localInertialBA cur_keyfrm_ id: "+std::to_string(mpCurrentKeyFrame->mnId)+" ("+std::to_string(mpCurrentKeyFrame->mnFrameId)+")", Verbose::VERBOSITY_DEBUG);
                        Verbose::PrintMess("\t\t[optimzie] bLarge: "+std::to_string(bLarge), Verbose::VERBOSITY_DEBUG);
                        Verbose::PrintMess("\t\t[optimize] abort_local_BA_: "+std::to_string(mbAbortBA), Verbose::VERBOSITY_DEBUG);
                        Verbose::PrintMess("\t\t[optimize] num_FixedKF_BA: "+std::to_string(num_FixedKF_BA), Verbose::VERBOSITY_DEBUG);
                        Verbose::PrintMess("\t\t[optimize] num_OptKF_BA: "+std::to_string(num_OptKF_BA), Verbose::VERBOSITY_DEBUG);
                        Verbose::PrintMess("\t\t[optimize] num_MPs_BA: "+std::to_string(num_MPs_BA), Verbose::VERBOSITY_DEBUG);
                        Verbose::PrintMess("\t\t[optimize] num_edges_BA: "+std::to_string(num_edges_BA), Verbose::VERBOSITY_DEBUG);
                        Verbose::PrintMess("\t\t[optimize] inertialBA2: "+std::to_string(mpCurrentKeyFrame->GetMap()->GetIniertialBA2()), Verbose::VERBOSITY_DEBUG);

                        b_doneLBA = true;
                    }
                    // Step 6.2：非IMU模式 或 IMU 未完成第一阶段初始化，进行纯视觉局部地图 BA
                    else {
                        // 局部地图 BA，不包括IMU数据。优化局部关键帧(一级共视关键帧)位姿、局部地图点。
                        // 注意这里的第二个参数是按地址传递的,当这里的 mbAbortBA 状态发生变化时，能够及时执行/停止BA
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                        b_doneLBA = true;
                    }
                }
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndLBA = std::chrono::steady_clock::now();

                if(b_doneLBA)
                {
                    timeLBA_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLBA - time_EndMPCreation).count();
                    vdLBA_ms.push_back(timeLBA_ms);

                    nLBA_exec += 1;
                    if(mbAbortBA)
                    {
                        nLBA_abort += 1;
                    }
                    vnLBA_edges.push_back(num_edges_BA);
                    vnLBA_KFopt.push_back(num_OptKF_BA);
                    vnLBA_KFfixed.push_back(num_FixedKF_BA);
                    vnLBA_MPs.push_back(num_MPs_BA);
                }

#endif

                // Step 7：进行IMU第一阶段初始化（2s内）。目的：快速初始化 IMU，尽快用IMU来跟踪。成功的标志 mbImuInitialized
                // IMU模式 且 IMU未完成第一阶段初始化
                if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
                {
                    cout << "start IMU第一阶段初始化" << endl;
                    if (mbMonocular)
                        InitializeIMU(1e2, 1e10, true);
                    else
                        InitializeIMU(1e2, 1e5, true);
                    cout << "end IMU第一阶段初始化" << endl;
                }

                // Step 8：检测并剔除 当前帧相邻的关键帧中冗余的关键帧
                // Tracking中关键帧插入条件比较松，交给LocalMapping线程的关键帧会比较密集(多)，这里要删除冗余。冗余的判定：该关键帧的90%的地图点可以被其它关键帧观测到
                KeyFrameCulling();

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndKFCulling = std::chrono::steady_clock::now();

                timeKFCulling_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndKFCulling - time_EndLBA).count();
                vdKFCulling_ms.push_back(timeKFCulling_ms);
#endif
                // Step 9: IMU模式 且 如果距离IMU第一阶段初始化成功累计时间差 < 50s，进行IMU第二、三阶段初始化
                if ((mTinit < 50.0f) && mbInertial)
                {
                    // IMU已完成第一阶段初始化 且 跟踪正常
                    if(mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState==Tracking::OK) // Enter here everytime local-mapping is called
                    {
                        // Step 9.1 进行IMU第二阶段初始化。目的：快速修正IMU，在短时间内使得IMU参数相对靠谱。成功标志为 mbIMU_BA1
                        // 累计时间差 > 5s 且 IMU未完成第二阶段初始化
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA1()) {
                            if (mTinit > 5.0f) {
                                cout << "start VIBA 1" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA1();
                                if (mbMonocular)
                                    InitializeIMU(1.f, 1e5, true);
                                else
                                    InitializeIMU(1.f, 1e5, true);
                                cout << "end VIBA 1" << endl;
                            }
                        }
                        // Step 9.2 进行IMU第三阶段初始化。目的：再次优化IMU，保证IMU参数的高精度。成功标志为 mbIMU_BA2
                        // 累计时间 > 15s 且 IMU未完成第三阶段初始化
                        else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2()) {
                            if (mTinit > 15.0f) {
                                cout << "start VIBA 2" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA2();
                                if (mbMonocular)
                                    InitializeIMU(0.f, 0.f, true);
                                else
                                    InitializeIMU(0.f, 0.f, true);
                                cout << "end VIBA 2" << endl;
                            }
                        }

                        // Step 9.3 进行IMU第四阶段初始化（仅单目）
                        // 单目+IMU模式，且关键帧个数 <= 200，且从25-75s内，每10s单独进行一次 尺度 与 重力方向 的优化
                        if ( ((mpAtlas->KeyFramesInMap()) <= 200) && ((mTinit>25.0f && mTinit<25.5f)||(mTinit>35.0f && mTinit<35.5f)||(mTinit>45.0f && mTinit<45.5f)||(mTinit>55.0f && mTinit<55.5f)||(mTinit>65.0f && mTinit<65.5f)||(mTinit>75.0f && mTinit<75.5f)) ) {
                            if (mbMonocular)
                                // 使用了所有关键帧，但只优化重力方向和尺度，以及速度和偏置（其实就是一切跟惯性相关的量）
                                ScaleRefinement();
                        }
                    }
                }
            }

#ifdef REGISTER_TIMES
            vdLBASync_ms.push_back(timeKFCulling_ms);
            vdKFCullingSync_ms.push_back(timeKFCulling_ms);
#endif

            // Step 10：将当前帧加入到闭环检测队列中
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndLocalMap = std::chrono::steady_clock::now();

            double timeLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLocalMap - time_StartProcessKF).count();
            vdLMTotal_ms.push_back(timeLocalMap);
#endif
        }
        // 当要终止当前线程的时候
        else if(Stop() && !mbBadImu)
        {
            // 如果还没有结束利索,那么等等它
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            // 然后确定终止了就跳出LocalMapping线程的主循环
            if(CheckFinish())
                break;
        }

        // 查看是否有复位线程的请求
        ResetIfRequested();

        // 告诉Tracking线程开始接收关键帧
        SetAcceptKeyFrames(true);

        // 如果当前线程已经结束了就跳出主循环
        if(CheckFinish())
            break;

        usleep(3000);
    }// 主循环结束

    // 设置线程已经终止
    SetFinish();
}

/**
 * @brief 插入关键帧,由外部（Tracking）线程调用;这里只是插入到列表中,等待线程主函数对其进行处理
 * @param pKF 新的关键帧
 */
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}

/**
 * @brief 查看列表 mlNewKeyFrames 中是否有等待被插入的关键帧：false: 列表为空,没有等待的； true: 列表不为空，有等待的
 */
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

/**
 * @brief 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
 */
void LocalMapping::ProcessNewKeyFrame()
{
    // Step 1：从缓冲队列中取出一帧关键帧
    // 该关键帧队列是Tracking线程向LocalMapping中插入的关键帧组成
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        // 取出列表中最前面的关键帧，作为当前要处理的关键帧
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        // 取出最前面的关键帧后，在原来的列表里删掉该关键帧
        mlNewKeyFrames.pop_front();
    }

    // Step 2：计算该关键帧特征点的Bow信息
    mpCurrentKeyFrame->ComputeBoW();

    // Step 3：处理当前关键帧中有效的地图点，更新normal，描述子等信息
    // TrackLocalMap中和当前帧新匹配上的地图点和当前关键帧进行关联绑定
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    int lm_count = 0;
    // 遍历当前关键帧所有匹配得到地图点
    for(size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                // 该地图点 不是 来自当前关键帧的观测，则为该地图点添加观测
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    // 更新该点的平均观测方向和观测距离范围
                    pMP->UpdateNormalAndDepth();
                    // 更新地图点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                // 该地图点 被 当前关键帧观测到，但这个地图点却没有保存这个关键帧的观测信息
                // 这些地图点可能来自双目或RGBD跟踪过程中新生成的地图点，或者是CreateNewMapPoints 中通过三角化产生
                else
                {
                    // 将上述地图点放入mlpRecentAddedMapPoints，等待后续MapPointCulling函数的检验
                    mlpRecentAddedMapPoints.push_back(pMP);
                    ++lm_count;
                }
            }
        }
    }
    std::cout << "\t\t[store_new_keyframe] 当前关键帧 "<< mpCurrentKeyFrame->mnId <<" ("<< mpCurrentKeyFrame->mnFrameId <<") 新添加了 "<< lm_count <<" 个地图点" << std::endl;

    // Update links in the Covisibility Graph
    // Step 4：更新关键帧间的连接关系（共视图）
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    // Step 5：将该关键帧插入到地图中
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
}

/**
 * @brief 处理新的关键帧，使队列为空，注意这里只是处理了关键帧，并没有生成MP
 */
void LocalMapping::EmptyQueue()
{
    while(CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

/**
 * @brief 检查新增地图点，根据地图点的观测情况剔除质量不好的新增的地图点
 * mlpRecentAddedMapPoints: 存储新增的地图点, 在ProcessNewKeyframe中更新, 这里是要删除其中不靠谱的
 */
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    // Step 1：根据相机类型设置不同的观测阈值
    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    int borrar = mlpRecentAddedMapPoints.size();
    std::cout << "\t\t新添加地图点的个数: " << borrar << std::endl;
    int num_removed = 0;
    int num1 = 0, num2 = 0, num3 = 0, num4 = 0;

    // Step 2：遍历检查的新添加的MapPoints
    while(lit != mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;

        // Step 2.1：已经是坏点的MapPoints直接从检查链表中删除
        if(pMP->isBad())
        {
            num1++;
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // Step 2.2：跟踪到该MapPoint的Frame数 相比 预计可观测到该MapPoint的Frame数 的比例 < 25%，删除
        // (mnFound / mnVisible） < 25%
        // mnFound ：地图点被多少帧（包括普通帧）看到，次数越多越好
        // mnVisible：地图点应该被看到的次数
        // (mnFound / mnVisible）：对于大FOV镜头这个比例会高，对于窄FOV镜头这个比例会低
        else if(pMP->GetFoundRatio() < 0.25f)
        {
            num2++;
            num_removed++;
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // Step 2.3：从该点建立开始，到现在已经过了 >= 2个关键帧 且 观测到该点的关键帧数 <= cnThObs帧，那么删除该点
        else if( ((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs )
        {
            num3++;
            num_removed++;
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // Step 2.4：从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
        // 因此没有SetBadFlag()，仅从队列中删除，放弃继续对该MapPoint的检测
        else if( ((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3 )
        {
            num4++;
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            lit++;
            borrar--;   // 是好的点时进行--，所以最后borrar是坏的点个数
        }
    }
    std::cout << "\t\t[remove_redundant_landmarks] 被标记需剔除的地图点个数: " << num_removed << ", 从链表中删除的地图点个数: " << borrar << ", 1: " << num1 << ", 2: " << num2 << ", 3: " << num3 << ", 3: " << num4 << std::endl;
}

/**
 * @brief 用当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
 */
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    // nn表示搜索最佳共视关键帧的数目。
    int nn = 10;
    // For monocular case
    // 不同传感器下要求不一样, 单目的时候需要有更多的具有较好共视关系的关键帧来建立地图 (0.4版本是20个，又加了许多)
    if(mbMonocular)
        nn = 30;

    // Step 1：获取当前关键帧的共视程度前10的一级相邻关键帧
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    // IMU模式，则再多添加当前关键帧 更多的上一关键帧、上上一关键帧
    if (mbInertial)
    {
        KeyFrame* pKF = mpCurrentKeyFrame;  // 当前关键帧
        int count = 0;
        // 相邻关键帧个数不够nn个 且 当前关键帧存在上一个关键帧 且 在该选项下添加的关键帧个数需<nn，则一直循环添加
        while((vpNeighKFs.size() <= nn) && (pKF->mPrevKF) && (count++ < nn))
        {
            // 在相邻关键帧中查找当前关键帧的上一个关键帧 pKF->mPrevKF，返回一个迭代器it
            vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
            // 没有找到，表明其没有加入过，则将其添加到相邻关键帧中
            if(it == vpNeighKFs.end())
                vpNeighKFs.push_back(pKF->mPrevKF);
            pKF = pKF->mPrevKF; // 更新当前关键帧为上一关键帧，用于重复添加上一关键帧
        }
    }

    float th = 0.6f;
    // 特征点匹配配置 最小距离 < 0.6*次小距离，比较苛刻了。不检查旋转
    ORBmatcher matcher(th,false);

    // 当前关键帧从世界坐标系到相机坐标系的变换矩阵
    Sophus::SE3<float> sophTcw1 = mpCurrentKeyFrame->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose();   // 旋转矩阵是一个正交阵，因此它的逆=它的转置
    Eigen::Vector3f tcw1 = sophTcw1.translation();
    // 当前关键帧（左目）光心 在世界坐标系中的坐标
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    // 当前关键帧的 内参
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    // 用于后面的点深度的验证; 这里的1.5是经验值
    const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;
    // 以下是统计点数用的，没啥作用
    int countStereo = 0;    // 双目建立的地图点个数
    int countStereoGoodProj = 0;    // 双目成功建立三维点个数
    int countStereoAttempt = 0;     // 想要使用双目恢复三维点的个数
    int totalStereoPts = 0;         // 双目特征点有效个数
    int num_mappoint_total = 0;   // liuzhi加，统计新建地图点个数

    // Search matches with epipolar restriction and triangulate，通过三角化和极线约束构建更多地图点
    // Step 2：遍历候选的相邻关键帧
    for(size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        // 下面的过程会比较耗费时间, 因此如果有新的关键帧需要处理的话, 就先去处理新的关键帧吧
        if(i > 0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i]; // 相邻关键帧

        GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        // 相邻关键帧光心 在世界坐标系中的坐标
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        // 两个关键帧间的相机位移向量
        Eigen::Vector3f vBaseline = Ow2 - Ow1;
        // 两个关键帧间的位移长度
        const float baseline = vBaseline.norm();

        // Step 3：判断相机运动的基线是不是足够长
        // 如果是双目相机，关键帧间距 < 相机左右目的基线时，不生成3D点。因为太短的基线下能够恢复的地图点不稳定
        if(!mbMonocular)
        {
            if(baseline < pKF2->mb)
                continue;
        }
        // 单目相机情况
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);  // 相邻关键帧的场景深度中值
            const float ratioBaselineDepth = baseline / medianDepthKF2;    // baseline与景深的比例
            // 如果比例特别小，基线太短恢复的3D点不准，那么跳过当前邻接的关键帧，不生成3D点
            if(ratioBaselineDepth < 0.01)
                continue;
        }

        // Search matches that fullfil epipolar constraint
        // Step 4：通过BoW对两关键帧的未匹配的特征点快速匹配，用极线约束抑制离群点，生成新的匹配点对

        // 只存储当前关键帧和相邻关键帧的匹配特征点对关系。key：当前关键帧匹配点id，value：相邻关键帧的匹配点id
        vector<pair<size_t,size_t> > vMatchedIndices;
        // IMU模式, 且 经过三次初始化, 且 为刚丢失状态
        bool bCoarse = mbInertial && mpTracker->mState==Tracking::RECENTLY_LOST && mpCurrentKeyFrame->GetMap()->GetIniertialBA2();

        // 通过极线约束的方式找到匹配点（且该点还没有成为MP，注意非单目已经生成的MP这里直接跳过不做匹配，所以最后并不会覆盖掉特征点对应的MP）
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices, false, bCoarse);
        std::cout << "\t\t   当前关键帧 "<<std::to_string(mpCurrentKeyFrame->mnId)<<" ("<<std::to_string(mpCurrentKeyFrame->mnFrameId)<<") 和 相邻关键帧 "<<std::to_string(pKF2->mnId)<<" ("<<std::to_string(pKF2->mnFrameId)<<") 匹配了 "<<std::to_string(vMatchedIndices.size())<<" 对";

        // 相邻关键帧的外参
        Sophus::SE3<float> sophTcw2 = pKF2->GetPose();
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4();
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0);
        Eigen::Matrix<float,3,3> Rwc2 = Rcw2.transpose();
        Eigen::Vector3f tcw2 = sophTcw2.translation();
        // 相邻关键帧的内参
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        // Step 5：对每个匹配点对通过三角化生成3D点, 和 Triangulate 函数差不多
        const int nmatches = vMatchedIndices.size();
        int num_mappoint = 0;   // liuzhi加，统计新建地图点个数
        // 遍历当前关键帧和相邻关键帧pKF2的每个匹配点对
        for(int ikp = 0; ikp < nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;   // 当前关键帧匹配点id
            const int &idx2 = vMatchedIndices[ikp].second;  // 相邻关键帧2的匹配点id

            // 当前关键帧 匹配特征点
            const cv::KeyPoint &kp1 = (mpCurrentKeyFrame -> NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
                                                                         : (idx1 < mpCurrentKeyFrame -> NLeft) ? mpCurrentKeyFrame -> mvKeys[idx1]
                                                                                                               : mpCurrentKeyFrame -> mvKeysRight[idx1 - mpCurrentKeyFrame -> NLeft];
            // mvuRight中存放着极线校准后双目特征点在右目对应的像素横坐标，如果不是极线校准的双目或者没有找到匹配点，其值将为-1（或者rgbd）
            const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
            // PinHole双目 && kp1_ur >= 0
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur >= 0);
            // 查看点idx1是否为 非立体匹配双目中右目的点 (pinHole为false)
            const bool bRight1 = (mpCurrentKeyFrame -> NLeft == -1 || idx1 < mpCurrentKeyFrame -> NLeft) ? false
                                                                                                         : true;
            // 相邻关键帧 匹配特征点
            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
            // mvuRight中存放着双目的深度值，如果不是双目，其值将为-1
            // mvuRight中存放着极线校准后双目特征点在右目对应的像素横坐标，如果不是极线校准的双目或者没有找到匹配点，其值将为-1（或者rgbd）
            const float kp2_ur = pKF2->mvuRight[idx2];
            // 立体匹配双目 && kp2_ur >= 0
            bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur>=0);
            // 查看点idx2是否为  非立体匹配双目中右目的点 (pinHole为false)
            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                               : true;

            // 如果为KB鱼眼相机，确定两个点所在相机之间的位姿关系
            if(mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2){
                if(bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if(bRight1 && !bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if(!bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else{
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
                eigTcw1 = sophTcw1.matrix3x4();
                Rcw1 = eigTcw1.block<3,3>(0,0);
                Rwc1 = Rcw1.transpose();
                tcw1 = sophTcw1.translation();

                eigTcw2 = sophTcw2.matrix3x4();
                Rcw2 = eigTcw2.block<3,3>(0,0);
                Rwc2 = Rcw2.transpose();
                tcw2 = sophTcw2.translation();
            }

            // Check parallax between rays
            // Step 5.4：利用匹配点反投影得到视差角
            // 特征点反投影, 其实得到的是在各自相机坐标系下的一个非归一化的方向向量, 和这个点的反投影射线重合
            Eigen::Vector3f xn1 = pCamera1->unprojectEig(kp1.pt);
            Eigen::Vector3f xn2 = pCamera2->unprojectEig(kp2.pt);

            // 由相机坐标系转到世界坐标系(得到的是那条反投影射线的一个同向向量在世界坐标系下的表示,还是只能够表示方向)，得到视差角余弦值
            Eigen::Vector3f ray1 = Rwc1 * xn1;
            Eigen::Vector3f ray2 = Rwc2 * xn2;
            // 这个就是求向量之间角度公式
            const float cosParallaxRays = ray1.dot(ray2) / ( ray1.norm() * ray2.norm() );

            // 加1是为了让cosParallaxStereo随便初始化为一个很大的值
            float cosParallaxStereo = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            // Step 5.5：对于双目，利用双目得到视差角；单目相机没有特殊操作
            // 传感器是双目相机, 并且当前的关键帧的这个点有对应的深度
            if(bStereo1)
                // 假设是平行的双目相机，计算出两个相机观察这个点的时候的视差角;
                // ? 感觉直接使用向量夹角的方式计算会准确一些啊（双目的时候），那么为什么不直接使用那个呢？
                // 回答：因为双目深度值、基线是更可靠的，比特征匹配再三角化出来的稳
                cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                //传感器是双目相机,并且邻接的关键帧的这个点有对应的深度，和上面一样操作
                cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2,pKF2->mvDepth[idx2]));

            // 统计用的
            if (bStereo1 || bStereo2) totalStereoPts++; // 双目特征点有效个数+1
            // 得到两个相机观测各自特征点的 双目最小视差角的余弦值
            cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);
            // Step 5.6：三角化恢复3D点
            Eigen::Vector3f x3D;

            bool goodProj = false;
            bool bPointStereo = false;
            // cosParallaxRays > 0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)：表明视线角正常
            // cosParallaxRays < cosParallaxStereo：前后帧视线角 > 两个相机双目视线角，则用前后帧三角化恢复地图点；否则使用双目的左右目恢复地图点；如果没有双目则跳过
            if(cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && (bStereo1 || bStereo2 ||
                                                                          (cosParallaxRays < 0.9996 && mbInertial) || (cosParallaxRays < 0.9998 && !mbInertial)))
            {
                // 用三角化恢复地图点
                goodProj = GeometricTools::Triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D);
                if(!goodProj)
                    continue;
            }
            else if(bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
            {
                // 用双目左右目恢复，相机1的视差角大，用它恢复
                countStereoAttempt++;   // 想要使用双目恢复三维点的个数 + 1
                bPointStereo = true;
                // 如果是双目，用视差角更大的那个双目信息来恢复，直接用已知3D点反投影了
                goodProj = mpCurrentKeyFrame->UnprojectStereo(idx1, x3D);
            }
            else if(bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
            {
                // 用双目左右目恢复，相机2的视差角大，用它恢复
                countStereoAttempt++;   // 想要使用双目恢复三维点的个数 + 1
                bPointStereo = true;
                // 如果是双目，用视差角更大的那个双目信息来恢复，直接用已知3D点反投影了
                goodProj = pKF2->UnprojectStereo(idx2, x3D);
            }
            else
            {
                // 没有双目则跳过
                continue; //No stereo and very low parallax
            }


            if(goodProj && bPointStereo)
                countStereoGoodProj++;  // 双目成功建立三维点个数+1

            if(!goodProj)
                continue;

            //Check triangulation in front of cameras
            // Step 5.7：检测生成的3D点是否在相机前方,不在的话就放弃这个点
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if(z1 <= 0)
                continue;

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2 <= 0)
                continue;

            //Check reprojection error in first keyframe
            // Step 5.8：计算3D点在当前关键帧下的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3D) + tcw1(0);    // 转换到当前相机的相机坐标系下
            const float y1 = Rcw1.row(1).dot(x3D) + tcw1(1);
            const float invz1 = 1.0 / z1;

            // 单目情况下
            if(!bStereo1)
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1)); // 投影到像素坐标系下
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;
                // 假设测量有一个像素的偏差，2自由度卡方检验阈值是5.991
                if((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                    continue;

            }
            // 双目情况
            else
            {
                float u1 = fx1 * x1 * invz1 + cx1;  // 相机坐标系
                // 根据视差公式计算假想的右目坐标
                float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                // 自由度为3，卡方检验阈值是7.8
                if((errX1*errX1 + errY1*errY1 + errX1_r*errX1_r) > 7.81473 * sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            // 计算3D点在另一个关键帧下的重投影误差，操作同上
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2(0);
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2));
                float errX2 = uv2.x - kp2.pt.x;
                float errY2 = uv2.y - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            // Step 5.8：检查尺度连续性
            // 世界坐标系下，3D点与相机间的向量，方向由相机指向3D点
            Eigen::Vector3f normal1 = x3D - Ow1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = x3D - Ow2;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            if(mbFarPoints && (dist1 >= mThFarPoints || dist2 >= mThFarPoints)) // MODIFICATION
                continue;

            // ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioDist = dist2 / dist1;
            // 金字塔尺度因子的比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

            // 距离的比例和图像金字塔的比例不应该差太多，否则就跳过
            if(ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                continue;

            // Triangulation is succesfull
            // Step 6：三角化生成3D点成功，构造成地图点
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());
            num_mappoint++;

            if (bPointStereo)
                countStereo++;  // 双目建立的地图点个数+1

            // Step 6.1：为该MapPoint添加属性：
            // a.观测到该MapPoint的关键帧
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            // b.该MapPoint的描述子
            pMP->ComputeDistinctiveDescriptors();

            // c.该MapPoint的平均观测方向和深度范围
            pMP->UpdateNormalAndDepth();

            // Step 7：将新产生的点放入检测队列
            // 这些MapPoints都会经过MapPointCulling函数的检验
            mpAtlas->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }// 遍历一个相邻关键帧结束
        num_mappoint_total += num_mappoint;
        std::cout << "，通过三角化生成新地图点 " << num_mappoint << " 个，加入到全局地图中" << std::endl;
    }// 遍历所有相邻关键帧结束
    Verbose::PrintMess("\t\tCreateNewMapPoints中，当前关键帧与其 " +std::to_string(vpNeighKFs.size())+" 个相邻关键帧，通过三角化生成新地图点 "+std::to_string(num_mappoint_total)+" 个，加入到全局地图中", Verbose::VERBOSITY_DEBUG);
}

/**
 * @brief 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
 */
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    // Step 1：获得当前关键帧在共视图中权重排名前nn的邻接关键帧+
    // 开始之前先定义几个概念
    // 当前关键帧的邻接关键帧，称为一级相邻关键帧，也就是邻居
    // 与一级相邻关键帧相邻的关键帧，称为二级相邻关键帧，也就是邻居的邻居

    // 单目情况要30个邻接关键帧 (0.4版本是20个，又加了许多)
    int nn = 10;
    if(mbMonocular)
        nn = 30;
    // 获取当前关键帧的共视程度前10的一级相邻关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    // Step 2：存储一级相邻关键帧 和 其二级相邻关键帧
    vector<KeyFrame*> vpTargetKFs;

    // Step 2.1: 存入一级共视关键帧
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;  // 某候选一级共视关键帧
        // 其是坏的 或 已和当前关键帧进行过融合操作，则跳过
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);    // 存入该一级共视关键帧
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;  // 将当前关键帧标记为 该一级共视关键帧的融合目标，标记已经加入，避免重复添加
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    // Step 2.2: 将一级相邻关键帧的共视关系最好的20个相邻关键帧 作为二级相邻关键帧 (0.4版本是5个)
    for(int i = 0, imax = vpTargetKFs.size(); i < imax; i++)
    {
        // 获取 第i个一级共视关键帧的 共视程度最高的20个相邻关键帧
        const vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        // 遍历二级相邻关键帧
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            // 该二级相邻关键帧是坏的 或 已和当前关键帧进行过融合操作 或 该二级相邻关键帧是当前关键帧，则跳过
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);   // 存入该二级共视关键帧
            pKFi2->mnFuseTargetForKF = mpCurrentKeyFrame->mnId; // 将当前关键帧标记为 该二级共视关键帧的融合目标，标记已经加入，避免重复添加
        }
        // 若放弃BA，则跳出循环
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    // IMU模式，则再多添加当前关键帧 更多的上一关键帧、上上一关键帧
    if(mbInertial)
    {
        KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF;
        // 若一级、二级相邻关键帧个数 < 20 且 当前关键帧的上一个关键帧存在，则一直添加
        while(vpTargetKFs.size() < 20 && pKFi)
        {
            // 当前关键帧的上一个关键帧是坏的 或 已和当前关键帧进行过融合操作，则跳过它，看它的上一关键帧
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            {
                pKFi = pKFi->mPrevKF;
                continue;
            }
            // 将其添加到相邻关键帧中
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId; // 将当前关键帧标记为它的融合目标，标记已经加入，避免重复添加
            pKFi = pKFi->mPrevKF;   // 更新
        }
    }

    // Search matches by projection from current KF in target KFs
    // 使用默认参数, 最优和次优比例0.6, 匹配时检查特征点的旋转
    ORBmatcher matcher;

    // Step 3：将当前帧的地图点分别与一级二级相邻关键帧地图点进行融合 -- 正向

    // 当前关键帧已匹配的地图点
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    // 遍历相邻关键帧
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;  // 相邻关键帧

        // 将地图点投影到关键帧中进行匹配和融合，融合策略如下：
        // 1.如果地图点能匹配相邻关键帧的特征点，并且该点有对应的地图点，那么选择观测数目多的替换两个地图点
        // 2.如果地图点能匹配相邻关键帧的特征点，并且该点没有对应的地图点，那么为该点添加该投影地图点
        // 注意这个时候对地图点融合的操作是立即生效的
        matcher.Fuse(pKFi,vpMapPointMatches);
        // KB鱼眼相机
        if(pKFi->NLeft != -1) matcher.Fuse(pKFi,vpMapPointMatches,true);
    }


    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    // Step 4：将一级二级相邻关键帧地图点分别与当前关键帧地图点进行融合 -- 反向

    // 存储要融合的 一级邻接和二级邻接关键帧 所有MapPoints的集合
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

    //  Step 4.1：遍历每一个一级邻接和二级邻接关键帧，收集他们的地图点存储到 vpFuseCandidates
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        // 遍历当前一级邻接和二级邻接关键帧中所有的MapPoints, 找出需要进行融合的并且加入到集合中
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    // Step 4.2：进行地图点投影融合, 和正向融合操作是完全相同的
    // 不同的是正向操作是"每个相邻关键帧和当前关键帧的地图点进行融合",而这里的是"当前关键帧和所有相邻关键帧的地图点进行融合"
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);
    // KB鱼眼相机
    if(mpCurrentKeyFrame->NLeft != -1) matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates,true);


    // Update points
    // Step 5：更新当前关键帧地图点的描述子、深度、观测主方向等属性
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                // 在所有找到pMP的关键帧中，获得最佳的描述子
                pMP->ComputeDistinctiveDescriptors();
                // 更新平均观测方向和观测距离
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    // Step 6：更新当前帧的MapPoints后更新与其它帧的连接关系
    // 更新共视图
    mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

/**
 * @brief 检测当前关键帧在共视图中的关键帧，根据地图点在共视图中的冗余程度剔除该共视关键帧
 *
 * 冗余关键帧的判定：90%以上的地图点能被其他关键帧（至少3个）观测到
 */
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points

    // 该函数里变量层层深入，这里列一下：
    // mpCurrentKeyFrame：   当前关键帧，本程序就是判断它是否需要删除
    // pKF：   mpCurrentKeyFrame的某一个共视关键帧
    // vpMapPoints：   pKF对应的所有地图点
    // pMP：   vpMapPoints中的某个地图点
    // observations：   所有能观测到pMP的关键帧
    // pKFi：   observations中的某个关键帧
    // scaleLeveli：   pKFi的金字塔尺度
    // scaleLevel：    pKF的金字塔尺度
    const int Nd = 21;
    // 更新当前关键帧的共视关系
    mpCurrentKeyFrame->UpdateBestCovisibles();
    // Step 1: 根据共视图Covisibility Graph获取当前关键帧的共视关键帧，且共视程度由高到低
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th;
    // 非IMU模式
    if(!mbInertial)
        redundant_th = 0.9;
    // IMU+单目
    else if (mbMonocular)
        redundant_th = 0.9;
    // IMU+双目/RGB-D
    else
        redundant_th = 0.5;

    // IMU是否完成初始化
    const bool bInitImu = mpAtlas->isImuInitialized();
    int count = 0;  // 遍历的共视关键帧个数

    // Compoute last KF from optimizable window:
    unsigned int last_ID;   // 当前关键帧前面第21关键帧的id，不足21个时为最前面关键帧的id
    // IMU模式
    if (mbInertial)
    {
        int count = 0;
        KeyFrame* aux_KF = mpCurrentKeyFrame;
        // 找到当前关键帧前面的第21个关键帧的id。如果不够21，则为最前面关键帧的id
        while(count < Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }

    // 遍历当前关键帧的 所有共视关键帧
    for(vector<KeyFrame*>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
    {
        count++;    // 遍历的共视关键帧个数+1
        KeyFrame* pKF = *vit;   // 该共视关键帧

        // 如果该共视关键帧是地图中的第1个关键帧 或 被标记为坏帧，则跳过
        if((pKF->mnId == pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        // Step 2：获取该共视关键帧的地图点
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        // 记录某个点被观测次数，后面并未使用
        int nObs = 3;
        const int thObs = nObs;   // 地图点被关键帧观测的次数阈值，默认为3
        // 记录冗余观测点的数目
        int nRedundantObservations = 0; // 冗余地图点个数 （被观测次数>3被认为冗余）
        int nMPs = 0;   // 有效共视关键帧的地图点个数

        // Step 3：遍历该共视关键帧的 所有地图点，判断是否90%以上的地图点能被其它至少3个关键帧（同样或者更低层级）观测到
        for(size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i]; // 该共视关键帧的一个地图点
            // 该地图点存在
            if(pMP)
            {
                // 该地图点不是坏点
                if(!pMP->isBad())
                {
                    // 对于双目，仅考虑近处（不超过基线的40倍 euroc为60倍，realsense为40倍 ）的地图点 0 < depth < 40倍基线
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
                            continue;
                    }

                    nMPs++; // 有效共视关键帧的地图点个数+1

                    // pMP->Observations() 是观测到该地图点的相机总数目（单目+1，双目+2）> 3, 是冗余点
                    if(pMP->Observations() > thObs)
                    {
                        // 该地图点 在该共视关键帧中对应特征点 的层级
                        const int &scaleLevel = (pKF -> NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                                     : (i < pKF -> NLeft) ? pKF -> mvKeys[i].octave
                                                                                          : pKF -> mvKeysRight[i].octave;
                        const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations(); // 观测到该地图点的 关键帧 和 该地图点在该关键帧中的 索引
                        int nObs = 0;   // 有效观测到该地图点的关键帧个数

                        // 遍历观测到该地图点的 所有关键帧
                        for(map<KeyFrame*, tuple<int,int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;    // 取出观测到该地图点的关键帧
                            // 如果是当前共视关键帧，则跳过
                            if(pKFi == pKF)
                                continue;
                            tuple<int,int> indexes = mit->second;   // 该地图点在其观测关键帧中的 索引。单目或立体匹配双目，则为<idx, -1>
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int scaleLeveli = -1;
                            // 单目或立体匹配双目
                            if(pKFi -> NLeft == -1)
                                scaleLeveli = pKFi->mvKeysUn[leftIndex].octave; // 该地图点在其观测关键帧中对应特征点 的层级
                            // 非立体匹配双目
                            else {
                                if (leftIndex != -1) {
                                    scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                                }
                                if (rightIndex != -1) {
                                    int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
                                    scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                  : scaleLeveli;
                                }
                            }

                            // 尺度约束：为什么pKF 尺度+1 要大于等于 pKFi 尺度？
                            // 该地图点在其观测关键帧中对应特征点 的层级 需<= 地图点在该共视关键帧中对应特征点 的层级 + 1
                            // 回答：因为同样或更低金字塔层级的地图点更准确
                            if(scaleLeveli <= scaleLevel + 1)
                            {
                                nObs++; // 有效观测到该地图点的关键帧个数+1
                                // 已经找到>3个满足条件的观测关键帧，就停止不找了
                                if(nObs > thObs)
                                    break;
                            }
                        }// 一个地图点的所有观测关键帧遍历结束
                        // 地图点至少被3个关键帧观测到，就记录为冗余点，更新冗余点计数数目
                        if(nObs > thObs)
                        {
                            nRedundantObservations++;   // 冗余地图点个数+1
                        }
                    }
                    // 该地图点被观测次数<=3，不是冗余点
                }
            }
            // 该共视关键帧的一个地图点遍历结束
        }// 该共视关键帧的所有地图点遍历结束，统计好了有效地图点个数 (包括冗余地图点和不冗余地图点)，和冗余地图点个数

        // Step 4：该共视关键帧90%以上的有效地图点被标记为冗余的，则删除该共视关键帧
        if(nRedundantObservations > redundant_th * nMPs)
        {
            // IMU模式，需要更改前后关键帧的连续性，且预积分要叠加起来
            if (mbInertial)
            {
                // 地图中关键帧个数 < 21个，则跳过，不删除
                if (mpAtlas->KeyFramesInMap() <= Nd)
                    continue;

                // 该共视关键帧与当前关键帧id差一个，则跳过，不删除
                if(pKF->mnId > (mpCurrentKeyFrame->mnId - 2))
                    continue;

                // 该共视关键帧有 前、后关键帧
                if(pKF->mPrevKF && pKF->mNextKF)
                {
                    // 其后关键帧时间戳 - 前关键帧时间戳
                    const float t = pKF->mNextKF->mTimeStamp - pKF->mPrevKF->mTimeStamp;

                    // 条件1：IMU已完成初始化，且距当前帧的ID超过21，且前后两个关键帧时间间隔 < 3s
                    // 条件2：时间间隔 < 0.5s
                    // 两个条件满足一个即可
                    if((bInitImu && (pKF->mnId < last_ID) && t<3.) || (t<0.5))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                        std::cout << "\t\t关键帧 " << pKF->mnId << " ( " << pKF->mnFrameId << " ) 被删除" << std::endl;
                    }
                    // 没经过IMU初始化的第三阶段，且关键帧与其前一个关键帧的距离小于0.02m，且前后两个关键帧时间间隔小于3s
                    else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && ((pKF->GetImuPosition() - pKF->mPrevKF->GetImuPosition()).norm()<0.02) && (t<3))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                        std::cout << "\t\t关键帧 " << pKF->mnId << " ( " << pKF->mnFrameId << " ) 被删除" << std::endl;
                    }
                }
            }
            // 非IMU模式，直接删除该共视关键帧
            else
            {
                pKF->SetBadFlag();
                std::cout << "\t\t关键帧 " << pKF->mnId << " ( " << pKF->mnFrameId << " ) 被删除" << std::endl;
            }
        }
        // 条件1：遍历共视关键帧个数 > 20 且放弃BA
        // 条件2：遍历共视关键帧个数 > 100
        // 两个条件满足一个，就直接退出
        if((count > 20 && mbAbortBA) || count > 100)
        {
            break;
        }
    }// 所有共视关键帧遍历结束
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
    cout << "LM: Map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Map reset, Done!!!" << endl;
}

void LocalMapping::RequestResetActiveMap(Map* pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
    cout << "LM: Active map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequestedActiveMap)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Active map reset, Done!!!" << endl;
}

void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            executed_reset = true;

            cout << "LM: Reseting Atlas in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mbResetRequested = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mIdxInit=0;

            cout << "LM: End reseting Local Mapping..." << endl;
        }

        if(mbResetRequestedActiveMap) {
            executed_reset = true;
            cout << "LM: Reseting current map in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mbResetRequested = false;
            mbResetRequestedActiveMap = false;
            cout << "LM: End reseting Local Mapping..." << endl;
        }
    }
    if(executed_reset)
        cout << "LM: Reset free the mutex" << endl;

}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

/**
 * @brief IMU 的初始化：获得重力方向和IMU零偏的初始值。有了正确的重力方向才能消除IMU预积分中加速度计关于重力的影响，得到的IMU预积分数据才能保证准确
 * @param priorG 陀螺仪偏置的信息矩阵系数，主动设置时一般 bInit 为 true，也就是只优化最后一帧的偏置，这个数会作为计算信息矩阵时使用
 * @param priorA 加速度计偏置的信息矩阵系数
 * @param bFIBA 是否进行 视觉+IMU全局BA优化 (目前都为true)
 */
void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    // 将所有关键帧放入列表及向量里，且查看是否满足初始化条件

    // Step 1: 下面是各种不满足IMU初始化的条件，直接返回
    // 如有置位请求，直接返回
    if (mbResetRequested)
        return;

    float minTime;  // 最后一个关键帧 与 第一个关键帧的时间戳间隔需 >= 该最小时间
    int nMinKF;     // 地图中至少存在的关键帧数目
    // 从时间及帧数上限制初始化，不满足下面条件的不进行初始化
    // 单目
    if (mbMonocular) {
        minTime = 2.0;
        nMinKF = 10;
    }
    // 双目、RGBD
    else {
        minTime = 1.0;
        nMinKF = 10;
    }

    // 当前地图关键帧个数需 >= 10，否则不进行初始化
    if(mpAtlas->KeyFramesInMap() < nMinKF)
        return;

    // 按照时间顺序存储 地图中的所有关键帧（包括当前关键帧）
    list<KeyFrame*> lpKF;   // 老的在前
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);

    // 同样内容再构建一个和lpKF一样的容器vpKF
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());
    if(vpKF.size() < nMinKF)
        return;

    // 头尾关键帧时间戳之差需 >= minTime (单目2s, 双目、RGBD 1s)
    mFirstTs = vpKF.front()->mTimeStamp;
    if(mpCurrentKeyFrame->mTimeStamp - mFirstTs < minTime)
        return;

    // Step 2: 为true，表示正在进行IMU初始化，该标志用于Tracking线程中判断是否添加关键帧
    bInitializing = true;

    // 将缓存队列中还未处理的 新关键帧也放进来，防止堆积且保证数据量充足
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();   // 处理该容器的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // Step 3: 正式开始IMU初始化
    const int N = vpKF.size();  // 目前所有关键帧的个数
    // 零偏初始值为0
    IMU::Bias b(0,0,0,0,0,0);

    // Compute and KF velocities mRwg estimation
    // IMU未进行第一阶段初始化，即IMU第一阶段初始化时，主要为了计算 重力坐标系到世界坐标系的 旋转矩阵 的初值mRwg
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized())
    {
        Eigen::Matrix3f Rwg;    // 世界坐标系(第一帧相机) 到 重力方向 的旋转矩阵
        Eigen::Vector3f dirG;   // 重力方向的估计值 （在经过Rwg变换前的坐标系下）
        dirG.setZero();

        int have_imu_num = 0;
        // 从老到新遍历每个关键帧
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            // 去掉不满足的条件的关键帧
            // 上一KF 到 当前KF的预积分不存在，则跳过
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            // 当前KF的上一KF不存在，则跳过
            if (!(*itKF)->mPrevKF)
                continue;

            have_imu_num++; // 因为只有一帧无mPrevKF，因此其最终值为N-1，至少为9

            // 初始化时关于速度的预积分定义 Ri.t() * (s*Vj - s*Vi - Rwg*g*tij) Note: 核心中的核心
            // dirG: 从参考关键帧到vpKF中最后插入的关键帧之间的速度增量（从IMU坐标系转换到世界坐标系下），其中ΔV_ij = Riw (V_j − V_i − g ∗ Δt)。即速度变化的估计值，这个向量表示在世界坐标系下的速度变化方向
            // dirG 相当于 把地图中所有关键帧的ΔV_ij转换到世界坐标系下，然后取反累加，即dirG = V(参考关键帧) - V(地图中最新的关键帧) + g∗Δt，得到初步估计的重力方向
            // 假设速度变化很小，故dirG = g∗Δt，这只是粗略的估计了一个初值，因为后面还会优化，这个值越精确越有助于后面的优化
            dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            // 求取实际的速度，位移 / 时间
            // _vel：当前关键帧的参考关键帧 到 当前关键帧的平均速度
            Eigen::Vector3f _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition()) / (*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);             // 设置优化前速度初值
            (*itKF)->mPrevKF->SetVelocity(_vel);    // 设置优化前速度初值
        }
        // ----detailed代码中加入的
        if (have_imu_num < 6)
        {
            cout << "IMU初始化失败, 由于带有IMU预积分信息的关键帧数量太少" << endl;
            bInitializing = false;
            mbBadImu = true;
            return;
        }
        // -----------

        // dirG = sV1 - sVn + n * Rwg * g * t
        // ≈ 重力 在世界坐标系下的方向 (对dirG归一化，即单位向量)
        dirG = dirG / dirG.norm();
        // 重力 在重力坐标系下的方向
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);

        // “重力在重力坐标系下的方向” 与 “重力在世界坐标系(纯视觉)下的方向” 的叉乘，即两个坐标系之间的 旋转轴，用于计算重力坐标系到世界坐标系的旋转矩阵的初值
        Eigen::Vector3f v = gI.cross(dirG);
        const float nv = v.norm();  // 旋转轴的长度

        // 两个坐标系之间的 夹角
        const float cosg = gI.dot(dirG);    // 夹角余弦值
        const float ang = acos(cosg);   // 夹角

        // 计算 重力坐标系到世界坐标系的 旋转向量 vzg
        Eigen::Vector3f vzg = v * ang / nv;     // v/nv 旋转轴的单位向量, ang表示该轴转的角度
        // 获得 重力坐标系 到 世界坐标系的 旋转矩阵 的初值
        Rwg = Sophus::SO3f::exp(vzg).matrix();  // exp: 指数映射，旋转向量 -> 旋转矩阵
        mRwg = Rwg.cast<double>();

        // 更新 mTinit = 当前关键帧时间戳（mlNewKeyFrames中最新的关键帧） - 初始化参考关键帧时间戳（mFirstTs）；
        mTinit = mpCurrentKeyFrame->mTimeStamp - mFirstTs;
    }
    // IMU第二、三阶段初始化时，mRwg初值设为单位矩阵，mbg、mba设为当前关键帧的零偏
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = mpCurrentKeyFrame->GetGyroBias().cast<double>();
        mba = mpCurrentKeyFrame->GetAccBias().cast<double>();
    }

    // 将尺度mScale设为1，然后进行优化
    mScale = 1.0;

    // 暂时没发现在别的地方出现过
    mInitTime = mpTracker->mLastFrame.mTimeStamp - vpKF.front()->mTimeStamp;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    //! 纯IMU优化：优化 重力方向、尺度、地图中所有关键帧的速度和零偏 (零偏先验为0，双目模式不优化尺度)
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba, mbMonocular, infoInertial, false, false, priorG, priorA);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 优化后的尺度 < 0.1，则认为初始化失败
    if (mScale < 1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing = false;
        return;
    }

    // 到此时为止，前面做的东西没有改变map
    // 后续改变地图，所以加锁

    {
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        // 非单目 或 尺度变化 > 设定值，则进行如下操作（无论带不带IMU，但这个函数只在带IMU时才执行，所以这个可以理解为双目IMU）
        if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular) {
            // 4.1 恢复地图的尺度及重力方向，即更新每个关键帧在重力坐标系下的 位姿 和 速度，更新每个地图点在重力坐标系下的 坐标
            Sophus::SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());  // 世界坐标系到重力坐标系的变换矩阵
            mpAtlas->GetCurrentMap()->ApplyScaledRotation(Twg, mScale, true);
            // 4.2 将IMU与图像数据进行融合，并更新跟踪线程中普通帧的位姿，主要是当前帧和上一帧的 IMU 位姿和速度
            mpTracker->UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
        }
        // 单目 且 尺度变化 <= 设定值，不更新关键帧在重力坐标系下的位姿，不更新普通帧的位姿

        // Check if initialization OK
        // 若之前IMU还未初始化（此时刚刚初始化成功），则遍历每个关键帧，将其bImu设为true
        // 后面的KF全部都在Tracking里面标记为true。也就是初始化之前的那些关键帧即使有IMU信息也不算
        if (!mpAtlas->isImuInitialized()) {
            // 遍历每个关键帧
            for (int i = 0; i < N; i++) {
                KeyFrame *pKF2 = vpKF[i];
                pKF2->bImu = true;
            }
        }
    }

    // Step 4: 初始化成功
    // TODO 这步更新是否有必要做待研究，0.4版本是放在FullInertialBA下面做的
    // 这个版本FullInertialBA不直接更新位姿及三维点了
    // 更新普通帧的位姿，主要是当前帧和上一帧的 IMU 位姿和速度
    mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);

    if (!mpAtlas->isImuInitialized())
    {
        //! 标记初始化成功，意为IMU已经完成了第一阶段初始化
        mpAtlas->SetImuInitialized();
        std::cout << "\t\t[Local Mapping] IMU初始化成功" << std::endl;
        mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
        mpCurrentKeyFrame->bImu = true;
    }

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    //! 视觉+IMU全局优化 (三个阶段的初始化都做)
    // 优化地图中所有关键帧的位姿、地图点、完成第一阶段IMU初始化之前所有关键帧的速度
    if (bFIBA)
    {
        // 5. 按照之前的结果更新了尺度信息及适应重力方向，在上一步纯IMU优化的基础上，结合地图进行一次视觉+IMU全局优化
        // 1.0版本里面不直接赋值了，而是将所有优化后的信息保存到变量里面
        // IMU第一、二阶段初始化进入
        if (priorA != 0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, true, priorG, priorA);
        // IMU第三阶段初始化进入
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    Verbose::PrintMess("Global Bundle Adjustment finished\nUpdating map ...", Verbose::VERBOSITY_NORMAL);

    // Get Map Mutex
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

    unsigned long GBAid = mpCurrentKeyFrame->mnId;

    // Process keyframes in the queue
    // 6. 处理一下新来的关键帧，这些关键帧没有参与优化，但是这部分bInitializing为true，只在第2次跟第3次初始化会有新的关键帧进来
    // 这部分关键帧也需要被更新
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // 引擎中没有7、8步，在对比时这里需注释
    // Correct keyframes starting at map first keyframe
    // 7. 更新地图中关键帧的位姿 和 地图点的坐标，删除并清空局部建图线程中缓存的关键帧
    // 获取地图中初始关键帧，第一帧肯定经过修正的
    list<KeyFrame*> lpKFtoCheck(mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.begin(),mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.end());

    // 初始就一个关键帧，顺藤摸瓜找到父子相连的所有关键帧
    // 类似于树的广度优先搜索，其实也就是根据父子关系遍历所有的关键帧，有的参与了FullInertialBA有的没参与
    while(!lpKFtoCheck.empty())
    {
        // 7.1 获得这个关键帧的子关键帧
        KeyFrame* pKF = lpKFtoCheck.front();
        const set<KeyFrame*> sChilds = pKF->GetChilds();
        Sophus::SE3f Twc = pKF->GetPoseInverse();   // 获得关键帧的优化前的位姿
        // 7.2 遍历这个关键帧所有的子关键帧
        for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
        {
            // 确认是否能用
            KeyFrame* pChild = *sit;
            if(!pChild || pChild->isBad())
                continue;

            // 这个判定为true表示pChild没有参与前面的优化，因此要根据已经优化过的更新，结果全部暂存至变量
            if(pChild->mnBAGlobalForKF!=GBAid)
            {
                // pChild->GetPose()也是优化前的位姿，Twc也是优化前的位姿
                // 7.3 因此他们的相对位姿是比较准的，可以用于更新pChild的位姿
                Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                // 使用相对位姿，根据pKF优化后的位姿更新pChild位姿，最后结果都暂时放于mTcwGBA
                pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;

                // 7.4 使用相同手段更新速度
                Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                if(pChild->isVelocitySet()){
                    pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                }
                else {
                    Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);
                }

                pChild->mBiasGBA = pChild->GetImuBias();
                pChild->mnBAGlobalForKF = GBAid;

            }
            // 加入到list中，再去寻找pChild的子关键帧
            lpKFtoCheck.push_back(pChild);
        }

        // 7.5 此时pKF的利用价值就没了，但是里面的数值还都是优化前的，优化后的全部放于类似mTcwGBA这样的变量中
        // 所以要更新到正式的状态里，另外mTcwBefGBA要记录更新前的位姿，用于同样的手段更新三维点用
        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);

        // 速度偏置同样更新
        if(pKF->bImu)
        {
            pKF->mVwbBefGBA = pKF->GetVelocity();
            pKF->SetVelocity(pKF->mVwbGBA);
            pKF->SetNewBias(pKF->mBiasGBA);
        } else {
            cout << "KF " << pKF->mnId << " not set to inertial!! \n";
        }

        lpKFtoCheck.pop_front();
    }

    // Correct MapPoints
    //  8. 更新三维点，三维点在优化后同样没有正式的更新，而是找了个中间变量保存了优化后的数值
    const vector<MapPoint*> vpMPs = mpAtlas->GetCurrentMap()->GetAllMapPoints();

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        // 8.1 如果这个点参与了全局优化，那么直接使用优化后的值来赋值
        if(pMP->mnBAGlobalForKF==GBAid)
        {
            // If optimized by Global BA, just update
            pMP->SetWorldPos(pMP->mPosGBA);
        }
        // 如果没有参与，与关键帧的更新方式类似
        else
        {
            // Update according to the correction of its reference keyframe
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

            if(pRefKF->mnBAGlobalForKF!=GBAid)
                continue;

            // Map to non-corrected camera
            // 8.2 根据优化前的世界坐标系下三维点的坐标以及优化前的关键帧位姿计算这个点在关键帧下的坐标
            Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

            // Backproject using corrected camera
            // 8.3 根据优化后的位姿转到世界坐标系下作为这个点优化后的三维坐标
            pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
        }
    }

    Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);

    mnKFs=vpKF.size();
    mIdxInit++;

    // 9. 再有新的来就不要了~不然陷入无限套娃了
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    mpTracker->mState=Tracking::OK;
    bInitializing = false;

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}

/**
 * @brief 通过BA优化进行尺度更新，关键帧小于100，使用了所有关键帧的信息，但只优化尺度和重力方向。每10s在这里的时间段内时多次进行尺度更新
 */
void LocalMapping::ScaleRefinement()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    // unique_lock<mutex> lock0(mMutexImuInit);
    if (mbResetRequested)
        return;

    // Retrieve all keyframes in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();

    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }
    
    Sophus::SO3d so3wg(mRwg);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if ((fabs(mScale-1.f)>0.002)||!mbMonocular)
    {
        Sophus::SE3f Tgw(mRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Tgw,mScale,true);
        mpTracker->UpdateFrameIMU(mScale,mpCurrentKeyFrame->GetImuBias(),mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}



bool LocalMapping::IsInitializing()
{
    return bInitializing;
}


double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

KeyFrame* LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

} //namespace ORB_SLAM
