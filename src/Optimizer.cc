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


#include "Optimizer.h"


#include <complex>

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "G2oTypes.h"
#include "Converter.h"

#include<mutex>

#include "OptimizableTypes.h"


namespace ORB_SLAM3
{
bool sortByVal(const pair<MapPoint*, int> &a, const pair<MapPoint*, int> &b)
{
    return (a.second < b.second);
}

/**
 * @brief 全局BA： pMap中所有的MapPoints和关键帧做bundle adjustment优化
 * 这个全局BA优化在本程序中有两个地方使用：
 * 1、单目初始化：CreateInitialMapMonocular函数
 * 2、闭环优化：RunGlobalBundleAdjustment函数
 * @param[in] pMap                  地图点
 * @param[in] nIterations           迭代次数
 * @param[in] pbStopFlag            外部控制BA结束标志
 * @param[in] nLoopKF               形成了闭环的当前关键帧的id
 * @param[in] bRobust               是否使用鲁棒核函数
 */
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();  // 获取地图中的所有关键帧
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();   // 获取地图中的所有地图点
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);  // 调用GBA
}

/**
 * @brief bundle adjustment 优化过程
 * @param[in] vpKFs                 参与BA的所有关键帧
 * @param[in] vpMP                  参与BA的所有地图点
 * @param[in] nIterations           优化迭代次数
 * @param[in] pbStopFlag            外部控制BA结束标志
 * @param[in] nLoopKF               形成了闭环的当前关键帧的id
 * @param[in] bRobust               是否使用核函数
 */
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    // 不参与优化的地图点，下面会用到
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    Map* pMap = vpKFs[0]->GetMap();

    // Step 1：初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    // 使用LM算法优化
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // 如果这个时候外部请求终止，那就结束
    // 注意这句执行之后，外部再请求结束BA，就结束不了了
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    // 记录添加到优化器中的顶点的最大关键帧id
    long unsigned int maxKFid = 0;

    const int nExpectedSize = (vpKFs.size())*vpMP.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);


    // Set KeyFrame vertices

    // Step 2：向优化器添加顶点
    // Set KeyFrame vertices
    // Step 2.1 ：向优化器添加关键帧位姿顶点
    // 对于当前地图中的所有关键帧
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        // 对于每一个能用的关键帧构造SE3顶点,其实就是当前关键帧的位姿
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKF->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKF->mnId);
        // 只有第0帧关键帧不优化（参考基准）
        vSE3->setFixed(pKF->mnId==pMap->GetInitKFid());
        // 向优化器中添加顶点，并且更新maxKFid
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    // 卡方分布 95% 以上可信度的时候的阈值
    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    // Step 2.2：向优化器添加MapPoints顶点
    // 遍历地图中的所有地图点
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        // 创建顶点
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        // 前面记录maxKFid 是在这里使用的
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        // g2o在做BA的优化时必须将其所有地图点全部schur掉，否则会出错。
        // 原因是使用了g2o::LinearSolver<BalBlockSolver::PoseMatrixType>这个类型来指定linearsolver,
        // 其中模板参数当中的位姿矩阵类型在程序中为相机姿态参数的维度，于是BA当中schur消元后解得线性方程组必须是只含有相机姿态变量。
        // Ceres库则没有这样的限制
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        // 边的关系，其实就是点和关键帧之间观测的关系
        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        //  Step 3：向优化器添加投影边（是在遍历地图点、添加地图点的顶点的时候顺便添加的）
        //  遍历观察到当前地图点的所有关键帧
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            // 滤出不合法的关键帧
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;
            if(optimizer.vertex(id) == NULL || optimizer.vertex(pKF->mnId) == NULL)
                continue;
            nEdges++;

            const int leftIndex = get<0>(mit->second);

            if(leftIndex != -1 && pKF->mvuRight[get<0>(mit->second)] < 0)
            {
                // 如果是单目相机按照下面操作
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->pCamera = pKF->mpCamera;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMP);
            }
            else if(leftIndex != -1 && pKF->mvuRight[leftIndex] >= 0) //Stereo observation
            {
                // 双目或RGBD相机按照下面操作
                // 双目相机的观测数据则是由三个部分组成：投影点的x坐标，投影点的y坐标，以及投影点在右目中的x坐标（默认y方向上已经对齐了）
                const cv::KeyPoint &kpUn = pKF->mvKeysUn[leftIndex];

                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMP);
            }

            if(pKF->mpCamera2){
                int rightIndex = get<1>(mit->second);

                if(rightIndex != -1 && rightIndex < pKF->mvKeysRight.size()){
                    rightIndex -= pKF->NLeft;

                    Eigen::Matrix<double,2,1> obs;
                    cv::KeyPoint kp = pKF->mvKeysRight[rightIndex];
                    obs << kp.pt.x, kp.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);

                    Sophus::SE3f Trl = pKF-> GetRelativePoseTrl();
                    e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                    e->pCamera = pKF->mpCamera2;

                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKF);
                    vpMapPointEdgeBody.push_back(pMP);
                }
            }
        }

        // 如果因为一些特殊原因,实际上并没有任何关键帧观测到当前的这个地图点,那么就删除掉这个顶点,并且这个地图点也就不参与优化
        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    // Step 4：开始优化
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);

    // Recover optimized data
    // Step 5：得到优化的结果

    // Step 5.1 Keyframes
    // 遍历所有的关键帧
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        // 获取到优化结果
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));

        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==pMap->GetOriginKF()->mnId)
        {
            // 原则上来讲不会出现"当前闭环关键帧是第0帧"的情况,如果这种情况出现,只能够说明是在创建初始地图点的时候调用的这个全局BA函数.
            // 这个时候,地图中就只有两个关键帧,其中优化后的位姿数据可以直接写入到帧的成员变量中
            // 老白：视觉初始化时
            pKF->SetPose(Sophus::SE3f(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>()));
        }
        else
        {
            // 正常的操作,先把优化后的位姿写入到帧的一个专门的成员变量mTcwGBA中备用
            pKF->mTcwGBA = Sophus::SE3d(SE3quat.rotation(),SE3quat.translation()).cast<float>();
            pKF->mnBAGlobalForKF = nLoopKF; // 标记这个关键帧参与了这次全局优化

            // 下面都是一些调试操作，计算优化前后的位移
            Sophus::SE3f mTwc = pKF->GetPoseInverse();
            Sophus::SE3f mTcGBA_c = pKF->mTcwGBA * mTwc;
            Eigen::Vector3f vector_dist =  mTcGBA_c.translation();
            double dist = vector_dist.norm();
            if(dist > 1)
            {
                int numMonoBadPoints = 0, numMonoOptPoints = 0;
                int numStereoBadPoints = 0, numStereoOptPoints = 0;
                vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;

                for(size_t i2=0, iend=vpEdgesMono.size(); i2<iend;i2++)
                {
                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i2];
                    MapPoint* pMP = vpMapPointEdgeMono[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                    {
                        continue;
                    }

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>5.991 || !e->isDepthPositive())
                    {
                        numMonoBadPoints++;

                    }
                    else
                    {
                        numMonoOptPoints++;
                        vpMonoMPsOpt.push_back(pMP);
                    }

                }

                for(size_t i2=0, iend=vpEdgesStereo.size(); i2<iend;i2++)
                {
                    g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i2];
                    MapPoint* pMP = vpMapPointEdgeStereo[i2];
                    KeyFrame* pKFedge = vpEdgeKFMono[i2];

                    if(pKF != pKFedge)
                    {
                        continue;
                    }

                    if(pMP->isBad())
                        continue;

                    if(e->chi2()>7.815 || !e->isDepthPositive())
                    {
                        numStereoBadPoints++;
                    }
                    else
                    {
                        numStereoOptPoints++;
                        vpStereoMPsOpt.push_back(pMP);
                    }
                }
            }
        }
    }

    //Points
    // Step 5.2 遍历所有地图点,去除其中没有参与优化过程的地图点
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        // 获取优化之后的地图点的位置
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==pMap->GetOriginKF()->mnId)
        {
            // 如果这个GBA是在创建初始地图的时候调用的话,那么地图点的位姿也可以直接写入
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            // 反之,如果是正常的闭环过程调用,就先临时保存一下
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
}

/**
 * @brief IMU初始化中 视觉+IMU全局优化。LocalMapping::InitializeIMU中使用 LoopClosing::RunGlobalBundleAdjustment
 * 地图全部做BA。也就是IMU版的GlobalBundleAdjustemnt
 * 根据上一步得到的Rwg和scale变换到整张地图，再进行IMU+视觉的整体优化
 * 优化的节点：关键帧位姿、速度、加速度和陀螺仪偏置、地图点
 * 优化的边：重投影、旋转变换、位置变化、速度变换、相邻关键帧偏置
 * @param pMap      当前活跃地图
 * @param its       迭代次数，100次
 * @param bFixLocal 是否固定局部，false
 * @param nLoopId   回环ID，当前关键帧ID
 * @param pbStopFlag 是否停止的标志，NULL
 * @param bInit     提供priorG、priorA时为 true，此时零偏只优化最后一关键帧的值，然后所有关键帧的零偏都赋值为优化后的值
 *                  若为false，则建立每两帧之间的零偏边，优化使其相差为0。顶点加入IMU第一阶段初始化之前关键帧的陀螺仪和加速度计零偏
 * @param priorG    陀螺仪偏置的信息矩阵系数。传入有值时 bInit 也为true，顶点加入地图中最后一个符合要求的关键帧的陀螺仪和加速度计偏置，这个数会作为计算信息矩阵时使用
 * @param priorA    加速度计偏置的信息矩阵系数。
 * @param vSingVal  没用，估计调试用的
 * @param bHess     没用，估计调试用的
 */
void Optimizer::FullInertialBA(Map *pMap, int its, const bool bFixLocal, const long unsigned int nLoopId, bool *pbStopFlag, bool bInit, float priorG, float priorA, Eigen::VectorXd *vSingVal, bool *bHess)
{
    // 获取地图里所有 地图点 与 关键帧，以及最大关键帧ID
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    // Setup optimizer
    // Step 1: 构造优化器
    g2o::SparseOptimizer optimizer; // 创建稀疏优化器
    g2o::BlockSolverX::LinearSolverType * linearSolver; // 声明线性求解器的类型
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>(); // 创建线性求解器
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);   // 创建块求解器，并用上面的线性求解器初始化
    // 创建总求解器 solver，并使用 LM (Levenberg-Marquardt)算法；再用上面的块求解器初始化
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e-5);    // 将求解器的初始阻尼因子设为0.00001
    optimizer.setAlgorithm(solver);     // 将上述求解器作为稀疏优化器的求解方法
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    int nNonFixed = 0;

    // 2. 为每个关键帧添加 位姿顶点；
    // 若是IMU第一阶段初始化后的关键帧，则添加其速度顶点；若处于第三阶段初始化，则为其添加陀螺仪、加速度计零偏顶点
    KeyFrame* pIncKF;   // vpKFs中最后一个ID符合要求的关键帧
    // 遍历每个关键帧
    for(size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId > maxKFid)
            continue;
        // 创建该关键帧 位姿顶点 (不固定)
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        pIncKF = pKFi;
        bool bFixed = false;
        if (bFixLocal)  // false，不进入
        {
            bFixed = (pKFi->mnBALocalForKF >= (maxKFid-1)) || (pKFi->mnBAFixedForKF >= (maxKFid-1));
            if(!bFixed)
                nNonFixed++;
            VP->setFixed(bFixed);
        }
        optimizer.addVertex(VP);

        // 若该关键帧 是 IMU第一阶段初始化后的关键帧 (bImu为true)，则为该关键帧创建 速度顶点 (不固定)
        if (pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3*(pKFi->mnId) + 1);
            VV->setFixed(bFixed);
            optimizer.addVertex(VV);

            // (bInit: IMU第一、二阶段初始化为true，第三阶段为false，也就是又加入了零偏节点)
            // 第三阶段初始化时进入，为该关键帧创建 陀螺仪、加速度计零偏顶点 (不固定)
            if (!bInit)
            {
                VertexGyroBias* VG = new VertexGyroBias(pKFi);
                VG->setId(maxKFid+3*(pKFi->mnId)+2);
                VG->setFixed(bFixed);
                optimizer.addVertex(VG);

                VertexAccBias* VA = new VertexAccBias(pKFi);
                VA->setId(maxKFid+3*(pKFi->mnId)+3);
                VA->setFixed(bFixed);
                optimizer.addVertex(VA);
            }
        }
    }// 每个关键帧遍历完毕

    // 第一、二阶段初始化时进入，为最后一个关键帧 创建 陀螺仪、加速度计零偏顶点
    if (bInit)
    {
        VertexGyroBias* VG = new VertexGyroBias(pIncKF);
        VG->setId(4*maxKFid+2);
        VG->setFixed(false);
        optimizer.addVertex(VG);

        VertexAccBias* VA = new VertexAccBias(pIncKF);
        VA->setId(4*maxKFid+3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }

    if (bFixLocal)  // false，不进入
    {
        if(nNonFixed < 3)
            return;
    }

    // 3. 添加关于IMU的边
    // 遍历所有关键帧
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        // 该关键帧没有上一关键帧，则跳过
        if (!pKFi->mPrevKF) {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        // 该关键帧存在上一关键帧
        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid)
        {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;

            // 该关键帧 及其 上一关键帧 都必须是 IMU第一阶段初始化后的关键帧
            if(pKFi->bImu && pKFi->mPrevKF->bImu)
            {
                // 3.1 将该关键帧的IMU预积分的零偏 设为 其上一关键帧的零偏
                pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
                // 3.2 设置顶点
                g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);   // 该关键帧的上一关键帧的 位姿顶点
                g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid + 3*(pKFi->mPrevKF->mnId) + 1); // 该关键帧的上一关键帧的 速度顶点

                g2o::HyperGraph::Vertex* VG1;
                g2o::HyperGraph::Vertex* VA1;
                g2o::HyperGraph::Vertex* VG2;
                g2o::HyperGraph::Vertex* VA2;

                // 第三阶段初始化时
                if (!bInit)
                {
                    VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);  // 该关键帧的上一关键帧的 陀螺仪零偏顶点
                    VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);  // 该关键帧的上一关键帧的 加速度计零偏顶点
                    VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);   // 该关键帧的 陀螺仪零偏顶点
                    VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);   // 该关键帧的 加速度计零偏顶点
                }
                // 第一、二阶段初始化时
                else
                {
                    VG1 = optimizer.vertex(4*maxKFid+2);    // 最后一个关键帧的 陀螺仪零偏顶点
                    VA1 = optimizer.vertex(4*maxKFid+3);    // 最后一个关键帧的 加速度计零偏顶点
                }

                g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);   // 该关键帧的 位姿顶点
                g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);  // 该关键帧的 速度顶点

                // 检查顶点是否存在，不存在则跳过
                if (!bInit) {
                    if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2) {
                        cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                        continue;
                    }
                } else {
                    if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2) {
                        cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<endl;
                        continue;
                    }
                }

                // 3.3 创建六元边，并设置相关的顶点
                EdgeInertial* ei = new EdgeInertial(pKFi->mpImuPreintegrated);
                ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
                ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
                ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
                ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
                ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
                ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

                // 设置鲁棒核函数
                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                ei->setRobustKernel(rki);
                // 9个自由度的卡方检验（0.05）
                rki->setDelta(sqrt(16.92));

                // 将该边添加到优化器中
                optimizer.addEdge(ei);

                // 第三阶段初始化时，添加一个IMU陀螺仪零偏、一个加速度计零链接的 二元边。(优化两关键帧之间零偏的误差)
                if (!bInit)
                {
                    EdgeGyroRW* egr= new EdgeGyroRW();
                    egr->setVertex(0,VG1);  // 该关键帧前一关键帧的 陀螺仪零偏顶点
                    egr->setVertex(1,VG2);  // 该关键帧的 陀螺仪零偏顶点
                    // 从预积分的协方差矩阵中提取的陀螺仪零偏的信息, 并作为该边的信息矩阵
                    Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
                    egr->setInformation(InfoG);
                    egr->computeError();
                    optimizer.addEdge(egr);

                    EdgeAccRW* ear = new EdgeAccRW();
                    ear->setVertex(0,VA1);  // 该关键帧前一关键帧的 加速度计零偏顶点
                    ear->setVertex(1,VA2);  // 该关键帧的 加速度计零偏顶点
                    // 从预积分的协方差矩阵中提取的加速度计零偏的信息, 并作为该边的信息矩阵
                    Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
                    ear->setInformation(InfoA);
                    ear->computeError();
                    optimizer.addEdge(ear);
                }
            }
            else
                cout << pKFi->mnId << " or " << pKFi->mPrevKF->mnId << " no imu" << endl;
        }
    }

    // 只加入pIncKF帧的偏置，优化偏置到0
    // 第一、二阶段初始化时，添加最后一个参考关键帧的 加速度计、陀螺仪零偏的 边
    if (bInit)
    {
        g2o::HyperGraph::Vertex* VG = optimizer.vertex(4*maxKFid+2);    // 最后一个关键帧的 陀螺仪零偏顶点
        g2o::HyperGraph::Vertex* VA = optimizer.vertex(4*maxKFid+3);    // 最后一个关键帧的 加速度计零偏顶点

        // Add prior to comon biases
        Eigen::Vector3f bprior;
        bprior.setZero();

        EdgePriorAcc* epa = new EdgePriorAcc(bprior);
        epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
        double infoPriorA = priorA; //
        epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);

        EdgePriorGyro* epg = new EdgePriorGyro(bprior);
        epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
        double infoPriorG = priorG; //
        epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);
    }

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    const unsigned long iniMPid = maxKFid*5;

    vector<bool> vbNotIncludedMP(vpMPs.size(),false);

    // 5. 添加关于地图点的顶点与边，这段比较好理解，很传统的视觉上的重投影误差
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();


        bool bAllFixed = true;

        //Set edges
        //  遍历所有能观测到这个点的关键帧
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnId>maxKFid)
                continue;

            if(!pKFi->isBad())
            {
                const int leftIndex = get<0>(mit->second);
                cv::KeyPoint kpUn;

                // 添加边
                if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)]<0) // Monocular observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono(0);

                    g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    if(bAllFixed)
                        if(!VP->fixed())
                            bAllFixed=false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                }
                else if(leftIndex != -1 && pKFi->mvuRight[leftIndex] >= 0) // stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];
                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo(0);

                    g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                    if(bAllFixed)
                        if(!VP->fixed())
                            bAllFixed=false;

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];

                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                }

                if(pKFi->mpCamera2){ // Monocular right observation
                    int rightIndex = get<1>(mit->second);

                    if(rightIndex != -1 && rightIndex < pKFi->mvKeysRight.size()){
                        rightIndex -= pKFi->NLeft;

                        Eigen::Matrix<double,2,1> obs;
                        kpUn = pKFi->mvKeysRight[rightIndex];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        EdgeMono *e = new EdgeMono(1);

                        g2o::OptimizableGraph::Vertex* VP = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId));
                        if(bAllFixed)
                            if(!VP->fixed())
                                bAllFixed=false;

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, VP);
                        e->setMeasurement(obs);
                        const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                    }
                }
            }
        }

        // false
        if(bAllFixed)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;


    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // 5. 取出优化结果，对应的值赋值
    // Recover optimized data
    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        if(nLoopId==0)
        {
            Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
            pKFi->SetPose(Tcw);
        }
        else
        {
            pKFi->mTcwGBA = Sophus::SE3f(VP->estimate().Rcw[0].cast<float>(),VP->estimate().tcw[0].cast<float>());
            pKFi->mnBAGlobalForKF = nLoopId;

        }
        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            if(nLoopId==0)
            {
                pKFi->SetVelocity(VV->estimate().cast<float>());
            }
            else
            {
                pKFi->mVwbGBA = VV->estimate().cast<float>();
            }

            VertexGyroBias* VG;
            VertexAccBias* VA;
            if (!bInit)
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            }
            else
            {
                VG = static_cast<VertexGyroBias*>(optimizer.vertex(4*maxKFid+2));
                VA = static_cast<VertexAccBias*>(optimizer.vertex(4*maxKFid+3));
            }

            Vector6d vb;
            vb << VG->estimate(), VA->estimate();
            IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
            if(nLoopId==0)
            {
                pKFi->SetNewBias(b);
            }
            else
            {
                pKFi->mBiasGBA = b;
            }
        }
    }

    //Points
    for(size_t i=0; i<vpMPs.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMPs[i];
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));

        if(nLoopId==0)
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopId;
        }

    }

    pMap->IncreaseChangeIndex();
}

/**
 * @brief 仅位姿优化，不优化地图点。
 * 主要在追踪线程中被调用，如 Tracking::TrackWithMotionModel()、Tracking::TrackReferenceKeyFrame()、Tracking::Relocalization()、Tracking::TrackLocalMap() 中都使用了该函数
 *
 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw * Pw)
 * 只优化普通帧的位姿Tcw，不优化3D地图点的坐标
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的位姿Tcw
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZOnlyPose()，BaseUnaryEdge     单目
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()，BaseUnaryEdge   PinHole双目
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 * @param pFrame 待优化的帧
 * @return  inliers数量
 */
int Optimizer::PoseOptimization(Frame *pFrame)
{
    // Step 1：构造g2o优化器
    // Step 1.1：声明线性求解器的类型。BlockSolver_6_3 表示：位姿_PoseDim为 6 维；观测点，即路标点_LandmarkDim是 3 维
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    // Step 1.2：创建线性求解器 linearSolver
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    // Step 1.3：创建块求解器 solver_ptr，并用上面的线性求解器初始化
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    // Step 1.4：创建总求解器 solver，并从GN、LM、DogLeg中选择一个，这里使用了 LM (Levenberg-Marquardt)算法；再用上面的块求解器初始化，
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    // Step 1.5：创建稀疏优化器 SparseOptimizer，并用已经定义的求解器作为求解方法
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 求解器solver设置给优化器optimizer，这样优化器就可以使用该求解器进行图优化

    // 输入的帧中，有效的、参与优化过程的2D - 3D 点对数，即边的个数
    int nInitialCorrespondences = 0;

    // Set Frame vertex
    // Step 2：添加顶点：将待优化当前帧的位姿Tcw 作为图的顶点添加到图中
    // Step 2.1：创建一个顶点
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    // Step 2.2：获取当前帧的位姿
    Sophus::SE3<float> Tcw = pFrame->GetPose();
    // Step 2.3：设定顶点的估计值：即当前帧位姿变为 单位四元数 + 平移向量 形式，再转换为g2o库中的SE3Quat类型
    vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
    // Step 2.4：设置顶点的编号ID：保证本次优化过程中id独立即可
    vSE3->setId(0);
    vSE3->setFixed(false);  // 该顶点是待优化的变量，不能固定住
    // Step 2.5：将该顶点添加到优化器中
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    // 当前帧的特征点数量
    const int N = pFrame->N;

    // 存放单目边
    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *> vpEdgesMono_FHR;   // 存放另一目的边
    vector<size_t> vnIndexEdgeMono, vnIndexEdgeRight;                       // 边对应特征点的id
    vpEdgesMono.reserve(N);
    vpEdgesMono_FHR.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeRight.reserve(N);

    // 存放双目边
    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    // 下面涉及卡方分布去除外点的知识，这里不做讲解
    const float deltaMono = sqrt(5.991);    // 自由度为 2 的卡方分布，显著性水平为 0.05，对应的临界阈值5.991。卡方值高于5.991 95%的几率为外点
    const float deltaStereo = sqrt(7.815);  // 自由度为 3 的卡方分布，显著性水平为 0.05，对应的临界阈值7.815
//    std::ofstream wfs2("单帧优化位姿.txt", std::ios::app);
//    wfs2 << "idx" << ", " << "obs_x" << ", "<< "obs_y" << ", " << "obs_xr" << ", " << "level" << ", " << "pos_w[0]" << ", " << "pos_w[1]" << ", " << "pos_w[2]" << endl;

//    std::ifstream ifs("/home/liuzhi/Project/Optimization_Results/单帧优化位姿.txt");
//    std::string header;
//    std::getline(ifs, header);
//    if (pFrame->mnId == 1)
//    {
//        while (!ifs.eof())
//        {
//            int i, level;
//            float x, y, xr;
//            double pos_w[3];
//            char comma;
//            if(!(ifs >> i >> comma >> x >> comma >> y >> comma >> xr >> comma >> level >> comma >> pos_w[0] >> comma >> pos_w[1] >> comma >> pos_w[2]))
//            {
//                break;
//            }
//            Eigen::Matrix<double, 3, 1> pose_world;
//            pose_world << pos_w[0], pos_w[1], pos_w[2];
//            if (xr < 0)
//            {
//                // Step 4.1：如果为单目(双目省略讲解)，创建一元边EdgeSE3ProjectXYZOnlyPose实例。设置顶点(当前帧位姿)、设置观测点(当前帧与地图点对应的特征点)。
//                nInitialCorrespondences++;
//                pFrame->mvbOutlier[i] = false;  // 先默认此地图点对应的特征点为内点
//
//                // 对这个地图点的观测，也就是特征点在当前帧的左目 去畸变像素坐标
////                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
//                Eigen::Matrix<double,2,1> obs;
//                obs << x, y;
//
//                // 新建一元边e：这个边只优化位姿 Pose（一个特征点对应一个一元边）
//                ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();
//
//                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0))); // 设置连接的顶点(当前帧位姿)
//                e->setMeasurement(obs);     // 边的观测数值：该特征点在当前帧的左目和右目的像素坐标
//
//                // Step 4.2：设置置性度(信息矩阵→与特征点所处金字塔层级相关)：现在来考虑另一种情况，比方说在一次优化中，对于某一次测量，我们有十足的把握，它非常的准确，所以优化时我们希望对于这次测量给予更高的权重。对应带代码 e->setInformation()
//                const float invSigma2 = pFrame->mvInvLevelSigma2[level];  // 这个点的置信度，其与特征点所在的图层有关。用信息矩阵（协方差矩阵得逆）来表示。层数越高，invSigma2越小，信息矩阵越小，表示误差越大，优化时考虑的比较少
//                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
//                // Step 4.3：设置鲁棒核函数，g2o中提供了鲁棒核函数来抑制某些误差特别大的点，避免拉偏整个优化结果。(鲁棒核函数不是g2o独有的，这是非线性优化方法中的一种常用手段)
//                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                e->setRobustKernel(rk);
//                rk->setDelta(deltaMono);
//
//                // Step 4.4：设置相机内参、地图点的空间位置
//                e->pCamera = pFrame->mpCamera;
//                // 地图点的空间位置 , 作为迭代的初始值，因为是一元边，所以不以节点的形式出现
//                e->Xw = pose_world;
//
//                optimizer.addEdge(e);   // 将此边加入优化器
//
//                vpEdgesMono.push_back(e);   // 记录边属于单目情况
//                vnIndexEdgeMono.push_back(i);   // 记录索引
//            }
//            else
//            {
//                nInitialCorrespondences++;
//                pFrame->mvbOutlier[i] = false;
//
//                // 该特征点在当前帧的左目 去畸变像素横纵坐标 和 在右目的横坐标
//                Eigen::Matrix<double,3,1> obs;
////        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
////        const float &kp_ur = pFrame->mvuRight[i];
//                obs << x, y, xr;
//
//                // 新建边e，注意这里也是只优化位姿 (双目)
//                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
//
//                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0))); // 设置连接的顶点(当前帧位姿)
//                e->setMeasurement(obs); // 边的观测数值：该特征点在当前帧的左目和右目的像素坐标
//
//                // 置信程度主要是看左目特征点所在的图层
//                const float invSigma2 = pFrame->mvInvLevelSigma2[level];
//                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
//                e->setInformation(Info);    // 信息矩阵：协方差矩阵之逆。边的权重设置为该特征点所在图层的置信度
//
//                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
//                e->setRobustKernel(rk);
//                rk->setDelta(deltaStereo);
//
//                // 双目下没有加入相机
//                e->fx = pFrame->fx;
//                e->fy = pFrame->fy;
//                e->cx = pFrame->cx;
//                e->cy = pFrame->cy;
//                e->bf = pFrame->mbf;
//                e->Xw = pose_world;
//
//                optimizer.addEdge(e);   // 该边加入到优化器中
////            wfs2 << "\tcur_id: " << pFrame->mnId << ": " << i << ", (" << kpUn.pt.x << ", "<< kpUn.pt.y << "), " << kpUn.octave << ", ("<< pMP->GetWorldPos()[0] << ", " << pMP->GetWorldPos()[1] << ", " << pMP->GetWorldPos()[2] << ")" << std::endl;
//                wfs2 << i << "," << x << "," << y << "," << xr << ","
//                     << level << "," << pos_w[0] << ","  << pos_w[1] << "," << pos_w[2] << std::endl;
//
//                vpEdgesStereo.push_back(e);
//                vnIndexEdgeStereo.push_back(i);
//            }
//        }
//    }
//    else
    // Step 3：添加一元边。边的误差为观测的特征点坐标和地图点在当前帧的投影之差。
    // 其中边的信息矩阵与该特征点所在的图像金字塔层级有关。
    {
    // 锁定地图点。因为系统是多线程，所以在取数据时要加锁才能保证线程安全。
    // 另一方面，需要使用地图点来构造顶点和边 , 因此不希望在构造的过程中部分地图点被改写造成不一致甚至是段错误
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    // Step 4：遍历当前帧所有特征点，如果特征点匹配到地图点，则添加一元边
    for(int i = 0; i < N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];

        // 如果这个地图点存在，没有被剔除掉
        if(pMP)
        {
            // Conventional SLAM
            // 不存在相机2，则有可能为单目与PinHole双目、RGBD
            if(!pFrame->mpCamera2)
            {
                // 单目边，创建单目一元边EdgeSE3ProjectXYZOnlyPose。设置顶点(当前帧位姿)、设置观测点(地图点在当前帧对应的特征点)。 (因为双目模式下pFrame->mvuRight[i]会>0)
                if(pFrame->mvuRight[i] < 0)
                {
                    nInitialCorrespondences++;  // 边的个数+1
                    pFrame->mvbOutlier[i] = false;  // 先默认此地图点对应的特征点为内点

                    // 获取对当前帧该地图点的观测值：就是其对应特征点的去畸变像素坐标
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    // 创建观测值，并写入像素坐标
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    // Step 3.1：创建单目一元边e：误差为观测的特征点坐标 - 地图点在当前帧像素投影坐标。一个特征点对应一个一元边
                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();
                    // Step 3.2：设置该边 连接的顶点: i: 为顶点的编号0，v: 表示顶点，这里是当前帧位姿对应的顶点
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    // Step 3.3：设置该边 的观测值：通常是观测到的特征点坐标 (其当前帧的左目的像素坐标)
                    e->setMeasurement(obs);

                    // Step 3.4：设置该边的信息矩阵 (反映观测值(该特征点的坐标)的可信度，与特征点所在金字塔层级相关)
                    // 计算置信度: 层数越高，invSigma2越小，置信度越小，表示误差越大，优化时考虑的比较少
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    // 比如在一次优化中，对于某一次测量，我们有十足的把握，它非常的准确，所以优化时我们希望对于这次测量给予更高的权重，则对其的信息矩阵设大一点，对应带代码 e->setInformation()
                    // 设置信息矩阵：= 观测值的协方差矩阵的逆
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);  // 设置该约束的信息矩阵

                    // Step 3.5：设置鲁棒核函数：g2o中提供了鲁棒核函数来抑制某些误差特别大的点，避免拉偏整个优化结果。(鲁棒核函数不是g2o独有的，这是非线性优化方法中的一种常用手段) 注：后续在优化2次后会用e->setRobustKernel(0)禁掉鲁棒核函数
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    // 重投影误差的自由度为2，设置对应的卡方阈值
                    rk->setDelta(deltaMono);

                    // Step 3.6：设置相机内参
                    e->pCamera = pFrame->mpCamera;
                    // Step 3.7：设置地图点的空间位置, 作为迭代的初始值。因为是一元边，所以不以节点的形式出现
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    // Step 3.8: 将该边添加到优化器中
                    optimizer.addEdge(e);   // 将此边加入优化器
//                    wfs2 << "\tcur_id: " << pFrame->mnId << ": " << i << ", (" << kpUn.pt.x << ", "<< kpUn.pt.y << "), " << kpUn.octave << ", ("<< pMP->GetWorldPos()[0] << ", " << pMP->GetWorldPos()[1] << ", " << pMP->GetWorldPos()[2] << ")" << std::endl;

                    vpEdgesMono.push_back(e);       // 存储到 单目边
                    vnIndexEdgeMono.push_back(i);   // 记录单目边对应 特征点索引
                }
                // PinHole双目、RGBD边
                else
                {
                    nInitialCorrespondences++;  // 边的个数+1
                    pFrame->mvbOutlier[i] = false;

                    // 该特征点在当前帧的左目 去畸变像素横纵坐标 和 在右目的横坐标
                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    // Step 3.1：创建双目一元边e (注意这里也只优化位姿)
                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                    // Step 3.2：设置该边 连接的顶点:
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    // Step 3.3：设置该边 的观测值：通常是观测到的特征点坐标 (其在当前帧的左目和右目的像素坐标)
                    e->setMeasurement(obs);

                    // Step 3.4：设置该边的信息矩阵。置信程度主要是看左目特征点所在的图层
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2; // 维度[3, 3]
                    e->setInformation(Info);    // 设置信息矩阵：观测值的协方差矩阵的逆

                    // Step 3.5：设置鲁棒核函数：
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    // 重投影误差的自由度为3，设置对应的卡方阈值
                    rk->setDelta(deltaStereo);

                    // Step 3.6：设置相机内参。双目下没有加入相机
                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    // Step 3.7：设置地图点的空间位置, 作为迭代的初始值。因为是一元边，所以不以节点的形式出现
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    // Step 3.8: 将该边添加到优化器中
                    optimizer.addEdge(e);
//                    wfs2 << "\tcur_id: " << pFrame->mnId << ": " << i << ", (" << kpUn.pt.x << ", "<< kpUn.pt.y << "), " << kpUn.octave << ", ("<< pMP->GetWorldPos()[0] << ", " << pMP->GetWorldPos()[1] << ", " << pMP->GetWorldPos()[2] << ")" << std::endl;

                    vpEdgesStereo.push_back(e);     // 存储到 PinHole双目边
                    vnIndexEdgeStereo.push_back(i); // 记录PinHole双目边对应 特征点索引
                }
            }
            // KB鱼眼双目的 在右目的边
            else
            {
                nInitialCorrespondences++;

                cv::KeyPoint kpUn;

                if (i < pFrame->Nleft) {    // Left camera observation
                    kpUn = pFrame->mvKeys[i];

                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else {
                    kpUn = pFrame->mvKeysRight[i - pFrame->Nleft];

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    pFrame->mvbOutlier[i] = false;

                    ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->pCamera = pFrame->mpCamera2;
                    e->Xw = pMP->GetWorldPos().cast<double>();

                    e->mTrl = g2o::SE3Quat(pFrame->GetRelativePoseTrl().unit_quaternion().cast<double>(), pFrame->GetRelativePoseTrl().translation().cast<double>());

                    optimizer.addEdge(e);

                    vpEdgesMono_FHR.push_back(e);
                    vnIndexEdgeRight.push_back(i);
                }
            }
        }// 该地图点处理完毕
    }// 遍历完当前帧所有地图点，添加边完毕
    }
//    wfs2 << nInitialCorrespondences << std::endl;

    // 如果没有足够的匹配点, 边数不够, 那么就只好放弃了
    if(nInitialCorrespondences < 3)
        return 0;


    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // Step 4：设置卡法阈值 (4次迭代的卡方阈值，用来判断内外点，假设有一个像素的偏差)
    const float chi2Mono[4] = {5.991,5.991,5.991,5.991};      // 单目
    const float chi2Stereo[4] = {7.815,7.815,7.815, 7.815};   // 双目
    const int its[4] = {10,10,10,10};     // 4次迭代，每次迭代的次数

    int nBad=0;     // bad 地图点的 个数
    // Step 5：开始优化，共优化4次，每次优化迭代 10 次。每次优化后，根据边的误差将对应的地图点分为外点和内点，外点不参与下次优化。
    // 每次优化是对所有的观测进行外点和内点的判别，因此之前被判别为外点的点有可能变为内点，反之亦然。
    for(size_t it = 0; it < 4; it++)
    {
        // 这里面计算了重投影误差
        // 设置估计值
        Tcw = pFrame->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));

        // 初始化优化器, 这里的参数默认为0, 也就是优化等级为0，只对 level 为 0 的边进行优化；不对外点进行优化（外点等级为1）
        optimizer.initializeOptimization(0);
        // 开始优化，迭代 10 次
        optimizer.optimize(its[it]);

        nBad = 0;
        // 一次优化结束后，遍历参与优化的每一条误差边，根据投影误差判别内、外点
        // 单目边
        for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];   // 误差边
            const size_t idx = vnIndexEdgeMono[i];  // 索引

            // 下面是卡方检验，由于每次优化后是对所有的观测进行卡方检验 outlier 和 inlier 判别，因此之前被判别为 outlier 有可能变成 inlier ，反之亦然

            // 如果这条误差边是来自于外点，重新计算误差。因为上一次外点这一次可能成为内点
            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();  // 计算重投影误差 = 二维观测值(特征点坐标) - 三维地图点经估计的相机位姿和相机内参K投影到图像坐标系的二维值
            }
            // 这个点的误差大小 (考虑置信度以后)，就是 error*\Omega*error, 表征了
            const float chi2 = e->chi2();
//            std::cout << "error.size: " << e->error() <<  ", information.size: " << e->information().size() << std::endl;

            // 误差较大，检验不通过，则设为外点
            if(chi2 > chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);     // level 1 对应为外点，上面的过程中我们设置其为不优化
                nBad++;
            }
            // 误差较小，则设为内点
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);     // level 0 对应为内点，上面的过程中我们就是要优化这些关系
            }

            if(it == 2)
                e->setRobustKernel(0);  // 除了前两次优化需要 RobustKernel 以外，其余的优化都不需要 -- 因为重投影的误差已经有明显的下降了
        }
        // 对于相机2，KB相机边
        for(size_t i = 0, iend = vpEdgesMono_FHR.size(); i < iend; i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZOnlyPoseToBody* e = vpEdgesMono_FHR[i];

            const size_t idx = vnIndexEdgeRight[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }
        // PinHole双目、RGBD边
        for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size() < 10)
            break;
    }// 4次优化结束

    // Recover optimized pose and return number of inliers
    // step 6: 用优化后的位姿更新当前帧的位姿，并返回内点个数
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    Sophus::SE3<float> pose(SE3quat_recov.rotation().cast<float>(), SE3quat_recov.translation().cast<float>()); // 优化后的位姿
    // 用优化后的位姿更新 当前帧的位姿
    pFrame->SetPose(pose);      // 设置优化后的位姿
    Verbose::PrintMess("\t\t有效的、参与优化的点对个数："+std::to_string(nInitialCorrespondences)+", 判定为外点的个数："+std::to_string(nBad), Verbose::VERBOSITY_DEBUG);
    return nInitialCorrespondences - nBad;  // 返回内点个数
}

/**
 * @brief 局部BA，局部建图线程 LocalMapping::Run() 使用，纯视觉。优化局部关键帧 和 局部地图点
 * 1. Vertex:
 *     - g2o::VertexSE3Expmap()，待优化的局部关键帧，即当前关键帧、当前关键帧的一级共视关键帧的位姿
 *     - g2o::VertexSE3Expmap()，不优化的固定关键帧，即当前关键帧的二级共视关键帧(能观测到局部地图点的关键帧)的位姿，用于增加约束关系，在优化中这些关键帧的位姿不变
 *     - g2o::VertexSBAPointXYZ()，待优化的局部地图点，即局部关键帧观测的所有地图点的位置
 * 2. Edge: 局部地图点与观测到它的关键帧的观测关系，为二元边。
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge    单目模式
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZ()，BaseBinaryEdge  PinHole双目、RGBD模式
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 * @param pKF        当前关键帧
 * @param pbStopFlag 是否停止优化的标志
 * @param pMap       在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 * @param 剩下的都是调试用的
 *
 * 总结下与ORBSLAM2的不同
 * 前面操作基本一样，但优化时2代去掉了误差大的点又进行优化了，3代只是统计但没有去掉继续优化，而后都是将误差大的点干掉
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    // Step 1：将当前关键帧 及其一级共视关键帧 加入局部关键帧 lLocalKeyFrames 中
    // 存储局部关键帧
    list<KeyFrame*> lLocalKeyFrames;

    // 当前关键帧加入局部关键帧
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;    // 标记当前关键帧 是 当前关键帧的局部关键帧
    Map* pCurrentMap = pKF->GetMap();

    // 找到与当前关键帧连接的 一级共视关键帧，加入局部关键帧中
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        // 标记该关键帧 是 当前关键帧的局部关键帧：把参与局部BA的每一个关键帧的 mnBALocalForKF 设置为当前关键帧的ID，防止重复添加
        pKFi->mnBALocalForKF = pKF->mnId;
        // 不是坏帧 且 属于当前地图，则加入局部关键帧
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    // 如果局部关键帧有一个关键帧是 当前地图的第一个关键帧，则将其也算作固定关键帧
    num_fixedKF = 0;

    // Step 2: 遍历局部关键帧，将它们观测的地图点 加入到局部地图点中
    // 存储局部地图点
    list<MapPoint*> lLocalMapPoints;

    set<MapPoint*> sNumObsMP;
    // 遍历局部关键帧
    for(list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin() , lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        // 该局部关键帧是地图中的第一个关键帧
        if(pKFi->mnId == pMap->GetInitKFid())
        {
            num_fixedKF = 1;
        }
        // 获取该局部关键帧匹配到的地图点
        vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        // 遍历该局部关键帧的每一个地图点，加入局部地图点中
        for(vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad() && pMP->GetMap() == pCurrentMap)
                {
                    // pMP->mnBALocalForKF 说明该地图点是当前关键帧的 局部BA中的 局部地图点
                    // 该地图点 还不是 当前关键帧的局部地图点，则将其加入局部地图点中。且把参与优化的地图点的 mnBALocalForKF 设为当前关键帧的ID，防止重复添加
                    if(pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
                }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // Step 3：获取固定关键帧：能观测到局部地图点，但不是局部关键帧，即为当前关键帧的二级共视关键帧，在局部BA优化时不参与优化，仅作为约束条件
    // 存储固定关键帧
    list<KeyFrame*> lFixedCameras;
    // 遍历局部地图点
    for(list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        // 获取该局部地图点的观测。key: 观测到该地图点的关键帧。value: 该地图点在该关键帧中的索引，默认为<-1,-1>；如果是单目或PinHole双目，则为<idx,-1>；如果是KB双目且idx在右目中，则为<-1,idx>
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        // 遍历该局部地图点的每一个观测
        for(map<KeyFrame*,tuple<int,int>>::iterator mit = observations.begin(), mend = observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;    // 取出观测到该局部地图点的关键帧

            // 该关键帧 不是 当前关键帧的局部关键帧 且 还不是 当前关键帧的固定关键帧
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId )
            {
                // 将其标记为当前关键帧的 固定关键帧，防止重复添加
                pKFi->mnBAFixedForKF = pKF->mnId;
                // 该关键帧是好的 且 属于当前地图，则将其加入 固定关键帧中
                if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // 固定关键帧的个数：原个数 + 当前地图中第一个关键帧
    num_fixedKF = lFixedCameras.size() + num_fixedKF;

    // 相比ORBSLAM2多出了判断固定关键帧的个数，最起码要两个固定的,如果实在没有就把lLocalKeyFrames中最早的KF固定，还是不够再加上第二早的KF固定
    // 0.4 版本有，1.0版本删除了这段
    // if (num_fixedKF < 2)
    // {
    //     list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin();
    //     int lowerId = pKF->mnId;
    //     KeyFrame *pLowerKf;
    //     int secondLowerId = pKF->mnId;
    //     KeyFrame *pSecondLowerKF;

    //     for (; lit != lLocalKeyFrames.end(); lit++)
    //     {
    //         KeyFrame *pKFi = *lit;
    //         if (pKFi == pKF || pKFi->mnId == pMap->GetInitKFid())
    //         {
    //             continue;
    //         }

    //         if (pKFi->mnId < lowerId)
    //         {
    //             lowerId = pKFi->mnId;
    //             pLowerKf = pKFi;
    //         }
    //         else if (pKFi->mnId < secondLowerId)
    //         {
    //             secondLowerId = pKFi->mnId;
    //             pSecondLowerKF = pKFi;
    //         }
    //     }
    //     lFixedCameras.push_back(pLowerKf);
    //     lLocalKeyFrames.remove(pLowerKf);
    //     num_fixedKF++;
    //     if (num_fixedKF < 2)
    //     {
    //         lFixedCameras.push_back(pSecondLowerKF);
    //         lLocalKeyFrames.remove(pSecondLowerKF);
    //         num_fixedKF++;
    //     }
    // }

    // 固定关键帧个数 = 0，不进行BA，返回
    if(num_fixedKF == 0)
    {
        Verbose::PrintMess("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted", Verbose::VERBOSITY_NORMAL);
        return;
    }

    // Step 4：构造g2o优化器
    // Step 4.1：创建稀疏优化器
    g2o::SparseOptimizer optimizer;
    // Step 4.2：声明线性求解器的类型。BlockSolver_6_3 表示：位姿_PoseDim为 6 维；观测点，即路标点_LandmarkDim是 3 维
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    // Step 4.3：创建线性求解器
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    // Step 4.4：创建块求解器，并用上面的线性求解器初始化
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    // Step 4.5：创建总求解器 solver，并从GN、LM、DogLeg中选择一个，这里使用了 LM (Levenberg-Marquardt)算法；再用上面的块求解器初始化
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    // Step 4.6：IMU模式，则将求解器的初始阻尼因子设为100。在优化问题中，阻尼因子用于平衡优化的速度和准确性，它会影响优化的收敛性。
    if (pMap->IsInertial())
        solver->setUserLambdaInit(100.0);

    // Step 4.7：将上述求解器作为稀疏优化器的求解方法
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // DEBUG LBA
    pCurrentMap->msOptKFs.clear();  // 参与局部BA优化的关键帧ID
    pCurrentMap->msFixedKFs.clear();    // 固定关键帧的ID

    // Step 5：添加待优化的位姿顶点：局部关键帧的位姿 (若是当前地图的初始关键帧，则固定住，不优化)
    for(list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        // 创建一个位姿顶点
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        // 获取该局部关键帧的位姿
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        // 设置该顶点的估计值：即该局部关键帧位姿变为 单位四元数 + 平移向量 形式，再转换为g2o库中的SE3Quat类型
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        // 设置该顶点的ID为 该局部关键帧的ID
        vSE3->setId(pKFi->mnId);

        // 该局部关键帧 是当前地图的初始关键帧，则固定住，不优化；
        //            不是则需优化
        vSE3->setFixed(pKFi->mnId==pMap->GetInitKFid());
        // 将该顶点添加到优化器中
        optimizer.addVertex(vSE3);

        // 更新最大局部关键帧ID
        if(pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msOptKFs.insert(pKFi->mnId);
    }
    // 参与优化的关键帧个数 = 局部关键帧的个数
    //! 是否应该 - 初始关键帧的1
    num_OptKF = lLocalKeyFrames.size();

    // Step 6：添加不优化的位姿顶点：固定关键帧的位姿。注意这里调用了vSE3->setFixed(true)，不参与优化
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);   // 不参与优化
        optimizer.addVertex(vSE3);

        if(pKFi->mnId > maxKFid)
            maxKFid=pKFi->mnId;
        // DEBUG LBA
        pCurrentMap->msFixedKFs.insert(pKFi->mnId);
    }


    // 存放的方式(举例)
    // 边id: 1 2 3 4 5 6 7 8 9
    // KFid: 1 2 3 4 1 2 3 2 3
    // MPid: 1 1 1 1 2 2 2 3 3
    // 所以边的最大数目 = 位姿数目 * 地图点数目，实际肯定比这个要少
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    // 存放单目时的二元边
    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    // 存放鱼眼双目时的 另一个相机的边
    vector<ORB_SLAM3::EdgeSE3ProjectXYZToBody*> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    // 存放单目时的关键帧
    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    // 存放鱼眼双目时另一个相机的关键帧
    vector<KeyFrame*> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    // 存放单目时的地图点
    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // 存放鱼眼双目时另一个相机的地图点
    vector<MapPoint*> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    // 存放双目时的边
    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    // 存放双目时的关键帧
    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    // 存放双目时的地图点
    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    int nPoints = 0;

    int nEdges = 0;

    // Step 7：添加待优化的地图点作为顶点
    for(list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        // 创建一个地图点顶点
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        // 设置该顶点的估计值：即该局部地图点的世界坐标
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());
        // 设置该顶点的ID: 该局部地图点的ID + 最大关键帧ID
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);

        // 设置该地图点顶点为边缘化，即在优化过程中，这个顶点不会被更新，而是被积分掉，从而减少优化问题的维度。
        // 好处是可以避免一些不稳定的情况，比如当地图点的深度很小或者很大时，它的雅可比矩阵会变得很奇异，导致优化不收敛或者发散。
        // 缺点是会增加海塞矩阵的稀疏性，使得求解线性方程更困难。
        // 这里的边缘化与滑动窗口不同，而是为了加速稀疏矩阵的计算。BlockSolver_6_3默认了6维度的不边缘化，3自由度的三维点被边缘化，所以所有三维点都设置边缘化
        vPoint->setMarginalized(true);
        // 将该顶点添加到优化器中
        optimizer.addVertex(vPoint);

        nPoints++;

        // 获取该局部地图点的观测情况
        // key: 观测到该地图点的关键帧
        // value: 该地图点在该关键帧中的索引，默认为<-1, -1>；如果是单目或PinHole双目，则为<idx, -1>；如果是KB双目且idx在右目中，则为<-1, idx>
        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        // Set edges
        // Step 8：添加边：每添加完一个局部地图点后，对每对关联的 地图点 和 观测到它的关键帧 创建边
        // 遍历所有观测到该局部地图点的关键帧
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            // 观测到该局部地图点的关键帧是好的 且 属于当前地图
            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);  // 该局部地图点 在 该观测到它的关键帧中的索引

                // Monocular observation
                // 单目
                if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] < 0)
                {
                    // 获取 该观测到它的关键帧 对 该局部地图点 的观测值：对应特征点的去畸变像素坐标
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    // 创建观测值，并写入像素坐标
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    // Step 8.1：创建单目二元边e
                    ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                    // Step 8.2：设置该边 连接的顶点: 顶点0 为该局部地图点顶点；顶点1 为观测到该局部地图点的一个关键帧位姿顶点
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    // Step 8.3：设置该边 的观测值：通常是观测到的特征点坐标 (其当前帧的左目的像素坐标)
                    e->setMeasurement(obs);

                    // Step 8.4：设置该边的信息矩阵 (与特征点所在金字塔层级相关)
                    // 计算置信度: invSigma2越小，信息矩阵越小，表示误差越大，优化时考虑的比较少
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    // 设置信息矩阵：信息矩阵 = 协方差矩阵的逆
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    // Step 8.5：设置鲁棒核函数：g2o中提供了鲁棒核函数来抑制某些误差特别大的点，如外点，避免拉偏整个优化结果。
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    // 重投影误差的自由度为2，设置对应的卡方阈值
                    rk->setDelta(thHuberMono);

                    // Step 8.6：设置相机内参
                    e->pCamera = pKFi->mpCamera;

                    // Step 8.7: 将该边添加到优化器中
                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);       // 存储到单目边中
                    vpEdgeKFMono.push_back(pKFi);   // 记录单目边的 关键帧 中
                    vpMapPointEdgeMono.push_back(pMP);  // 存储到单目边的 地图点 中

                    nEdges++;   // 边个数 + 1
                }
                // PinHole双目、RGBD
                else if(leftIndex != -1 && pKFi->mvuRight[get<0>(mit->second)] >= 0)// Stereo observation
                {
                    // 获取 该观测到它的关键帧 对 该局部地图点 的观测值：对应特征点的去畸变像素坐标
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[leftIndex];
                    // 创建观测值，并写入左目去畸变特征点坐标 和 右目对应特征点横坐标
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    // Step 8.1：创建PinHole双目二元边e
                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    // Step 8.2：设置该边 连接的顶点: 顶点0 为该局部地图点顶点；顶点1 为观测到该局部地图点的一个关键帧位姿顶点
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    // Step 8.3：设置该边 的观测值：通常是观测到的特征点坐标 (其当前帧的左目的像素坐标)
                    e->setMeasurement(obs);

                    // Step 8.4：设置该边的信息矩阵 (与特征点所在金字塔层级相关)
                    // 计算置信度: invSigma2越小，信息矩阵越小，表示误差越大，优化时考虑的比较少
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    // 设置信息矩阵：信息矩阵 = 协方差矩阵的逆
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    // Step 8.5：设置鲁棒核函数：g2o中提供了鲁棒核函数来抑制某些误差特别大的点，如外点，避免拉偏整个优化结果。
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    // 重投影误差的自由度为3，设置对应的卡方阈值
                    rk->setDelta(thHuberStereo);

                    // Step 8.6：设置相机内参。双目下没有加入相机
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    // Step 8.7: 将该边添加到优化器中
                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);         // 存储到PinHole双目边中
                    vpEdgeKFStereo.push_back(pKFi);     // 记录PinHole双目边的 关键帧 中
                    vpMapPointEdgeStereo.push_back(pMP);    // 存储到PinHole双目边的 地图点 中

                    nEdges++;   // 边个数 + 1
                }
                // KB鱼眼双目的 右目
                if(pKFi->mpCamera2)
                {
                    int rightIndex = get<1>(mit->second);   // 该局部地图点 在 该观测到它的关键帧的 在右目的特征点的去畸变坐标
                    // 右目观测到了
                    if(rightIndex != -1 )
                    {
                        rightIndex -= pKFi->NLeft;  // 实际在右目的特征点索引

                        Eigen::Matrix<double,2,1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        ORB_SLAM3::EdgeSE3ProjectXYZToBody *e = new ORB_SLAM3::EdgeSE3ProjectXYZToBody();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        Sophus::SE3f Trl = pKFi-> GetRelativePoseTrl();
                        e->mTrl = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                        e->pCamera = pKFi->mpCamera2;

                        optimizer.addEdge(e);
                        vpEdgesBody.push_back(e);
                        vpEdgeKFBody.push_back(pKFi);
                        vpMapPointEdgeBody.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }// 遍历一个局部地图点的观测完毕
    }// 遍历所有局部地图点完毕，添加完所有地图点顶点，也添加完边

    // 所有边的个数
    num_edges = nEdges;

    // 执行BA优化前，再次确认是否有外部请求停止优化
    // 因为这个变量是引用传递，所以会随外部变化。可能在Tracking::NeedNewKeyFrame(), mpLocalMapper->InsertKeyFrame()中修改了该变量的值
    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    // Step 9：开始优化，只优化一次，迭代10次
    // 初始化优化器, 这里的参数默认为0, 也就是优化等级为0，只对 level 为 0 的边进行优化；不对外点进行优化（外点等级为1）
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // 存储要删除的边 (实际为对应的关键帧和地图点的双向观测关系)
    vector<pair<KeyFrame*, MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());

    // Check inlier observations
    // Step 10：一次优化结束后，遍历参与优化的每一条误差边，根据投影误差检测外点
    // 在优化后重新计算误差，剔除连接误差比较大的关键帧和MapPoint
    // 单目边
    for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];   // // 误差边
        MapPoint* pMP = vpMapPointEdgeMono[i];  // 该边对应的 地图点

        if(pMP->isBad())
            continue;

        // 基于卡方检验计算出的阈值(假设有一个像素的偏差)
        // /这个边的误差较大 或者 边连接的地图点深度 < 0，该边有问题，缓存到删除数组中
        //! SLAM2使用两次优化 (第一次迭代5次，第二次迭代10次)，其在第一优化有，将外点设置为不参与第二次优化优化e->setLevel(1)；
        //! SLAM3因只有一次优化，所以在所有优化后重新计算误差，剔除边连接误差比较大的关键帧和地图点
        auto x_mono = e->chi2();
        if(e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];   // 取出该边对应的 关键帧
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // KB鱼眼相机双目 的右目边
    for(size_t i=0, iend=vpEdgesBody.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZToBody* e = vpEdgesBody[i];
        MapPoint* pMP = vpMapPointEdgeBody[i];

        if(pMP->isBad())
            continue;

        auto x_bino = e->chi2();
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFBody[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }
    // PinHole双目、RGBD边
    for(size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Step 11：删除被标记为外点的关键帧与地图点间的双向观测
    // Get Map Mutex
    // 锁住当前地图
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // 有需要去除的关键帧和地图点的观测，则进行双向删除
    if(!vToErase.empty())
    {
        for(size_t i=0; i < vToErase.size(); i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi); // 先删除该关键帧对该地图点的观测
            pMPi->EraseObservation(pKFi);   // 再删除当前地图点对该关键帧的观测
        }
    }

    // Recover optimized data
    // Step 12：更新优化后的局部关键帧位姿 和 局部地图点的位置、平均观测方向、最大距离、最小距离
    // 局部关键帧
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        // 用优化后的该局部关键帧的位姿 更新 该局部关键帧的位姿
        pKFi->SetPose(Tiw);
    }

    // 局部地图点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        // 优化后的该局部地图点的位置 更新 其平均观测方向、最大距离、最小距离
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();    // 增加该地图的变化次数
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{   
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    vector<Eigen::Vector3d> vZvectors(nMaxKFid+1); // For debugging
    Eigen::Vector3d z_vec;
    z_vec << 0.0, 0.0, 1.0;

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF->mnId==pMap->GetInitKFid())
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);
        vZvectors[nIDi]=vScw[nIDi].rotation()*z_vec; // For debugging

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    int count_loop = 0;
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);
            count_loop++;
            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) /*&& !sLoopEdges.count(pKFn)*/)
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }

        // Inertial edges if inertial
        if(pKF->bImu && pKF->mPrevKF)
        {
            g2o::Sim3 Spw;
            LoopClosing::KeyFrameAndPose::const_iterator itp = NonCorrectedSim3.find(pKF->mPrevKF);
            if(itp!=NonCorrectedSim3.end())
                Spw = itp->second;
            else
                Spw = vScw[pKF->mPrevKF->mnId];

            g2o::Sim3 Spi = Spw * Swi;
            g2o::EdgeSim3* ep = new g2o::EdgeSim3();
            ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mPrevKF->mnId)));
            ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            ep->setMeasurement(Spi);
            ep->information() = matLambda;
            optimizer.addEdge(ep);
        }
    }


    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);
    optimizer.computeActiveErrors();
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();

        Sophus::SE3f Tiw(CorrectedSiw.rotation().cast<float>(), CorrectedSiw.translation().cast<float>() / s);
        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double,3,1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

        pMP->UpdateNormalAndDepth();
    }

    // TODO Check this changeindex
    pMap->IncreaseChangeIndex();
}

void Optimizer::OptimizeEssentialGraph(KeyFrame* pCurKF, vector<KeyFrame*> &vpFixedKFs, vector<KeyFrame*> &vpFixedCorrectedKFs,
                                       vector<KeyFrame*> &vpNonFixedKFs, vector<MapPoint*> &vpNonCorrectedMPs)
{
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedKFs.size()) + " KFs fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedCorrectedKFs.size()) + " KFs fixed in the old map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonFixedKFs.size()) + " KFs non-fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonCorrectedMPs.size()) + " MPs non-corrected in the merged map", Verbose::VERBOSITY_DEBUG);

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    Map* pMap = pCurKF->GetMap();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    vector<bool> vpGoodPose(nMaxKFid+1);
    vector<bool> vpBadPose(nMaxKFid+1);

    const int minFeat = 100;

    for(KeyFrame* pKFi : vpFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vCorrectedSwc[nIDi]=Siw.inverse();
        VSim3->setEstimate(Siw);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = true;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = false;
    }
    Verbose::PrintMess("Opt_Essential: vpFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

    set<unsigned long> sIdKF;
    for(KeyFrame* pKFi : vpFixedCorrectedKFs)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vCorrectedSwc[nIDi]=Siw.inverse();
        VSim3->setEstimate(Siw);

        Sophus::SE3d Tcw_bef = pKFi->mTcwBefMerge.cast<double>();
        vScw[nIDi] = g2o::Sim3(Tcw_bef.unit_quaternion(),Tcw_bef.translation(),1.0);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = true;
    }

    for(KeyFrame* pKFi : vpNonFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        if(sIdKF.count(nIDi)) // It has already added in the corrected merge KFs
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vScw[nIDi] = Siw;
        VSim3->setEstimate(Siw);

        VSim3->setFixed(false);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = false;
        vpBadPose[nIDi] = true;
    }

    vector<KeyFrame*> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
    vpKFs.insert(vpKFs.end(),vpFixedKFs.begin(),vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(),vpFixedCorrectedKFs.begin(),vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(),vpNonFixedKFs.begin(),vpNonFixedKFs.end());
    set<KeyFrame*> spKFs(vpKFs.begin(), vpKFs.end());

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    for(KeyFrame* pKFi : vpKFs)
    {
        int num_connections = 0;
        const int nIDi = pKFi->mnId;

        g2o::Sim3 correctedSwi;
        g2o::Sim3 Swi;

        if(vpGoodPose[nIDi])
            correctedSwi = vCorrectedSwc[nIDi];
        if(vpBadPose[nIDi])
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKFi = pKFi->GetParent();

        // Spanning tree edge
        if(pParentKFi && spKFs.find(pParentKFi) != spKFs.end())
        {
            int nIDj = pParentKFi->mnId;

            g2o::Sim3 Sjw;
            bool bHasRelation = false;

            if(vpGoodPose[nIDi] && vpGoodPose[nIDj])
            {
                Sjw = vCorrectedSwc[nIDj].inverse();
                bHasRelation = true;
            }
            else if(vpBadPose[nIDi] && vpBadPose[nIDj])
            {
                Sjw = vScw[nIDj];
                bHasRelation = true;
            }

            if(bHasRelation)
            {
                g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3* e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;
                optimizer.addEdge(e);
                num_connections++;
            }

        }

        // Loop edges
        const set<KeyFrame*> sLoopEdges = pKFi->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(spKFs.find(pLKF) != spKFs.end() && pLKF->mnId<pKFi->mnId)
            {
                g2o::Sim3 Slw;
                bool bHasRelation = false;

                if(vpGoodPose[nIDi] && vpGoodPose[pLKF->mnId])
                {
                    Slw = vCorrectedSwc[pLKF->mnId].inverse();
                    bHasRelation = true;
                }
                else if(vpBadPose[nIDi] && vpBadPose[pLKF->mnId])
                {
                    Slw = vScw[pLKF->mnId];
                    bHasRelation = true;
                }


                if(bHasRelation)
                {
                    g2o::Sim3 Sli = Slw * Swi;
                    g2o::EdgeSim3* el = new g2o::EdgeSim3();
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                    num_connections++;
                }
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn) && spKFs.find(pKFn) != spKFs.end())
            {
                if(!pKFn->isBad() && pKFn->mnId<pKFi->mnId)
                {

                    g2o::Sim3 Snw =  vScw[pKFn->mnId];
                    bool bHasRelation = false;

                    if(vpGoodPose[nIDi] && vpGoodPose[pKFn->mnId])
                    {
                        Snw = vCorrectedSwc[pKFn->mnId].inverse();
                        bHasRelation = true;
                    }
                    else if(vpBadPose[nIDi] && vpBadPose[pKFn->mnId])
                    {
                        Snw = vScw[pKFn->mnId];
                        bHasRelation = true;
                    }

                    if(bHasRelation)
                    {
                        g2o::Sim3 Sni = Snw * Swi;

                        g2o::EdgeSim3* en = new g2o::EdgeSim3();
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                        num_connections++;
                    }
                }
            }
        }

        if(num_connections == 0 )
        {
            Verbose::PrintMess("Opt_Essential: KF " + to_string(pKFi->mnId) + " has 0 connections", Verbose::VERBOSITY_DEBUG);
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(KeyFrame* pKFi : vpNonFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();
        Sophus::SE3d Tiw(CorrectedSiw.rotation(),CorrectedSiw.translation() / s);

        pKFi->mTcwBefMerge = pKFi->GetPose();
        pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
        pKFi->SetPose(Tiw.cast<float>());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(MapPoint* pMPi : vpNonCorrectedMPs)
    {
        if(pMPi->isBad())
            continue;

        KeyFrame* pRefKF = pMPi->GetReferenceKeyFrame();
        while(pRefKF->isBad())
        {
            if(!pRefKF)
            {
                Verbose::PrintMess("MP " + to_string(pMPi->mnId) + " without a valid reference KF", Verbose::VERBOSITY_DEBUG);
                break;
            }

            pMPi->EraseObservation(pRefKF);
            pRefKF = pMPi->GetReferenceKeyFrame();
        }

        if(vpBadPose[pRefKF->mnId])
        {
            Sophus::SE3f TNonCorrectedwr = pRefKF->mTwcBefMerge;
            Sophus::SE3f Twr = pRefKF->GetPoseInverse();

            Eigen::Vector3f eigCorrectedP3Dw = Twr * TNonCorrectedwr.inverse() * pMPi->GetWorldPos();
            pMPi->SetWorldPos(eigCorrectedP3Dw);

            pMPi->UpdateNormalAndDepth();
        }
        else
        {
            cout << "ERROR: MapPoint has a reference KF from another map" << endl;
        }

    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2,
                            const bool bFixScale, Eigen::Matrix<double,7,7> &mAcumHessian, const bool bAllPoints)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Camera poses
    const Eigen::Matrix3f R1w = pKF1->GetRotation();
    const Eigen::Vector3f t1w = pKF1->GetTranslation();
    const Eigen::Matrix3f R2w = pKF2->GetRotation();
    const Eigen::Vector3f t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    ORB_SLAM3::VertexSim3Expmap * vSim3 = new ORB_SLAM3::VertexSim3Expmap();
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->pCamera1 = pKF1->mpCamera;
    vSim3->pCamera2 = pKF2->mpCamera;
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<ORB_SLAM3::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;
    vector<bool> vbIsInKF2;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);
    vbIsInKF2.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;
    int nBadMPs = 0;
    int nInKF2 = 0;
    int nOutKF2 = 0;
    int nMatchWithoutMP = 0;

    vector<int> vIdsOnlyInKF2;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = get<0>(pMP2->GetIndexInKeyFrame(pKF2));

        Eigen::Vector3f P3D1c;
        Eigen::Vector3f P3D2c;

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad())
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D1w = pMP1->GetWorldPos();
                P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(P3D1c.cast<double>());
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
            {
                nBadMPs++;
                continue;
            }
        }
        else
        {
            nMatchWithoutMP++;

            //TODO The 3D position in KF1 doesn't exist
            if(!pMP2->isBad())
            {
                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);

                vIdsOnlyInKF2.push_back(id2);
            }
            continue;
        }

        if(i2<0 && !bAllPoints)
        {
            Verbose::PrintMess("    Remove point -> i2: " + to_string(i2) + "; bAllPoints: " + to_string(bAllPoints), Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if(P3D2c(2) < 0)
        {
            Verbose::PrintMess("Sim3: Z coordinate is negative", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = new ORB_SLAM3::EdgeSim3ProjectXYZ();

        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        cv::KeyPoint kpUn2;
        bool inKF2;
        if(i2 >= 0)
        {
            kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;
            inKF2 = true;

            nInKF2++;
        }
        else
        {
            float invz = 1/P3D2c(2);
            float x = P3D2c(0)*invz;
            float y = P3D2c(1)*invz;

            obs2 << x, y;
            kpUn2 = cv::KeyPoint(cv::Point2f(x, y), pMP2->mnTrackScaleLevel);

            inKF2 = false;
            nOutKF2++;
        }

        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = new ORB_SLAM3::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);

        vbIsInKF2.push_back(inKF2);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    int nBadOutKF2 = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<ORB_SLAM3::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;

            if(!vbIsInKF2[i])
            {
                nBadOutKF2++;
            }
            continue;
        }

        //Check if remove the robust adjustment improve the result
        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    mAcumHessian = Eigen::MatrixXd::Zero(7, 7);
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();

        if(e12->chi2()>th2 || e21->chi2()>th2){
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else{
            nIn++;
        }
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}

/**
 * @brief 局部地图＋IMU BA。LocalMapping IMU中使用，地图经过imu初始化时用这个函数代替LocalBundleAdjustment
 *
 * @param[in] pKF           当前关键帧
 * @param[in] pbStopFlag    是否停止的标志
 * @param[in] pMap          地图
 * @param[in] num_fixedKF   固定不优化关键帧的个数
 * @param[in] num_OptKF     优化关键帧的个数
 * @param[in] num_MPs
 * @param[in] num_edges
 * @param[in] bLarge        成功跟踪匹配的点数是否足够多
 * @param[in] bRecInit      是否未完成IMU第三阶段初始化，完成这个变量变为false
 */
void Optimizer::LocalInertialBA(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF, int& num_MPs, int& num_edges, bool bLarge, bool bRecInit)
{
    // Step 1:  确定待优化的关键帧们
    Map* pCurrentMap = pKF->GetMap();

    int maxOpt = 10;  // 最大优化关键帧数目
    int opt_it = 10;  // 每次优化的迭代次数
    // 成功跟踪的点数足够多，则增多优化关键帧的个数，但减少每次优化迭代次数
    if (bLarge)
    {
        maxOpt = 25;
        opt_it = 4;
    }
    // 预计待优化的关键帧数，min函数是为了控制优化关键帧的数量
    const int Nd = std::min((int)pCurrentMap->KeyFramesInMap() - 2, maxOpt);
    const unsigned long maxKFid = pKF->mnId;    // 当前关键帧ID

    vector<KeyFrame*> vpOptimizableKFs;     // 存储 待优化的关键帧
    const vector<KeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();   // 获取当前关键帧的 一级共视关键帧
    list<KeyFrame*> lpOptVisKFs;        // 存储 待优化的一级共视关键帧

    vpOptimizableKFs.reserve(Nd);

    // 当前关键帧加入 待优化关键帧向量
    vpOptimizableKFs.push_back(pKF);    // 将当前关键帧 加入待优化关键帧向量的尾
    pKF->mnBALocalForKF = pKF->mnId;    // 标记当前关键帧 是 当前关键帧的局部关键帧

    // 循环将当前关键帧的前一关键帧 加入 待优化关键帧向量的尾部，即vpOptimizableKFs: [当前KF, preKF1, preKF2, ...]
    for (int i = 1; i < Nd; i++)
    {
        // 当前KF的 前一KF 存在，则加入到待优化关键帧向量的尾，且标记其是当前KF的 局部关键帧
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
        }
        // 不存在，则退出循环
        else
            break;
    }

    // 待优化关键帧的实际个数
    int N = vpOptimizableKFs.size();

    // Optimizable points seen by temporal optimizable keyframes
    list<MapPoint*> lLocalMapPoints;    // 存储临时优化关键帧的 地图点(待优化)
    // Step 2: 将上述关键帧匹配的地图点，加入局部地图点列表中
    for(int i = 0; i < N; i++)
    {
        // 获取该待优化关键帧匹配到的 地图点
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    // 若该地图点 还不是 当前关键帧的局部地图点，则将其加入到局部地图点列表中，且标记其是当前KF的 局部地图点
                    if(pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframe: First frame previous KF to optimization window)
    // Step 3: 固定一关键帧：其为待优化关键帧向量中 最早的那一关键帧的前一关键帧；如果其不存在，则将其设为最早的那一关键帧 (目前得到的地图虽然有尺度但并不是绝对的位置)
    // 存储固定关键帧
    list<KeyFrame*> lFixedKeyFrames;
    // 最早的关键帧 有 上一关键帧，则将其上一关键帧加入到 固定关键帧列表尾，且标记其是当前关键帧的 固定关键帧
    if (vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF = pKF->mnId;
    }
    // 最早的关键帧 没有 上一关键帧，则将其自己加入到 固定关键帧列表尾，且标记其是当前关键帧的 固定关键帧
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF = 0;
        vpOptimizableKFs.back()->mnBAFixedForKF = pKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();    // 从优化关键帧中删除它
    }

    // Optimizable visual KFs
    // Step 4: // 将当前KF的一级共视关键帧 加入到 待优化的共视关键帧列表lpOptVisKFs中，并将它们匹配的地图点加入局部地图点列表lLocalMapPoints中
    // 做了一系列操作发现最后lpOptVisKFs为空。这段应该是调试遗留代码，如果实现的话其实就是把共视图中在前面没有加过的关键帧们加进来，
    // 但作者可能发现之前就把共视图的全部帧加进来了，也有可能发现优化的效果不好浪费时间
    const int maxCovKF = 0;
    // 遍历当前KF的一级共视关键帧
    for (int i = 0, iend = vpNeighsKFs.size(); i < iend; i++)
    {
        // 实际未添加
        if(lpOptVisKFs.size() >= maxCovKF)
            break;

        KeyFrame* pKFi = vpNeighsKFs[i];    // 该一级共视关键帧
        // 该一级共视关键帧 是 当前KF的 局部(优化)关键帧 或 固定关键帧，则跳过
        if(pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId)
            continue;
        pKFi->mnBALocalForKF = pKF->mnId;   // 标记该一级共视关键帧 是 当前KF的局部关键帧
        // 该一级共视关键帧是好的 且 属于当前地图
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            lpOptVisKFs.push_back(pKFi);    // 加入到 待优化的共视关键帧列表

            vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            // 遍历该一级共视关键帧匹配的 地图点，加入局部地图点列表中
            for(vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
            }
        }
    }

    // Fixed KFs which are not covisible optimizable
    // Step 5: 添加固定关键帧：遍历所有局部地图点，每个地图点只将一个观测到它的关键帧 添加为 固定关键帧。(不优化，仅约束)
    const int maxFixKF = 200;   // 最大固定关键帧个数
    // 遍历局部地图点
    for(list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        // 获取该局部地图点的观测。key: 观测到该地图点的关键帧。value: 该地图点在该关键帧中的索引，默认为<-1,-1>；如果是单目或PinHole双目，则为<idx,-1>；如果是KB双目且idx在右目中，则为<-1,idx>
        map<KeyFrame*,tuple<int,int>> observations = (*lit)->GetObservations();
        // 遍历该局部地图点的每一个观测
        for (map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;    // 观测到该局部地图点的关键帧
            // 该关键帧 不是 当前关键帧的局部关键帧 且 也不是 当前关键帧的固定关键帧
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
            {
                pKFi->mnBAFixedForKF = pKF->mnId; // 将其标记为当前关键帧的 固定关键帧，防止重复添加
                if(!pKFi->isBad())
                {
                    lFixedKeyFrames.push_back(pKFi);    // 该关键帧是好的，则将其加入 固定关键帧中，直接退出，即一个地图点只将一个观测到它的关键帧 添加为 固定关键帧
                    break;
                }
            }
        }
        // 固定关键帧个数 > 200，则退出
        if (lFixedKeyFrames.size() >= maxFixKF)
            break;
    }

    // 是否 没有固定关键帧 标志
    bool bNonFixed = (lFixedKeyFrames.size() == 0);

    // Setup optimizer
    // Step 6: 构造优化器，正式开始优化
    // Step 6.1：创建稀疏优化器
    g2o::SparseOptimizer optimizer;
    // Step 6.2：声明线性求解器的类型
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    // Step 6.3：创建线性求解器
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    // Step 6.4：创建块求解器，并用上面的线性求解器初始化
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // 成功跟踪的点数足够多
    if(bLarge)
    {
        // Step 6.5：创建总求解器 solver，并使用 LM (Levenberg-Marquardt)算法；再用上面的块求解器初始化
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        // Step 6.6：将求解器的初始阻尼因子设为0.01。在优化问题中，阻尼因子用于平衡优化的速度和准确性，它会影响优化的收敛性。
        solver->setUserLambdaInit(1e-2); // to avoid iterating for finding optimal lambda
        // Step 6.7：将上述求解器作为稀疏优化器的求解方法
        optimizer.setAlgorithm(solver);
    }
    else
    {
        // Step 6.5：创建总求解器 solver，并使用 LM (Levenberg-Marquardt)算法；再用上面的块求解器初始化
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        // Step 6.6：将求解器的初始阻尼因子设为1
        solver->setUserLambdaInit(1e0);
        // Step 6.7：将上述求解器作为稀疏优化器的求解方法
        optimizer.setAlgorithm(solver);
    }

    // Set Local temporal KeyFrame vertices
    // Step 7: 添加关于待优化关键帧的位姿、速度、陀螺仪零偏、加速度计零偏顶点
    N = vpOptimizableKFs.size();
    // 遍历待优化的关键帧
    for(int i = 0; i < N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];   // 该待优化的关键帧

        // 创建一个关键帧 位姿顶点
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);    // 优化
        optimizer.addVertex(VP);    // 添加到优化器中

        // 有IMU信息
        if (pKFi->bImu)
        {
            // 创建一个关键帧 速度顶点
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3*(pKFi->mnId) + 1);
            VV->setFixed(false);
            optimizer.addVertex(VV);

            // 创建一个关键帧 陀螺仪零偏顶点
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3*(pKFi->mnId) + 2);
            VG->setFixed(false);
            optimizer.addVertex(VG);

            // 创建一个关键帧 加速度计零偏顶点
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3*(pKFi->mnId) + 3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local visual KeyFrame vertices
    // Step 8: 添加关于共视关键帧的位姿顶点 （但这里实际为空）
    for(list<KeyFrame*>::iterator it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        VertexPose * VP = new VertexPose(pKFi); // 位姿顶点
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);
    }

    // Set Fixed KeyFrame vertices
    // Step 9: 添加关于固定关键帧的位姿、速度、陀螺仪零偏、加速度计零偏顶点
    for(list<KeyFrame*>::iterator lit = lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if(pKFi->bImu) // This should be done only for keyframe just before temporal window
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid + 3*(pKFi->mnId) + 1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid + 3*(pKFi->mnId) + 2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid + 3*(pKFi->mnId) + 3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    // 暂时没看到有什么用
    vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);

    // Step 10: 遍历每个待优化的关键帧，建立惯性边，没有IMU跳过
    // 遍历待优化关键帧
    for(int i = 0; i < N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];   // 该待优化关键帧

        // 该待优化KF不存在前一关键帧，则跳过
        if(!pKFi->mPrevKF)
        {
            cout << "NOT INERTIAL LINK TO PREVIOUS FRAME!!!!" << endl;
            continue;
        }
        // 该待优化KF有IMU 且 其前一KF有IMU 且 该待优化KF有IMU预积分
        if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            // Step 10.1: 检查顶点指针是否为空
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());  // 将该关键帧pKFi的IMU预积分的零偏 设为 上一关键帧的零偏
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);   // 该关键帧前一关键帧的 位姿顶点
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1); // 该关键帧前一关键帧的 速度顶点
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2); // 该关键帧前一关键帧的 陀螺仪零偏顶点
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3); // 该关键帧前一关键帧的 加速度计零偏顶点
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);           // 该关键帧的 位姿顶点
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);  // 该关键帧的 速度顶点
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);  // 该关键帧的 陀螺仪零偏顶点
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);  // 该关键帧的 加速度计零偏顶点

            // 若为空，则跳过
            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            // Step 10.2: 创建一个IMU信息链接的 多边，并设置相关的顶点
            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1)); // 该关键帧前一关键帧的 位姿顶点
            vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1)); // 该关键帧前一关键帧的 速度顶点
            vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1)); // 该关键帧前一关键帧的 陀螺仪零偏顶点
            vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1)); // 该关键帧前一关键帧的 加速度计零偏顶点
            vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2)); // 该关键帧的 位姿顶点
            vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2)); // 该关键帧的 速度顶点

            // Step 10.2.1：若为最早的一个可优化的关键帧 或 地图没有完成IMU第三阶段初始化，则添加鲁棒核函数
            if(i == N-1 || bRecInit)
            {
                // All inertial residuals are included without robust cost function, but not that one linking the last optimizable keyframe inside of the local window and the first fixed keyframe out.
                // The information matrix for this measurement is also downweighted. This is done to avoid accumulating error due to fixing variables.
                // 所有惯性残差都没有鲁棒核，但不包括窗口内最早一个可优化关键帧与第一个固定关键帧链接起来的惯性残差。该度量的信息矩阵也被降权。这样做是为了避免由于固定变量而累积误差

                // 设置鲁棒核函数：g2o中提供了鲁棒核函数来抑制某些误差特别大的点，如外点，避免拉偏整个优化结果。
                g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
                vei[i]->setRobustKernel(rki);

                // 若是最早一个可优化关键帧，则降低其对应边的信息矩阵的权重
                if (i == N-1)
                    vei[i]->setInformation(vei[i]->information() * 1e-2);

                // 设置核函数的
                rki->setDelta(sqrt(16.92));
            }
            // Step 10.2.2: 将该边添加到优化器中
            optimizer.addEdge(vei[i]);

            // Step 10.3: 创建一个IMU陀螺仪零偏链接的 二元边
            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0,VG1);  // 该关键帧前一关键帧的 陀螺仪零偏顶点
            vegr[i]->setVertex(1,VG2);  // 该关键帧的 陀螺仪零偏顶点
            // 从预积分的协方差矩阵中提取的陀螺仪零偏的信息, 并作为该边的信息矩阵
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            // Step 10.4: 创建一个IMU加速度计零偏链接的 二元边
            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0,VA1);  // 该关键帧前一关键帧的 加速度计零偏顶点
            vear[i]->setVertex(1,VA2);  // 该关键帧的 加速度计零偏顶点
            // 从预积分的协方差矩阵中提取的加速度计零偏的信息, 并作为该边的信息矩阵
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);
            optimizer.addEdge(vear[i]);
        }
        else
            cout << "ERROR building inertial edge" << endl;
    }

    // Set MapPoint vertices
    // 边的最大数目 = 关键帧数目 * 地图点数目，实际肯定比这个要少
    const int nExpectedSize = (N + lFixedKeyFrames.size()) * lLocalMapPoints.size();

    // Mono
    // 存放单目时的二元边
    vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    // 存放单目时的关键帧
    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    // 存放单目时的地图点
    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    // 存放双目时的边
    vector<EdgeStereo*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    // 存放双目时的关键帧
    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    // 存放双目时的地图点
    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    // Step 11: 建立视觉边
    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid * 5;

    map<int,int> mVisEdges;
    // 遍历待优化关键帧
    for(int i = 0; i < N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];
        mVisEdges[pKFi->mnId] = 0;
    }
    // 遍历固定关键帧
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        mVisEdges[(*lit)->mnId] = 0;
    }

    // 添加待 优化的地图点 作为 顶点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;   // 该局部地图点

        // 创建一个地图点顶点
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>()); // 设置该顶点的估计值：即该局部地图点的世界坐标

        unsigned long id = pMP->mnId + iniMPid + 1; // 设置该顶点的ID: 该局部地图点的ID + 最大关键帧ID*5 + 1
        vPoint->setId(id);
        // 设置该地图点顶点为 边缘化，即在优化过程中，这个顶点不会被更新，而是被积分掉，从而减少优化问题的维度, 加速稀疏矩阵的计算
        // 好处: 可以避免一些不稳定的情况，比如当地图点的深度很小或者很大时，它的雅可比矩阵会变得很奇异，导致优化不收敛或者发散
        // 缺点: 会增加海塞矩阵的稀疏性，使得求解线性方程更困难
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);    // 将该顶点添加到优化器中

        // 获取该局部地图点的观测情况。key: 观测到该地图点的关键帧；value: 该地图点在该关键帧中的索引，默认为<-1, -1>；如果是单目或PinHole双目，则为<idx, -1>；如果是KB双目且idx在右目中，则为<-1, idx>
        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        // Create visual constraints
        // Step 8：添加边：每添加完一个局部地图点后，对每对关联的 地图点 和 观测到它的关键帧 创建边
        // 遍历所有观测到该局部地图点的关键帧
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            // 观测到该局部地图点的关键帧 不是当前KF的待优化关键帧 也不是固定关键帧，则跳过
            // 即必须是当前KF的 待优化关键帧 或 固定关键帧
            if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                continue;

            // 观测到该局部地图点的关键帧 是好的 且 属于当前地图
            if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                cv::KeyPoint kpUn;

                // Monocular left observation
                // 单目
                if(leftIndex != -1 && pKFi->mvuRight[leftIndex] < 0)
                {
                    mVisEdges[pKFi->mnId]++;

                    kpUn = pKFi->mvKeysUn[leftIndex];   // 观测到它的关键帧 对 该局部地图点 的观测值：对应特征点的去畸变像素坐标
                    Eigen::Matrix<double,2,1> obs;      // 创建观测值，并写入像素坐标
                    obs << kpUn.pt.x, kpUn.pt.y;

                    // 创建单目二元边e
                    EdgeMono* e = new EdgeMono(0);
                    // 设置该边 连接的顶点: 顶点0 为该局部地图点顶点；顶点1 为观测到该局部地图点的一个关键帧位姿顶点
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs); // 设置该边 的观测值：通常是观测到的特征点坐标 (其当前帧的左目的像素坐标)

                    // Add here uncerteinty
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;    // 计算置信度: invSigma2越小，信息矩阵越小，表示误差越大，优化时考虑的比较少
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);   // 设置信息矩阵：信息矩阵 = 协方差矩阵的逆

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;    // 设置鲁棒核函数
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);  // 自由度为2对应的卡方阈值

                    optimizer.addEdge(e);   // 将该边添加到优化器中
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                // Stereo-observation
                // PinHole双目、RGBD
                else if(leftIndex != -1)// Stereo observation
                {
                    kpUn = pKFi->mvKeysUn[leftIndex];   // 获取 该观测到它的关键帧 对 该局部地图点 的观测值：对应特征点的去畸变像素坐标
                    mVisEdges[pKFi->mnId]++;

                    const float kp_ur = pKFi->mvuRight[leftIndex];
                    Eigen::Matrix<double,3,1> obs;      // 创建观测值，并写入左目去畸变特征点坐标 和 右目对应特征点横坐标
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    // 创建PinHole双目二元边e
                    EdgeStereo* e = new EdgeStereo(0);
                    // 设置该边 连接的顶点: 顶点0 为该局部地图点顶点；顶点1 为观测到该局部地图点的一个关键帧位姿顶点
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs); // 设置该边 的观测值

                    // Add here uncerteinty
                    // 设置该边的信息矩阵
                    const float unc2 = pKFi->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;    // 设置鲁棒核函数
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);    // 自由度为3对应的卡方阈值

                    optimizer.addEdge(e);           // 将该边添加到优化器中
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }

                // Monocular right observation
                // KB鱼眼双目的 右目
                if(pKFi->mpCamera2) {
                    int rightIndex = get<1>(mit->second);   // 该局部地图点 在 该观测到它的关键帧的 在右目的特征点的去畸变坐标

                    // 右目观测到了
                    if(rightIndex != -1 ) {
                        rightIndex -= pKFi->NLeft;
                        mVisEdges[pKFi->mnId]++;

                        Eigen::Matrix<double,2,1> obs;
                        cv::KeyPoint kp = pKFi->mvKeysRight[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        EdgeMono* e = new EdgeMono(1);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = pKFi->mpCamera->uncertainty2(obs);

                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave]/unc2;
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                }
            }
        }// 遍历一个局部地图点的观测完毕
    }// 遍历所有局部地图点完毕，添加完所有地图点顶点，也添加完边

    // cout << "Total map points: " << lLocalMapPoints.size() << endl;
    //! TODO debug会报错先注释掉
    for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
    {
//        assert(mit->second>=3);
    }

    // Step 12: 开始优化
    // 初始化优化器, 这里的参数默认为0, 也就是优化等级为0，只对 level 为 0 的边进行优化；不对外点进行优化（外点等级为1）
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();    // 计算当前图中活跃边的误差。活跃边：那些在优化过程中需要被优化的边，它们的误差会影响目标函数的值，也就是所有重投影误差的平方和。目标函数越小，表示优化结果越好
    float err = optimizer.activeRobustChi2();   // 计算活跃边优化前的 误差 (这个误差是经过鲁棒核函数处理的)

    optimizer.optimize(opt_it); // 迭代优化opt_it次
    float err_end = optimizer.activeRobustChi2();    // 计算活跃边优化后的 误差

    // Step 14: BA优化后，检查外部是否请求停止优化，收到则强制停止优化器
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    // 保存待删除的关键帧和地图点
    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Step 13: 优化结束后，遍历参与优化的每一条误差边，根据投影误差检测外点，确认待删除的连接关系
    // Check inlier observations
    // 单目边
    for(size_t i = 0, iend = vpEdgesMono.size(); i<iend;i++)
    {
        EdgeMono* e = vpEdgesMono[i];   // 单目边
        MapPoint* pMP = vpMapPointEdgeMono[i];  // 对应地图点
        bool bClose = pMP->mTrackDepth < 10.f;  // 深度 < 10则为近点

        if(pMP->isBad())
            continue;
        // (是远点 且 当前边的投影误差>chi2Mono2) 或 (是近点 且 投影误差>1.5倍的chi2Mono2) 或 深度为负，则认为该地图点是一个外点
        if((e->chi2() > chi2Mono2 && !bClose) || (e->chi2() > 1.5f*chi2Mono2 && bClose) || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));    // 将与该边关联的关键帧和地图点标记为待删除
        }
    }

    // 双目边
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        EdgeStereo* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2() > chi2Stereo2)
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);


    // TODO: Some convergence problems have been detected here
    // (活跃边优化前的误差*2 < 优化后的误差 或 优化前后误差包含NaN值) 且 没有匹配到足够的点，说明IMU 局部BA失败，返回
    if((2 * err < err_end || isnan(err) || isnan(err_end)) && !bLarge) //bGN)
    {
        cout << "FAIL LOCAL-INERTIAL BA!!!!" << endl;
        return;
    }

    // Step 14: 删除连接关系
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi); // 删除关键帧pKFi 对 地图点pMPi 的观测
            pMPi->EraseObservation(pKFi);   // 删除地图点pMPi 对 关键帧pKFi 的观测
        }
    }

    // Step 15: 清空当前关键帧的 固定关键帧列表
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
        (*lit)->mnBAFixedForKF = 0;

    // Step 16: 取出结果
    // Recover optimized data
    // Local temporal Keyframes
    N = vpOptimizableKFs.size();
    // 遍历优化的关键帧
    for(int i = 0; i < N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));    // 该优化关键帧的 位姿 顶点指针
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw); // 更新 该关键帧优化后的 位姿
        pKFi->mnBALocalForKF = 0;   // 清空其与当前KF的BA优化关系

        // 有IMU
        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));  // 该优化关键帧的 速度 顶点指针
            pKFi->SetVelocity(VV->estimate().cast<float>());    // 更新该关键帧优化后的 速度
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));  // 该优化关键帧的 陀螺仪零偏 顶点指针
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));    // 该优化关键帧的 加速度计零偏 顶点指针
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2])); // 更新该关键帧优化后的 零偏

        }
    }

    // 遍历优化的一级共视关键帧，实际为空
    for(list<KeyFrame*>::iterator it=lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it!=itEnd; it++)
    {
        KeyFrame* pKFi = *it;
        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);
        pKFi->mnBALocalForKF=0;
    }

    // 遍历优化的地图点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));   // 该地图点的 世界坐标 顶点
        pMP->SetWorldPos(vPoint->estimate().cast<float>()); // 更新该地图点的 世界坐标
        pMP->UpdateNormalAndDepth();    // 更新 其平均观测方向、最大距离、最小距离
    }

    pMap->IncreaseChangeIndex();    // 增加该地图的变化次数
}

/**
 * @brief PoseInertialOptimizationLastFrame 中使用 Marginalize(H, 0, 14);
 * 使用舒尔补的方式边缘化海森矩阵，边缘化。
 * 列数 6            3                    3                            3                         6           3             3              3
 * --------------------------------------------------------------------------------------------------------------------------------------------------- 行数
 * |  Jp1.t * Jp1  Jp1.t * Jv1         Jp1.t * Jg1                 Jp1.t * Ja1            |  Jp1.t * Jp2  Jp1.t * Jv2        0              0        |  6
 * |  Jv1.t * Jp1  Jv1.t * Jv1         Jv1.t * Jg1                 Jv1.t * Ja1            |  Jv1.t * Jp2  Jv1.t * Jv2        0              0        |  3
 * |  Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1 + Jgr1.t * Jgr1        Jg1.t * Ja1            |  Jg1.t * Jp2  Jg1.t * Jv2  Jgr1.t * Jgr2        0        |  3
 * |  Ja1.t * Jp1  Ja1.t * Jv1         Ja1.t * Jg1           Ja1.t * Ja1 + Jar1.t * Jar1  |  Ja1.t * Jp2  Ja1.t * Jv2  Jar1.t * Jar2        0        |  3
 * |--------------------------------------------------------------------------------------------------------------------------------------------------
 * |  Jp2.t * Jp1  Jp2.t * Jv1         Jp2.t * Jg1                 Jp2.t * Ja1            |  Jp2.t * Jp2  Jp2.t * Jv2        0              0        |  6
 * |  Jv2.t * Jp1  Jv2.t * Jv1         Jv2.t * Jg1                 Jv2.t * Ja1            |  Jv2.t * Jp2  Jv2.t * Jv2        0              0        |  3
 * |      0            0              Jgr2.t * Jgr1                      0                |        0           0       Jgr2.t * Jgr2        0        |  3
 * |      0            0                    0                     Jar2.t * Jar1           |        0           0             0        Jar2.t * Jar2  |  3
 * ---------------------------------------------------------------------------------------------------------------------------------------------------
 * @param H 30*30的海森矩阵
 * @param start 开始位置
 * @param end 结束位置
 */
Eigen::MatrixXd Optimizer::Marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    // Goal
    // a  | ab | ac       a*  | 0 | ac*
    // ba | b  | bc  -->  0   | 0 | 0
    // ca | cb | c        ca* | 0 | c*

    // Size of block before block to marginalize
    const int a = start;
    // Size of block to marginalize
    const int b = end-start+1;
    // Size of block after block to marginalize
    const int c = H.cols() - (end+1);

    // Reorder as follows:
    // a  | ab | ac       a  | ac | ab
    // ba | b  | bc  -->  ca | c  | cb
    // ca | cb | c        ba | bc | b

    Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        Hn.block(0,0,a,a) = H.block(0,0,a,a);
        Hn.block(0,a+c,a,b) = H.block(0,a,a,b);
        Hn.block(a+c,0,b,a) = H.block(a,0,b,a);
    }
    if(a>0 && c>0)
    {
        Hn.block(0,a,a,c) = H.block(0,a+b,a,c);
        Hn.block(a,0,c,a) = H.block(a+b,0,c,a);
    }
    if(c>0)
    {
        Hn.block(a,a,c,c) = H.block(a+b,a+b,c,c);
        Hn.block(a,a+c,c,b) = H.block(a+b,a,c,b);
        Hn.block(a+c,a,b,c) = H.block(a,a+b,b,c);
    }
    Hn.block(a+c,a+c,b,b) = H.block(a,a,b,b);

    // Perform marginalization (Schur complement)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hn.block(a+c,a+c,b,b),Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv=svd.singularValues();
    for (int i=0; i<b; ++i)
    {
        if (singularValues_inv(i)>1e-6)
            singularValues_inv(i)=1.0/singularValues_inv(i);
        else singularValues_inv(i)=0;
    }
    Eigen::MatrixXd invHb = svd.matrixV()*singularValues_inv.asDiagonal()*svd.matrixU().transpose();
    Hn.block(0,0,a+c,a+c) = Hn.block(0,0,a+c,a+c) - Hn.block(0,a+c,a+c,b)*invHb*Hn.block(a+c,0,b,a+c);
    Hn.block(a+c,a+c,b,b) = Eigen::MatrixXd::Zero(b,b);
    Hn.block(0,a+c,a+c,b) = Eigen::MatrixXd::Zero(a+c,b);
    Hn.block(a+c,0,b,a+c) = Eigen::MatrixXd::Zero(b,a+c);

    // Inverse reorder
    // a*  | ac* | 0       a*  | 0 | ac*
    // ca* | c*  | 0  -->  0   | 0 | 0
    // 0   | 0   | 0       ca* | 0 | c*
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(),H.cols());
    if(a>0)
    {
        res.block(0,0,a,a) = Hn.block(0,0,a,a);
        res.block(0,a,a,b) = Hn.block(0,a+c,a,b);
        res.block(a,0,b,a) = Hn.block(a+c,0,b,a);
    }
    if(a>0 && c>0)
    {
        res.block(0,a+b,a,c) = Hn.block(0,a,a,c);
        res.block(a+b,0,c,a) = Hn.block(a,0,c,a);
    }
    if(c>0)
    {
        res.block(a+b,a+b,c,c) = Hn.block(a,a,c,c);
        res.block(a+b,a,c,b) = Hn.block(a,a+c,c,b);
        res.block(a,a+b,b,c) = Hn.block(a+c,a,b,c);
    }

    res.block(a,a,b,b) = Hn.block(a+c,a+c,b,b);

    return res;
}

/**************************************以下为尺度与重力优化**************************************************************/

/**
 * @brief IMU初始化中 纯IMU优化。LocalMapping::InitializeIMU中使用
 * 固定 关键帧位姿
 * 优化 重力方向、尺度、关键帧的速度和零偏
 * @param pMap 地图
 * @param Rwg 重力坐标系到世界坐标系的旋转矩阵 的初值，(重力方向到速度方向的转角)
 * @param scale 尺度（输出cout用）
 * @param bg 陀螺仪偏置（输出cout用）
 * @param ba 加速度计偏置（输出cout用）
 * @param bMono 是否为单目
 * @param covInertial IMU协方差矩阵 (暂时没用，9*9的0矩阵)
 * @param bFixedVel 是否固定速度，不优化。false则优化速度
 * @param bGauss  没用，false
 * @param priorG 陀螺仪零偏的信息矩阵 系数  (初始化第一阶段为1e2, 第二阶段为1.f, 第三阶段为0.f)
 * @param priorA 加速度计零偏的信息矩阵 系数  (初始化第一阶段为1e10, 第二阶段为1e5, 第三阶段为0.f)
 */
void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, bool bMono, Eigen::MatrixXd  &covInertial, bool bFixedVel, bool bGauss, float priorG, float priorA)
{
    Verbose::PrintMess("inertial optimization", Verbose::VERBOSITY_NORMAL);
    int its = 200;
    long unsigned int maxKFid = pMap->GetMaxKFid(); // 当前地图关键帧的最大ID
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();    // 当前地图的所有关键帧

    // Setup optimizer
    // Step 1: 构建优化器
    // Step 1.1：创建稀疏优化器
    g2o::SparseOptimizer optimizer;
    // Step 1.2：声明线性求解器的类型
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    // Step 1.3：创建线性求解器
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    // Step 1.4：创建块求解器，并用上面的线性求解器初始化
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // Step 1.5：创建总求解器 solver，并从GN、LM、DogLeg中选择一个，这里使用了 LM (Levenberg-Marquardt)算法；再用上面的块求解器初始化
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    // Step 1.6：陀螺仪零偏的信息矩阵系数 != 0时，设置求解器的初始正则化参数（阻尼因子）为1000。其用于 控制求解器在解决非线性优化问题时收敛速度和结果精度之间的平衡。
    // 正则化参数越高，求解器达到收敛的速度就越快，但是结果的精度可能会变得更差。相反，如果正则化参数越低，结果的精度可能会更高，但收敛速度会变慢，同时求解器可能会发生数值上的不稳定。
    if (priorG != 0.f)
        solver->setUserLambdaInit(1e3);

    // Step 1.7：将上述求解器作为稀疏优化器的求解方法
    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    // Step 2: 添加当前地图所有关键帧的 位姿顶点(固定,不优化) 和 速度顶点(优化)
    // 遍历当前地图中的所有关键帧
    for(size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        // 跳过 ID>当前地图最大ID的 关键帧
        if(pKFi->mnId > maxKFid)
            continue;

        // 创建关键帧的 位姿顶点 (固定)
        VertexPose * VP = new VertexPose(pKFi); // 继承于public g2o::BaseVertex<6, ImuCamPose>
        VP->setId(pKFi->mnId);
        VP->setFixed(true);     // 固定住，不优化；
        optimizer.addVertex(VP);    // 将该顶点添加到优化器中

        // 创建关键帧的 速度顶点 (不固定)
        VertexVelocity* VV = new VertexVelocity(pKFi);  // 继承于public g2o::BaseVertex<3, Eigen::Vector3d>
        VV->setId(maxKFid + (pKFi->mnId) + 1);
        if (bFixedVel)
            VV->setFixed(true);
        else
            VV->setFixed(false);    // 优化速度
        optimizer.addVertex(VV);
    }

    // Biases
    // Step 3: 添加初始参考关键帧（第一个关键帧）的 陀螺仪、加速度计的 零偏 顶点 (不固定)
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front()); // 创建陀螺仪零偏 顶点
    VG->setId(maxKFid * 2 + 2);
    if (bFixedVel)
        VG->setFixed(true);
    else
        VG->setFixed(false);    // 优化陀螺仪 零偏
    optimizer.addVertex(VG);

    VertexAccBias* VA = new VertexAccBias(vpKFs.front());   // 创建加速度计零偏顶点
    VA->setId(maxKFid * 2 + 3);
    if (bFixedVel)
        VA->setFixed(true);
    else
        VA->setFixed(false);    // 优化加速度计 零偏
    optimizer.addVertex(VA);

    // prior acc bias
    Eigen::Vector3f bprior;
    bprior.setZero();

    // Step 4: 添加初始参考关键帧（第一个关键帧）的 加速度计、陀螺仪零偏的 边
    EdgePriorAcc* epa = new EdgePriorAcc(bprior);   // 创建加速度计零偏 一元边
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA)); // 该边 连接的顶点: 顶点0为加速度计零偏 顶点
    double infoPriorA = priorA;     // 传入的加速度计零偏的 信息矩阵 系数 (初始化第一阶段为1e10, 第二阶段为1e5, 第三阶段为0.f)
    epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());  // 该边的 信息矩阵
    optimizer.addEdge(epa);     // 将该边添加到优化器中

    EdgePriorGyro* epg = new EdgePriorGyro(bprior); // 创建陀螺仪零偏 一元边
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG)); // 该边 连接的顶点: 顶点0为陀螺仪零偏 顶点
    double infoPriorG = priorG;     // 传入的陀螺仪零偏的 信息矩阵 系数 (初始化第一阶段为1e2, 第二阶段为1.f, 第三阶段为0.f)
    epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());  // 该边的 信息矩阵
    optimizer.addEdge(epg);

    // Gravity and scale
    // Step 5: 添加 重力方向、尺度的 顶点 (不固定)
    VertexGDir* VGDir = new VertexGDir(Rwg);    // 创建重力方向 顶点
    VGDir->setId(maxKFid * 2 + 4);
    VGDir->setFixed(false);     // 优化
    optimizer.addVertex(VGDir);

    VertexScale* VS = new VertexScale(scale);   // 创建尺度 顶点
    VS->setId(maxKFid * 2 + 5);
    VS->setFixed(!bMono);       // 单目不固定，双目固定
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    // Step 6: 添加 关键帧之间与IMU信息、重力方向、尺度信息链接的 八元边。将关键帧之间的IMU测量信息与重力方向和尺度进行优化
    // 存储与IMU信息、重力方向、尺度信息链接的八元边
    vector<EdgeInertialGS*> vpei;
    vpei.reserve(vpKFs.size());

    // 存储使用的某关键帧的前一关键帧 与 其的 关键帧对
    vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());
    //std::cout << "build optimization graph" << std::endl;

    // 遍历每个关键帧
    for(size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        // 该关键帧存在前一关键帧 且 其ID <= 最大关键帧ID
        if(pKFi->mPrevKF && pKFi->mnId <= maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;
            // 该关键帧没有IMU预积分，显示信息但不跳过
            if(!pKFi->mpImuPreintegrated)
                std::cout << "Not preintegrated measurement" << std::endl;

            // 到这里的条件: pKFi是好的，且它有上一个关键帧，且它的ID<最大ID
            // Step 6.1: 检查顶点指针是否为空
            // 将该关键帧的IMU预积分的零偏 设为 上一关键帧的零偏
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());

            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);   // 该关键帧前一关键帧的 位姿顶点
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1); // 该关键帧前一关键帧的 速度顶点
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);   // 该关键帧的 位姿顶点
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);    // 该关键帧的 速度顶点
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);    // 陀螺仪零偏顶点
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);    // 加速度计零偏顶点
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4); // 重力方向顶点
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);    // 尺度顶点
            // 若为空，则跳过
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;
                continue;
            }
            // Step 6.2: 创建一个IMU信息链接 八元边，并设置相关的顶点。这是一个大边，包含了上面所有信息。(注意到前面的两个零偏也做了两个一元边加入)
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            vpei.push_back(ei); // 将该边加入 vpei 中
            vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi)); // 将该关键帧的前一关键帧 与 该关键帧 对 加入vppUsedKF

            optimizer.addEdge(ei);  // 将该边添加到优化器中
        }
    }

    // Compute error for different scales
    // 获取优化器中的所有边
    std::set<g2o::HyperGraph::Edge*> setEdges = optimizer.edges();

    optimizer.setVerbose(false);    // 关闭冗长的输出信息
    // Step 7：开始优化，只优化一次，迭代200次。（优化后更新了所有关键帧的零偏）
    optimizer.initializeOptimization(); // 初始化优化器, 这里的参数默认为0, 也就是优化等级为0，只对 level 为 0 的边进行优化；不对外点进行优化（外点等级为1）
    optimizer.optimize(its);

    // Step 8: 获取优化后的结果
    scale = VS->estimate(); // 优化后的尺度

    // Recover optimized data
    // Biases
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));   // 优化后的 初始关键帧的 陀螺仪零偏顶点指针，并转换为VertexGyroBias类型
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));    // 优化后的 初始关键帧的 加速度计零偏顶点指针，并转换为VertexAccBias类型
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();   // 优化后的 初始关键帧的 陀螺仪、加速度计零偏
    bg << VG->estimate();
    ba << VA->estimate();
    scale = VS->estimate();     // 更新尺度

    // 使用优化后的 初始关键帧的 零偏向量vb 构建一个IMU::Bias对象 b
    IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);
    // 从优化后的结果中获取 重力坐标系到世界坐标系的 旋转矩阵初值 Rwg
    Rwg = VGDir->estimate().Rwg;

    // Keyframes velocities and biases
    // 更新每个关键帧的 速度 和 零偏(优化后的初始关键帧的零偏)
    const int N = vpKFs.size();
    for(size_t i = 0; i < N; i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId > maxKFid)
            continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));    // 速度顶点的指针，并转换为 VertexVelocity类型
        Eigen::Vector3d Vw = VV->estimate();    // 优化后的速度，Velocity is scaled after
        pKFi->SetVelocity(Vw.cast<float>());    // 更新关键帧的速度

        // 检查该关键帧的陀螺仪零偏是否发生变化 (即关键帧的陀螺仪零偏 - 优化后的陀螺仪零偏的 模 是否>0.01)，
        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->SetNewBias(b);    // 更新该关键帧的零偏 为 优化后的初始关键帧的零偏
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();    // 根据新的零偏 更新该关键帧的预积分
        }
        else
            pKFi->SetNewBias(b);    // 只更新零偏，但不重新预积分
    }
}

/**
 * @brief LoopClosing::MergeLocal2 中使用
 * 跟参数最多的那个同名函数不同的地方在于很多节点不可选是否固定，优化的目标有：
 * 速度，偏置
 * @param pMap 地图
 * @param bg 陀螺仪偏置
 * @param ba 加速度计偏置
 * @param priorG 陀螺仪偏置的信息矩阵系数
 * @param priorA 加速度计偏置的信息矩阵系数
 */
void Optimizer::InertialOptimization(Map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG, float priorA)
{
    int its = 200; // Check number of iterations
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (fixed poses and optimizable velocities)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+(pKFi->mnId)+1);
        VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
    VG->setId(maxKFid*2+2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    VertexAccBias* VA = new VertexAccBias(vpKFs.front());
    VA->setId(maxKFid*2+3);
    VA->setFixed(false);

    optimizer.addVertex(VA);
    // prior acc bias
    Eigen::Vector3f bprior;
    bprior.setZero();

    EdgePriorAcc* epa = new EdgePriorAcc(bprior);
    epa->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    EdgePriorGyro* epg = new EdgePriorGyro(bprior);
    epg->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG*Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Eigen::Matrix3d::Identity());
    VGDir->setId(maxKFid*2+4);
    VGDir->setFixed(true);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(1.0);
    VS->setId(maxKFid*2+5);
    VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    vector<EdgeInertialGS*> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<KeyFrame*,KeyFrame*> > vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());

    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;

            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(maxKFid*2+2);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(maxKFid*2+3);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(maxKFid*2+4);
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(maxKFid*2+5);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                cout << "Error" << VP1 << ", "<< VV1 << ", "<< VG << ", "<< VA << ", " << VP2 << ", " << VV2 <<  ", "<< VGDir << ", "<< VS <<endl;

                continue;
            }
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->mPrevKF,pKFi));
            optimizer.addEdge(ei);

        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);


    // Recover optimized data
    // Biases
    VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid*2+2));
    VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid*2+3));
    Vector6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();

    IMU::Bias b (vb[3],vb[4],vb[5],vb[0],vb[1],vb[2]);

    //Keyframes velocities and biases
    const int N = vpKFs.size();
    for(size_t i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;

        VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+(pKFi->mnId)+1));
        Eigen::Vector3d Vw = VV->estimate();
        pKFi->SetVelocity(Vw.cast<float>());

        if ((pKFi->GetGyroBias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->SetNewBias(b);
            if (pKFi->mpImuPreintegrated)
                pKFi->mpImuPreintegrated->Reintegrate();
        }
        else
            pKFi->SetNewBias(b);
    }
}

/**
 * @brief 优化重力方向与尺度，LocalMapping::ScaleRefinement()中使用，优化目标有：
 * 重力方向与尺度
 * @param pMap 地图
 * @param Rwg 重力方向到速度方向的转角
 * @param scale 尺度
 */
void Optimizer::InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale)
{
    int its = 10;
    long unsigned int maxKFid = pMap->GetMaxKFid();
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Set KeyFrame vertices (all variables are fixed)
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKFi = vpKFs[i];
        if(pKFi->mnId>maxKFid)
            continue;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        VertexVelocity* VV = new VertexVelocity(pKFi);
        VV->setId(maxKFid+1+(pKFi->mnId));
        VV->setFixed(true);
        optimizer.addVertex(VV);

        // Vertex of fixed biases
        VertexGyroBias* VG = new VertexGyroBias(vpKFs.front());
        VG->setId(2*(maxKFid+1)+(pKFi->mnId));
        VG->setFixed(true);
        optimizer.addVertex(VG);
        VertexAccBias* VA = new VertexAccBias(vpKFs.front());
        VA->setId(3*(maxKFid+1)+(pKFi->mnId));
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }

    // Gravity and scale
    VertexGDir* VGDir = new VertexGDir(Rwg);
    VGDir->setId(4*(maxKFid+1));
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    VertexScale* VS = new VertexScale(scale);
    VS->setId(4*(maxKFid+1)+1);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    // Graph edges
    int count_edges = 0;
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        if(pKFi->mPrevKF && pKFi->mnId<=maxKFid)
        {
            if(pKFi->isBad() || pKFi->mPrevKF->mnId>maxKFid)
                continue;
                
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex((maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VP2 =  optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex((maxKFid+1)+pKFi->mnId);
            g2o::HyperGraph::Vertex* VG = optimizer.vertex(2*(maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VA = optimizer.vertex(3*(maxKFid+1)+pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VGDir = optimizer.vertex(4*(maxKFid+1));
            g2o::HyperGraph::Vertex* VS = optimizer.vertex(4*(maxKFid+1)+1);
            if(!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                Verbose::PrintMess("Error" + to_string(VP1->id()) + ", " + to_string(VV1->id()) + ", " + to_string(VG->id()) + ", " + to_string(VA->id()) + ", " + to_string(VP2->id()) + ", " + to_string(VV2->id()) +  ", " + to_string(VGDir->id()) + ", " + to_string(VS->id()), Verbose::VERBOSITY_NORMAL);

                continue;
            }
            count_edges++;
            EdgeInertialGS* ei = new EdgeInertialGS(pKFi->mpImuPreintegrated);
            ei->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            ei->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            ei->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG));
            ei->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA));
            ei->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            ei->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));
            ei->setVertex(6,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VGDir));
            ei->setVertex(7,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VS));
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            ei->setRobustKernel(rk);
            rk->setDelta(1.f);
            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(its);
    optimizer.computeActiveErrors();
    float err_end = optimizer.activeRobustChi2();
    // Recover optimized data
    scale = VS->estimate();
    Rwg = VGDir->estimate().Rwg;
}

/**
 * @brief Local Bundle Adjustment LoopClosing::MergeLocal() 融合地图时使用，纯视觉
 * 优化目标： 1. vpAdjustKF; 2.vpAdjustKF与vpFixedKF对应的MP点
 * 优化所有的当前关键帧共视窗口里的关键帧和地图点, 固定所有融合帧共视窗口里的帧
 * @param pMainKF        mpCurrentKF 当前关键帧
 * @param vpAdjustKF     vpLocalCurrentWindowKFs 待优化的KF
 * @param vpFixedKF      vpMergeConnectedKFs 固定的KF
 * @param pbStopFlag     false
 */
void Optimizer::LocalBundleAdjustment(KeyFrame* pMainKF,vector<KeyFrame*> vpAdjustKF, vector<KeyFrame*> vpFixedKF, bool *pbStopFlag)
{
    bool bShowImages = false;

    vector<MapPoint*> vpMPs;

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(true);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;
    set<KeyFrame*> spKeyFrameBA;

    Map* pCurrentMap = pMainKF->GetMap();

    // Set fixed KeyFrame vertices
    int numInsertedPoints = 0;
    for(KeyFrame* pKFi : vpFixedKF)
    {
        if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
        {
            Verbose::PrintMess("ERROR LBA: KF is bad or is not in the current map", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        pKFi->mnBALocalForMerge = pMainKF->mnId;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;

        set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
        for(MapPoint* pMPi : spViewMPs)
        {
            if(pMPi)
                if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)

                    if(pMPi->mnBALocalForMerge!=pMainKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge=pMainKF->mnId;
                        numInsertedPoints++;
                    }
        }

        spKeyFrameBA.insert(pKFi);
    }

    // Set non fixed Keyframe vertices
    set<KeyFrame*> spAdjustKF(vpAdjustKF.begin(), vpAdjustKF.end());
    numInsertedPoints = 0;
    for(KeyFrame* pKFi : vpAdjustKF)
    {
        if(pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
            continue;

        pKFi->mnBALocalForMerge = pMainKF->mnId;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float> Tcw = pKFi->GetPose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->mnId);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;

        set<MapPoint*> spViewMPs = pKFi->GetMapPoints();
        for(MapPoint* pMPi : spViewMPs)
        {
            if(pMPi)
            {
                if(!pMPi->isBad() && pMPi->GetMap() == pCurrentMap)
                {
                    if(pMPi->mnBALocalForMerge != pMainKF->mnId)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->mnBALocalForMerge = pMainKF->mnId;
                        numInsertedPoints++;
                    }
                }
            }
        }

        spKeyFrameBA.insert(pKFi);
    }

    const int nExpectedSize = (vpAdjustKF.size()+vpFixedKF.size())*vpMPs.size();

    vector<ORB_SLAM3::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    map<KeyFrame*, int> mpObsKFs;
    map<KeyFrame*, int> mpObsFinalKFs;
    map<MapPoint*, int> mpObsMPs;
    for(unsigned int i=0; i < vpMPs.size(); ++i)
    {
        MapPoint* pMPi = vpMPs[i];
        if(pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMPi->GetWorldPos().cast<double>());
        const int id = pMPi->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);


        const map<KeyFrame*,tuple<int,int>> observations = pMPi->GetObservations();
        int nEdges = 0;
        //SET EDGES
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid || pKF->mnBALocalForMerge != pMainKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[get<0>(mit->second)];

            if(pKF->mvuRight[get<0>(mit->second)]<0) //Monocular
            {
                mpObsMPs[pMPi]++;
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                ORB_SLAM3::EdgeSE3ProjectXYZ* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber2D);

                e->pCamera = pKF->mpCamera;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsMPs[pMPi]+=2;
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[get<0>(mit->second)];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber3D);

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    map<unsigned long int, int> mWrongObsKF;
    if(bDoMore)
    {
        // Check inlier observations
        int badMonoMP = 0, badStereoMP = 0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badMonoMP++;
            }
            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badStereoMP++;
            }

            e->setRobustKernel(0);
        }
        Verbose::PrintMess("[BA]: First optimization(Huber), there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " stereo bad edges", Verbose::VERBOSITY_DEBUG);

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    set<MapPoint*> spErasedMPs;
    set<KeyFrame*> spErasedKFs;

    // Check inlier observations
    int badMonoMP = 0, badStereoMP = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            mWrongObsKF[pKFi->mnId]++;
            badMonoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
            mWrongObsKF[pKFi->mnId]++;
            badStereoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    Verbose::PrintMess("[BA]: Second optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " sterero bad edges", Verbose::VERBOSITY_DEBUG);

    // Get Map Mutex
    unique_lock<mutex> lock(pMainKF->GetMap()->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }
    for(unsigned int i=0; i < vpMPs.size(); ++i)
    {
        MapPoint* pMPi = vpMPs[i];
        if(pMPi->isBad())
            continue;

        const map<KeyFrame*,tuple<int,int>> observations = pMPi->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid || pKF->mnBALocalForKF != pMainKF->mnId || !pKF->GetMapPoint(get<0>(mit->second)))
                continue;

            if(pKF->mvuRight[get<0>(mit->second)]<0) //Monocular
            {
                mpObsFinalKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsFinalKFs[pKF]++;
            }
        }
    }

    // Recover optimized data
    // Keyframes
    for(KeyFrame* pKFi : vpAdjustKF)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKFi->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        Sophus::SE3f Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());

        int numMonoBadPoints = 0, numMonoOptPoints = 0;
        int numStereoBadPoints = 0, numStereoOptPoints = 0;
        vector<MapPoint*> vpMonoMPsOpt, vpStereoMPsOpt;
        vector<MapPoint*> vpMonoMPsBad, vpStereoMPsBad;

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ORB_SLAM3::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];
            KeyFrame* pKFedge = vpEdgeKFMono[i];

            if(pKFi != pKFedge)
            {
                continue;
            }

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                numMonoBadPoints++;
                vpMonoMPsBad.push_back(pMP);

            }
            else
            {
                numMonoOptPoints++;
                vpMonoMPsOpt.push_back(pMP);
            }

        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];
            KeyFrame* pKFedge = vpEdgeKFMono[i];

            if(pKFi != pKFedge)
            {
                continue;
            }

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                numStereoBadPoints++;
                vpStereoMPsBad.push_back(pMP);
            }
            else
            {
                numStereoOptPoints++;
                vpStereoMPsOpt.push_back(pMP);
            }
        }

        pKFi->SetPose(Tiw);
    }

    //Points
    for(MapPoint* pMPi : vpMPs)
    {
        if(pMPi->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMPi->mnId+maxKFid+1));
        pMPi->SetWorldPos(vPoint->estimate().cast<float>());
        pMPi->UpdateNormalAndDepth();

    }
}

/**
 * @brief 这里面进行visual inertial ba
 * LoopClosing::MergeLocal2 中用到
 * 优化目标：相关帧的位姿，速度，偏置，还有涉及点的坐标，可以理解为跨地图的局部窗口优化
 * @param[in] pCurrKF 当前关键帧
 * @param[in] pMergeKF 融合帧
 * @param[in] pbStopFlag 是否优化
 * @param[in] pMap 当前地图
 * @param[out] corrPoses 所有的Sim3 矫正
 */
void Optimizer::MergeInertialBA(KeyFrame* pCurrKF, KeyFrame* pMergeKF, bool *pbStopFlag, Map *pMap, LoopClosing::KeyFrameAndPose &corrPoses)
{
    const int Nd = 6;
    const unsigned long maxKFid = pCurrKF->mnId;

    vector<KeyFrame*> vpOptimizableKFs;
    vpOptimizableKFs.reserve(2*Nd);

    // For cov KFS, inertial parameters are not optimized
    const int maxCovKF = 30;
    vector<KeyFrame*> vpOptimizableCovKFs;
    vpOptimizableCovKFs.reserve(maxCovKF);

    // Add sliding window for current KF
    vpOptimizableKFs.push_back(pCurrKF);
    pCurrKF->mnBALocalForKF = pCurrKF->mnId;
    for(int i=1; i<Nd; i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    list<KeyFrame*> lFixedKeyFrames;
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBALocalForKF=pCurrKF->mnId;
    }
    else
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Add temporal neighbours to merge KF (previous and next KFs)
    vpOptimizableKFs.push_back(pMergeKF);
    pMergeKF->mnBALocalForKF = pCurrKF->mnId;

    // Previous KFs
    for(int i=1; i<(Nd/2); i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    // We fix just once the old map
    if(vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF=pCurrKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF=0;
        vpOptimizableKFs.back()->mnBAFixedForKF=pCurrKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Next KFs
    if(pMergeKF->mNextKF)
    {
        vpOptimizableKFs.push_back(pMergeKF->mNextKF);
        vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
    }

    while(vpOptimizableKFs.size()<(2*Nd))
    {
        if(vpOptimizableKFs.back()->mNextKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mNextKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pCurrKF->mnId;
        }
        else
            break;
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by optimizable keyframes
    list<MapPoint*> lLocalMapPoints;
    map<MapPoint*,int> mLocalObs;
    for(int i=0; i<N; i++)
    {
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            // Using mnBALocalForKF we avoid redundance here, one MP can not be added several times to lLocalMapPoints
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pCurrKF->mnId)
                    {
                        mLocalObs[pMP]=1;
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pCurrKF->mnId;
                    }
                    else {
                        mLocalObs[pMP]++;
                    }
        }
    }

    std::vector<std::pair<MapPoint*, int>> pairs;
    pairs.reserve(mLocalObs.size());
    for (auto itr = mLocalObs.begin(); itr != mLocalObs.end(); ++itr)
        pairs.push_back(*itr);
    sort(pairs.begin(), pairs.end(),sortByVal);

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    int i=0;
    for(vector<pair<MapPoint*,int>>::iterator lit=pairs.begin(), lend=pairs.end(); lit!=lend; lit++, i++)
    {
        map<KeyFrame*,tuple<int,int>> observations = lit->first->GetObservations();
        if(i>=maxCovKF)
            break;
        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pCurrKF->mnId && pKFi->mnBAFixedForKF!=pCurrKF->mnId) // If optimizable or already included...
            {
                pKFi->mnBALocalForKF=pCurrKF->mnId;
                if(!pKFi->isBad())
                {
                    vpOptimizableCovKFs.push_back(pKFi);
                    break;
                }
            }
        }
    }

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Set Local KeyFrame vertices
    N=vpOptimizableKFs.size();
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local cov keyframes vertices
    int Ncov=vpOptimizableCovKFs.size();
    for(int i=0; i<Ncov; i++)
    {
        KeyFrame* pKFi = vpOptimizableCovKFs[i];

        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        VertexPose * VP = new VertexPose(pKFi);
        VP->setId(pKFi->mnId);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if(pKFi->bImu)
        {
            VertexVelocity* VV = new VertexVelocity(pKFi);
            VV->setId(maxKFid+3*(pKFi->mnId)+1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            VertexGyroBias* VG = new VertexGyroBias(pKFi);
            VG->setId(maxKFid+3*(pKFi->mnId)+2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            VertexAccBias* VA = new VertexAccBias(pKFi);
            VA->setId(maxKFid+3*(pKFi->mnId)+3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    vector<EdgeInertial*> vei(N,(EdgeInertial*)NULL);
    vector<EdgeGyroRW*> vegr(N,(EdgeGyroRW*)NULL);
    vector<EdgeAccRW*> vear(N,(EdgeAccRW*)NULL);
    for(int i=0;i<N;i++)
    {
        //cout << "inserting inertial edge " << i << endl;
        KeyFrame* pKFi = vpOptimizableKFs[i];

        if(!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!!!!", Verbose::VERBOSITY_NORMAL);
            continue;
        }
        if(pKFi->bImu && pKFi->mPrevKF->bImu && pKFi->mpImuPreintegrated)
        {
            pKFi->mpImuPreintegrated->SetNewBias(pKFi->mPrevKF->GetImuBias());
            g2o::HyperGraph::Vertex* VP1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* VV1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+1);
            g2o::HyperGraph::Vertex* VG1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+2);
            g2o::HyperGraph::Vertex* VA1 = optimizer.vertex(maxKFid+3*(pKFi->mPrevKF->mnId)+3);
            g2o::HyperGraph::Vertex* VP2 = optimizer.vertex(pKFi->mnId);
            g2o::HyperGraph::Vertex* VV2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+1);
            g2o::HyperGraph::Vertex* VG2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+2);
            g2o::HyperGraph::Vertex* VA2 = optimizer.vertex(maxKFid+3*(pKFi->mnId)+3);

            if(!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                cerr << "Error " << VP1 << ", "<< VV1 << ", "<< VG1 << ", "<< VA1 << ", " << VP2 << ", " << VV2 <<  ", "<< VG2 << ", "<< VA2 <<endl;
                continue;
            }

            vei[i] = new EdgeInertial(pKFi->mpImuPreintegrated);

            vei[i]->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP1));
            vei[i]->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV1));
            vei[i]->setVertex(2,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VG1));
            vei[i]->setVertex(3,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VA1));
            vei[i]->setVertex(4,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VP2));
            vei[i]->setVertex(5,dynamic_cast<g2o::OptimizableGraph::Vertex*>(VV2));

            // TODO Uncomment
            g2o::RobustKernelHuber* rki = new g2o::RobustKernelHuber;
            vei[i]->setRobustKernel(rki);
            rki->setDelta(sqrt(16.92));
            optimizer.addEdge(vei[i]);

            vegr[i] = new EdgeGyroRW();
            vegr[i]->setVertex(0,VG1);
            vegr[i]->setVertex(1,VG2);
            Eigen::Matrix3d InfoG = pKFi->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new EdgeAccRW();
            vear[i]->setVertex(0,VA1);
            vear[i]->setVertex(1,VA2);
            Eigen::Matrix3d InfoA = pKFi->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);
            optimizer.addEdge(vear[i]);
        }
        else
            Verbose::PrintMess("ERROR building inertial edge", Verbose::VERBOSITY_NORMAL);
    }

    Verbose::PrintMess("end inserting inertial edges", Verbose::VERBOSITY_NORMAL);


    // Set MapPoint vertices
    const int nExpectedSize = (N+Ncov+lFixedKeyFrames.size())*lLocalMapPoints.size();

    // Mono
    vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<EdgeStereo*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid*5;

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        if (!pMP)
            continue;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());

        unsigned long id = pMP->mnId+iniMPid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        // Create visual constraints
        for(map<KeyFrame*,tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if (!pKFi)
                continue;

            if ((pKFi->mnBALocalForKF!=pCurrKF->mnId) && (pKFi->mnBAFixedForKF!=pCurrKF->mnId))
                continue;

            if (pKFi->mnId>maxKFid){
                continue;
            }


            if(optimizer.vertex(id)==NULL || optimizer.vertex(pKFi->mnId)==NULL)
                continue;

            if(!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[get<0>(mit->second)];

                if(pKFi->mvuRight[get<0>(mit->second)]<0) // Monocular observation
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // stereo observation
                {
                    const float kp_ur = pKFi->mvuRight[get<0>(mit->second)];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(8);

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        EdgeMono* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Mono2)
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Stereo
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        EdgeStereo* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>chi2Stereo2)
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }


    // Recover optimized data
    //Keyframes
    for(int i=0; i<N; i++)
    {
        KeyFrame* pKFi = vpOptimizableKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);

        Sophus::SE3d Tiw = pKFi->GetPose().cast<double>();
        g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
        corrPoses[pKFi] = g2oSiw;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
        }
    }

    for(int i=0; i<Ncov; i++)
    {
        KeyFrame* pKFi = vpOptimizableCovKFs[i];

        VertexPose* VP = static_cast<VertexPose*>(optimizer.vertex(pKFi->mnId));
        Sophus::SE3f Tcw(VP->estimate().Rcw[0].cast<float>(), VP->estimate().tcw[0].cast<float>());
        pKFi->SetPose(Tcw);

        Sophus::SE3d Tiw = pKFi->GetPose().cast<double>();
        g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
        corrPoses[pKFi] = g2oSiw;

        if(pKFi->bImu)
        {
            VertexVelocity* VV = static_cast<VertexVelocity*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+1));
            pKFi->SetVelocity(VV->estimate().cast<float>());
            VertexGyroBias* VG = static_cast<VertexGyroBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+2));
            VertexAccBias* VA = static_cast<VertexAccBias*>(optimizer.vertex(maxKFid+3*(pKFi->mnId)+3));
            Vector6d b;
            b << VG->estimate(), VA->estimate();
            pKFi->SetNewBias(IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]));
        }
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+iniMPid+1));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    pMap->IncreaseChangeIndex();
}

/**
 * @brief 使用上一关键帧+当前帧的视觉信息+IMU信息，优化当前帧位姿
 *
 * 可分为以下几个步骤：
 * // Step 1：创建g2o优化器，初始化顶点和边
 * // Step 2：启动多轮优化，剔除外点
 * // Step 3：更新当前帧位姿、速度、IMU偏置
 * // Step 4：记录当前帧的优化状态，包括参数信息和对应的海森矩阵
 *
 * @param[in] pFrame 当前帧，也是待优化的帧
 * @param[in] bRecInit 调用这个函数的位置并没有传这个参数，因此它的值默认为false
 * @return int 返回优化后的内点数
 */
int Optimizer::PoseInertialOptimizationLastKeyFrame(Frame *pFrame, bool bRecInit)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    int nInitialMonoCorrespondences=0;
    int nInitialStereoCorrespondences=0;
    int nInitialCorrespondences=0;

    // Set Frame vertex
    VertexPose* VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set MapPoint vertices
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;
    const bool bRight = (Nleft!=-1);

    vector<EdgeMonoOnlyPose*> vpEdgesMono;
    vector<EdgeStereoOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeMono;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);
    int null_point = 0;
    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for(int i=0; i<N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                cv::KeyPoint kpUn;

                // Left monocular observation
                if((!bRight && pFrame->mvuRight[i]<0) || i < Nleft)
                {
                    if(i < Nleft) // pair left-right
                        kpUn = pFrame->mvKeys[i];
                    else
                        kpUn = pFrame->mvKeysUn[i];

                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else if(!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                if(bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
            else
                null_point++;
        }
    }
    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    Verbose::PrintMess("\t\t[optimize] nInitialCorrespondences: "+std::to_string(nInitialCorrespondences), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("\t\t[optimize] total landmarks: "+std::to_string(N), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("\t\t[optimize] null point: "+std::to_string(null_point), Verbose::VERBOSITY_DEBUG);

    // 4. 上一个关键帧节点
    // pKF为上一关键帧
    KeyFrame* pKF = pFrame->mpLastKeyFrame;

    // 上一关键帧的位姿，旋转+平移，6-dim
    VertexPose* VPk = new VertexPose(pKF);
    VPk->setId(4);
    VPk->setFixed(true);
    optimizer.addVertex(VPk);
    // 上一关键帧的速度，3-dim
    VertexVelocity* VVk = new VertexVelocity(pKF);
    VVk->setId(5);
    VVk->setFixed(true);
    optimizer.addVertex(VVk);
    // 上一关键帧的陀螺仪偏置，3-dim
    VertexGyroBias* VGk = new VertexGyroBias(pKF);
    VGk->setId(6);
    VGk->setFixed(true);
    optimizer.addVertex(VGk);
    // 上一关键帧的加速度偏置，3-dim
    VertexAccBias* VAk = new VertexAccBias(pKF);
    VAk->setId(7);
    VAk->setFixed(true);
    optimizer.addVertex(VAk);
    // setFixed(true)这个设置使以上四个顶点（15个参数）的值在优化时保持固定
    // 既然被选为关键帧，就不能太善变

    Verbose::PrintMess("\t[optimize] add inertial edge", Verbose::VERBOSITY_DEBUG);
    // 5. 第二种边（IMU预积分约束）：两帧之间位姿的变化量与IMU预积分的值偏差尽可能小
    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegrated);

    // 将上一关键帧四个顶点（P、V、BG、BA）和当前帧两个顶点（P、V）加入第二种边
    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);  // 把第二种边加入优化器

//    std::cout << "pKF gyro bias: " << pKF->transpose() << std::endl;
//    std::cout << "frame gyro bias: " << frame->get_gyro_bias().transpose() << std::endl;
//
//    std::cout << "pKF acc bias: " << pKF->get_acc_bias().transpose() << std::endl;
//    std::cout << "frame acc bias: " << frame->get_acc_bias().transpose() << std::endl;

    // 6. 第三种边（陀螺仪随机游走约束）：陀螺仪的随机游走值在相近帧间不会相差太多  residual=VG-VGk
    // 用大白话来讲就是用固定的VGK拽住VG，防止VG在优化中放飞自我
    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0,VGk);  // 将上一关键帧的BG加入第三种边
    egr->setVertex(1,VG);   // 将当前帧的BG加入第三种边
    // C值在预积分阶段更新，range(9,12)对应陀螺仪偏置的协方差，最终cvInfoG值为inv(∑(GyroRW^2/freq))
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr); // 把第三种边加入优化器
    std::cout << "infoG: \n" << InfoG << std::endl;

    // 7. 第四种边（加速度随机游走约束）：加速度的随机游走值在相近帧间不会相差太多  residual=VA-VAk
    // 用大白话来讲就是用固定的VAK拽住VA，防止VA在优化中放飞自我
    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0,VAk);  // 将上一关键帧的BA加入第四种边
    ear->setVertex(1,VA);   // 将当前帧的BA加入第四种边
    // C值在预积分阶段更新，range(12,15)对应加速度偏置的协方差，最终cvInfoG值为inv(∑(AccRW^2/freq))
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear); // 把第四种边加入优化器
    std::cout << "infoA: \n" << InfoA << std::endl;

    // 8. 启动多轮优化，剔除外点

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 卡方检验值呈递减趋势，目的是让检验越来越苛刻
    float chi2Mono[4]={12,7.5,5.991,5.991};
    float chi2Stereo[4]={15.6,9.8,7.815,7.815};

    // 4次优化的迭代次数都为10
    int its[4]={10,10,10,10};

    int nBad = 0;
    int nBadMono = 0;
    int nBadStereo = 0;
    int nInliersMono = 0;
    int nInliersStereo = 0;
    int nInliers = 0;

    // 进行4次优化
    for(size_t it=0; it<4; it++)
    {
        // 初始化优化器,这里的参数0代表只对level为0的边进行优化（不传参数默认也是0）
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);    // 每次优化迭代十次

        nBad = 0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers = 0;
        nInliersMono = 0;
        nInliersStereo = 0;
        // 使用1.5倍的chi2Mono作为“近点”的卡方检验值，意味着地图点越近，检验越宽松
        // 地图点如何定义为“近点”在下面的代码中有解释
        float chi2close = 1.5 * chi2Mono[it];

        // For monocular observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();
//            Verbose::PrintMess("\t\tid: "+std::to_string(idx)+", chi2: "+std::to_string(chi2), Verbose::VERBOSITY_DEBUG);
            std::stringstream ss_info;
            ss_info << e->information();
//            std::cout << "\t\tinfo: " << ss_info.str().c_str() << std::endl;
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth < 10.f;

            if((chi2>chi2Mono[it]&&!bClose)||(bClose && chi2>chi2close)||!e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it==2)
                e->setRobustKernel(0);
        }

        // For stereo observations
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1); // not included in next optimization
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        // 内点总数=单目内点数+双目内点数
        nInliers = nInliersMono + nInliersStereo;
        // 坏点数=单目坏点数+双目坏点数
        nBad = nBadMono + nBadStereo;

        if(optimizer.edges().size()<10)
        {
            cout << "\t\t局部地图跟踪中，优化当前帧位姿: NOT ENOUGH EDGES" << endl;
            break;
        }// 一次优化迭代结束
    }// 4次迭代优化结束

    std::cout << "bRecInit: " << bRecInit << std::endl;
    // If not too much tracks, recover not too bad points
    // 若4次优化后内点数小于30，尝试恢复一部分不那么糟糕的坏点
    if ((nInliers < 30) && !bRecInit)
    {
        nBad=0;
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;
        EdgeMonoOnlyPose* e1;
        EdgeStereoOnlyPose* e2;
        for(size_t i=0, iend=vnIndexEdgeMono.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1 = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2()<chi2MonoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
        for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2 = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2()<chi2StereoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
    }

    // 更新当前帧位姿、速度、IMU偏置

    // Recover optimized pose, velocity and biases
    pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), VP->estimate().twb.cast<float>(), VV->estimate().cast<float>());
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

    // Recover Hessian, marginalize keyFframe states and generate new prior for frame
    Eigen::Matrix<double,15,15> H;
    H.setZero();

    H.block<9,9>(0,0)+= ei->GetHessian2();
    H.block<3,3>(9,9) += egr->GetHessian2();
    H.block<3,3>(12,12) += ear->GetHessian2();

    int tot_in = 0, tot_out = 0;
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
    {
        EdgeMonoOnlyPose* e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(0,0) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
    {
        EdgeStereoOnlyPose* e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if(!pFrame->mvbOutlier[idx])
        {
            H.block<6,6>(0,0) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    Verbose::PrintMess("\t\t[optimize] frame id: "+std::to_string(pFrame->mnId)+", set_constrain_pose_imu", Verbose::VERBOSITY_DEBUG);
    std::stringstream ss_H;
    ss_H << H;
    std::cout << "\t\t[optimize] Hessian: \n" << ss_H.str().c_str() << std::endl;

    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H);

    std::cout << "Rwb: " << VP->estimate().Rwb << std::endl;
    std::cout << "twb: " << VP->estimate().twb << std::endl;
    std::cout << "velo: " << VV->estimate() << std::endl;
    std::cout << "gyro: " << VG->estimate() << std::endl;
    std::cout << "accel: " << VA->estimate() << std::endl;

    return nInitialCorrespondences - nBad;
}

/**
 * @brief 使用上一帧+当前帧的视觉信息+IMU信息，优化当前帧位姿
 *
 * 可分为以下几个步骤：
 * // Step 1：创建g2o优化器，初始化顶点和边
 * // Step 2：启动多轮优化，剔除外点
 * // Step 3：更新当前帧位姿、速度、IMU偏置
 * // Step 4：记录当前帧的优化状态，包括参数信息和边缘化后的海森矩阵
 *
 * @param[in] pFrame 当前帧，也是待优化的帧
 * @param[in] bRecInit 调用这个函数的位置并没有传这个参数，因此它的值默认为false
 * @return int 返回优化后的内点数
 */
int Optimizer::PoseInertialOptimizationLastFrame(Frame *pFrame, bool bRecInit)
{
    // Step 1：创建g2o优化器，初始化顶点和边
    // 构建一个稀疏求解器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver; // 声明线性求解器的类型

    // 创建线性求解器, 且使用dense的求解器，（常见非dense求解器有cholmod线性求解器和shur补线性求解器）
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    // 创建块求解器，并用上面的线性求解器初始化
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // 创建总求解器, 并用上面的块求解器初始化，这里使用高斯牛顿算法
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver); // 求解器solver设置给优化器optimizer，这样优化器就可以使用该求解器进行图优化
    optimizer.setVerbose(false);

    // 当前帧单（左）目地图点数目
    int nInitialMonoCorrespondences = 0;
    int nInitialStereoCorrespondences = 0;
    int nInitialCorrespondences = 0;

    // Set Current Frame vertex
    // 当前帧的位姿，旋转+平移，6-dim
    VertexPose* VP = new VertexPose(pFrame);
    VP->setId(0);
    VP->setFixed(false);    // 需要优化，所以不固定
    optimizer.addVertex(VP);
    // 当前帧的速度，3-dim
    VertexVelocity* VV = new VertexVelocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    // 当前帧的陀螺仪偏置，3-dim
    VertexGyroBias* VG = new VertexGyroBias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    // 当前帧的加速度偏置，3-dim
    VertexAccBias* VA = new VertexAccBias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set MapPoint vertices
    // 当前帧的特征点总数：单目、立体匹配双目、RGB-D: N != 0, Nleft = -1，Nright = -1
    //                  非立体匹配双目: N != 0, Nleft != -1，Nright != -1。N = Nleft + Nright
    const int N = pFrame->N;
    const int Nleft = pFrame->Nleft;    // 当前帧左目的特征点数，非立体匹配双目(Kar)有值
    const bool bRight = (Nleft != -1);  // 当前帧是否为非立体匹配双目，存在右相机，存在为true

    vector<EdgeMonoOnlyPose*> vpEdgesMono;
    vector<EdgeStereoOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeMono;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
    const float thHuberMono = sqrt(5.991);
    // 自由度为3的卡方分布，显著性水平为0.05，对应的临界阈值7.815
    const float thHuberStereo = sqrt(7.815);

    {
        // 锁定地图点。由于需要使用地图点来构造顶点和边,因此不希望在构造的过程中部分地图点被改写造成不一致甚至是段错误
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        // 遍历当前帧中所有有效地图点，加边
        for(int i = 0; i < N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                cv::KeyPoint kpUn;
                // 单（左）目 Left monocular observation
                // 包含两种情况：1.单目情况 2.非立体匹配双目情况下的左目
                if((!bRight && pFrame->mvuRight[i] < 0) || i < Nleft)
                {
                    // 非立体匹配双目情况下的左目，使用未畸变校正的特征点
                    if(i < Nleft) // pair left-right
                        kpUn = pFrame->mvKeys[i];
                    // 单目，使用畸变校正过的特征点
                    else
                        kpUn = pFrame->mvKeysUn[i];

                    nInitialMonoCorrespondences++;  // 单目地图点计数增加
                    pFrame->mvbOutlier[i] = false;  // 当前地图点默认设置为不是外点

                    // 观测的特征点
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    // 第一种边(视觉重投影约束)：地图点投影到该帧图像的坐标与特征点坐标偏差尽可能小
                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),0);

                    // 将位姿作为第一个顶点
                    e->setVertex(0,VP);
                    // 设置观测值，即去畸变后的像素坐标
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    // 获取不确定度，这里调用uncertainty2返回固定值1.0
                    // ?这里返回1.0是作为缺省值，是否可以根据对视觉信息的信任度动态修改这个值，比如标定的误差？
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    // invSigma2 = (Inverse(协方差矩阵))^2，表明该约束在各个维度上的可信度
                    //  图像金字塔层数越高，可信度越差
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                    // 设置该约束的信息矩阵
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    // 设置鲁棒核函数，避免其误差的平方项出现数值过大的增长 注：后续在优化2次后会用e->setRobustKernel(0)禁掉鲁棒核函数
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    // 重投影误差的自由度为2，设置对应的卡方阈值
                    rk->setDelta(thHuberMono);

                    // 将第一种边加入优化器
                    optimizer.addEdge(e);

                    // 将第一种边加入vpEdgesMono
                    vpEdgesMono.push_back(e);
                    // 将对应的特征点索引加入vnIndexEdgeMono
                    vnIndexEdgeMono.push_back(i);
                }
                // 立体匹配双目 Stereo observation
                else if(!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    // 该特征点在当前帧的左目 去畸变像素横纵坐标 和 在右目的横坐标
                    kpUn = pFrame->mvKeysUn[i];
                    const float kp_ur = pFrame->mvuRight[i];
                    Eigen::Matrix<double,3,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    // 新建边e，
                    EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                    e->setVertex(0, VP);    // 设置连接的顶点(当前帧位姿)
                    e->setMeasurement(obs); // 边的观测数值：该特征点在当前帧的左目和右目的像素坐标

                    // Add here uncerteinty
                    // 置信程度主要是看左目特征点所在的图层
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix3d::Identity()*invSigma2);   // 设置信息矩阵

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);     // 记录边属于双目情况
                    vnIndexEdgeStereo.push_back(i); // 记录索引
                }

                // 非立体匹配双目中的右目 Right monocular observation
                if(bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    kpUn = pFrame->mvKeysRight[i - Nleft];
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos(),1);

                    e->setVertex(0,VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = pFrame->mpCamera->uncertainty2(obs);

                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave]/unc2;
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }// 一个地图点结束
            // 该地图点不存在
        }// 遍历完所有地图点
    }

    // 统计参与优化的地图点总数目
    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    // Set Previous Frame Vertex
    // pFp为上一帧
    Frame* pFp = pFrame->mpPrevFrame;


    std::cout << "prev frame pose： \n" << pFp->GetPose().matrix() << std::endl;
    std::cout << "curr frame pose： \n" << pFrame->GetPose().matrix() << std::endl;

    // 上一帧的位姿，旋转+平移，6-dim
    VertexPose* VPk = new VertexPose(pFp);
    VPk->setId(4);
    VPk->setFixed(false);
    optimizer.addVertex(VPk);
    // 上一帧的速度，3-dim
    VertexVelocity* VVk = new VertexVelocity(pFp);
    VVk->setId(5);
    VVk->setFixed(false);
    optimizer.addVertex(VVk);
    // 上一帧的陀螺仪偏置，3-dim
    VertexGyroBias* VGk = new VertexGyroBias(pFp);
    VGk->setId(6);
    VGk->setFixed(false);
    optimizer.addVertex(VGk);
    // 上一帧的加速度偏置，3-dim
    VertexAccBias* VAk = new VertexAccBias(pFp);
    VAk->setId(7);
    VAk->setFixed(false);
    optimizer.addVertex(VAk);
    // setFixed(false)这个设置使以上四个顶点（15个参数）的值随优化而变，这样做会给上一帧再提供一些优化空间
    // 但理论上不应该优化过多，否则会有不良影响，故后面的代码会用第五种边来约束上一帧的变化量

    // 第二种边（IMU预积分约束）：两帧之间位姿的变化量与IMU预积分的值偏差尽可能小
    EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);

    // 将上一帧四个顶点（P、V、BG、BA）和当前帧两个顶点（P、V）加入第二种边
    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    // 把第二种边加入优化器
    optimizer.addEdge(ei);

    // 第三种边（陀螺仪随机游走约束）：陀螺仪的随机游走值在相邻帧间不会相差太多  residual=VG-VGk
    // 用大白话来讲就是用固定的VGK拽住VG，防止VG在优化中放飞自我
    EdgeGyroRW* egr = new EdgeGyroRW();
    egr->setVertex(0,VGk);   // 将上一帧的BG加入第三种边
    egr->setVertex(1,VG);   // 将当前帧的BG加入第三种边

    // C值在预积分阶段更新，range(9,12)对应陀螺仪偏置的协方差，最终cvInfoG值为inv(∑(GyroRW^2/freq))
    Eigen::Matrix3d InfoG = pFrame->mpImuPreintegrated->C.block<3,3>(9,9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr); // 把第三种边加入优化器

    // 第四种边（加速度随机游走约束）：加速度的随机游走值在相近帧间不会相差太多  residual=VA-VAk
    // 用大白话来讲就是用固定的VAK拽住VA，防止VA在优化中放飞自我
    EdgeAccRW* ear = new EdgeAccRW();
    ear->setVertex(0,VAk);  // 将上一帧的BA加入第四种边
    ear->setVertex(1,VA);   // 将当前帧的BA加入第四种边
    // C值在预积分阶段更新，range(12,15)对应加速度偏置的协方差，最终cvInfoG值为inv(∑(AccRW^2/freq))
    Eigen::Matrix3d InfoA = pFrame->mpImuPreintegrated->C.block<3,3>(12,12).cast<double>().inverse();
    ear->setInformation(InfoA);
    // 把第四种边加入优化器
    optimizer.addEdge(ear);

    // ?既然有判空的操作，可以区分一下有先验信息（五条边）和无先验信息（四条边）的情况
    if (!pFp->mpcpi)
        Verbose::PrintMess("pFp->mpcpi does not exist!!!\nPrevious Frame " + to_string(pFp->mnId), Verbose::VERBOSITY_NORMAL);

    // 第五种边（先验约束）：上一帧信息随优化的改变量要尽可能小
    EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);

    // 将上一帧的四个顶点（P、V、BG、BA）加入第五种边
    ep->setVertex(0,VPk);
    ep->setVertex(1,VVk);
    ep->setVertex(2,VGk);
    ep->setVertex(3,VAk);
    g2o::RobustKernelHuber* rkp = new g2o::RobustKernelHuber;
    ep->setRobustKernel(rkp);
    rkp->setDelta(5);
    // 把第五种边加入优化器
    optimizer.addEdge(ep);

    // Step 2：启动多轮优化，剔除外点

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 与PoseInertialOptimizationLastKeyFrame函数对比，区别在于：在优化过程中保持卡方阈值不变
    // 以下参数的解释
    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
    // 自由度为3的卡方分布，显著性水平为0.05，对应的临界阈值7.815
    // 自由度为3的卡方分布，显著性水平为0.02，对应的临界阈值9.8
    // 自由度为3的卡方分布，显著性水平为0.001，对应的临界阈值15.6
    // 计算方法：https://stattrek.com/online-calculator/chi-square.aspx
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={15.6f,9.8f,7.815f,7.815f};
    // 4次优化的迭代次数都为10
    const int its[4]={10,10,10,10};

    int nBad=0;             // 坏点数
    int nBadMono = 0;       // 单目坏点数
    int nBadStereo = 0;     // 双目坏点数
    int nInliersMono = 0;   // 单目内点数
    int nInliersStereo = 0; // 双目内点数
    int nInliers = 0;         // 内点数

    // 迭代优化4次
    for(size_t it = 0; it < 4; it++)
    {
        // 初始化优化器,这里的参数0代表只对level为0的边进行优化（不传参数默认也是0）
        optimizer.initializeOptimization(0);
        // 每次优化迭代十次
        optimizer.optimize(its[it]);

        // 每次优化都重新统计各类点的数目
        nBad = 0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers = 0;
        nInliersMono = 0;
        nInliersStereo = 0;

        // 使用1.5倍的chi2Mono作为“近点”的卡方检验值，意味着地图点越近，检验越宽松
        // 地图点如何定义为“近点”在下面的代码中有解释
        float chi2close = 1.5 * chi2Mono[it];

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            EdgeMonoOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            // 当地图点在当前帧的深度值小于10时，该地图点属于close（近点）
            // mTrackDepth是在Frame.cc的isInFrustum函数中计算出来的
            bool bClose = pFrame->mvpMapPoints[idx]->mTrackDepth<10.f;

            // 如果这条误差边是来自于outlier
            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();  // 计算本次优化后的误差
            }

            // 就是error *\ Omega * error，表示了这条边考虑置信度以后的误差大小
            const float chi2 = e->chi2();

            // 判断某地图点为外点的条件有以下三种：
            // 1.该地图点不是近点并且误差大于卡方检验值chi2Mono[it]
            // 2.该地图点是近点并且误差大于卡方检验值chi2close
            // 3.深度不为正
            // 每次优化后，用更小的卡方检验值，原因是随着优化的进行，对划分为内点的信任程度越来越低
            if((chi2 > chi2Mono[it] && !bClose)||(bClose && chi2 > chi2close) || !e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx]=true;   // 将该点设置为外点
                e->setLevel(1);     // 外点不参与下一轮优化
                nBadMono++;             // 单目坏点数+1
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;  // 将该点设置为内点（暂时）
                e->setLevel(0);         // 内点继续参与下一轮优化
                nInliersMono++;         // 单目内点数+1
            }

            // 从第三次优化开始就不设置鲁棒核函数了，原因是经过两轮优化已经趋向准确值，不会有太大误差
            if (it==2)
                e->setRobustKernel(0);

        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        // 内点总数 = 单目内点数 + 双目内点数
        nInliers = nInliersMono + nInliersStereo;
        // 坏点数 = 单目坏点数 + 双目坏点数
        nBad = nBadMono + nBadStereo;

        if(optimizer.edges().size() < 10)
        {
            break;
        }
    }

    // 若4次优化后内点数 < 30，尝试恢复一部分不那么糟糕的坏点
    if ((nInliers < 30) && !bRecInit)
    {
        // 重新从0开始统计坏点数
        nBad = 0;
        // 单目可容忍的卡方检验最大值（如果误差比这还大就不要挣扎了...）
        const float chi2MonoOut = 18.f;
        const float chi2StereoOut = 24.f;
        EdgeMonoOnlyPose* e1;
        EdgeStereoOnlyPose* e2;
        // 遍历所有单目特征点
        for(size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            // 获取这些特征点对应的边
            e1 = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2()<chi2MonoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;

        }
        for(size_t i=0, iend=vnIndexEdgeStereo.size(); i<iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2 = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2()<chi2StereoOut)
                pFrame->mvbOutlier[idx]=false;
            else
                nBad++;
        }
    }

    // ?多此一举？优化过程中nInliers这个值已经计算过了，nInliersMono和nInliersStereo在后续代码中一直保持不变
    nInliers = nInliersMono + nInliersStereo;

    // Step 3：更新当前帧位姿、速度、IMU偏置

    // Recover optimized pose, velocity and biases
    // 给当前帧设置优化后的旋转、位移、速度，用来更新位姿
    pFrame->SetImuPoseVelocity(VP->estimate().Rwb.cast<float>(), VP->estimate().twb.cast<float>(), VV->estimate().cast<float>());
    Vector6d b;
    b << VG->estimate(), VA->estimate();
    // 给当前帧设置优化后的bg，ba
    pFrame->mImuBias = IMU::Bias(b[3],b[4],b[5],b[0],b[1],b[2]);

    // Step 4：记录当前帧的优化状态，包括参数信息和边缘化后的海森矩阵
    // Recover Hessian, marginalize previous frame states and generate new prior for frame
    // 包含本次优化所有信息矩阵的和，它代表本次优化对确定性的影响
    Eigen::Matrix<double,30,30> H;
    H.setZero();

    //第1步，加上EdgeInertial（IMU预积分约束）的海森矩阵
    // ei的定义
    // EdgeInertial* ei = new EdgeInertial(pFrame->mpImuPreintegratedFrame);
    // ei->setVertex(0, VPk);
    // ei->setVertex(1, VVk);
    // ei->setVertex(2, VGk);
    // ei->setVertex(3, VAk);
    // ei->setVertex(4, VP);  // VertexPose* VP = new VertexPose(pFrame);
    // ei->setVertex(5, VV);  // VertexVelocity* VV = new VertexVelocity(pFrame);
    // ei->GetHessian()  =  J.t * J 下同，不做详细标注了
    // J
    //       rn + tn  vn    gn   an   rn+1 + tn+1  vn+1
    // er      Jp1    Jv1  Jg1   Ja1      Jp2      Jv2
    // 角标1表示上一帧，2表示当前帧
    //      6            3             3           3            6            3
    // Jp1.t * Jp1  Jp1.t * Jv1  Jp1.t * Jg1  Jp1.t * Ja1  Jp1.t * Jp2  Jp1.t * Jv2     6
    // Jv1.t * Jp1  Jv1.t * Jv1  Jv1.t * Jg1  Jv1.t * Ja1  Jv1.t * Jp2  Jv1.t * Jv2     3
    // Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1  Jg1.t * Ja1  Jg1.t * Jp2  Jg1.t * Jv2     3
    // Ja1.t * Jp1  Ja1.t * Jv1  Ja1.t * Jg1  Ja1.t * Ja1  Ja1.t * Jp2  Ja1.t * Jv2     3
    // Jp2.t * Jp1  Jp2.t * Jv1  Jp2.t * Jg1  Jp2.t * Ja1  Jp2.t * Jp2  Jp2.t * Jv2     6
    // Jv2.t * Jp1  Jv2.t * Jv1  Jv2.t * Jg1  Jv2.t * Ja1  Jv2.t * Jp2  Jv2.t * Jv2     3
    // 所以矩阵是24*24 的
    H.block<24,24>(0,0)+= ei->GetHessian();

    // 经过这步H变成了
    // 列数 6            3             3           3            6           3        6
    // ---------------------------------------------------------------------------------- 行数
    // Jp1.t * Jp1  Jp1.t * Jv1  Jp1.t * Jg1  Jp1.t * Ja1  Jp1.t * Jp2  Jp1.t * Jv2   0 |  6
    // Jv1.t * Jp1  Jv1.t * Jv1  Jv1.t * Jg1  Jv1.t * Ja1  Jv1.t * Jp2  Jv1.t * Jv2   0 |  3
    // Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1  Jg1.t * Ja1  Jg1.t * Jp2  Jg1.t * Jv2   0 |  3
    // Ja1.t * Jp1  Ja1.t * Jv1  Ja1.t * Jg1  Ja1.t * Ja1  Ja1.t * Jp2  Ja1.t * Jv2   0 |  3
    // Jp2.t * Jp1  Jp2.t * Jv1  Jp2.t * Jg1  Jp2.t * Ja1  Jp2.t * Jp2  Jp2.t * Jv2   0 |  6
    // Jv2.t * Jp1  Jv2.t * Jv1  Jv2.t * Jg1  Jv2.t * Ja1  Jv2.t * Jp2  Jv2.t * Jv2   0 |  3
    //     0            0            0            0            0           0          0 |  6
    // ----------------------------------------------------------------------------------

    //第2步，加上EdgeGyroRW（陀螺仪随机游走约束）的信息矩阵：
    //|   0~8   |       9~11     | 12~23 |     24~26     |27~29
    // 9~11是上一帧的bg(3-dim)，24~26是当前帧的bg(3-dim)
    //  egr的定义
    //  EdgeGyroRW* egr = new EdgeGyroRW();
    //  egr->setVertex(0, VGk);
    //  egr->setVertex(1, VG);
    Eigen::Matrix<double,6,6> Hgr = egr->GetHessian();
    H.block<3,3>(9,9) += Hgr.block<3,3>(0,0);
    H.block<3,3>(9,24) += Hgr.block<3,3>(0,3);
    H.block<3,3>(24,9) += Hgr.block<3,3>(3,0);
    H.block<3,3>(24,24) += Hgr.block<3,3>(3,3);

    // 经过这步H变成了
    // 列数 6            3                    3                      3            6           3             3         3
    // ----------------------------------------------------------------------------------------------------------------- 行数
    // Jp1.t * Jp1  Jp1.t * Jv1         Jp1.t * Jg1           Jp1.t * Ja1  Jp1.t * Jp2  Jp1.t * Jv2        0         0 |  6
    // Jv1.t * Jp1  Jv1.t * Jv1         Jv1.t * Jg1           Jv1.t * Ja1  Jv1.t * Jp2  Jv1.t * Jv2        0         0 |  3
    // Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1 + Jgr1.t * Jgr1  Jg1.t * Ja1  Jg1.t * Jp2  Jg1.t * Jv2  Jgr1.t * Jgr2   0 |  3
    // Ja1.t * Jp1  Ja1.t * Jv1         Ja1.t * Jg1           Ja1.t * Ja1  Ja1.t * Jp2  Ja1.t * Jv2        0         0 |  3
    // Jp2.t * Jp1  Jp2.t * Jv1         Jp2.t * Jg1           Jp2.t * Ja1  Jp2.t * Jp2  Jp2.t * Jv2        0         0 |  6
    // Jv2.t * Jp1  Jv2.t * Jv1         Jv2.t * Jg1           Jv2.t * Ja1  Jv2.t * Jp2  Jv2.t * Jv2        0         0 |  3
    //     0            0             Jgr2.t * Jgr1                 0            0           0       Jgr2.t * Jgr2   0 |  3
    //     0            0                    0                      0            0           0             0         0 |  3
    // -----------------------------------------------------------------------------------------------------------------

    //第3步，加上EdgeAccRW（加速度随机游走约束）的信息矩阵：
    //|   0~11   |      12~14    | 15~26 |     27~29     |30
    // 12~14是上一帧的ba(3-dim)，27~29是当前帧的ba(3-dim)
    // ear的定义
    // EdgeAccRW* ear = new EdgeAccRW();
    // ear->setVertex(0, VAk);
    // ear->setVertex(1, VA);
    Eigen::Matrix<double,6,6> Har = ear->GetHessian();
    H.block<3,3>(12,12) += Har.block<3,3>(0,0);
    H.block<3,3>(12,27) += Har.block<3,3>(0,3);
    H.block<3,3>(27,12) += Har.block<3,3>(3,0);
    H.block<3,3>(27,27) += Har.block<3,3>(3,3);

    // 经过这步H变成了
    // 列数 6            3                    3                            3                         6           3             3              3
    // --------------------------------------------------------------------------------------------------------------------------------------------------- 行数
    // |  Jp1.t * Jp1  Jp1.t * Jv1         Jp1.t * Jg1                 Jp1.t * Ja1            |  Jp1.t * Jp2  Jp1.t * Jv2        0              0        |  6
    // |  Jv1.t * Jp1  Jv1.t * Jv1         Jv1.t * Jg1                 Jv1.t * Ja1            |  Jv1.t * Jp2  Jv1.t * Jv2        0              0        |  3
    // |  Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1 + Jgr1.t * Jgr1        Jg1.t * Ja1            |  Jg1.t * Jp2  Jg1.t * Jv2  Jgr1.t * Jgr2        0        |  3
    // |  Ja1.t * Jp1  Ja1.t * Jv1         Ja1.t * Jg1           Ja1.t * Ja1 + Jar1.t * Jar1  |  Ja1.t * Jp2  Ja1.t * Jv2        0       Jar1.t * Jar2   |  3
    // |--------------------------------------------------------------------------------------------------------------------------------------------------
    // |  Jp2.t * Jp1  Jp2.t * Jv1         Jp2.t * Jg1                 Jp2.t * Ja1            |  Jp2.t * Jp2  Jp2.t * Jv2        0              0        |  6
    // |  Jv2.t * Jp1  Jv2.t * Jv1         Jv2.t * Jg1                 Jv2.t * Ja1            |  Jv2.t * Jp2  Jv2.t * Jv2        0              0        |  3
    // |      0            0              Jgr2.t * Jgr1                      0                |        0           0       Jgr2.t * Jgr2        0        |  3
    // |      0            0                    0                     Jar2.t * Jar1           |        0           0             0        Jar2.t * Jar2  |  3
    // ---------------------------------------------------------------------------------------------------------------------------------------------------

    //第4步，加上EdgePriorPoseImu（先验约束）的信息矩阵：
    //|   0~14   |  15~29
    // 0~14是上一帧的P(6-dim)、V(3-dim)、BG(3-dim)、BA(3-dim)
    // ep定义
    // EdgePriorPoseImu* ep = new EdgePriorPoseImu(pFp->mpcpi);
    // ep->setVertex(0, VPk);
    // ep->setVertex(1, VVk);
    // ep->setVertex(2, VGk);
    // ep->setVertex(3, VAk);
    //      6            3             3           3
    // Jp1.t * Jp1  Jp1.t * Jv1  Jp1.t * Jg1  Jp1.t * Ja1     6
    // Jv1.t * Jp1  Jv1.t * Jv1  Jv1.t * Jg1  Jv1.t * Ja1     3
    // Jg1.t * Jp1  Jg1.t * Jv1  Jg1.t * Jg1  Jg1.t * Ja1     3
    // Ja1.t * Jp1  Ja1.t * Jv1  Ja1.t * Jg1  Ja1.t * Ja1     3
    H.block<15,15>(0,0) += ep->GetHessian();    // 上一帧 的H矩阵，矩阵太大了不写了。。。总之就是加到下标为1相关的了

    int tot_in = 0, tot_out = 0;
    // 第5步：关于位姿的海森
    for(size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        EdgeMonoOnlyPose* e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if(!pFrame->mvbOutlier[idx])
        {
            //  0~14  |     15~20   | 21~29
            // 15~20是当前帧的P(6-dim)
            H.block<6,6>(15,15) += e->GetHessian(); // 当前帧的H矩阵，矩阵太大了不写了。。。总之就是加到p2相关的了
            tot_in++;
        }
        else
            tot_out++;
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
    {
        EdgeStereoOnlyPose* e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if(!pFrame->mvbOutlier[idx])
        {
            //  0~14  |     15~20   | 21~29
            H.block<6,6>(15,15) += e->GetHessian();
            tot_in++;
        }
        else
            tot_out++;
    }

    // H  = |B  E.t| ------> |0             0|
    //      |E    A|         |0 A-E*B.inv*E.t|
    H = Marginalize(H,0,14);

    /*
    Marginalize里的函数在此处的调用等效于：
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H.block(0, 0, 15, 15), Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv = svd.singularValues();
    for (int i = 0; i < 15; ++i)
    {
        if (singularValues_inv(i) > 1e-6)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else
            singularValues_inv(i) = 0;
    }
    Eigen::MatrixXd invHb = svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose();
    H.block(15, 15, 15, 15) = H.block(15, 15, 15, 15) - H.block(15, 0, 15, 15) * invHb - H.block(0, 15, 15, 15);
    */

    // 构造一个ConstraintPoseImu对象，为下一帧提供先验约束
    // 构造对象所使用的参数是当前帧P、V、BG、BA的估计值和边缘化后的H矩阵
    pFrame->mpcpi = new ConstraintPoseImu(VP->estimate().Rwb,VP->estimate().twb,VV->estimate(),VG->estimate(),VA->estimate(),H.block<15,15>(15,15));
    // 下一帧使用的EdgePriorPoseImu参数来自于此
    delete pFp->mpcpi;
    pFp->mpcpi = NULL;

    // 返回值：内点数 = 总地图点数目 - 坏点（外点）数目
    return nInitialCorrespondences - nBad;
}

/**
 * @brief  LoopClosing::MergeLocal() 融合地图时使用，优化当前帧没有参与融合的元素，本质图优化
 * 优化目标： 1. vpNonFixedKFs; 2.vpNonCorrectedMPs
 * @param pCurKF                 mpCurrentKF 融合时当前关键帧
 * @param vpFixedKFs             vpMergeConnectedKFs 融合地图中的关键帧，也就是上面函数中的 vpFixedKF
 * @param vpFixedCorrectedKFs    vpLocalCurrentWindowKFs 当前地图中经过矫正的关键帧，也就是Optimizer::LocalBundleAdjustment中优化过的vpAdjustKF
 * @param vpNonFixedKFs          vpCurrentMapKFs 当前地图中剩余的关键帧，待优化
 * @param vpNonCorrectedMPs      vpCurrentMapMPs 当前地图中剩余的MP点，待优化
 */
void Optimizer::OptimizeEssentialGraph4DoF(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections)
{
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<4, 4> > BlockSolver_4_4;

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolverX::LinearSolverType * linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);

    vector<VertexPose4DoF*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;
    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;

        VertexPose4DoF* V4DoF;

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            const g2o::Sim3 Swc = it->second.inverse();
            Eigen::Matrix3d Rwc = Swc.rotation().toRotationMatrix();
            Eigen::Vector3d twc = Swc.translation();
            V4DoF = new VertexPose4DoF(Rwc, twc, pKF);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

            vScw[nIDi] = Siw;
            V4DoF = new VertexPose4DoF(pKF);
        }

        if(pKF==pLoopKF)
            V4DoF->setFixed(true);

        V4DoF->setId(nIDi);
        V4DoF->setMarginalized(false);

        optimizer.addVertex(V4DoF);
        vpVertices[nIDi]=V4DoF;
    }
    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    // Edge used in posegraph has still 6Dof, even if updates of camera poses are just in 4DoF
    Eigen::Matrix<double,6,6> matLambda = Eigen::Matrix<double,6,6>::Identity();
    matLambda(0,0) = 1e3;
    matLambda(1,1) = 1e3;
    matLambda(0,0) = 1e3;

    // Set Loop edges
    Edge4DoF* e_loop;
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sij = Siw * Sjw.inverse();
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3) = 1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));

            e->information() = matLambda;
            e_loop = e;
            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // 1. Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Siw;

        // Use noncorrected poses for posegraph edges
        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Siw = iti->second;
        else
            Siw = vScw[nIDi];

        // 1.1.0 Spanning tree edge
        KeyFrame* pParentKF = static_cast<KeyFrame*>(NULL);
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Swj;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj =  vScw[nIDj].inverse();

            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3)=1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.1.1 Inertial edges
        KeyFrame* prevKF = pKF->mPrevKF;
        if(prevKF)
        {
            int nIDj = prevKF->mnId;

            g2o::Sim3 Swj;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(prevKF);

            if(itj!=NonCorrectedSim3.end())
                Swj = (itj->second).inverse();
            else
                Swj =  vScw[nIDj].inverse();

            g2o::Sim3 Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3,3>(0,0) = Sij.rotation().toRotationMatrix();
            Tij.block<3,1>(0,3) = Sij.translation();
            Tij(3,3)=1.;

            Edge4DoF* e = new Edge4DoF(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.2 Loop edges
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Swl;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Swl = itl->second.inverse();
                else
                    Swl = vScw[pLKF->mnId].inverse();

                g2o::Sim3 Sil = Siw * Swl;
                Eigen::Matrix4d Til;
                Til.block<3,3>(0,0) = Sil.rotation().toRotationMatrix();
                Til.block<3,1>(0,3) = Sil.translation();
                Til(3,3) = 1.;

                Edge4DoF* e = new Edge4DoF(Til);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                e->information() = matLambda;
                optimizer.addEdge(e);
            }
        }

        // 1.3 Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && pKFn!=prevKF && pKFn!=pKF->mNextKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Swn;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Swn = itn->second.inverse();
                    else
                        Swn = vScw[pKFn->mnId].inverse();

                    g2o::Sim3 Sin = Siw * Swn;
                    Eigen::Matrix4d Tin;
                    Tin.block<3,3>(0,0) = Sin.rotation().toRotationMatrix();
                    Tin.block<3,1>(0,3) = Sin.translation();
                    Tin(3,3) = 1.;
                    Edge4DoF* e = new Edge4DoF(Tin);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    e->information() = matLambda;
                    optimizer.addEdge(e);
                }
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        VertexPose4DoF* Vi = static_cast<VertexPose4DoF*>(optimizer.vertex(nIDi));
        Eigen::Matrix3d Ri = Vi->estimate().Rcw[0];
        Eigen::Vector3d ti = Vi->estimate().tcw[0];

        g2o::Sim3 CorrectedSiw = g2o::Sim3(Ri,ti,1.);
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();

        Sophus::SE3d Tiw(CorrectedSiw.rotation(),CorrectedSiw.translation());
        pKFi->SetPose(Tiw.cast<float>());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;

        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
        nIDr = pRefKF->mnId;

        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double,3,1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

        pMP->UpdateNormalAndDepth();
    }
    pMap->IncreaseChangeIndex();
}

} //namespace ORB_SLAM
