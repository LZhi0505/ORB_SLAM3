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


#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>
#include "System.h" // liuzhi加

using namespace std;

namespace ORB_SLAM3
{

    const int ORBmatcher::TH_HIGH = 100;
    const int ORBmatcher::TH_LOW = 50;
    const int ORBmatcher::HISTO_LENGTH = 30;

    /**
     * @brief 构造函数
     * @param nnratio   默认值为0.6
     * @param checkOri  true
     */
    ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

    /**
     * @brief 将局部地图中新增的、在当前帧视野范围内的 地图点，投影到当前帧,进行搜索匹配，得到更多的匹配关系
     *        用于Tracking::SearchLocalPoints中匹配更多地图点
     * @param F
     * @param vpMapPoints
     * @param th
     * @param bFarPoints
     * @param thFarPoints
     * @return
     */
    int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
    {
        int nmatches=0, left = 0, right = 0;

        // 如果 th != 1 (RGBD 相机或者刚刚进行过重定位), 需要扩大范围搜索
        const bool bFactor = th!=1.0;
        int count_4 = 0;
        // Step 1：遍历有效的局部地图点
        for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
        {
            MapPoint* pMP = vpMapPoints[iMP];
            // 跳过不在视野范围内的
            if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
                continue;
            // 跳过距离>thFarPoints的远点
            if(bFarPoints && pMP->mTrackDepth > thFarPoints)    // bFarPoints, thFarPoints: 配置yaml文件中未指明，则分别为false, 0.0000
                continue;

            if(pMP->isBad())
                continue;

            // 对当前帧视野范围内的地图点，即mbTrackInView = true进行投影
            if(pMP->mbTrackInView)
            {
                // 通过距离预测的金字塔层数，该层数相对于当前的帧
                const int &nPredictedLevel = pMP->mnTrackScaleLevel;

                // The size of the window will depend on the viewing direction
                // Step 2：设定搜索窗口的大小。取决于视角, 若当前视角和平均视角夹角较小时, r取一个较小的值
                float r = RadiusByViewingCos(pMP->mTrackViewCos);
                if (r == 4.0) count_4++;

                // 如果需要扩大范围搜索，则乘以阈值th
                if(bFactor)
                    r *= th;

                // Step 3：通过投影点以及搜索窗口和预测的尺度进行搜索, 找出搜索半径内的候选匹配点索引
                const vector<size_t> vIndices = F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,      // 该地图点投影到一帧上的坐标
                                                                    r * F.mvScaleFactors[nPredictedLevel],        // 认为搜索窗口的大小和该特征点被追踪到时所处的尺度也有关系
                                                                    nPredictedLevel-1,nPredictedLevel); // 搜索的图层范围

                if(!vIndices.empty()){
                    const cv::Mat MPdescriptor = pMP->GetDescriptor();  // 该局部地图点的 描述子

                    // 最优的次优的描述子距离和index
                    int bestDist=256;
                    int bestLevel= -1;
                    int bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // Get best and second matches with near keypoints
                    // Step 4：寻找候选匹配点中的最佳和次佳匹配点
                    // 遍历该局部地图点 在 当前帧投影位置 周围区域内的 候选匹配特征点
                    for(vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
                    {
                        const size_t idx = *vit;    // 取出一个 候选特征点

                        // 如果当前帧中的该候选特征点已经有对应的地图点了,则退出该次循环
                        if(F.mvpMapPoints[idx])
                            if(F.mvpMapPoints[idx]->Observations() > 0)
                                continue;

                        // 如果是立体匹配双目数据 且 特征点在右目的横坐标
                        if(F.Nleft == -1 && F.mvuRight[idx] > 0)
                        {
                            //计算在X轴上的投影误差
                            const float er = fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
                            // 超过阈值,说明这个点不行,丢掉.
                            // 这里的阈值定义是以给定的搜索范围r为参考,然后考虑到越近的点(nPredictedLevel越大), 相机运动时对其产生的影响也就越大,
                            // 因此需要扩大其搜索空间.
                            // 当给定缩放倍率为1.2的时候, mvScaleFactors 中的数据是: 1 1.2 1.2^2 1.2^3 ...
                            if(er > r * F.mvScaleFactors[nPredictedLevel])
                                continue;
                        }

                        const cv::Mat &d = F.mDescriptors.row(idx); // 当前帧 该特征点 的 描述子

                        // 计算该局部地图点 和 候选特征点的描述子距离
                        const int dist = DescriptorDistance(MPdescriptor,d);

                        // 寻找描述子距离最小和次小的特征点和索引
                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                        : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                          : F.mvKeysRight[idx - F.Nleft].octave;
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                         : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                           : F.mvKeysRight[idx - F.Nleft].octave;
                            bestDist2=dist;
                        }
                    } // 该局部地图点在其投影位置周围区域内，找到最佳距离、次佳距离、最佳距离特征点索引

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    // 最佳匹配距离 < 设定的阈值
                    if(bestDist <= TH_HIGH)
                    {
                        // 条件1：bestLevel == bestLevel2 表示 最佳和次佳在同一金字塔层级
                        // 条件2：bestDist > mfNNratio*bestDist2 表示最佳和次佳距离不满足阈值比例。理论来说 bestDist/bestDist2 越小越好
                        if(bestLevel==bestLevel2 && bestDist > mfNNratio*bestDist2)
                            continue;

                        // 最佳距离 明显比 次佳距离好
                        if(bestLevel!=bestLevel2 || bestDist <= mfNNratio*bestDist2){
                            F.mvpMapPoints[bestIdx] = pMP;      // 将该局部地图点 作为 其匹配最佳距离特征点 的地图点

                            if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                                F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                                nmatches++;
                                right++;
                            }

                            nmatches++;
                            left++;
                        }
                    }
                }
            }
            // 不是 单目、立体匹配双目、RGB-D
            if(F.Nleft != -1 && pMP->mbTrackInViewR){
                const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
                if(nPredictedLevel != -1){
                    float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                    const vector<size_t> vIndices =
                            F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

                    if(vIndices.empty())
                        continue;

                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    int bestDist=256;
                    int bestLevel= -1;
                    int bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // Get best and second matches with near keypoints
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;

                        if(F.mvpMapPoints[idx + F.Nleft])
                            if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                                continue;


                        const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

                        const int dist = DescriptorDistance(MPdescriptor,d);

                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = F.mvKeysRight[idx].octave;
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = F.mvKeysRight[idx].octave;
                            bestDist2=dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if(bestDist<=TH_HIGH)
                    {
                        if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                            continue;

                        if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                            F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                            nmatches++;
                            left++;
                        }


                        F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
                        nmatches++;
                        right++;
                    }
                }
            }// 一个地图点投影匹配完毕
        }// 遍历局部地图点完毕
        std::cout << "\t\t[局部地图跟踪中的 投影匹配] r = 4个数: " << count_4 << std::endl;
        Verbose::PrintMess("\t\t投影匹配点对数: "+std::to_string(nmatches), Verbose::VERBOSITY_DEBUG);
        return nmatches;
    }

    // 根据观察的视角来计算匹配的时的搜索窗口大小
    float ORBmatcher::RadiusByViewingCos(const float &viewCos)
    {
        // 当视角相差小于3.6°，对应cos(3.6°)=0.998，搜索范围是2.5，否则是4
        if(viewCos>0.998)
            return 2.5;
        else
            return 4.0;
    }

    /**
     * @brief 通过词袋，对关键帧的特征点进行跟踪
     *
     * @param[in] pKF               关键帧
     * @param[in] F                 当前帧
     * @param[in,out] vpMapPointMatches F中地图点对应的匹配，NULL表示未匹配
     * @return int                  成功匹配的数量
     *
     * 1. 只匹配同一词袋node的特征点，计算关键帧的 特征点的描述子 与 当前帧 特征点的描述子的 距离，计算最小距离下的特征点
     * 2. 将关键帧 特征点对应的地图点 设为 当前帧最佳匹配特征点的 地图点
     */
    int ORBmatcher::SearchByBoW(KeyFrame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches)
    {
        // 获取该关键帧的 地图点
        const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();  // return mvpMapPoints. key：2D特征点索引，value：对应3D地图点

        // 存储 匹配的地图点 vector，和 当前帧的特征点的 索引一致，初始化为 NULL
        vpMapPointMatches = vector<MapPoint*>(F.N, static_cast<MapPoint*>(NULL));

        // 取出关键帧的 词袋特征向量
        const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec; // 特征向量 mFeatVec，记录 node id（距离叶子节点深度为level up对应的node的Id）及其 对应的图像feature的IDs（该节点下所有叶子节点对应的feature的id）

        int nmatches = 0;

        // 用于统计 特征点角度旋转差 的直方图
        vector<int> rotHist[HISTO_LENGTH];
        for(int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);    // 为每个元素分配500个int类型的空间

        // 将 0 ~ 360 的数转换到 0 ~ HISTO_LENGTH 的系数
        //! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码
        // const float factor = HISTO_LENGTH/360.0f;
        const float factor = 1.0f / HISTO_LENGTH;

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
        // 确保属于同一词袋节点（特定层）的 ORB 特征进行匹配
        DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
        DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
        DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
        DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

        while(KFit != KFend && Fit != Fend)
        {
            // Step 1：分别取出属于同一 node 的 ORB 特征点 ( 只有属于同一 node ，才有可能是匹配点 )
            if(KFit->first == Fit->first)   // first 对应的是 node id；second 对应的是一个vector，存储与图像feature相关联的特征点索引
            {
                const vector<unsigned int> vIndicesKF = KFit->second;   // 属于同一node下的 参考关键帧的 特征点索引集合
                const vector<unsigned int> vIndicesF = Fit->second;     // 属于同一node下的 当前帧的 特征点索引集合

                // 遍历关键帧 KF 属于该node的每个特征点，再用当前帧 F 上相对应node的每个特征点和该特征点进行比对，找到描述子距离最小那个
                // Step 2：遍历关键帧 KF 中属于该 node 的特征点
                for(size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
                {
                    const unsigned int realIdxKF = vIndicesKF[iKF]; // 关键帧KF中属于该node的特征点的 索引

                    MapPoint* pMP = vpMapPointsKF[realIdxKF];   // 关键帧KF中该特征点对应的 地图点

                    if(!pMP)
                        continue;

                    if(pMP->isBad())
                        continue;

                    const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);   // 关键帧KF中该特征点对应的 描述子

                    int bestDist1 = 256;    // 最好的距离（最小距离）
                    int bestIdxF  = -1 ;
                    int bestDist2 = 256;    // 次好距离 （倒数第二小距离）

                    int bestDist1R = 256;
                    int bestIdxFR  = -1 ;
                    int bestDist2R = 256;

                    // Step 3：遍历当前帧F中属于该node的特征点，寻找最佳匹配点
                    for(size_t iF = 0; iF < vIndicesF.size(); iF++)
                    {
                        // 单目、双目立体匹配（未提供Camera2，PinHole不提供）、RGBD
                        if(F.Nleft == -1)
                        {
                            const unsigned int realIdxF = vIndicesF[iF];    // 当前帧F中属于该node的单目或左目特征点的 索引

                            if(vpMapPointMatches[realIdxF]) // 如果地图点已存在，说明这个点已经被匹配过了，不再匹配，加快速度
                                continue;

                            const cv::Mat &dF = F.mDescriptors.row(realIdxF);   // 当前帧F中该特征的 描述子

                            const int dist =  DescriptorDistance(dKF,dF);   // 计算KF与F的描述子之间的 距离

                            // 记录最佳距离、最佳距离对应的 索引、次佳距离
                            // 如果 dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
                            if(dist < bestDist1)
                            {
                                bestDist2 = bestDist1;
                                bestDist1 = dist;
                                bestIdxF  = realIdxF;
                            }
                            // 如果bestDist1 < dist < bestDist2，更新bestDist2
                            else if(dist < bestDist2)
                            {
                                bestDist2 = dist;
                            }
                        }
                        // 双目（提供Camera2，KannalaBrandt8提供）
                        else{
                            const unsigned int realIdxF = vIndicesF[iF];    // 当前帧F中属于该node的特征点的 索引

                            if(vpMapPointMatches[realIdxF])
                                continue;

                            const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                            const int dist =  DescriptorDistance(dKF,dF);

                            // 该特征点属于 左目
                            if(realIdxF < F.Nleft && dist < bestDist1){
                                bestDist2=bestDist1;
                                bestDist1=dist;
                                bestIdxF=realIdxF;
                            }
                            else if(realIdxF < F.Nleft && dist < bestDist2){
                                bestDist2=dist;
                            }

                            // 该特征点属于 右目
                            if(realIdxF >= F.Nleft && dist < bestDist1R){
                                bestDist2R=bestDist1R;
                                bestDist1R=dist;
                                bestIdxFR=realIdxF;
                            }
                            else if(realIdxF >= F.Nleft && dist < bestDist2R){
                                bestDist2R=dist;
                            }
                        }
                    }// 遍历结束，在当前帧中找到 与 该参考关键帧该特征点的 最佳匹配点，获取到最佳距离bestDist1，次佳距离bestDist2，以及 对应的当前帧特征点索引 bestIdxF

                    // 开始剔除外点：首先根据阈值，然后是角度一致性
                    // Step 4：根据阈值 和 角度投票剔除误匹配
                    // Step 4.1：第一关筛选：匹配距离必须小于设定阈值
                    if(bestDist1 <= TH_LOW)
                    {
                        // Step 4.2：第二关筛选：最佳匹配 比 次佳匹配明显要好，那么最佳匹配
                        if(static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                        {
                            // Step 4.3：记录成功匹配特征点的对应的地图点 (来自关键帧KF)
                            vpMapPointMatches[bestIdxF] = pMP;  // 将来自关键帧的地图点 设为 当前帧最佳匹配特征点的 地图点

                            // 这里的 realIdxKF 是当前关键帧的特征点id
                            const cv::KeyPoint &kp =
                                    (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                    (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                : pKF -> mvKeys[realIdxKF];
                            // Step 4.4：计算匹配点 旋转角度差 所在的直方图
                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint &Fkp =
                                        (!pKF->mpCamera2 || F.Nleft == -1) ? F.mvKeys[bestIdxF] :
                                        (bestIdxF >= F.Nleft) ? F.mvKeysRight[bestIdxF - F.Nleft]
                                                              : F.mvKeys[bestIdxF];
                                // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                                float rot = kp.angle - Fkp.angle;
                                if(rot < 0.0)
                                    rot += 360.0f;
                                int bin = round(rot * factor);    // 将 rot 分配到 bin 组 , 四舍五入 , 其实就是离散到对应的直方图组中
                                if(bin == HISTO_LENGTH)
                                    bin = 0;
                                assert(bin >= 0 && bin < HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxF);
                            }
                            nmatches++;
                        }

                        if(bestDist1R <= TH_LOW)
                        {
                            if(static_cast<float>(bestDist1R)<mfNNratio*static_cast<float>(bestDist2R) || true)
                            {
                                vpMapPointMatches[bestIdxFR]=pMP;

                                const cv::KeyPoint &kp =
                                        (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                        (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                    : pKF -> mvKeys[realIdxKF];

                                if(mbCheckOrientation)
                                {
                                    cv::KeyPoint &Fkp =
                                            (!F.mpCamera2) ? F.mvKeys[bestIdxFR] :
                                            (bestIdxFR >= F.Nleft) ? F.mvKeysRight[bestIdxFR - F.Nleft]
                                                                   : F.mvKeys[bestIdxFR];

                                    float rot = kp.angle - Fkp.angle;
                                    if(rot<0.0)
                                        rot += 360.0f;
                                    int bin = round(rot * factor);
                                    if(bin == HISTO_LENGTH)
                                        bin = 0;
                                    std::cout << "rot = " << rot << ", bin = " << bin << std::endl << std::endl;
                                    assert(bin >= 0 && bin < HISTO_LENGTH);
                                    rotHist[bin].push_back(bestIdxFR);
                                }
                                nmatches++;
                            }
                        }
                    }// 内层遍历结束，关键帧的一个特征点在当前帧中找到了最佳匹配特征点

                }// 外层遍历结束，关键帧KF中所有属于该 node的 且拥有有效地图点的特征点 都在 F 找到了与其最小距离的特征点。但只有满足条件的才会使得 nmatches + 1。

                KFit++;
                Fit++;
            }
            else if(KFit->first < Fit->first)
            {
                KFit = vFeatVecKF.lower_bound(Fit->first);  // 对齐
            }
            else
            {
                Fit = F.mFeatVec.lower_bound(KFit->first);  // 对齐
            }
        }// 遍历所有 node 结束
        Verbose::PrintMess("\t\t\t当前帧与参考关键帧匹配的个数 num matches = "+std::to_string(nmatches), Verbose::VERBOSITY_DEBUG);

        // Step 5：根据方向剔除误匹配的点
        if(mbCheckOrientation)
        {
//            for(int bin = 0; bin < HISTO_LENGTH; bin++)
//            {
//                std::cout << "bin " << bin << ": " << rotHist[bin].size() << std::endl;
//            }

            // index
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            // 筛选出 在旋转角度差 落在直方图区间内数量最多的前三个 bin 的索引
            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for(int i = 0; i < HISTO_LENGTH; i++)
            {
                // 如果特征点的旋转角度变化量属于这三个组，则保留
                if(i == ind1 || i == ind2 || i == ind3)
                    continue;

                // 剔除掉不在前三的匹配对，因为他们不符合 “ 主流旋转方向 ”
                for(size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    vpMapPointMatches[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
        Verbose::PrintMess("\t\t\t筛选主流旋转方向后，匹配的个数 num matches: "+std::to_string(nmatches), Verbose::VERBOSITY_DEBUG);

        return nmatches;
    }

    int ORBmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3f &Scw, const vector<MapPoint*> &vpPoints,
                                       vector<MapPoint*> &vpMatched, int th, float ratioHamming)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0)
                continue;

            // Project into Image
            const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
                continue;

            // Depth must be inside the scale invariance region of the point
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist = PO.norm();

            if(dist<minDistance || dist>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;
                if(vpMatched[idx])
                    continue;

                const int &kpLevel= pKF->mvKeysUn[idx].octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW*ratioHamming)
            {
                vpMatched[bestIdx]=pMP;
                nmatches++;
            }

        }

        return nmatches;
    }

    int ORBmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                                       std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];
            KeyFrame* pKFi = vpPointsKFs[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0)
                continue;

            // Project into Image
            const float invz = 1/p3Dc(2);
            const float x = p3Dc(0)*invz;
            const float y = p3Dc(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF->IsInImage(u,v))
                continue;

            // Depth must be inside the scale invariance region of the point
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist = PO.norm();

            if(dist<minDistance || dist>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;
                if(vpMatched[idx])
                    continue;

                const int &kpLevel= pKF->mvKeysUn[idx].octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW*ratioHamming)
            {
                vpMatched[bestIdx] = pMP;
                vpMatchedKF[bestIdx] = pKFi;
                nmatches++;
            }

        }

        return nmatches;
    }

    /**
     * @brief   参考帧 F1 和 当前帧 F2 的特征点匹配
     * @param F1    参考帧
     * @param F2    当前帧
     * @param vbPrevMatched 初始化前 存储的是 参考帧 F1(初始化第一帧)的特征点坐标，匹配后 存储的是 与 F1 匹配好的 F2 特征点坐标
     * @param vnMatches12   参考帧 F1 中特征点与 F2中特征点的匹配关系，匹配上则为 匹配特征点的ID，未匹配上则为-1。key：是 F1 对应特征点ID，value：是匹配好的 F2 的特征点ID
     * @param windowSize    匹配窗口大小
     * @return
     */
    int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
    {
        int nmatches = 0;

        // F1 中特征点和 F2中匹配关系，注意是按照 F1 特征点数目分配空间，初始化为-1
        vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

        // Step 1：构建旋转直方图，HISTO_LENGTH = 30
        vector<int> rotHist[HISTO_LENGTH];

        for(int i = 0; i < HISTO_LENGTH; i++)   // 每个bin里预分配500个，因为使用的是vector不够的话可以自动扩展容量
            rotHist[i].reserve(500);

        const float factor = 1.0f / HISTO_LENGTH;

        // 匹配点对的 距离，注意是按照F2特征点数目分配空间
        vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);

        // 从 F2 到 F1 的反向匹配，注意是按照 F2特征点数目分配空间
        vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

        // 遍历 F1 中的所有去畸变特征点
        for(size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++)
        {
            cv::KeyPoint kp1 = F1.mvKeysUn[i1];
            int level1 = kp1.octave;    // 该特征点所在的金字塔图层数

            // 只使用 原始图像上提取的特征点
            if(level1 > 0)
                continue;

            // Step 2：获取该特征点搜索窗口内的 F2中 所有候选特征点的索引，存储在vIndices2中
            // vbPrevMatched[i] F1 中 特征点i 的坐标
            // windowSize = 100，输入最大最小金字塔层级 均为0
            vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize,level1,level1);

            // 没有候选特征点，跳过
            if(vIndices2.empty())
                continue;

            // 取出该特征点的 描述子
            cv::Mat d1 = F1.mDescriptors.row(i1);

            int bestDist = INT_MAX;     // 该特征点与搜索窗口内的 F2中匹配特征点的 最佳描述子匹配距离，越小越好
            int bestDist2 = INT_MAX;    // 次佳描述子匹配距离
            int bestIdx2 = -1;          // 最佳候选特征点在 F2 中的 index

            // Step 3：遍历搜索窗口中的 所有候选匹配特征点，找到 最优距离和其对应特征点的索引 和 次优距离
            for(vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
            {
                size_t i2 = *vit;   // 在窗口内的F2中 该候选点的索引

                cv::Mat d2 = F2.mDescriptors.row(i2);   // 该候选特征点 的描述子

                int dist = DescriptorDistance(d1,d2);   // 计算两个特征点描述子距离

                if(vMatchedDistance[i2] <= dist)    // 距离过大，舍弃
                    continue;

                // 当前距离 < 最佳距离，更新最佳
                if(dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestIdx2 = i2;
                }
                // 当前距离 < 次佳距离，更新次佳
                else if(dist < bestDist2)
                {
                    bestDist2 = dist;
                }
            }

            // Step 4：对最优、次优结果进行检查，满足阈值、最优/次优比例，删除重复匹配
            // 即使算出了最佳描述子匹配距离，也不一定保证配对成功。要小于设定阈值
            if(bestDist <= TH_LOW)
            {
                // 最优距离 < 次佳距离 * mfNNratio，这样特征点辨识度更高
                if(bestDist < (float)bestDist2 * mfNNratio)
                {
                    // 如果找到的候选特征点 对应 F1中特征点已经匹配过了，说明发生了重复匹配，将原来的匹配删掉，更新为最新的匹配
                    if(vnMatches21[bestIdx2] >= 0)
                    {
                        vnMatches12[vnMatches21[bestIdx2]] = -1;
                        nmatches--;
                    }

                    // 次优的匹配关系，双向建立
                    // vnMatches12 保存 参考帧F1 和 F2匹配关系，index 保存是F1对应特征点索引，value 保存的是匹配的F2特征点索引
                    // vnMatches21 与之相反
                    vnMatches12[i1] = bestIdx2;
                    vnMatches21[bestIdx2] = i1;
                    vMatchedDistance[bestIdx2] = bestDist;
                    nmatches++;

                    // Step 5：计算匹配点旋转角度差所在的直方图
                    if(mbCheckOrientation)
                    {
                        // 计算匹配特征点的角度差，这里单位是角度°，不是弧度
                        float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;
                        if(rot < 0.0)
                            rot += 360.0f;

                        // 前面factor = HISTO_LENGTH / 360.0f
                        // bin = rot / 360.of * HISTO_LENGTH 表示当前rot被分配在第几个直方图bin
                        int bin = round(rot * factor);

                        // 如果bin 满了又是一个轮回
                        if(bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(i1);
                    }
                }
            } // 该特征点匹配结束
        } // F1中所有特征点匹配结束，可能在F2中找到了匹配的特征点

        // Step 6：筛除旋转直方图中“非主流”部分
        if(mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;
            // 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin（主流方向）的索引
            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            // 遍历直方图的每个bin
            for(int i = 0; i < HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                // 剔除掉不在前三的匹配对，因为他们不符合“主流旋转方向”
                for(size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    int idx1 = rotHist[i][j];
                    if(vnMatches12[idx1] >= 0)
                    {
                        vnMatches12[idx1] = -1;
                        nmatches--;
                    }
                }
            }
        }

        //Update prev matched
        // Step 7：将最后通过筛选的 匹配好的特征点保存到 vbPrevMatched
        for(size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
            if(vnMatches12[i1] >= 0)
                vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;

        return nmatches;
    }

    int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
    {
        const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        const cv::Mat &Descriptors1 = pKF1->mDescriptors;

        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        const cv::Mat &Descriptors2 = pKF2->mDescriptors;

        vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
        vector<bool> vbMatched2(vpMapPoints2.size(),false);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        int nmatches = 0;

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                        continue;
                    }

                    MapPoint* pMP1 = vpMapPoints1[idx1];
                    if(!pMP1)
                        continue;
                    if(pMP1->isBad())
                        continue;

                    const cv::Mat &d1 = Descriptors1.row(idx1);

                    int bestDist1=256;
                    int bestIdx2 =-1 ;
                    int bestDist2=256;

                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                            continue;
                        }

                        MapPoint* pMP2 = vpMapPoints2[idx2];

                        if(vbMatched2[idx2] || !pMP2)
                            continue;

                        if(pMP2->isBad())
                            continue;

                        const cv::Mat &d2 = Descriptors2.row(idx2);

                        int dist = DescriptorDistance(d1,d2);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }

                    if(bestDist1<TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2]=true;

                            if(mbCheckOrientation)
                            {
                                float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

    /**
     * @brief 通过极线约束的方式找到还没有成为 地图点的 匹配点
     * @param pKF1  当前关键帧
     * @param pKF2  相邻关键帧
     * @param vMatchedPairs 当前关键帧和相邻关键帧2的匹配特征点对关系。key：当前关键帧匹配点id，value：相邻关键帧的匹配点id
     * @param bOnlyStereo   为 false
     * @param bCoarse   非IMU模式为 false
     * @return
     */
    int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
    {
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        //Compute epipole in second image
        // Step 1；计算KF1的相机中心在KF2图像平面的二维像素坐标
        Sophus::SE3f T1w = pKF1->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();
        Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
        Eigen::Vector3f Cw = pKF1->GetCameraCenter();
        Eigen::Vector3f C2 = T2w * Cw;  // KF1的相机中心在KF2相机平面的三维坐标
        Eigen::Vector2f ep = pKF2->mpCamera->project(C2);   // KF1的相机中心在KF2图像平面的二维坐标
        std::cout << "\t\t相机1的光心坐标" << Cw(0) << ", " << Cw(1) << ", " << Cw(2) <<
            ", 在相机2的三维坐标" << C2(0) << ", " << C2(1) << ", " << C2(2) << ", 在相机2的投影坐标" << ep(0)<< ", " << ep(1)<< std::endl;

        Sophus::SE3f T12;
        Sophus::SE3f Tll, Tlr, Trl, Trr;
        Eigen::Matrix3f R12; // for fastest computation
        Eigen::Vector3f t12; // for fastest computation

        GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;
        // 立体匹配双目
        if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
            T12 = T1w * Tw2;    // KF2到KF1的相对变换
            R12 = T12.rotationMatrix();
            t12 = T12.translation();
        }
        // 非立体匹配双目
        else{
            Sophus::SE3f Tr1w = pKF1->GetRightPose();
            Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
            Tll = T1w * Tw2;
            Tlr = T1w * Twr2;
            Trl = Tr1w * Tw2;
            Trr = Tr1w * Twr2;
        }

        Eigen::Matrix3f Rll = Tll.rotationMatrix(), Rlr  = Tlr.rotationMatrix(), Rrl  = Trl.rotationMatrix(), Rrr  = Trr.rotationMatrix();
        Eigen::Vector3f tll = Tll.translation(), tlr = Tlr.translation(), trl = Trl.translation(), trr = Trr.translation();

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node
        int nmatches = 0;
        // 记录匹配是否成功，避免重复匹配
        vector<bool> vbMatched2(pKF2->N,false);
        vector<int> vMatches12(pKF1->N,-1);
        // 用于统计匹配点对旋转差的直方图
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;
        //! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码
        // const float factor = HISTO_LENGTH/360.0f;

        // Step 2：利用BoW加速匹配：只对属于同一节点(特定层)的ORB特征进行匹配
        // FeatureVector其实就是一个map类，那就可以直接获取它的迭代器进行遍历
        // FeatureVector的数据结构类似于：{(node1,feature_vector1) (node2,feature_vector2)...}
        // f1it->first对应node id，f1it->second对应属于该node的所有特征点id
        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        // Step 2.1：遍历pKF1和pKF2中的node节点
        while(f1it != f1end && f2it != f2end)
        {
            // 如果f1it和f2it属于同一个node节点才会进行匹配，这就是BoW加速匹配原理
            if(f1it->first == f2it->first)
            {
                // Step 2.2：遍历关键帧1同一node节点下的所有特征点
                for(size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];   // 关键帧1的特征点索引 idx1

                    MapPoint* pMP1 = pKF1->GetMapPoint(idx1);   // 关键帧1 该特征点对应的 地图点

                    // If there is already a MapPoint skip。由于寻找的是未匹配的特征点，所以已经有地图点则跳过
                    if(pMP1)
                    {
                        continue;
                    }
                    // 如果mvuRight中的值大于0，表示是双目，且为立体匹配双目，该特征点有深度值
                    const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1] >= 0);

                    if(bOnlyStereo)
                        if(!bStereo1)
                            continue;

                    // Step 2.4：通过特征点索引idx1在pKF1中取出对应的 特征点
                    const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                    : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                             : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                    const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                       : true;

                    const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);   // 关键帧1 特征点idx1对应的 描述子

                    int bestDist = TH_LOW;
                    int bestIdx2 = -1;
                    // Step 2.5：遍历该node节点下 关键帧2中的所有特征点
                    for(size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                    {
                        size_t idx2 = f2it->second[i2]; // 关键帧2的特征点索引idx2

                        MapPoint* pMP2 = pKF2->GetMapPoint(idx2);// 关键帧2的特征点索引idx2 对应的地图点

                        // If we have already matched or there is a MapPoint skip。如果pKF2当前特征点索引idx2 已经被匹配过 || 对应的3d点非空，那么跳过
                        if(vbMatched2[idx2] || pMP2)
                            continue;

                        const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2] >= 0);

                        if(bOnlyStereo)
                            if(!bStereo2)
                                continue;

                        const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);   // 关键帧2的特征点索引idx2 对应的描述子

                        // Step 2.6 计算idx1与idx2在两个关键帧中对应特征点的描述子距离
                        const int dist = DescriptorDistance(d1,d2);

                        if(dist > TH_LOW || dist > bestDist)
                            continue;

                        // 通过特征点索引idx2在pKF2中取出对应的特征点
                        const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                        : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                 : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                        const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                           : true;

                        //? 为什么双目就不需要判断像素点到极点的距离的判断？
                        // 因为双目模式下可以左右互匹配恢复三维点
                        // 立体匹配双目
                        if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                        {
                            const float distex = ep(0) - kp2.pt.x;
                            const float distey = ep(1) - kp2.pt.y;
//                            std::cout << "\t\t相机1在相机2的投影坐标" << ep(0)<< ", " << ep(1)<< ", " << distex << ", " << distey << std::endl;
                            // Step 2.7 极点e2到kp2的像素距离 < 阈值th, 认为kp2对应的MapPoint距离pKF1相机太近，跳过该匹配点对
                            // 作者根据kp2金字塔尺度因子(scale^n，scale=1.2，n为层数)定义阈值th
                            // 金字塔层数从0到7，对应距离 sqrt(100 * pKF2->mvScaleFactors[kp2.octave]) 是10-20个像素
                            //? 对这个阈值的有效性持怀疑态度
                            if(distex * distex + distey * distey < 100 * pKF2->mvScaleFactors[kp2.octave])
                            {
                                continue;
                            }
                        }

                        if(pKF1->mpCamera2 && pKF2->mpCamera2){
                            if(bRight1 && bRight2){
                                R12 = Rrr;
                                t12 = trr;
                                T12 = Trr;

                                pCamera1 = pKF1->mpCamera2;
                                pCamera2 = pKF2->mpCamera2;
                            }
                            else if(bRight1 && !bRight2){
                                R12 = Rrl;
                                t12 = trl;
                                T12 = Trl;

                                pCamera1 = pKF1->mpCamera2;
                                pCamera2 = pKF2->mpCamera;
                            }
                            else if(!bRight1 && bRight2){
                                R12 = Rlr;
                                t12 = tlr;
                                T12 = Tlr;

                                pCamera1 = pKF1->mpCamera;
                                pCamera2 = pKF2->mpCamera2;
                            }
                            else{
                                R12 = Rll;
                                t12 = tll;
                                T12 = Tll;

                                pCamera1 = pKF1->mpCamera;
                                pCamera2 = pKF2->mpCamera;
                            }

                        }

                        // Step 2.8 计算特征点kp2到kp1对应极线的距离是否小于阈值
                        if(bCoarse || pCamera1->epipolarConstrain(pCamera2, kp1, kp2, R12, t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])) // MODIFICATION_2
                        {
                            // bestIdx2，bestDist 是 kp1 对应 KF2中的最佳匹配点 index及匹配距离
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }// 关键帧1的一个特征点 遍历完毕关键帧2属于同该node节点的特征点

                    if(bestIdx2 >= 0)
                    {
                        const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                        : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                     : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                        // 记录匹配结果
                        vMatches12[idx1] = bestIdx2;
                        // !记录已经匹配，避免重复匹配。原作者漏掉！可以添加下面代码
                         vbMatched2[bestIdx2] = true;
                        nmatches++;

                        // 记录旋转差直方图信息
                        if(mbCheckOrientation)
                        {
                            // angle：角度，表示匹配点对的方向差。
                            float rot = kp1.angle-kp2.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }// 关键帧1同属于这个node节点的特征点遍历完毕

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }// 所有遍历结束

        // Step 3：用旋转差直方图来筛掉错误匹配对
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vMatches12[rotHist[i][j]]=-1;
                    nmatches--;
                }
            }
        }

        // Step 4：存储匹配关系，下标是关键帧1的特征点id，存储的是关键帧2的特征点id
        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for(size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            if(vMatches12[i] < 0)
                continue;
            vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
        }

        return nmatches;
    }

    /**
     *
     * @param pKF   相邻关键帧
     * @param vpMapPoints   当前关键帧地图点
     * @param th    默认为3
     * @param bRight    默认为false
     * @return
     */
    int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
    {
        GeometricCamera* pCamera;
        Sophus::SE3f Tcw;
        Eigen::Vector3f Ow;

        if(bRight){
            Tcw = pKF->GetRightPose();
            Ow = pKF->GetRightCameraCenter();
            pCamera = pKF->mpCamera2;
        }
        else{
            Tcw = pKF->GetPose();
            Ow = pKF->GetCameraCenter();
            pCamera = pKF->mpCamera;
        }

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        int nFused = 0;

        const int nMPs = vpMapPoints.size();

        // For debbuging
        int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;
        // 遍历当前关键帧的地图点
        for(int i = 0; i < nMPs; i++)
        {
            MapPoint* pMP = vpMapPoints[i];

            if(!pMP)
            {
                count_notMP++;
                continue;
            }

            if(pMP->isBad())
            {
                count_bad++;
                continue;
            }
            else if(pMP->IsInKeyFrame(pKF))
            {
                count_isinKF++;
                continue;
            }

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();  // 地图点的世界坐标
            Eigen::Vector3f p3Dc = Tcw * p3Dw;          // 地图点在该相邻关键帧坐标系下的三维坐标

            // Depth must be positive
            if(p3Dc(2) < 0.0f)
            {
                count_negdepth++;
                continue;
            }

            const float invz = 1 / p3Dc(2);

            const Eigen::Vector2f uv = pCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
            {
                count_notinim++;
                continue;
            }

            const float ur = uv(0) - bf * invz;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw - Ow; // 相邻关键帧相机光心 到 该地图点 的向量
            const float dist3D = PO.norm();

            // Depth must be inside the scale pyramid of the image
            if(dist3D < minDistance || dist3D > maxDistance) {
                count_dist++;
                continue;
            }

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist3D)
            {
                count_normal++;
                continue;
            }

            int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

            // Search in a radius
            const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius,bRight);

            if(vIndices.empty())
            {
                count_notidx++;
                continue;
            }

            // Match to the most similar keypoint in the radius

            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                size_t idx = *vit;
                const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                              : (!bRight) ? pKF -> mvKeys[idx]
                                                                          : pKF -> mvKeysRight[idx];

                const int &kpLevel= kp.octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                if(pKF->mvuRight[idx]>=0)
                {
                    // Check reprojection error in stereo
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float &kpr = pKF->mvuRight[idx];
                    const float ex = uv(0)-kpx;
                    const float ey = uv(1)-kpy;
                    const float er = ur-kpr;
                    const float e2 = ex*ex+ey*ey+er*er;

                    if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                        continue;
                }
                else
                {
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float ex = uv(0)-kpx;
                    const float ey = uv(1)-kpy;
                    const float e2 = ex*ex+ey*ey;

                    if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                        continue;
                }

                if(bRight) idx += pKF->NLeft;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if(bestDist<=TH_LOW)
            {
                MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if(pMPinKF)
                {
                    if(!pMPinKF->isBad())
                    {
                        if(pMPinKF->Observations()>pMP->Observations())
                            pMP->Replace(pMPinKF);
                        else
                            pMPinKF->Replace(pMP);
                    }
                }
                else
                {
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP,bestIdx);
                }
                nFused++;
            }
            else
                count_thcheck++;

        }

        return nFused;
    }

    int ORBmatcher::Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

        int nFused=0;

        const int nPoints = vpPoints.size();

        // For each candidate MapPoint project and match
        for(int iMP=0; iMP<nPoints; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0f)
                continue;

            // Project into Image
            const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
                continue;

            // Depth must be inside the scale pyramid of the image
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist3D = PO.norm();

            if(dist3D<minDistance || dist3D>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist3D)
                continue;

            // Compute predicted scale level
            const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius

            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
            {
                const size_t idx = *vit;
                const int &kpLevel = pKF->mvKeysUn[idx].octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if(bestDist<=TH_LOW)
            {
                MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if(pMPinKF)
                {
                    if(!pMPinKF->isBad())
                        vpReplacePoint[iMP] = pMPinKF;
                }
                else
                {
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP,bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

    int ORBmatcher::SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th)
    {
        const float &fx = pKF1->fx;
        const float &fy = pKF1->fy;
        const float &cx = pKF1->cx;
        const float &cy = pKF1->cy;

        // Camera 1 & 2 from world
        Sophus::SE3f T1w = pKF1->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();

        //Transformation between cameras
        Sophus::Sim3f S21 = S12.inverse();

        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        const int N1 = vpMapPoints1.size();

        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        const int N2 = vpMapPoints2.size();

        vector<bool> vbAlreadyMatched1(N1,false);
        vector<bool> vbAlreadyMatched2(N2,false);

        for(int i=0; i<N1; i++)
        {
            MapPoint* pMP = vpMatches12[i];
            if(pMP)
            {
                vbAlreadyMatched1[i]=true;
                int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
                if(idx2>=0 && idx2<N2)
                    vbAlreadyMatched2[idx2]=true;
            }
        }

        vector<int> vnMatch1(N1,-1);
        vector<int> vnMatch2(N2,-1);

        // Transform from KF1 to KF2 and search
        for(int i1=0; i1<N1; i1++)
        {
            MapPoint* pMP = vpMapPoints1[i1];

            if(!pMP || vbAlreadyMatched1[i1])
                continue;

            if(pMP->isBad())
                continue;

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc1 = T1w * p3Dw;
            Eigen::Vector3f p3Dc2 = S21 * p3Dc1;

            // Depth must be positive
            if(p3Dc2(2)<0.0)
                continue;

            const float invz = 1.0/p3Dc2(2);
            const float x = p3Dc2(0)*invz;
            const float y = p3Dc2(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF2->IsInImage(u,v))
                continue;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const float dist3D = p3Dc2.norm();

            // Depth must be inside the scale invariance region
            if(dist3D<minDistance || dist3D>maxDistance )
                continue;

            // Compute predicted octave
            const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

            // Search in a radius
            const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

                if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_HIGH)
            {
                vnMatch1[i1]=bestIdx;
            }
        }

        // Transform from KF2 to KF2 and search
        for(int i2=0; i2<N2; i2++)
        {
            MapPoint* pMP = vpMapPoints2[i2];

            if(!pMP || vbAlreadyMatched2[i2])
                continue;

            if(pMP->isBad())
                continue;

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc2 = T2w * p3Dw;
            Eigen::Vector3f p3Dc1 = S12 * p3Dc2;

            // Depth must be positive
            if(p3Dc1(2)<0.0)
                continue;

            const float invz = 1.0/p3Dc1(2);
            const float x = p3Dc1(0)*invz;
            const float y = p3Dc1(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF1->IsInImage(u,v))
                continue;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const float dist3D = p3Dc1.norm();

            // Depth must be inside the scale pyramid of the image
            if(dist3D<minDistance || dist3D>maxDistance)
                continue;

            // Compute predicted octave
            const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

            // Search in a radius of 2.5*sigma(ScaleLevel)
            const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

                if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_HIGH)
            {
                vnMatch2[i2]=bestIdx;
            }
        }

        // Check agreement
        int nFound = 0;

        for(int i1=0; i1<N1; i1++)
        {
            int idx2 = vnMatch1[i1];

            if(idx2>=0)
            {
                int idx1 = vnMatch2[idx2];
                if(idx1==i1)
                {
                    vpMatches12[i1] = vpMapPoints2[idx2];
                    nFound++;
                }
            }
        }

        return nFound;
    }

    /**
     * @brief 将 上一帧跟踪的地图点投影到当前帧，并且搜索匹配特征点。
     * 步骤
     * Step 1 建立旋转直方图，用于检测旋转一致性
     * Step 2 计算当前帧和前一帧的平移向量
     * Step 3 对于前一帧的每一个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
     * Step 4 根据相机的前后前进方向来判断搜索尺度范围
     * Step 5 遍历候选匹配 特征点，寻找距离最小的最佳匹配特征点
     * Step 6 计算匹配点旋转角度差所在的直方图
     * Step 7 进行旋转一致检测，剔除不一致的匹配特征点
     * @param[in] CurrentFrame          当前帧
     * @param[in] LastFrame             上一帧
     * @param[in] th                    搜索范围阈值，默认单目为7，双目15
     * @param[in] bMono                 是否为单目
     * @return int                      成功匹配特征点的数量
     */
    // 利用投影找到当前帧与上一帧特征点的匹配关系
    // 根据当前帧位姿的初始值，把上一帧的 3D 点投影到当前帧上，然后在这个投影点所在的格子里依次比较所有特征点与该 3D 点描述子的距离，选择最小的那个。
    // 然后比较两者的朝向的夹角，并放到对应的直方图里。
    // 所有匹配关系都找到后，它们的朝向应该是一致的，所以把直方图排名前三的方向的点找出来，这是最后保留的匹配结果。
    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
    {
        int nmatches = 0;
        int nmp = 0;        // liuzhi加，记录上一帧特征点对应有地图点的个数
        int nmpvalid = 0;   // liuzhi加，记录上一帧有效的地图点个数

        // Rotation Histogram (to check rotation consistency)
        // Step 1：建立旋转直方图，用于检测旋转一致性
        vector<int> rotHist[HISTO_LENGTH];  // HISTO_LENGTH = 30
        for(int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);    // 为每个数组元素分配了500个int类型的内存空间

        const float factor = 1.0f / HISTO_LENGTH;

        // Step 2：计算当前帧和前一帧的平移向量
        const Sophus::SE3f Tcw = CurrentFrame.GetPose();    // 当前帧的相机位姿
        const Eigen::Vector3f twc = Tcw.inverse().translation();

        const Sophus::SE3f Tlw = LastFrame.GetPose();   // 上一帧的相机位姿
        const Eigen::Vector3f tlc = Tlw * twc;          // 当前帧 -> 上一帧的平移向量

        // 判断前进还是后退（近大远小）（尺度越大，图像越小)
        const bool bForward = tlc(2) > CurrentFrame.mb && !bMono;   // 非单目 且 前进：z > 基线b，物体在当前帧的图像上变大，因此对于上一帧的特征点，需要在当前帧更高的尺度上搜索，在当前帧的尺寸小的金字塔图像层上搜索
        const bool bBackward = -tlc(2) > CurrentFrame.mb && !bMono; // 非单目 且 后退：z < -b，物体在当前帧的图像上变小，因此对于上一帧的特征点，需要在当前帧更低的尺度上搜索

        //  Step 3：遍历上一帧的每个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
        for(int i = 0; i < LastFrame.N; i++)
        {
            MapPoint* pMP = LastFrame.mvpMapPoints[i];  // 上一帧的特征点i对应的 地图点

            // 对应的地图点存在
            if(pMP)
            {
                nmp++;  // liuzhi加
                // 该特征点不是外点
                if(!LastFrame.mvbOutlier[i])
                {
                    nmpvalid++;     // liuzhi加
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();  // 上一帧特征点i对应的地图点的 世界坐标
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;  // 上一帧特征点i对应的地图点 在当前帧的相机坐标系下的 三维坐标

                    const float xc = x3Dc(0);
                    const float yc = x3Dc(1);
                    const float invzc = 1.0 / x3Dc(2);

                    if(invzc < 0)
                        continue;

                    // 将上一帧有效的地图点投影到 当前帧的 像素坐标系
                    Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    // 跳过超出图像边界的
                    if(uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX)
                        continue;
                    if(uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
                        continue;

                    // 认为投影前后地图点的尺度信息不变
                    // cv::KeyPoint::octave中表示的是从金字塔的哪一层提取的数据
                    int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                     : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                    // Search in a window. Size depends on scale
                    // 仅双目：th = 7，单目、RGB-D、IMU+单目、IMU+双目、IMU+RGBD：th = 15
                    float radius = th * CurrentFrame.mvScaleFactors[nLastOctave]; // 搜索半径 = th * 上一帧地图点所在层级的scale (尺度越大，搜索范围越大)

                    // 记录候选匹配的 当前帧特征点的 ID
                    vector<size_t> vIndices2;

                    // Step 4：根据相机的前后前进方向来判断搜索尺度范围
                    // 前进 (在更高尺度搜索)，则当前帧搜索尺度：nCurOctave >= nLastOctave
                    if(bForward)
                        // 根据上一帧地图点的投影坐标(uv(0), uv(1))，返回周围半径为radius的区域内所有当前帧的 特征点索引
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave);
                    // 后退 (在更低尺度搜索)，则当前帧搜索尺度： 0 <= nCurOctave <= nLastOctave
                    else if(bBackward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave);
                    // 在相邻尺度搜索，在[nLastOctave - 1, nLastOctave + 1]中搜索
                    else
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave - 1, nLastOctave + 1);

                    if(vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();   // 获取上一帧 地图点的 描述子：该地图点在观测到它的关键帧中，各描述子的中位数

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    // Step 5:遍历当前帧 候选匹配的 特征点，寻找距离最小的最佳匹配特征点
                    for(vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                    {
                        const size_t i2 = *vit;

                        // 如果该候选特征点已经有对应的地图点了，且 其被观测次数 > 0，则跳过
                        if(CurrentFrame.mvpMapPoints[i2])
                            if(CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
                                continue;

                        // PinHole双目和RGBD，需保证右目的特征点也在搜索半径以内
                        if(CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2] > 0)
                        {

                            const float ur = uv(0) - CurrentFrame.mbf * invzc;
                            const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                            if(er > radius)
                                continue;
                        }

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);   // 该候选特征点的 描述子

                        const int dist = DescriptorDistance(dMP,d);

                        // 更新最佳距离 和 最佳索引
                        if(dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }// 当前帧中，在上一帧的某个地图点的投影位置 周围区域内，找到与该地图点 拥有最小距离的 特征点索引和最小距离

                    // 最佳匹配距离 < 阈值
                    if(bestDist <= TH_HIGH)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2] = pMP;  // 将该地图点设为当前帧最佳匹配特征点 对应的 地图点
                        nmatches++;

                        // Step 6：计算匹配点旋转角度差所在的直方图
                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                        : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                            cv::KeyPoint kpCF = (CurrentFrame.Nleft == -1) ? CurrentFrame.mvKeysUn[bestIdx2]
                                                                           : (bestIdx2 < CurrentFrame.Nleft) ? CurrentFrame.mvKeys[bestIdx2]
                                                                                                             : CurrentFrame.mvKeysRight[bestIdx2 - CurrentFrame.Nleft];
                            float rot = kpLF.angle - kpCF.angle;
                            if(rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if(bin == HISTO_LENGTH)
                                bin = 0;
//                            std::cout << "上一帧特征点 " << i << " 的坐标: (" << kpLF.pt.x << ", " << kpLF.pt.y << "), 当前帧特征点 " << bestIdx2 << " 的坐标: (" << kpCF.pt.x << ", " << kpCF.pt.y << ")的rot = " << rot << ", bin = " << bin << std::endl;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                    // KB鱼眼相机
                    if(CurrentFrame.Nleft != -1) {
                        Eigen::Vector3f x3Dr = CurrentFrame.GetRelativePoseTrl() * x3Dc;
                        Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dr);

                        int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                             : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                        // Search in a window. Size depends on scale
                        float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                        vector<size_t> vIndices2;

                        if(bForward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave, -1,true);
                        else if(bBackward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave, true);
                        else
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1, true);

                        const cv::Mat dMP = pMP->GetDescriptor();

                        int bestDist = 256;
                        int bestIdx2 = -1;

                        for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                        {
                            const size_t i2 = *vit;
                            if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                                if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations()>0)
                                    continue;

                            const cv::Mat &d = CurrentFrame.mDescriptors.row(i2 + CurrentFrame.Nleft);

                            const int dist = DescriptorDistance(dMP,d);

                            if(dist<bestDist)
                            {
                                bestDist=dist;
                                bestIdx2=i2;
                            }
                        }

                        if(bestDist<=TH_HIGH)
                        {
                            CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft]=pMP;
                            nmatches++;
                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                            : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                    : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                                cv::KeyPoint kpCF = CurrentFrame.mvKeysRight[bestIdx2];

                                float rot = kpLF.angle-kpCF.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdx2  + CurrentFrame.Nleft);
                            }
                        }

                    }// 结束 在当前帧中，找与上一帧地图点匹配的特征点
                }
                // 该地图点是外点
            }
            // 该特征点没有对应的地图点
        }// 遍历完上一帧的所有地图点，每个有效的地图点都在当前帧匹配到了一个特征点
        Verbose::PrintMess("\t\t\t上一帧特征点对应有地图点的个数：" + std::to_string(nmp) + "，其中内点个数：" + std::to_string(nmpvalid) + "，当前帧与上一帧地图点匹配的个数 nmatches = " + to_string(nmatches), Verbose::VERBOSITY_DEBUG);

        // Apply rotation consistency
        // Step 7：进行旋转一致检测，剔除不一致的匹配
        if(mbCheckOrientation)
        {
//            for(int bin = 0; bin < HISTO_LENGTH; bin++)
//            {
//                std::cout << "bin " << bin << ": " << rotHist[bin].size() << std::endl;
//            }

            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            // 检查个数最多的前三个方向，且当第二第三个方向的个数 < 0.1 * 第一个方向的个数则舍弃
            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
            // 遍历所有bin
            for(int i = 0; i < HISTO_LENGTH; i++)
            {
                // 剔除旋转方向不是前3多的特征点对
                if(i != ind1 && i != ind2 && i != ind3)
                {
                    for(size_t j = 0, jend = rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }
        }
        Verbose::PrintMess("\t\t\t筛选主流旋转方向后，匹配的个数 nmatches = " + to_string(nmatches), Verbose::VERBOSITY_DEBUG);

        return nmatches;
    }

    /**
     * @brief （重定位中第二次匹配使用）通过投影的方式将关键帧中未匹配的地图点投影到当前帧中,进行匹配，并通过旋转直方图进行筛选
     *
     * @param[in] CurrentFrame          当前帧
     * @param[in] pKF                   参考关键帧
     * @param[in] sAlreadyFound         已经找到的地图点集合，不会用于PNP
     * @param[in] th                    匹配时搜索范围，会乘以金字塔尺度
     * @param[in] ORBdist               匹配的ORB描述子距离应该小于这个阈值
     * @return int                      成功匹配的数量
     */
    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
    {
        int nmatches = 0;

        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMPs[i];

            if(pMP)
            {
                if(!pMP->isBad() && !sAlreadyFound.count(pMP))
                {
                    //Project
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;

                    const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    if(uv(0)<CurrentFrame.mnMinX || uv(0)>CurrentFrame.mnMaxX)
                        continue;
                    if(uv(1)<CurrentFrame.mnMinY || uv(1)>CurrentFrame.mnMaxY)
                        continue;

                    // Compute predicted scale level
                    Eigen::Vector3f PO = x3Dw-Ow;
                    float dist3D = PO.norm();

                    const float maxDistance = pMP->GetMaxDistanceInvariance();
                    const float minDistance = pMP->GetMinDistanceInvariance();

                    // Depth must be inside the scale pyramid of the image
                    if(dist3D<minDistance || dist3D>maxDistance)
                        continue;

                    int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                    // Search in a window
                    const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                    const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, nPredictedLevel-1, nPredictedLevel+1);

                    if(vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                    {
                        const size_t i2 = *vit;
                        if(CurrentFrame.mvpMapPoints[i2])
                            continue;

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        const int dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=ORBdist)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }

                }
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }

    /**
     * @brief 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
     *
     * @param[in] histo         匹配特征点对旋转方向差直方图
     * @param[in] L             直方图尺寸
     * @param[in & out] ind1          bin值第一大对应的索引
     * @param[in & out] ind2          bin值第二大对应的索引
     * @param[in & out] ind3          bin值第三大对应的索引
     */
    void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1 = 0;   // 第一多的个数
        int max2 = 0;
        int max3 = 0;
        // 遍历所有的直方图索引 bin (bin的value存储着 匹配的特征点索引)
        for(int i = 0; i < L; i++)
        {
            const int s = histo[i].size();  // 该bin下匹配特征点的个数
            if(s > max1)
            {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            }
            else if(s > max2)
            {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            }
            else if(s > max3)
            {
                max3 = s;
                ind3 = i;
            }
        }

        // 如果个数距离max1太大了, 说明次优方向非常不好, 直接放弃, 都置为-1
        if(max2 < 0.1f * (float)max1)
        {
            ind2 = -1;
            ind3 = -1;
        }
        else if(max3 < 0.1f * (float)max1)
        {
            ind3 = -1;
        }
    }


    // Bit set count operation from http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    // Hamming distance：两个二进制串之间的汉明距离，指的是其不同位数的个数
    int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        // 8 * 32 = 256bit
        for(int i = 0; i < 8; i++, pa++, pb++)
        {
            unsigned  int v = *pa ^ *pb;    // 相等为0，不等为1
            // 计算v中 bit为1的个数
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

} //namespace ORB_SLAM
