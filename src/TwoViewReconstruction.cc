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

#include "TwoViewReconstruction.h"

#include "Converter.h"
#include "GeometricTools.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include<thread>

#include "Tracking.h"


using namespace std;
namespace ORB_SLAM3
{
    TwoViewReconstruction::TwoViewReconstruction(const Eigen::Matrix3f& k, float sigma, int iterations)
    {
        mK = k;

        mSigma = sigma;
        mSigma2 = sigma*sigma;
        mMaxIterations = iterations;
    }

    /**
     * @brief 单目初始化重要的环节，先获得当前帧相机位姿R t，再通过三角化恢复3D点坐标
     * @param[in] vKeys1    第一帧的特征点
     * @param[in] vKeys2    第二帧的特征点
     * @param[in] vMatches12    两帧之间特征点的匹配关系，index 保存的是 vKeys1 中特征点索引；如果匹配上，value 保存的是 匹配上的 vKeys2 的特征点索引，未匹配上则为-1
     * @param[in,out] T21       相机从位置1到位置2的位姿的 变换矩阵
     * @param[in,out] vP3D      恢复出的三维点
     * @param[in,out] vbTriangulated    匹配点是否可以被三角化成功
     * @return true         初始化成功
     * @return false
     */
    bool TwoViewReconstruction::Reconstruct(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2, const vector<int> &vMatches12,
                                             Sophus::SE3f &T21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
    {
        // Step 1. 准备工作，筛选出成功匹配的特征点对的索引
        mvKeys1.clear();
        mvKeys2.clear();

        mvKeys1 = vKeys1;
        mvKeys2 = vKeys2;

        // Fill structures with current keypoints and matches with reference frame
        // 参考帧 Reference Frame: 1, 当前帧 Current Frame: 2

        // mvMatches12：只存储 vKeys1，vKeys2 成功匹配的 特征点对的 索引
        mvMatches12.clear();
        mvMatches12.reserve(mvKeys2.size());

        // mvbMatched1：记录所有vKeys1中的特征点 与 vKeys2中点的匹配关系，若匹配上，则置 true；否则置 false
        mvbMatched1.resize(mvKeys1.size());

        // 遍历 vMatches12，筛选出成功匹配的特征点对的索引
        for(size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            if(vMatches12[i] >= 0)
            {
                mvMatches12.push_back(make_pair(i,vMatches12[i]));
                mvbMatched1[i] = true;
            }
            else
                mvbMatched1[i] = false;
        }

        const int N = mvMatches12.size();   // 匹配好的 点对数

        // Indices for minimum set selection
        vector<size_t> vAllIndices;
        vAllIndices.reserve(N);
        vector<size_t> vAvailableIndices;

        // 使用 vAllIndices 为了保证8个点选不到同一个点
        for(int i = 0; i < N; i++)
        {
            vAllIndices.push_back(i);   // 匹配点对 的索引
        }

        // mvSets：用来存储 RANSAC 迭代200次，每次迭代所需的8个点的 索引集合
        mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

        DUtils::Random::SeedRandOnce(0);    // 设置随机种子

        // Step 2. 遍历200次，把每次迭代需要的8个点的索引选好
        for(int it = 0; it < mMaxIterations; it++)
        {
            vAvailableIndices = vAllIndices;

            // Select a minimum set 随机选 不重复的8个点对的索引
            for(size_t j = 0; j < 8; j++)
            {
                int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size() - 1);  // 在 vAvailableIndices中 随机选择一个索引
                int idx = vAvailableIndices[randi];     // 这句不多余，防止重复选择

                mvSets[it][j] = idx;    // 将索引存储在 mvSets 的相应位置

                // 并将其从 vAvailableIndices 中移除，以防止重复选择
                vAvailableIndices[randi] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }
        } // 生成的 mvSets 向量包含 mMaxIterations 个集合，每个集合都是8个不重复点的索引

        // Launch threads to compute in parallel a fundamental matrix F and a homography H
        vector<bool> vbMatchesInliersH, vbMatchesInliersF;  // 存储 匹配点内点的布尔向量
        float SH, SF;
        Eigen::Matrix3f H, F;   // 存储 单应矩阵 H 和 基础矩阵 F

        // Step 3. 双线程分别计算迭代200次后，拥有最高得分的单应矩阵 H 和 基础矩阵 F
        // 加ref为了提供引用
        thread threadH(&TwoViewReconstruction::FindHomography,  // 该线程的主线程
                       this,                       // 由于主函数为类的成员函数，所以第一个参数就应该是当前对象的this指针
                       ref(vbMatchesInliersH),  // 输出：经H矩阵验证，匹配点对 是否为内点的标志位
                       ref(SH),                 // 输出：最高得分 SH（阈值 - 重投影误差 的累加值 最大，说明H矩阵估计越准确）
                       ref(H));                 // 输出：单应矩阵 H
        thread threadF(&TwoViewReconstruction::FindFundamental, this, ref(vbMatchesInliersF), ref(SF), ref(F));

        // Wait until both threads have finished 等待两个线程完成计算
        threadH.join();
        threadF.join();


        if(SH + SF == 0.f)
        {
            Verbose::PrintMess("\t\t\t\tSF + HF == 0.f，不能使用H或F矩阵恢复相机运动R,t", Verbose::VERBOSITY_DEBUG);
            return false;
        }
        // Step 4. 计算得分的比例 RH，判断选取哪个模型来求位姿R,t
        float RH = SH / (SH + SF);

        float minParallax = 1.0;

        // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
        if(RH > 0.50) // H 矩阵分高（H矩阵对于 特征点多在一个平面上时好）
        {
//            cout << "Initialization from Homography" << endl;
            Verbose::PrintMess("\t\t\t\t使用单应矩阵 H 恢复相机运动R,t", Verbose::VERBOSITY_DEBUG);
            return ReconstructH(vbMatchesInliersH,  // 输入：匹配点对 的内点标记
                                H,          // 输入：H矩阵
                                mK,         // 输入：相机内参
                                T21,        // 输出：变换矩阵
                                vP3D,       // 输出：恢复出的三维点空间坐标
                                vbTriangulated, // 输出：特征点对 是否被三角化的标记
                                minParallax,    // 输入：要求的最小视差角
                                50);    // 输入：要求的最少恢复出的三维点的数目
        }
        else
        {
            Verbose::PrintMess("\t\t\t\t使用基础矩阵 F 恢复相机运动R,t", Verbose::VERBOSITY_DEBUG);
            //cout << "Initialization from Fundamental" << endl;
            return ReconstructF(vbMatchesInliersF,F,mK,T21,vP3D,vbTriangulated,minParallax,50);
        }
    }

    /**
     * @brief 计算单应矩阵 H，同时计算得分与内点
     * @param vbMatchesInliers 经过 H21矩阵 验证，匹配点对是否为内点，大小为 mvMatches12
     * @param score 得分
     * @param H21 1到2的 H矩阵
     */
    void TwoViewReconstruction::FindHomography(vector<bool> &vbMatchesInliers, float &score, Eigen::Matrix3f &H21)
    {
        // Number of putative matches
        // 匹配点对 的数量
        const int N = mvMatches12.size();

        // Normalize coordinates
        // 归一化坐标
        vector<cv::Point2f> vPn1, vPn2;     // 归一化后的特征点坐标
        Eigen::Matrix3f T1, T2;         // 归一化变化矩阵
        Normalize(mvKeys1,vPn1, T1);
        Normalize(mvKeys2,vPn2, T2);
        Eigen::Matrix3f T2inv = T2.inverse();

        // Best Results variables
        score = 0.0;
        vbMatchesInliers = vector<bool>(N,false);   // 匹配点对 是否是内点

        // Iteration variables
        vector<cv::Point2f> vPn1i(8);   // 8对 匹配点
        vector<cv::Point2f> vPn2i(8);
        Eigen::Matrix3f H21i, H12i;
        vector<bool> vbCurrentInliers(N,false);
        float currentScore;

        // Perform all RANSAC iterations and save the solution with highest score
        // 进行 RANSAC 迭代，选择得分最高的单应矩阵H21和对应的内点vbMatchesInliers。
        for(int it = 0; it < mMaxIterations; it++)
        {
            // Select a minimum set，获取 8对 匹配点
            for(size_t j = 0; j < 8; j++)
            {
                int idx = mvSets[it][j];

                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            // 计算标准化归一化后的单应矩阵 Hn。
            Eigen::Matrix3f Hn = ComputeH21(vPn1i,vPn2i);   // 计算归一化后的H矩阵 p2 = H21 * p1
            // 恢复到原始像素坐标系下的单应矩阵 H21i，H12i
            H21i = T2inv * Hn * T1;
            H12i = H21i.inverse();

            // 计算单应矩阵 H21i 和 H12i 的得分 currentScore（阈值 - 重投影误差 的累加值，越大，说明H矩阵估计越准确）
            // 并确定vbCurrentInliers：匹配点对 在当前 单应矩阵H下，两个重投影误差是否都 < 阈值，如果是，则是内点，其长度 = 匹配点对的数目 mvMatches12
            currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);   // mSigma默认为1

            // 更新最高得分、对应的 H 矩阵、在当前H验证下，匹配点对 是内点的标志
            if(currentScore > score)
            {
                H21 = H21i;
                vbMatchesInliers = vbCurrentInliers;
                score = currentScore;
            }
        } // 迭代200次结束
    }

    /**
     * @brief 计算基础矩阵 F，同时计算得分与内点
     * @param vbMatchesInliers 经过F矩阵验证，是否为内点，大小为mvMatches12
     * @param score 得分
     * @param F21 1到2的F矩阵
     */
    void TwoViewReconstruction::FindFundamental(vector<bool> &vbMatchesInliers, float &score, Eigen::Matrix3f &F21)
    {
        // Number of putative matches
        const int N = vbMatchesInliers.size();

        // Normalize coordinates
        vector<cv::Point2f> vPn1, vPn2;
        Eigen::Matrix3f T1, T2;
        Normalize(mvKeys1,vPn1, T1);
        Normalize(mvKeys2,vPn2, T2);
        Eigen::Matrix3f T2t = T2.transpose();

        // Best Results variables
        score = 0.0;
        vbMatchesInliers = vector<bool>(N,false);

        // Iteration variables
        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);
        Eigen::Matrix3f F21i;
        vector<bool> vbCurrentInliers(N,false);
        float currentScore;

        // Perform all RANSAC iterations and save the solution with highest score
        for(int it = 0; it < mMaxIterations; it++)
        {
            // Select a minimum set
            for(int j=0; j<8; j++)
            {
                int idx = mvSets[it][j];

                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            Eigen::Matrix3f Fn = ComputeF21(vPn1i,vPn2i);

            // FN得到的是基于归一化坐标的F矩阵，必须乘上归一化过程矩阵才是最后的基于像素坐标的F
            F21i = T2t * Fn * T1;

            currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

            if(currentScore > score)
            {
                F21 = F21i;
                vbMatchesInliers = vbCurrentInliers;
                score = currentScore;
            }
        }
    }

    /**
     * @brief 从8对特征点求 单应矩阵 H（normalized DLT）
     * |x2|     | h1 h2 h3 ||x1|
     * |y2| = a | h4 h5 h6 ||y1|  简写: x2 = a H x1, a为一个尺度因子
     * |1 |     | h7 h8 h9 ||1|
     * 使用DLT(direct linear tranform)求解该模型
     * x2 = a H x1
     * ---> (x2) 叉乘 (H x1)  = 0
     * ---> x2^ H x1 = 0，参考本质矩阵，利用矩阵内积，可化成下面的式子
     * ---> Ah = 0 (h 是 H的向量形式)
     * A = | 0  0  0 -x1 -y1 -1 x1y2 y1y2 y2|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
     *     |-x1 -y1 -1  0  0  0 x1x2 y1x2 x2|
     * 通过SVD求解Ah = 0，A'A最小特征值对应的特征向量即为解
     * @param  vP1 归一化后的点, in reference frame
     * @param  vP2 归一化后的点, in current frame
     * @return     单应矩阵
     * @see        Multiple View Geometry in Computer Vision - Algorithm 4.2 p109
     */
    Eigen::Matrix3f TwoViewReconstruction::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
    {
        const int N = vP1.size();   // 8

        Eigen::MatrixXf A(2 * N, 9);

        for(int i = 0; i < N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            A(2*i,0) = 0.0;
            A(2*i,1) = 0.0;
            A(2*i,2) = 0.0;
            A(2*i,3) = -u1;
            A(2*i,4) = -v1;
            A(2*i,5) = -1;
            A(2*i,6) = v2*u1;
            A(2*i,7) = v2*v1;
            A(2*i,8) = v2;

            A(2*i+1,0) = u1;
            A(2*i+1,1) = v1;
            A(2*i+1,2) = 1;
            A(2*i+1,3) = 0.0;
            A(2*i+1,4) = 0.0;
            A(2*i+1,5) = 0.0;
            A(2*i+1,6) = -u2*u1;
            A(2*i+1,7) = -u2*v1;
            A(2*i+1,8) = -u2;

        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);

        Eigen::Matrix<float,3,3,Eigen::RowMajor> H(svd.matrixV().col(8).data());

        return H;
    }

    /**
     * @brief 根据特征点匹配求fundamental matrix（normalized 8点法）
     * 注意F矩阵有秩为2的约束，所以需要两次SVD分解
     *
     * @param  vP1 参考帧中归一化后的特征点
     * @param  vP2 当前帧中归一化后的特征点
     * @return     最后计算得到的基础矩阵F
     * @see        Multiple View Geometry in Computer Vision - Algorithm 11.1 p282 (中文版 p191)
     *
     * x'Fx = 0 整理可得：Af = 0
     * A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
     * 通过SVD求解Af = 0，A'A最小特征值对应的特征向量即为解
     */
    Eigen::Matrix3f TwoViewReconstruction::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
    {
        //获取参与计算的特征点对数
        const int N = vP1.size();

        //初始化A矩阵
        Eigen::MatrixXf A(N, 9);    // N*9维

        // 构造矩阵A，将每个特征点添加到矩阵A中的元素
        for(int i =0; i < N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            A(i,0) = u2*u1;
            A(i,1) = u2*v1;
            A(i,2) = u2;
            A(i,3) = v2*u1;
            A(i,4) = v2*v1;
            A(i,5) = v2;
            A(i,6) = u1;
            A(i,7) = v1;
            A(i,8) = 1;
        }
        // 使用JacobiSVD进行矩阵分解，得到矩阵A的奇异值分解结果
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // 转换成基础矩阵的形式
        Eigen::Matrix<float,3,3,Eigen::RowMajor> Fpre(svd.matrixV().col(8).data());

        // 基础矩阵的秩为2, 而我们不敢保证计算得到的这个结果的秩为2, 所以需要通过第二次奇异值分解,来强制使其秩为2
        // 对初步得来的基础矩阵进行第2次奇异值分解
        Eigen::JacobiSVD<Eigen::Matrix3f> svd2(Fpre, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // 将奇异值向量中的第三个值强制设为0，以确保F矩阵是一个秩为2的矩阵
        Eigen::Vector3f w = svd2.singularValues();
        // 这里注意计算完要强制让第三个奇异值为0
        w(2) = 0;

        // 重新组合好满足秩约束的基础矩阵F，并返回
        return svd2.matrixU() * Eigen::DiagonalMatrix<float,3>(w) * svd2.matrixV().transpose();
    }

    /**
     * @brief 检查H矩阵结果
     *
     * @param H21 1到2的 H矩阵
     * @param H12 2到1的 H矩阵
     * @param vbMatchesInliers 匹配点对在当前 单应矩阵H下，两个重投影误差是否都 < 阈值，是则是内点，大小为 匹配点对的数目 mvMatches12
     * @param sigma 默认为1
     * @return float 返回得分 score
     */
    float TwoViewReconstruction::CheckHomography(const Eigen::Matrix3f &H21, const Eigen::Matrix3f &H12, vector<bool> &vbMatchesInliers, float sigma)
    {
        const int N = mvMatches12.size();

        const float h11 = H21(0,0);
        const float h12 = H21(0,1);
        const float h13 = H21(0,2);
        const float h21 = H21(1,0);
        const float h22 = H21(1,1);
        const float h23 = H21(1,2);
        const float h31 = H21(2,0);
        const float h32 = H21(2,1);
        const float h33 = H21(2,2);

        const float h11inv = H12(0,0);
        const float h12inv = H12(0,1);
        const float h13inv = H12(0,2);
        const float h21inv = H12(1,0);
        const float h22inv = H12(1,1);
        const float h23inv = H12(1,2);
        const float h31inv = H12(2,0);
        const float h32inv = H12(2,1);
        const float h33inv = H12(2,2);

        vbMatchesInliers.resize(N);

        float score = 0;    // 得分：阈值 - 重投影误差 的累加值，越大，说明H矩阵估计越准确

        const float th = 5.991; // 阈值

        const float invSigmaSquare = 1.0 / (sigma * sigma);

        // 遍历所有匹配点对（在所有特征点中，检查用随机选取的8对匹配点计算出的H矩阵）
        for(int i = 0; i < N; i++)
        {
            bool bIn = true;

            const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            // Reprojection error in first image 将 kp2 投影到 kp1
            // kp2in1 "=" H12 * kp2

            // 使用 H21矩阵的第三行将非零因子去掉，参考《SLAM十四讲 P171》
            const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
            const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
            const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

            // 重投影误差：坐标间距离的平方
            const float squareDist1 = (u1 - u2in1)*(u1 - u2in1) + (v1 - v2in1)*(v1 - v2in1);
            // 乘以invSigmaSquare得到 卡方值
            const float chiSquare1 = squareDist1 * invSigmaSquare;

            if(chiSquare1 > th)
                bIn = false;
            else
                score += th - chiSquare1;   // 得分累加 (阈值 - 卡方值)

            // Reprojection error in second image 将 kp1 投影到 kp2
            // kp1in2 "=" H21 * kp1

            // 使用 H21矩阵的第三行将非零因子去掉，参考《SLAM十四讲 P171》
            const float w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
            const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
            const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

            const float squareDist2 = (u2 - u1in2)*(u2 - u1in2) + (v2 - v1in2)*(v2 - v1in2);
            const float chiSquare2 = squareDist2 * invSigmaSquare;

            if(chiSquare2 > th)
                bIn = false;
            else
                score += th - chiSquare2;

            // 如果两个重投影误差都 < 阈值，则说明当前特征点是 内点
            if(bIn)
                vbMatchesInliers[i] = true;
            else
                vbMatchesInliers[i] = false;
        }

        return score;
    }

    /**
     * @brief 检查结果
     * @param F21 顾名思义
     * @param vbMatchesInliers 匹配是否合法，大小为mvMatches12
     * @param sigma 默认为1
     */
    float TwoViewReconstruction::CheckFundamental(const Eigen::Matrix3f &F21, vector<bool> &vbMatchesInliers, float sigma)
    {
        const int N = mvMatches12.size();

        const float f11 = F21(0,0);
        const float f12 = F21(0,1);
        const float f13 = F21(0,2);
        const float f21 = F21(1,0);
        const float f22 = F21(1,1);
        const float f23 = F21(1,2);
        const float f31 = F21(2,0);
        const float f32 = F21(2,1);
        const float f33 = F21(2,2);

        vbMatchesInliers.resize(N);

        float score = 0;

        const float th = 3.841;
        const float thScore = 5.991;

        const float invSigmaSquare = 1.0/(sigma*sigma);

        for(int i=0; i<N; i++)
        {
            bool bIn = true;

            const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            // Reprojection error in second image
            // l2=F21x1=(a2,b2,c2)

            const float a2 = f11*u1+f12*v1+f13;
            const float b2 = f21*u1+f22*v1+f23;
            const float c2 = f31*u1+f32*v1+f33;

            const float num2 = a2*u2+b2*v2+c2;

            const float squareDist1 = num2*num2/(a2*a2+b2*b2);

            const float chiSquare1 = squareDist1*invSigmaSquare;

            if(chiSquare1>th)
                bIn = false;
            else
                score += thScore - chiSquare1;

            // Reprojection error in second image
            // l1 =x2tF21=(a1,b1,c1)

            const float a1 = f11*u2+f21*v2+f31;
            const float b1 = f12*u2+f22*v2+f32;
            const float c1 = f13*u2+f23*v2+f33;

            const float num1 = a1*u1+b1*v1+c1;

            const float squareDist2 = num1*num1/(a1*a1+b1*b1);

            const float chiSquare2 = squareDist2*invSigmaSquare;

            if(chiSquare2>th)
                bIn = false;
            else
                score += thScore - chiSquare2;

            if(bIn)
                vbMatchesInliers[i]=true;
            else
                vbMatchesInliers[i]=false;
        }

        return score;
    }

    /**
     * @brief 通过基础矩阵F重建
     * @param vbMatchesInliers 匹配是否合法，大小为mvMatches12
     * @param F21 顾名思义
     * @param K 相机内参
     * @param T21 旋转平移（要输出的）
     * @param vP3D 恢复的三维点（要输出的）
     * @param vbTriangulated 大小与mvKeys1一致，表示哪个点被重建了
     * @param minParallax 1
     * @param minTriangulated 50
     */
    bool TwoViewReconstruction::ReconstructF(vector<bool> &vbMatchesInliers, Eigen::Matrix3f &F21, Eigen::Matrix3f &K,
                                             Sophus::SE3f &T21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
    {
        int N=0;
        for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
            if(vbMatchesInliers[i])
                N++;

        // Compute Essential Matrix from Fundamental Matrix
        Eigen::Matrix3f E21 = K.transpose() * F21 * K;

        Eigen::Matrix3f R1, R2;
        Eigen::Vector3f t;

        // Recover the 4 motion hypotheses
        DecomposeE(E21,R1,R2,t);

        Eigen::Vector3f t1 = t;
        Eigen::Vector3f t2 = -t;

        // Reconstruct with the 4 hyphoteses and check
        vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
        vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
        float parallax1,parallax2, parallax3, parallax4;

        int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
        int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
        int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
        int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

        int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

        int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

        int nsimilar = 0;
        if(nGood1>0.7*maxGood)
            nsimilar++;
        if(nGood2>0.7*maxGood)
            nsimilar++;
        if(nGood3>0.7*maxGood)
            nsimilar++;
        if(nGood4>0.7*maxGood)
            nsimilar++;

        // If there is not a clear winner or not enough triangulated points reject initialization
        if(maxGood<nMinGood || nsimilar>1)
        {
            return false;
        }

        // If best reconstruction has enough parallax initialize
        if(maxGood==nGood1)
        {
            if(parallax1>minParallax)
            {
                vP3D = vP3D1;
                vbTriangulated = vbTriangulated1;

                T21 = Sophus::SE3f(R1, t1);
                return true;
            }
        }else if(maxGood==nGood2)
        {
            if(parallax2>minParallax)
            {
                vP3D = vP3D2;
                vbTriangulated = vbTriangulated2;

                T21 = Sophus::SE3f(R2, t1);
                return true;
            }
        }else if(maxGood==nGood3)
        {
            if(parallax3>minParallax)
            {
                vP3D = vP3D3;
                vbTriangulated = vbTriangulated3;

                T21 = Sophus::SE3f(R1, t2);
                return true;
            }
        }else if(maxGood==nGood4)
        {
            if(parallax4>minParallax)
            {
                vP3D = vP3D4;
                vbTriangulated = vbTriangulated4;

                T21 = Sophus::SE3f(R2, t2);
                return true;
            }
        }

        return false;
    }


    /**
     * @brief 从 H 恢复 R t
     * H矩阵分解常见有两种方法：Faugeras SVD-based decomposition 和 Zhang SVD-based decomposition
     * 参考文献：Motion and structure from motion in a piecewise plannar environment
     * 这篇参考文献和下面的代码使用了Faugeras SVD-based decomposition算法
     * @see
     * - Faugeras et al, Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988.
     * - Deeper understanding of the homography decomposition for vision-based control
     * 设平面法向量 n = (a, b, c)^t 有一点(x, y, z)在平面上，则ax + by + cz = d  即 1/d * n^t * x = 1 其中d表示
     * x' = R*x + t  从下面开始x 与 x'都表示归一化坐标
     * λ2*x' = R*(λ1*x) + t = R*(λ1*x) + t * 1/d * n^t * (λ1*x)
     * x' = λ*(R + t * 1/d * n^t) * x = H^ * x
     * 对应图像坐标   u' = G * u   G = KH^K.inv
     * H^ ~=  d*R + t * n^t = U∧V^t    ∧ = U^t * H^ * V = d*U^t * R * V + (U^t * t) * (V^t * n)^t
     * s = det(U) * det(V)  s 有可能是 1 或 -1  ∧ = s^2 * d*U^t * R * V + (U^t * t) * (V^t * n)^t = (s*d) * s * U^t * R * V + (U^t * t) * (V^t * n)^t
     * 令 R' = s * U^t * R * V      t' = U^t * t   n' = V^t * n    d' = s * d
     * ∧ = d' * R' + t' * n'^t    所以∧也可以认为是一个单应矩阵，其中加入s只为说明有符号相反的可能
     * 设∧ = | d1, 0, 0 |    取e1 = (1, 0, 0)^t   e2 = (0, 1, 0)^t   e3 = (0, 0, 1)^t
     *      | 0, d2, 0 |
     *      | 0, 0, d3 |
     * n' = (a1, 0, 0)^t + (0, b1, 0)^t + (0, 0, c1)^t = a1*e1 + b1*e2 + c1*e3
     *
     * ∧ = [d1*e1, d2*e2, d3*e3] = [d' * R' * e1, d' * R' * e2, d' * R' * e3] + [t' * a1, t' * b1, t' * c1]
     * ==> d1*e1 = d' * R' * e1 + t' * a1
     *     d2*e2 = d' * R' * e2 + t' * b1
     *     d3*e3 = d' * R' * e3 + t' * c1
     *
     *
     * 上面式子每两个消去t可得
     * d'R'(b1e1 - a1e2) = d1b1e1 - d2a1e2
     * d'R'(c1e2 - b1e3) = d2c1e1 - d3b1e3          同时取二范数，因为旋转对二范数没影响，所以可以约去
     * d'R'(a1e3 - c1e1) = d3a1e3 - d1c1e1
     *
     * (d'^2 - d2^2)*a1^2 + (d'^2 - d1^2)*b1^2 = 0
     * (d'^2 - d3^2)*b1^2 + (d'^2 - d2^2)*c1^2 = 0   令 d'^2 - d1^2 = x1       d'^2 - d2^2 = x2       d'^2 - d3^2 = x3
     * (d'^2 - d1^2)*c1^2 + (d'^2 - d3^2)*a1^2 = 0
     *
     *
     * | x2  x1  0 |     | a1^2 |
     * | 0   x3 x2 |  *  | b1^2 |  =  0    ===>  x1 * x2 * x3 = 0      有(d'^2 - d1^2) * (d'^2 - d2^2) * (d'^2 - d3^2) = 0
     * | x3  0  x1 |     | c1^2 |
     * 由于d1 >= d2 >= d3  所以d' = d2 or -d2
     *
     * -----------------------------------------------------------------------------------------------------------------------------------------------------------------
     * 下面分情况讨论，当d' > 0   再根据a1^2 + b1^2 + c1^2 = 1  ??????哪来的根据，不晓得
     * 能够求得a1 = ε1 * sqrt((d1^2 - d2^2) / (d1^2 - d3^2))
     *       b1 = 0
     *       c1 = ε2 * sqrt((d2^2 - d3^2) / (d1^2 - d3^2))  其中ε1 ε2  可以为正负1
     * 结果带入 d2*e2 = d' * R' * e2 + t' * b1    => R' * e2 = e2
     *           | cosθ 0 -sinθ |
     * ==> R' =  |  0   1   0   |      n'  与   R' 带入  d'R'(a1e3 - c1e1) = d3a1e3 - d1c1e1
     *           | sinθ 0  cosθ |
     *      | cosθ 0 -sinθ |   | -c1 |    | -d1c1 |
     * d' * |  0   1   0   | * |  0  | =  |   0   |   能够解出 sinθ  与 cosθ
     *      | sinθ 0  cosθ |   |  a1 |    |  d3a1 |
     *
     * 到此为止得到了 R'   再根据 d1*e1 = d' * R' * e1 + t' * a1
     *                         d2*e2 = d' * R' * e2 + t' * b1
     *                         d3*e3 = d' * R' * e3 + t' * c1
     *
     * 求得 t' = (d1 - d3) * (a1, 0, c1)^t
     * @param[in] vbMatchesInliers  匹配点对 的内点标记，长度为 mvMatches12
     * @param[in] H21           单应矩阵 H
     * @param[in] K             相机内参
     * @param[in,out] T21       计算出来的 相机的变换矩阵
     * @param[in,out] vP3D      世界坐标系下，三角化测量特征点对之后得到的特征点的空间坐标，实际好像没有对其进行更新
     * @param[in,out] vbTriangulated    特征点是否成功三角化的标记，大小与vbMatchesInliers
     * @param[in] minParallax   对特征点的三角化测量中，认为其测量有效时需要满足的 最小视差角（如果视差角过小则会引起非常大的观测误差）,单位是角度
     * @param[in] minTriangulated   为了进行运动恢复，所需要的 最少的三角化测量成功的点个数，默认50
     * @return true             单应矩阵成功计算出位姿和三维点
     * @return false            初始化失败
     */
    bool TwoViewReconstruction::ReconstructH(vector<bool> &vbMatchesInliers, Eigen::Matrix3f &H21, Eigen::Matrix3f &K,
                                             Sophus::SE3f &T21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
    {
        // 流程:
        //      1. 根据H矩阵的奇异值d'= d2 或者 d' = -d2 分别计算 H 矩阵分解的 8 组解
        //        1.1 讨论 d' > 0 时的 4 组解
        //        1.2 讨论 d' < 0 时的 4 组解
        //      2. 对 8 组解进行验证，并选择产生相机前方最多3D点的解为最优解

        // 匹配点对 是内点的个数
        int N = 0;
        for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
            if(vbMatchesInliers[i])
                N++;

        // We recover 8 motion hypotheses using the method of Faugeras et al.
        // Motion and structure from motion in a piecewise planar environment.
        // International Journal of Pattern Recognition and Artificial Intelligence, 1988

        // 参考SLAM十四讲第二版p170-p171
        // H = K * (R - t * n / d) * K_inv
        // 其中: K 表示内参数矩阵
        //       K_inv 表示内参数矩阵的逆
        //       R 和 t 表示旋转和平移向量
        //       n 表示平面法向量
        // 令 H = K * A * K_inv
        // 则 A = K_inv * H * K

        // step 1：奇异值SVD 分解 H 矩阵
        Eigen::Matrix3f invK = K.inverse();
        Eigen::Matrix3f A = invK * H21 * K;

        // 对矩阵A进行SVD分解，A = U * w * Vt
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(A,    // 待分解的矩阵
                                              Eigen::ComputeFullU | Eigen::ComputeFullV);   // 表示要计算完整的U和V矩阵
        Eigen::Matrix3f U = svd.matrixU();  // 奇异值分解左矩阵，正交阵
        Eigen::Matrix3f V = svd.matrixV();  // 奇异值分解右矩阵，正交阵
        Eigen::Matrix3f Vt = V.transpose();
        Eigen::Vector3f w = svd.singularValues();   // 奇异值矩阵，对角阵

        // 计算变量s = det(U) * det(V)。因为det(V)==det(Vt), 所以 s = det(U) * det(Vt)
        float s = U.determinant() * Vt.determinant();

        // 取得矩阵的各个奇异值
        float d1 = w(0);
        float d2 = w(1);
        float d3 = w(2);

        // // SVD分解正常情况下特征值di应该是正的，且满足d1 >= d2 >= d3
        if(d1/d2 < 1.00001 || d2/d3 < 1.00001)
        {
            Verbose::PrintMess("\t\t\t\t奇异值分解后，奇异值矩阵不满足d1 >= d2 >= d3，H矩阵恢复相机运动 R,t 失败！！", Verbose::VERBOSITY_DEBUG);
            return false;
        }

        // 定义 8 种情况下的旋转矩阵、平移向量和空间向量
        vector<Eigen::Matrix3f> vR;
        vector<Eigen::Vector3f> vt, vn;
        vR.reserve(8);
        vt.reserve(8);
        vn.reserve(8);

        // 一、讨论 d' > 0 时的 4 组解
        // step 2：计算法向量
        // 法向量 n'= [x1 0 x3]

        // 根据论文eq.(12)有
        // x1 = e1 * sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3))
        // x2 = 0
        // x3 = e3 * sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3))
        // 令 aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3))
        //    aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3))
        // 则
        // x1 = e1 * aux1
        // x3 = e3 * aux2

        // 因为 e1, e2, e3 = 1 or -1
        // 所以有 x1 和 x3 有 4 种组合
        // x1 =  {aux1, aux1,-aux1,-aux1}
        // x3 =  {aux3,-aux3,aux3, -aux3}
        float aux1 = sqrt((d1*d1 - d2*d2) / (d1*d1 - d3*d3));
        float aux3 = sqrt((d2*d2 - d3*d3) / (d1*d1 - d3*d3));
        float x1[] = {aux1,aux1,-aux1,-aux1};
        float x3[] = {aux3,-aux3,aux3,-aux3};

        // step 3：恢复旋转矩阵 R
        // step 3.1：计算 sin(theta)和cos(theta)，case d'= d2
        // 根据论文eq.(13)有
        // sin(theta) = e1 * e3 * sqrt(( d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) /(d1 + d3)/d2
        // cos(theta) = (d2* d2 + d1 * d3) / (d1 + d3) / d2
        // 令  aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2)
        // 则  sin(theta) = e1 * e3 * aux_stheta
        //     cos(theta) = (d2*d2+d1*d3)/((d1+d3)*d2)
        // 因为 e1 e2 e3 = 1 or -1
        // 所以 sin(theta) = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta}
        float aux_stheta = sqrt((d1*d1 - d2*d2) * (d2*d2 - d3*d3)) / ((d1 + d3) * d2);

        float ctheta = (d2*d2 + d1*d3) / ((d1 + d3) * d2);
        float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

        // step 3.2：计算 4 种旋转矩阵 R，t
        //根据不同的e1 e3组合所得出来的 4 种R t的解
        //      | ctheta      0   -aux_stheta|       | aux1|
        // Rp = |    0        1       0      |  tp = |  0  |
        //      | aux_stheta  0    ctheta    |       |-aux3|

        //      | ctheta      0    aux_stheta|       | aux1|
        // Rp = |    0        1       0      |  tp = |  0  |
        //      |-aux_stheta  0    ctheta    |       | aux3|

        //      | ctheta      0    aux_stheta|       |-aux1|
        // Rp = |    0        1       0      |  tp = |  0  |
        //      |-aux_stheta  0    ctheta    |       |-aux3|

        //      | ctheta      0   -aux_stheta|       |-aux1|
        // Rp = |    0        1       0      |  tp = |  0  |
        //      | aux_stheta  0    ctheta    |       | aux3|

        // 开始遍历这 4 种情况中的每一种
        for(int i = 0; i < 4; i++)
        {
            // 生成Rp，就是eq.(8) 的 R'
            Eigen::Matrix3f Rp;
            Rp.setZero();
            Rp(0,0) = ctheta;
            Rp(0,2) = -stheta[i];
            Rp(1,1) = 1.f;
            Rp(2,0) = stheta[i];
            Rp(2,2) = ctheta;
            // eq.(8) 恢复出原来的 R
            Eigen::Matrix3f R = s*U*Rp*Vt;
            vR.push_back(R);        // 保存

            // eq. (14) 生成 tp
            Eigen::Vector3f tp;
            tp(0) = x1[i];
            tp(1) = 0;
            tp(2) = -x3[i];
            tp *= d1 - d3;
            // 这里虽然对t有归一化，并没有决定单目整个SLAM过程的尺度
            // 因为 CreateInitialMapMonocular函数 对3D点深度会缩放，然后反过来对 t 有改变
            // eq.(8)恢复原始的 t
            Eigen::Vector3f t = U*tp;
            vt.push_back(t / t.norm());

            // 构造法向量 np
            Eigen::Vector3f np;
            np(0) = x1[i];
            np(1) = 0;
            np(2) = x3[i];
            // eq.(8) 恢复原始的法向量 n
            Eigen::Vector3f n = V*np;
            // 看PPT 16页的图，保持平面法向量向上
            if(n(2) < 0)
                n = -n;
            vn.push_back(n);
        }

        // 二、讨论 d' < 0 时的 4 组解
        // step 3.3 ：计算 sin(theta)和cos(theta)，case d' = -d2
        float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

        float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
        float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

        // step3.4 ：计算 4 种旋转矩阵R，t
        // 计算旋转矩阵 R‘
        for(int i = 0; i < 4; i++)
        {
            // 计算旋转矩阵 R'
            Eigen::Matrix3f Rp;
            Rp.setZero();
            Rp(0,0) = cphi;
            Rp(0,2) = sphi[i];
            Rp(1,1) = -1;
            Rp(2,0) = sphi[i];
            Rp(2,2) = -cphi;
            // 恢复出原来的 R
            Eigen::Matrix3f R = s*U*Rp*Vt;
            vR.push_back(R);

            // 构造tp
            Eigen::Vector3f tp;
            tp(0) = x1[i];
            tp(1) = 0;
            tp(2) = x3[i];
            tp *= d1+d3;
            // 恢复出原来的 t
            Eigen::Vector3f t = U*tp;
            vt.push_back(t / t.norm());

            // 构造法向量 np
            Eigen::Vector3f np;
            np(0) = x1[i];
            np(1) = 0;
            np(2) = x3[i];
            // 恢复出原来的法向量 n
            Eigen::Vector3f n = V*np;
            if(n(2) < 0)
                n = -n;
            vn.push_back(n);
        }

        int bestGood = 0;           // 最好的good点的数
        int secondBestGood = 0;     // 次好的good点的数
        int bestSolutionIdx = -1;   // 最好的解的索引，初始值为-1
        float bestParallax = -1;    // 最大的视差角
        vector<cv::Point3f> bestP3D;    // 最好解对应的 对特征点对，进行三角化测量的结果
        vector<bool> bestTriangulated;  // 最好解对应的 可以被三角化测量的点的标记

        // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
        // We reconstruct all hypotheses and check in terms of triangulated points and parallax

        // step 4：d' = d2和d' = -d2分别对应 8 组(R t)，通过恢复3D点进行验证，选择产生相机前方最多3D点的解为最优解
        for(size_t i = 0; i < 8; i++)
        {
            float parallaxi;                // 第i组解对应的 在三角化测量的时候的比较大的视差角
            vector<cv::Point3f> vP3Di;      // 第i组解对应的 三角化测量之后的特征点的空间坐标
            vector<bool> vbTriangulatedi;   // 第i组解对应的 特征点对 是否被三角化的标记

            // 通过R，t，恢复3D点，并计算特征点对中 成功被三角化点的数目(good点的数目)
            int nGood = CheckRT(vR[i], vt[i],   // 输入：当前解对应的旋转矩阵，平移向量
                                mvKeys1, mvKeys2,   // 输入：特征点
                                mvMatches12, vbMatchesInliers,  // 输入：特征匹配关系以及 是否是内点的标记
                                K,                      // 输入：相机内参矩阵
                                vP3Di,               // 输出：恢复出的三维点的空间坐标
                                4.0*mSigma2,        // 输入：三角化过程中允许的最大重投影误差
                                vbTriangulatedi,     // 输出：特征点对 是否被三角化的标记
                                parallaxi);          // 输出：在三角化测量的时候的比较大的视差角

            // 更新历史最优和次优的解
            // 保留最优的和次优的解.保存次优解的目的是看看最优解是否突出
            if(nGood > bestGood)
            {
                secondBestGood = bestGood;
                bestGood = nGood;
                bestSolutionIdx = i;
                bestParallax = parallaxi;
                bestP3D = vP3Di;
                bestTriangulated = vbTriangulatedi;
            }
            // 当前组的good计数 < 历史最优，但 > 历史次优，则更新当前组解为历史次优
            else if(nGood > secondBestGood)
            {
                secondBestGood = nGood;
            }
        }

        // step 5：选择最优解。通过判断最优是否明显好于次优，从而判断该次Homography分解是否成功
        // 要满足下面的四个条件：
        // 1. 最优解的 good点数 明显大于次优解，这里取0.75经验值
        // 2. 视角差大于规定的阈值
        // 3. good点数要大于规定的最小的被三角化的点数量
        // 4. good数要足够多，达到总数的90%以上
        if(bestGood*0.75 > secondBestGood && bestParallax >= minParallax && bestGood > minTriangulated && bestGood > 0.9*N)
        {
            T21 = Sophus::SE3f(vR[bestSolutionIdx], vt[bestSolutionIdx]);   // 第一帧到第二帧的变换矩阵
            vbTriangulated = bestTriangulated;  // 特征点对 是否被三角化的标记

            Verbose::PrintMess("\t\t\t\t8组解R,t 中的最优解满足条件，H矩阵恢复相机运动 R,t 成功！", Verbose::VERBOSITY_DEBUG);
            return true;
        }

        Verbose::PrintMess("\t\t\t\t8组解R,t 中的最优解满足条件，H矩阵恢复相机运动 R,t 失败！！", Verbose::VERBOSITY_DEBUG);
        return false;
    }

    /**
     * @brief 像素坐标归一化，计算点集的横纵均值，与均值偏差的均值。最后返回的是变化矩阵T 直接乘以像素坐标的齐次向量即可获得去中心去均值后的特征点坐标
     * @param vKeys 特征点
     * @param vNormalizedPoints 归一化后的特征点的 坐标：去中心、去均值后
     * @param T  归一化变化矩阵
     */
    void TwoViewReconstruction::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, Eigen::Matrix3f &T)
    {

        const int N = vKeys.size();
        vNormalizedPoints.resize(N);

        // 1. 特征点的 x坐标和y坐标的均值
        float meanX = 0;
        float meanY = 0;
        for(int i = 0; i < N; i++)
        {
            meanX += vKeys[i].pt.x;
            meanY += vKeys[i].pt.y;
        }
        meanX = meanX / N;
        meanY = meanY / N;

        // 2. 确定新坐标与旧坐标的距离均值：平均绝对偏差(偏差的均值)
        float meanDevX = 0;
        float meanDevY = 0;
        for(int i = 0; i < N; i++)
        {
            // 将每个特征点的 x,y坐标 减去 均值，即偏差，并存储在 vNormalizedPoints 中
            vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
            vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;
            // 计算偏差的绝对值之和
            meanDevX += fabs(vNormalizedPoints[i].x);
            meanDevY += fabs(vNormalizedPoints[i].y);
        }
        meanDevX = meanDevX / N;
        meanDevY = meanDevY / N;

        // 3. 去均值化：标准化偏差 = (原坐标 - 均值) / 平均绝对偏差
        float sX = 1.0 / meanDevX;  // 缩放因子
        float sY = 1.0 / meanDevY;
        for(int i = 0; i < N; i++)
        {
            vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
            vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
        }

        // 4. 计算变化矩阵T：对角线元素为缩放因子sX和sY，平移元素为-meanX * sX和-meanY * sY
        T.setZero();
        T(0,0) = sX;
        T(1,1) = sY;
        T(0,2) = -meanX * sX;
        T(1,2) = -meanY * sY;
        T(2,2) = 1.f;
    }

    /**
     * @brief 进行 cheirality check，即给出特征点对及其R t，通过三角化检查解的有效性，进一步找出H,F分解后最合适的解
     * @param[in] R         旋转矩阵
     * @param[in] t         平移向量
     * @param[in] vKeys1    参考帧特征点
     * @param[in] vKeys2    当前帧特征点
     * @param[in] vMatches12 两帧特征点间的匹配关系
     * @param[in] vbMatchesInliers 匹配特征点对 是否是内点的标记
     * @param[in] K         内参
     * @param[in,out] vP3D  恢复出的 三维点的 空间坐标
     * @param[in] th2       重投影误差阈值
     * @param[in,out] vbGood    特征点对 是否被三角化、被重建的标记
     * @param[in,out] parallax  计算出来的 比较大的视差角（不是最大）
     * @return int nGood        返回被三角化点的数目
     */

    int TwoViewReconstruction::CheckRT(const Eigen::Matrix3f &R, const Eigen::Vector3f &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                                       const Eigen::Matrix3f &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
    {
        // Calibration parameters
        // 内参
        const float fx = K(0,0);
        const float fy = K(1,1);
        const float cx = K(0,2);
        const float cy = K(1,2);

        // 参考帧中的特征点 是否是good点的标记
        vbGood = vector<bool>(vKeys1.size(),false);
        // 存储 空间点的三维坐标
        vP3D.resize(vKeys1.size());

        // 存储 计算出来的每对特征点的 视差
        vector<float> vCosParallax;
        vCosParallax.reserve(vKeys1.size());

        // Camera 1 Projection Matrix K[I|0]
        // Step 1：得到第一个相机的投影矩阵 P1 = K*[I|0]
        // 投影矩阵P是一个 3x4 的矩阵，可以将空间中的一个点投影到平面上，获得其平面坐标，这里均指的是齐次坐标
        // 以第一个相机的光心作为世界坐标系, 定义相机的投影矩阵
        Eigen::Matrix<float, 3, 4> P1;
        P1.setZero();
        P1.block<3,3>(0,0) = K; // 将整个K矩阵拷贝到P1矩阵的左侧3x3矩阵，因为 K*I = K

        // 第一个相机的光心设置为世界坐标系下的原点
        Eigen::Vector3f O1;
        O1.setZero();

        // Camera 2 Projection Matrix K[R|t]
        // Step 2：得到第二个相机的投影矩阵 P2 = K*[R|t]
        Eigen::Matrix<float,3,4> P2;
        P2.block<3,3>(0,0) = R;
        P2.block<3,1>(0,3) = t;
        P2 = K * P2;

        // 第二个相机的光心在世界坐标系下的坐标
        Eigen::Vector3f O2 = - R.transpose() * t;

        int nGood = 0;

        // 遍历所有 匹配特征点对
        for(size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            // 跳过外点 outliers
            if(!vbMatchesInliers[i])
                continue;

            // kp1和kp2是匹配特征点
            const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
            const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];

            // 存储三维点的 空间坐标（？参考帧相机坐标系下）
            Eigen::Vector3f p3dC1;
            // 特征点的齐次坐标
            Eigen::Vector3f x_p1(kp1.pt.x, kp1.pt.y, 1);
            Eigen::Vector3f x_p2(kp2.pt.x, kp2.pt.y, 1);

            // Step 3：调用Triangulate() 函数进行三角化，，得到三角化测量之后的3D点坐标 p3dC1
            GeometricTools::Triangulate(x_p1, x_p2, P1, P2,
                                        p3dC1); // 输出：三角化测量之后特征点的空间坐标

            // 第一关：检查三角化的三维点坐标是否合法（非无穷值）
            // 只要三角测量的结果中有一个是无穷大的就说明三角化失败，跳过对当前点的处理，进行下一对特征点的遍历
            if(!isfinite(p3dC1(0)) || !isfinite(p3dC1(1)) || !isfinite(p3dC1(2)))
            {
                vbGood[vMatches12[i].first] = false;
                continue;
            }

            // Check parallax
            // 第二关：通过三维点深度值正负、两相机光心视差角余弦值大小来检查是否合法
            // Step 4：计算视差角余弦值
            Eigen::Vector3f normal1 = p3dC1 - O1;   // 得到向量PO1
            float dist1 = normal1.norm();           // 求取模长，其实就是距离

            Eigen::Vector3f normal2 = p3dC1 - O2;   // 构造向量PO2
            float dist2 = normal2.norm();           // //求模长

            // 根据公式：a .* b = |a||b|cos_theta 可以推导出来下面的式子
            float cosParallax = normal1.dot(normal2) / (dist1*dist2);

            // Step 5：判断3D点是否在两个摄像头前方
            // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            // Step 5.1：3D点深度为负，在第一个摄像头后方，淘汰
            // ?视差比较小时，重投影误差比较大。这里0.99998 对应的角度为0°21'。为了剔除视差较小的，这里不应该是 cosParallax > 0.99998 吗？
            // !可能导致初始化不稳定
            if(p3dC1(2) <= 0 && cosParallax < 0.99998)
                continue;

            // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            // Step 5.2：3D点深度为负，在第二个摄像头后方，淘汰
            Eigen::Vector3f p3dC2 = R * p3dC1 + t;  // 将空间点p3dC1变换到第2个相机坐标系下，变为p3dC2

            if(p3dC2(2) <= 0 && cosParallax < 0.99998)
                continue;

            // 第三关：计算空间点在参考帧和当前帧上的重投影误差，如果大于阈值则舍弃
            // Step 6：计算重投影误差
            // Check reprojection error in first image
            // 计算3D点在第一个图像上的投影误差
            float im1x, im1y;   // 投影到参考帧图像上的点的坐标x,y
            float invZ1 = 1.0 / p3dC1(2);
            im1x = fx * p3dC1(0) * invZ1 + cx;  // 投影到参考帧图像上。因为参考帧下的相机坐标系和世界坐标系重合，因此这里就直接进行投影就可以了
            im1y = fy * p3dC1(1) * invZ1 + cy;

            // 参考帧上的重投影误差，这个的确就是按照定义来的
            float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

            // Step 6.1：重投影误差太大，淘汰
            // 一般视差角比较小时，重投影误差比较大
            if(squareError1 > th2)
                continue;

            // Check reprojection error in second image
            // 计算3D点在第二个图像上的投影误差
            float im2x, im2y;
            float invZ2 = 1.0 / p3dC2(2); // // 注意这里的p3dC2已经是第二个相机坐标系下的三维点了
            im2x = fx*p3dC2(0)*invZ2+cx;
            im2y = fy*p3dC2(1)*invZ2+cy;

            float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

            // Step 6.2：重投影误差太大，跳过淘汰
            // 一般视差角比较小时重投影误差比较大
            if(squareError2 > th2)
                continue;

            // Step 7：统计经过检验的3D点个数，记录3D点视差角
            vCosParallax.push_back(cosParallax);
            vP3D[vMatches12[i].first] = cv::Point3f(p3dC1(0), p3dC1(1), p3dC1(2));
            nGood++;

            //? bug 我觉得这个写的位置不太对。你的good点计数都++了然后才判断视角角如果大，才置True，不是会让 good点标志 和 good点计数 不一样吗
            if(cosParallax < 0.99998)
                vbGood[vMatches12[i].first] = true;
        }

        // 得到3D点中较大的视差角，并且转换成为角度制表示
        if(nGood > 0)
        {
            // 从小到大排序，注意vCosParallax值越大，视差越小
            sort(vCosParallax.begin(),vCosParallax.end());

            // !排序后并没有取最小的视差角，而是取一个较小的视差角
            // 作者的做法：如果经过检验过后的有效3D点小于50个，那么就取最后那个最小的视差角(cos值最大)
            // 如果大于50个，就取排名第50个的较小的视差角即可，为了避免3D点太多时出现太小的视差角
            size_t idx = min(50, int(vCosParallax.size() - 1));
            // 将这个选中的角弧度制转换为角度制
            parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
        }
        else
            parallax = 0;   // //如果没有good点那么这个就直接设置为0了

        return nGood;   // 返回good点计数
    }

    /**
     * @brief 分解Essential矩阵
     * 解释的比较好的博客：https://blog.csdn.net/weixin_44580210/article/details/90344511
     * F矩阵通过结合内参K可以得到 E矩阵，分解E矩阵将得到4组解
     * 这4组解分别为[R1,t],[R1,-t],[R2,t],[R2,-t]
     * ## 反对称矩阵性质
     * 多视图几何上定义：一个3×3的矩阵是本质矩阵的充要条件是它的奇异值中有两个相等而第三个是0，为什么呢？
     * 首先我们知道 E=[t]×​R=SR其中S为反对称矩阵，反对称矩阵有什么性质呢？
     * 结论1：如果 S 是实的反对称矩阵，那么S=UBU^T，其中 B 为形如diag(a1​Z，a2​Z...am​Z，0，0...0)的分块对角阵，其中 Z = [0, 1; -1, 0]
     * 反对称矩阵的特征矢量都是纯虚数并且奇数阶的反对称矩阵必是奇异的
     * 那么根据这个结论我们可以将 S 矩阵写成 S=kUZU^⊤，而 Z 为
     * | 0, 1, 0 |
     * |-1, 0, 0 |
     * | 0, 0, 0 |
     * Z = diag(1, 1, 0) * W     W 为
     * | 0,-1, 0 |
     * | 1, 0, 0 |
     * | 0, 0, 1 |
     * E=SR=Udiag(1,1,0)(WU^⊤R)  这样就证明了 E 拥有两个相等的奇异值
     *
     * ## 恢复相机矩阵
     * 假定第一个摄像机矩阵是P=[I∣0]，为了计算第二个摄像机矩阵P′，必须把 E 矩阵分解为反对成举着和旋转矩阵的乘积 SR。
     * 还是根据上面的结论1，我们在相差一个常数因子的前提下有 S=UZU^T，我们假设旋转矩阵分解为UXV^T，注意这里是假设旋转矩阵分解形式为UXV^T，并不是旋转矩阵的svd分解，
     * 其中 UV都是E矩阵分解出的
     * Udiag(1,1,0)V^T = E = SR = (UZU^T)(UXV^⊤) = U(ZX)V^T
     * 则有 ZX = diag(1,1,0)，因此 x=W或者 X=W^T
     * 结论：如果 E 的SVD分解为 Udiag(1,1,0)V^⊤，E = SR有两种分解形式，分别是： S = UZU^⊤    R = UWVTor UW^TV^⊤

     * 接着分析，又因为St=0（自己和自己叉乘肯定为0嘛）以及∥t∥=1（对两个摄像机矩阵的基线的一种常用归一化），因此 t = U(0,0,1)^T = u3​，
     * 即矩阵 U 的最后一列，这样的好处是不用再去求S了，应为t的符号不确定，R矩阵有两种可能，因此其分解有如下四种情况：
     * P′=[UWV^T ∣ +u3​] or [UWV^T ∣ −u3​] or [UW^TV^T ∣ +u3​] or [UW^TV^T ∣ −u3​]
     * @param E  Essential Matrix
     * @param R1 Rotation Matrix 1
     * @param R2 Rotation Matrix 2
     * @param t  Translation
     * @see Multiple View Geometry in Computer Vision - Result 9.19 p259
     */
    void TwoViewReconstruction::DecomposeE(const Eigen::Matrix3f &E, Eigen::Matrix3f &R1, Eigen::Matrix3f &R2, Eigen::Vector3f &t)
    {

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f Vt = svd.matrixV().transpose();

        t = U.col(2);
        t = t / t.norm();

        Eigen::Matrix3f W;
        W.setZero();
        W(0,1) = -1;
        W(1,0) = 1;
        W(2,2) = 1;

        R1 = U * W * Vt;
        if(R1.determinant() < 0)
            R1 = -R1;

        R2 = U * W.transpose() * Vt;
        if(R2.determinant() < 0)
            R2 = -R2;
    }

} //namespace ORB_SLAM
