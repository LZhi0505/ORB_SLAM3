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

#include "ImuTypes.h"
#include "Converter.h"

#include "GeometricTools.h"

#include <iostream>

namespace ORB_SLAM3 {

namespace IMU {

const float eps = 1e-4;

/**
 * @brief 强制让R变成一个正交矩阵
 * @param R 待优化的旋转矩阵
 * @return 优化后的矩阵
 */
Eigen::Matrix3f NormalizeRotation(const Eigen::Matrix3f &R) {
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

/**
 * @brief 计算右雅可比矩阵
 * @param xyz 李代数
 * @return Jr
 */
Eigen::Matrix3f RightJacobianSO3(const float &x, const float &y, const float &z) {
    Eigen::Matrix3f I;
    I.setIdentity();
    const float d2 = x * x + y * y + z * z;
    const float d = sqrt(d2);
    Eigen::Vector3f v;
    v << x, y, z;
    Eigen::Matrix3f W = Sophus::SO3f::hat(v);
    if (d < eps) {
        return I;
    } else {
        return I - W * (1.0f - cos(d)) / d2 + W * W * (d - sin(d)) / (d2 * d);
    }
}

/**
 * @brief 计算右雅可比
 * @param v so3
 * @return Jr
 */
Eigen::Matrix3f RightJacobianSO3(const Eigen::Vector3f &v) { return RightJacobianSO3(v(0), v(1), v(2)); }

/**
 * @brief 计算右雅可比矩阵的 逆
 * @param xyz so3
 * @return Jr^-1
 */
Eigen::Matrix3f InverseRightJacobianSO3(const float &x, const float &y, const float &z) {
    Eigen::Matrix3f I;
    I.setIdentity();
    const float d2 = x * x + y * y + z * z;
    const float d = sqrt(d2);
    Eigen::Vector3f v;
    v << x, y, z;
    Eigen::Matrix3f W = Sophus::SO3f::hat(v);

    if (d < eps) {
        return I;
    } else {
        return I + W / 2 + W * W * (1.0f / d2 - (1.0f + cos(d)) / (2.0f * d * sin(d)));
    }
}

/**
 * @brief 计算右雅可比的逆
 * @param v so3
 * @return Jr^-1
 */
Eigen::Matrix3f InverseRightJacobianSO3(const Eigen::Vector3f &v) { return InverseRightJacobianSO3(v(0), v(1), v(2)); }

/**
 * @brief 计算 旋转角度积分量 (旋转矩阵 与 右乘雅可比矩阵)
 * 对应Bias更新对预积分的影响一节
 * @param[in] angVel    陀螺仪数据 (角速度)
 * @param[in] imuBias   陀螺仪零偏 (角速度零偏)
 * @param[in] time      两帧间的时间差, derta t
 */
IntegratedRotation::IntegratedRotation(const Eigen::Vector3f &angVel, const Bias &imuBias, const float &time) {
    // 得到考虑零偏后的角度旋转：去零偏后的 旋转向量
    const float x = (angVel(0) - imuBias.bwx) * time;
    const float y = (angVel(1) - imuBias.bwy) * time;
    const float z = (angVel(2) - imuBias.bwz) * time;

    // 计算旋转矩阵的模值，后面用罗德里格公式计算旋转矩阵时会用到
    const float d2 = x * x + y * y + z * z;
    const float d = sqrt(d2);

    Eigen::Vector3f v;
    v << x, y, z;

    // 角度转成叉积的矩阵形式：旋转向量的反对称矩阵
    Eigen::Matrix3f W = Sophus::SO3f::hat(v);

    // 模长 < eps (eps = 1e-4 是一个小量，根据罗德里格斯公式求极限，后面的高阶小量忽略掉得到此式)
    if (d < eps) {
        deltaR = Eigen::Matrix3f::Identity() + W; // 当前时刻 到 下一时刻的 旋转矩阵 (3*3 单位矩阵 + W(叉积矩阵的形式))
        rightJ = Eigen::Matrix3f::Identity();     // 右乘雅可比矩阵 E
    } else {
        // 经典预积分计算公式
        deltaR = Eigen::Matrix3f::Identity() + W * sin(d) / d + W * W * (1.0f - cos(d)) / d2;
        // 右雅可比矩阵 公式1.6
        rightJ = Eigen::Matrix3f::Identity() - W * (1.0f - cos(d)) / d2 + W * W * (d - sin(d)) / (d2 * d);
    }
}

/**
 * @brief 两帧之间的预积分类的构造函数
 *
 * 将 协方差矩阵赋值；根据输入的零偏初始化预积分参数
 * @param b_    上一帧的零偏类（6个值）
 * @param calib 当前帧的IMU标定参数类（噪声和随机游走）
 */
Preintegrated::Preintegrated(const Bias &b_, const Calib &calib) {
    Nga = calib.Cov;    // 噪声的协方差矩阵（6维对角阵）
    NgaWalk = calib.CovWalk;    // 随机游走的协方差矩阵（6维对角阵）

    // 初始化预积分器相关参数
    Initialize(b_);
}

// Copy constructor
/**
 * @brief 两帧之间的预积分类的 构造函数
 * @param pImuPre   上一段时间的预积分
 * 这一段就是把上段预积分的每个变量取出来赋值到当前的预积分类的成员变量里面
 */
Preintegrated::Preintegrated(Preintegrated *pImuPre)
    : dT(pImuPre->dT), C(pImuPre->C), Info(pImuPre->Info), Nga(pImuPre->Nga), NgaWalk(pImuPre->NgaWalk), b(pImuPre->b), dR(pImuPre->dR), dV(pImuPre->dV), dP(pImuPre->dP), JRg(pImuPre->JRg),
      JVg(pImuPre->JVg), JVa(pImuPre->JVa), JPg(pImuPre->JPg), JPa(pImuPre->JPa), avgA(pImuPre->avgA), avgW(pImuPre->avgW), bu(pImuPre->bu), db(pImuPre->db), mvMeasurements(pImuPre->mvMeasurements) {}

/**
 * @brief 复制上一帧的预积分
 * @param pImuPre 上一帧的预积分
 */
void Preintegrated::CopyFrom(Preintegrated *pImuPre) {
    dT = pImuPre->dT;
    C = pImuPre->C;
    Info = pImuPre->Info;
    Nga = pImuPre->Nga;
    NgaWalk = pImuPre->NgaWalk;
    b.CopyFrom(pImuPre->b);
    dR = pImuPre->dR;
    dV = pImuPre->dV;
    dP = pImuPre->dP;
    // 旋转关于陀螺仪偏置变化的雅克比，以此类推
    JRg = pImuPre->JRg;
    JVg = pImuPre->JVg;
    JVa = pImuPre->JVa;
    JPg = pImuPre->JPg;
    JPa = pImuPre->JPa;
    avgA = pImuPre->avgA;
    avgW = pImuPre->avgW;
    bu.CopyFrom(pImuPre->bu);
    db = pImuPre->db;
    mvMeasurements = pImuPre->mvMeasurements;
}

/**
 * @brief 初始化预积分器：仅更新零偏
 * @param b_ 上一帧的零偏类
 */
void Preintegrated::Initialize(const Bias &b_) {
    // 除了dR设置为单位矩阵，其他均设置为0矩阵
    dR.setIdentity();

    dV.setZero();
    dP.setZero();
    JRg.setZero();
    JVg.setZero();
    JVa.setZero();
    JPg.setZero();
    JPa.setZero();
    C.setZero();
    Info.setZero();
    db.setZero();
    b = b_;
    bu = b_;        // 更新后的零偏
    avgA.setZero(); // 平均加速度
    avgW.setZero(); // 平均角速度
    dT = 0.0f;
    mvMeasurements.clear(); // 存放imu数据及dt
}

/**
 * @brief 根据新的零偏 重新积分 mvMeasurements里的数据 Optimizer::InertialOptimization调用
 * 重新积分函数：将这段时间的IMU数据取出来放到容器中，然后用更新后的偏置bu更新偏置，其他参数全部初始化，然后重新计算这段时间的预积分（这个函数是最重要的函数）
 */
void Preintegrated::Reintegrate() {
    std::unique_lock<std::mutex> lock(mMutex);
    // 将这段时间的所有imu数据取出来放在aux向量中
    const std::vector<integrable> aux = mvMeasurements;
    // 根据最新的偏置bu重新初始化
    Initialize(bu);
    // 计算预积分
    for (size_t i = 0; i < aux.size(); i++)
        IntegrateNewMeasurement(aux[i].a, aux[i].w, aux[i].t);
}

/**
 * @brief 计算两帧之间的 每个预积分量的 预积分；根据预积分噪声的递推公式 更新 预积分噪声Noise
 *
 * @param[in] acceleration  IMU当前时刻 到 下一时刻的 平均加速度 (在这里作为 当前时刻的IMU 加速度)
 * @param[in] angVel        IMU当前时刻 到 下一时刻的 平均角速度 (当前时刻的IMU 角速度)
 * @param[in] dt            两数据间的 时间差
 */
void Preintegrated::IntegrateNewMeasurement(const Eigen::Vector3f &acceleration, const Eigen::Vector3f &angVel, const float &dt) {
    // 将 中值积分后的结果 封装成一个结构体，先保存在 mvMeasurements 中；
    mvMeasurements.push_back(integrable(acceleration, angVel, dt));

    // 先更新 位置P，因为其依赖于之前计算的 速度V 和 旋转R；
    // 再更新 速度V，因为其依赖于之前计算的 旋转R；
    // 最后更新 旋转R

    // Step 1: 构造协方差矩阵

    // Step 1.1: 定义 预积分噪声矩阵 的传递矩阵，A 用于计算 i 到 j-1 的历史噪声或者协方差; B用于计算 j-1 新的噪声或协方差；
    // 这两个矩阵里面的数都是当前时刻的，计算主要是为了下一时刻使用
    Eigen::Matrix<float, 9, 9> A;
    A.setIdentity();
    Eigen::Matrix<float, 9, 6> B;
    B.setZero();

    // 去零偏：当前时刻的 平均加速度、角速度的 测量值 - 上一帧的零偏（a~_j-1 - b^a_i）
    Eigen::Vector3f acc, accW;
    acc << acceleration(0) - b.bax, acceleration(1) - b.bay, acceleration(2) - b.baz;
    accW << angVel(0) - b.bwx, angVel(1) - b.bwy, angVel(2) - b.bwz;

    //! dR、dV、dP: 旋转、速度、位置预积分的测量值 (= 理想值 + 噪声) △R~_i,j-1

    // 记录平均加速度和角速度
    avgA = (dT * avgA + dR * acc * dt) / (dT + dt);
    avgW = (dT * avgW + accW * dt) / (dT + dt);

    // 更新位置和速度：根据 没有更新的dR 来更新 dP与dV (15-19)
    dP = dP + dV * dt + 0.5f * dR * acc * dt * dt; // △R~_i,j-2 + △V~_i,j-1 * dt + 1/2 * △R~_i,j-1 * (△a_j-1 - b^a_j-1) * dt^2
    dV = dV + dR * acc * dt;

    // Compute velocity and position parts of matrices A and B (rely on non-updated delta rotation)
    // Step 3: 根据 η_ij = A * η_i,j-1 + B_j-1 * η_j-1 中的Ａ矩阵和Ｂ矩阵 更新 速度和位移 (15-48)
    // Step 3.1: 加速度噪声分离项的 反对称矩阵 (加速度测量值 - 加速度零偏)
    Eigen::Matrix<float, 3, 3> Wacc = Sophus::SO3f::hat(acc); // (a~_j-1 - b^a_i)^

    A.block<3, 3>(3, 0) = -dR * dt * Wacc;
    A.block<3, 3>(6, 0) = -0.5f * dR * dt * dt * Wacc;
    A.block<3, 3>(6, 3) = Eigen::DiagonalMatrix<float, 3>(dt, dt, dt);
    B.block<3, 3>(3, 3) = dR * dt;
    B.block<3, 3>(6, 3) = 0.5f * dR * dt * dt;

    // Update position and velocity jacobians wrt bias correction
    // 因为随着时间推移，不可能每次都重新计算雅克比矩阵，所以需要做J(k+1) = j(k) + (~)这类事，分解方式与A、B矩阵相同
    // 论文作者对forster论文公式的基础上做了变形，然后递归更新，参见 https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/212
    // 即零偏更新后 各预积分量对零偏的偏导的递推模型 (雅可比矩阵)  (计算预积分相对于偏置的雅可比矩阵)
    JPa = JPa + JVa * dt - 0.5f * dR * dt * dt;              // (15-67)
    JPg = JPg + JVg * dt - 0.5f * dR * dt * dt * Wacc * JRg; // (15-66)
    JVa = JVa - dR * dt;                                     // (15-62)
    JVg = JVg - dR * dt * Wacc * JRg;                        // (15-63)

    // Step 2: 根据更新后的bias进行角度积分，更新 旋转预积分量
    IntegratedRotation dRi(angVel, b, dt); // (构造函数，会根据更新后的bias进行角度积分) 计算去零偏后的 旋转增量对应的旋转矩阵 与 右乘雅可比矩阵    dRi.deltaR: △R~_j-1,j
    dR = NormalizeRotation(dR * dRi.deltaR); // 更新旋转预积分的测量值 △R~_i,j = △R~_i,j-1 * △R~_j-1,j，并强制归一化使其符合旋转矩阵的格式

    // 用最新的预积分测量值补充更新A、B矩阵中剩余的元素    (15-48)
    // 小量delta初始为0，更新后通常也为0，故省略了小量的更新
    A.block<3, 3>(0, 0) = dRi.deltaR.transpose(); // △R~_j,j-1
    B.block<3, 3>(0, 0) = dRi.rightJ * dt;        // J^j-1_r

    // 小量delta初始为0，更新后通常也为0，故省略了小量的更新

    // Step 3: 更新预积分测量 协方差 (15-51)
    // B矩阵为 9x6 矩阵；Nga: IMU测量噪声n^d_k的协方差，6x6 对角矩阵，3个陀螺仪噪声的平方，3个加速度计噪声的平方
    // C矩阵为 预积分噪声 n^△_i,j 的协方差
    C.block<9, 9>(0, 0) = A * C.block<9, 9>(0, 0) * A.transpose() + B * Nga * B.transpose();
    // 这一部分最开始是0矩阵，随着积分次数增加，每次都加上 6x6 随机游走的信息矩阵
    C.block<6, 6>(9, 9) += NgaWalk;

    // 最后更新旋转的雅可比矩阵 (15-60)
    // 计算偏置的雅克比矩阵，r对bg的导数，∂ΔRij/∂bg = (ΔRjj-1) * ∂ΔRij-1/∂bg - Jr(j-1)*t
    // 论文作者对forster论文公式的基础上做了变形，然后递归更新，参见 https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/212
    // ? 为什么先更新JPa、JPg、JVa、JVg最后更新JRg? 答：这里必须先更新dRi才能更新到这个值，但是为什么JPg和JVg依赖的上一个JRg值进行更新的？
    JRg = dRi.deltaR.transpose() * JRg - dRi.rightJ * dt;

    // 累加两帧之间IMU数据的 总积分时间
    dT += dt;
}

/**
 * @brief 融合两个预积分，发生在删除关键帧的时候，3帧变2帧，需要把两段预积分融合
 * @param pPrev 前面的预积分 (上一段时间的预积分，即上上一帧 -> 上一帧)
 * 先取到最新的积分利用偏置的改变重新初始化预积分
 */
void Preintegrated::MergePrevious(Preintegrated *pPrev) {
    if (pPrev == this)
        return;

    std::unique_lock<std::mutex> lock1(mMutex);
    std::unique_lock<std::mutex> lock2(pPrev->mMutex);
    // 取到最新的偏置信息
    Bias bav;
    bav.bwx = bu.bwx;
    bav.bwy = bu.bwy;
    bav.bwz = bu.bwz;
    bav.bax = bu.bax;
    bav.bay = bu.bay;
    bav.baz = bu.baz;

    // 存放上一段预积分中的所有数据
    const std::vector<integrable> aux1 = pPrev->mvMeasurements;
    const std::vector<integrable> aux2 = mvMeasurements;

    // 重新初始化预积分
    Initialize(bav);
    // 好像没啥用呢
    for (size_t i = 0; i < aux1.size(); i++)
        IntegrateNewMeasurement(aux1[i].a, aux1[i].w, aux1[i].t);
    for (size_t i = 0; i < aux2.size(); i++)
        IntegrateNewMeasurement(aux2[i].a, aux2[i].w, aux2[i].t);
}

/**
 * @brief 更新偏置
 * @param bu_ 偏置
 */
void Preintegrated::SetNewBias(const Bias &bu_) {
    std::unique_lock<std::mutex> lock(mMutex);
    bu = bu_;

    db(0) = bu_.bwx - b.bwx;
    db(1) = bu_.bwy - b.bwy;
    db(2) = bu_.bwz - b.bwz;
    db(3) = bu_.bax - b.bax;
    db(4) = bu_.bay - b.bay;
    db(5) = bu_.baz - b.baz;
}

/**
 * @brief 获得 当前零偏 与 输入零偏的改变量
 * @param b_ 输入零偏
 * @return 改变量
 */
IMU::Bias Preintegrated::GetDeltaBias(const Bias &b_) {
    std::unique_lock<std::mutex> lock(mMutex);
    return IMU::Bias(b_.bax - b.bax, b_.bay - b.bay, b_.baz - b.baz, b_.bwx - b.bwx, b_.bwy - b.bwy, b_.bwz - b.bwz);
}

/**
 * @brief 根据新的零偏计算新的dR
 * @param b_ 新的零偏
 * @return dR
 */
Eigen::Matrix3f Preintegrated::GetDeltaRotation(const Bias &b_) {
    std::unique_lock<std::mutex> lock(mMutex);
    // 计算偏置的变化量
    Eigen::Vector3f dbg;
    dbg << b_.bwx - b.bwx, b_.bwy - b.bwy, b_.bwz - b.bwz;

    // 考虑偏置后，dR对偏置线性化的近似求解,邱笑晨《预积分总结与公式推导》P13～P14
    // Forster论文公式（44）yP17也有结果（但没有推导），后面两个函数GetDeltaPosition和GetDeltaPosition也是基于此推导的
    // 更新之前的dR * Exp(雅可比矩阵 * 偏置改变量)
    return NormalizeRotation(dR * Sophus::SO3f::exp(JRg * dbg).matrix());
}

/**
 * @brief 根据新的零偏计算新的dV
 * @param b_ 新的零偏
 * @return dV
 */
Eigen::Vector3f Preintegrated::GetDeltaVelocity(const Bias &b_) {
    std::unique_lock<std::mutex> lock(mMutex);
    Eigen::Vector3f dbg, dba;
    dbg << b_.bwx - b.bwx, b_.bwy - b.bwy, b_.bwz - b.bwz;
    dba << b_.bax - b.bax, b_.bay - b.bay, b_.baz - b.baz;

    // 考虑偏置后，dV对偏置线性化的近似求解,邱笑晨《预积分总结与公式推导》P13，JPg和JPa在预积分处理中更新
    return dV + JVg * dbg + JVa * dba;
}

/**
 * @brief 根据新的零偏计算新的dP
 * @param b_ 新的零偏
 * @return dP
 */
Eigen::Vector3f Preintegrated::GetDeltaPosition(const Bias &b_) {
    std::unique_lock<std::mutex> lock(mMutex);
    Eigen::Vector3f dbg, dba;
    dbg << b_.bwx - b.bwx, b_.bwy - b.bwy, b_.bwz - b.bwz;
    dba << b_.bax - b.bax, b_.bay - b.bay, b_.baz - b.baz;

    // 考虑偏置后，dP对偏置线性化的近似求解,邱笑晨《预积分总结与公式推导》P13，JPg和JPa在预积分处理中更新
    return dP + JPg * dbg + JPa * dba;
}

/**
 * @brief 返回经过db(δba, δbg)更新后的dR,与上面是一个意思
 * @return dR
 */
Eigen::Matrix3f Preintegrated::GetUpdatedDeltaRotation() {
    std::unique_lock<std::mutex> lock(mMutex);
    return NormalizeRotation(dR * Sophus::SO3f::exp(JRg * db.head(3)).matrix());
}

/**
 * @brief 返回经过db(δba, δbg)更新后的dV,与上面是一个意思
 * @return dV
 */
Eigen::Vector3f Preintegrated::GetUpdatedDeltaVelocity() {
    std::unique_lock<std::mutex> lock(mMutex);
    return dV + JVg * db.head(3) + JVa * db.tail(3);
}

/**
 * @brief 返回经过db(δba, δbg)更新后的dP,与上面是一个意思
 * @return dP
 */
Eigen::Vector3f Preintegrated::GetUpdatedDeltaPosition() {
    std::unique_lock<std::mutex> lock(mMutex);
    return dP + JPg * db.head(3) + JPa * db.tail(3);
}

/**
 * @brief 获取dR
 * @return dR
 */
Eigen::Matrix3f Preintegrated::GetOriginalDeltaRotation() {
    std::unique_lock<std::mutex> lock(mMutex);
    return dR;
}

/**
 * @brief 获取dV
 * @return dV
 */
Eigen::Vector3f Preintegrated::GetOriginalDeltaVelocity() {
    std::unique_lock<std::mutex> lock(mMutex);
    return dV;
}

/**
 * @brief 获取dP
 * @return dP
 */
Eigen::Vector3f Preintegrated::GetOriginalDeltaPosition() {
    std::unique_lock<std::mutex> lock(mMutex);
    return dP;
}

/**
 * @brief 获取b,更新前的偏置
 * @return b
 */
Bias Preintegrated::GetOriginalBias() {
    std::unique_lock<std::mutex> lock(mMutex);
    return b;
}

/**
 * @brief 获取bu,更新后的偏置
 * @return bu
 */
Bias Preintegrated::GetUpdatedBias() {
    std::unique_lock<std::mutex> lock(mMutex);
    return bu;
}

/**
 * @brief 获取db,更新前后的偏置差
 * @return db
 */
Eigen::Matrix<float, 6, 1> Preintegrated::GetDeltaBias() {
    std::unique_lock<std::mutex> lock(mMutex);
    return db;
}

/**
 * @brief 赋值新的零偏
 * @param b 零偏
 */
void Bias::CopyFrom(Bias &b) {
    bax = b.bax;
    bay = b.bay;
    baz = b.baz;
    bwx = b.bwx;
    bwy = b.bwy;
    bwz = b.bwz;
}

std::ostream &operator<<(std::ostream &out, const Bias &b) {
    if (b.bwx > 0)
        out << " ";
    out << b.bwx << ",";
    if (b.bwy > 0)
        out << " ";
    out << b.bwy << ",";
    if (b.bwz > 0)
        out << " ";
    out << b.bwz << ",";
    if (b.bax > 0)
        out << " ";
    out << b.bax << ",";
    if (b.bay > 0)
        out << " ";
    out << b.bay << ",";
    if (b.baz > 0)
        out << " ";
    out << b.baz;

    return out;
}

/**
 * @brief 设置参数
 * @param Tbc_ 左目到IMU的变换矩阵
 * @param ng 陀螺仪的 噪声
 * @param na 加速度计的 噪声
 * @param ngw 陀螺仪的 随机游走
 * @param naw 加速度的 计随机游走
 */
void Calib::Set(const Sophus::SE3<float> &sophTbc, const float &ng, const float &na, const float &ngw, const float &naw) {
    mbIsSet = true;
    const float ng2 = ng * ng;
    const float na2 = na * na;
    const float ngw2 = ngw * ngw;
    const float naw2 = naw * naw;

    // Sophus/Eigen
    mTbc = sophTbc;
    mTcb = mTbc.inverse();
    // 噪声 协方差
    Cov.diagonal() << ng2, ng2, ng2, na2, na2, na2;
    // 随机游走 协方差
    CovWalk.diagonal() << ngw2, ngw2, ngw2, naw2, naw2, naw2;
}

/**
 * @brief IMU标定参数的构造函数
 * @param calib IMU标定参数
 */
Calib::Calib(const Calib &calib) {
    mbIsSet = calib.mbIsSet;
    // Sophus/Eigen parameters
    mTbc = calib.mTbc;
    mTcb = calib.mTcb;
    Cov = calib.Cov;
    CovWalk = calib.CovWalk;
}

} // namespace IMU

} // namespace ORB_SLAM3
