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


#ifndef IMUTYPES_H
#define IMUTYPES_H

#include <vector>
#include <utility>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <mutex>

#include "SerializationUtils.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace ORB_SLAM3
{

namespace IMU
{

const float GRAVITY_VALUE = 9.81;

// IMU measurement (gyro, accelerometer and timestamp)
/**
 * 指代一个IMU测量数据：加速度、角速度及所对应的时间戳
 * 其构造函数有两个形式，其都是构造一个imu的测量数据的
 */
class Point
{
public:
    Point(const float &acc_x, const float &acc_y, const float &acc_z,
             const float &ang_vel_x, const float &ang_vel_y, const float &ang_vel_z,
             const double &timestamp): a(acc_x,acc_y,acc_z), w(ang_vel_x,ang_vel_y,ang_vel_z), t(timestamp){}
    Point(const cv::Point3f Acc, const cv::Point3f Gyro, const double &timestamp):
        a(Acc.x,Acc.y,Acc.z), w(Gyro.x,Gyro.y,Gyro.z), t(timestamp){}
public:
    Eigen::Vector3f a;  // 加速度
    Eigen::Vector3f w;  // 角速度
    double t;   // 时间戳
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * IMU 零偏类 (保存加速度和角速度的零偏)
 * 空构造时传入的都是0，否则传来的是加速度的零偏和角速度的零偏
 */
class Bias
{
    friend class boost::serialization::access;
    template<class Archive>
    // 保存地图的时候序列化反序列化的时候用到的。零偏里面也是保存六个数，关于加速度的零偏和关于角速度的零偏
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & bax;
        ar & bay;
        ar & baz;

        ar & bwx;
        ar & bwy;
        ar & bwz;
    }

public:
    Bias():bax(0),bay(0),baz(0),bwx(0),bwy(0),bwz(0){}
    Bias(const float &b_acc_x, const float &b_acc_y, const float &b_acc_z,
            const float &b_ang_vel_x, const float &b_ang_vel_y, const float &b_ang_vel_z):
            bax(b_acc_x), bay(b_acc_y), baz(b_acc_z), bwx(b_ang_vel_x), bwy(b_ang_vel_y), bwz(b_ang_vel_z){}

    // 赋值新的零偏 (六个数据)
    void CopyFrom(Bias &b);
    friend std::ostream& operator<< (std::ostream &out, const Bias &b);

public:
    float bax, bay, baz;
    float bwx, bwy, bwz;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * 保存标定好的 IMU外参 (包括Tbc, Tcb, noise)
 */
class Calib
{
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        serializeSophusSE3(ar,mTcb,version);
        serializeSophusSE3(ar,mTbc,version);

        ar & boost::serialization::make_array(Cov.diagonal().data(), Cov.diagonal().size());
        ar & boost::serialization::make_array(CovWalk.diagonal().data(), CovWalk.diagonal().size());

        ar & mbIsSet;
    }

public:

    Calib(const Sophus::SE3<float> &Tbc, const float &ng, const float &na, const float &ngw, const float &naw)
    {
        Set(Tbc,ng,na,ngw,naw);
    }

    Calib(const Calib &calib);
    Calib(){mbIsSet = false;}

    //void Set(const cv::Mat &cvTbc, const float &ng, const float &na, const float &ngw, const float &naw);
    void Set(const Sophus::SE3<float> &sophTbc, const float &ng, const float &na, const float &ngw, const float &naw);

public:
    // Sophus/Eigen implementation
    Sophus::SE3<float> mTcb;    // imu到相机的变换矩阵
    Sophus::SE3<float> mTbc;    // 相机到imu的变换矩阵
    Eigen::DiagonalMatrix<float,6> Cov, CovWalk;    // 误差的协方差、随机游走(零偏)的协方差矩阵    （协方差矩阵是用来算信息矩阵的!!!)
    bool mbIsSet;
};

// Integration of 1 gyro measurement
/**
 * 陀螺仪测量的积分类 (包括旋转增量、右雅可比、积分时间)
 * 角度的预积分类：单纯两个IMU数据之间的积分，不存在累计关系
 */
class IntegratedRotation
{
public:
    IntegratedRotation(){}
    IntegratedRotation(const Eigen::Vector3f &angVel, const Bias &imuBias, const float &time);

public:
    float deltaT;   // integration time, 一次积分中的间隔
    Eigen::Matrix3f deltaR; // Exp 里面中的值
    Eigen::Matrix3f rightJ; // right jacobian, 右雅可比
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Preintegration of Imu Measurements
/**
 * 预积分类
 */
class Preintegrated
{
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & dT;
        ar & boost::serialization::make_array(C.data(), C.size());
        ar & boost::serialization::make_array(Info.data(), Info.size());
        ar & boost::serialization::make_array(Nga.diagonal().data(), Nga.diagonal().size());
        ar & boost::serialization::make_array(NgaWalk.diagonal().data(), NgaWalk.diagonal().size());
        ar & b;
        ar & boost::serialization::make_array(dR.data(), dR.size());
        ar & boost::serialization::make_array(dV.data(), dV.size());
        ar & boost::serialization::make_array(dP.data(), dP.size());
        ar & boost::serialization::make_array(JRg.data(), JRg.size());
        ar & boost::serialization::make_array(JVg.data(), JVg.size());
        ar & boost::serialization::make_array(JVa.data(), JVa.size());
        ar & boost::serialization::make_array(JPg.data(), JPg.size());
        ar & boost::serialization::make_array(JPa.data(), JPa.size());
        ar & boost::serialization::make_array(avgA.data(), avgA.size());
        ar & boost::serialization::make_array(avgW.data(), avgW.size());

        ar & bu;
        ar & boost::serialization::make_array(db.data(), db.size());
        ar & mvMeasurements;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Preintegrated(const Bias &b_, const Calib &calib);
    Preintegrated(Preintegrated* pImuPre);
    Preintegrated() {}
    ~Preintegrated() {}
    void CopyFrom(Preintegrated* pImuPre);
    void Initialize(const Bias &b_);
    void IntegrateNewMeasurement(const Eigen::Vector3f &acceleration, const Eigen::Vector3f &angVel, const float &dt);
    void Reintegrate();
    void MergePrevious(Preintegrated* pPrev);
    void SetNewBias(const Bias &bu_);
    IMU::Bias GetDeltaBias(const Bias &b_);

    Eigen::Matrix3f GetDeltaRotation(const Bias &b_);
    Eigen::Vector3f GetDeltaVelocity(const Bias &b_);
    Eigen::Vector3f GetDeltaPosition(const Bias &b_);

    Eigen::Matrix3f GetUpdatedDeltaRotation();
    Eigen::Vector3f GetUpdatedDeltaVelocity();
    Eigen::Vector3f GetUpdatedDeltaPosition();

    Eigen::Matrix3f GetOriginalDeltaRotation();
    Eigen::Vector3f GetOriginalDeltaVelocity();
    Eigen::Vector3f GetOriginalDeltaPosition();

    Eigen::Matrix<float,6,1> GetDeltaBias();

    Bias GetOriginalBias();
    Bias GetUpdatedBias();

    void printMeasurements() const {
        std::cout << "pint meas:\n";
        for(int i=0; i<mvMeasurements.size(); i++){
            std::cout << "meas " << mvMeasurements[i].t << std::endl;
        }
        std::cout << "end pint meas:\n";
    }

public:
    float dT;   // 这一段的总时间
    Eigen::Matrix<float,15,15> C;   // 协方差矩阵
    Eigen::Matrix<float,15,15> Info;    // 信息矩阵
    Eigen::DiagonalMatrix<float,6> Nga, NgaWalk;    // 误差、随机游走

    // Values for the original bias (when integration was computed)
    // 初始零偏 (初始化的时候给的零偏)
    Bias b;

    // 预积分的值
    Eigen::Matrix3f dR; // 旋转预积分
    Eigen::Vector3f dV, dP; // 速度、位置预积分

    // 旋转对陀螺仪零偏的雅可比、速度对陀螺仪零偏的雅可比、速度对加速度计零偏的雅可比、位置对陀螺仪零偏的雅可比、位置对加速度计零偏的雅可比
    Eigen::Matrix3f JRg, JVg, JVa, JPg, JPa;
    // 加速度的平均值、角速度的平均值
    Eigen::Vector3f avgA, avgW;


private:
    // Updated bias
    // 更新后的零偏 (优化的时候会更新)
    Bias bu;
    // Dif between original and updated bias
    // This is used to compute the updated values of the preintegration
    // 零偏的 增量
    Eigen::Matrix<float,6,1> db;

    // 预积分类Preintegrated的一个结构体, 其实就是保存了一段时间的实测数据，加速度角速度和时间间隔
    struct integrable
    {
        template<class Archive>
        // 没用
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::make_array(a.data(), a.size());
            ar & boost::serialization::make_array(w.data(), w.size());
            ar & t;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        integrable(){}
        integrable(const Eigen::Vector3f &a_, const Eigen::Vector3f &w_ , const float &t_):a(a_),w(w_),t(t_){}
        Eigen::Vector3f a, w;
        float t;
    };

    // 这段时间内的所有imu数据
    std::vector<integrable> mvMeasurements;

    // 锁
    std::mutex mMutex;
};

// Lie Algebra Functions
Eigen::Matrix3f RightJacobianSO3(const float &x, const float &y, const float &z);
Eigen::Matrix3f RightJacobianSO3(const Eigen::Vector3f &v);

Eigen::Matrix3f InverseRightJacobianSO3(const float &x, const float &y, const float &z);
Eigen::Matrix3f InverseRightJacobianSO3(const Eigen::Vector3f &v);

Eigen::Matrix3f NormalizeRotation(const Eigen::Matrix3f &R);

}

} //namespace ORB_SLAM3

#endif // IMUTYPES_H
