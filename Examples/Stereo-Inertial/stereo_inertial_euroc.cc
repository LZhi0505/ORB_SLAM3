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

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>

#include "ImuTypes.h"
#include "Optimizer.h"
#include <System.h>

using namespace std;

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes, vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

/**
 * @param argc 可执行文件路径；词袋路径；配置文件yaml路径；数据集路径1；时间戳路径1；保存的轨迹文件名称
 * @param argv 提供的参数的个数
 * @return
 */
int main(int argc, char **argv) {
    if (argc < 5) {
        cerr << endl
             << "Usage: ./stereo_inertial_euroc path_to_vocabulary path_to_settings path_to_sequence_folder_1 path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... "
                "path_to_image_folder_N path_to_times_file_N) "
             << endl;
        return 1;
    }

    const int num_seq = (argc - 3) / 2; // 数据集序列个数，可以同时加载多个数据集的路径和时间戳路径，(6 - 3) / 2 = 1
    cout << "num_seq = " << num_seq << endl;

    bool bFileName = (((argc - 3) % 2) == 1); // true，表示有提供需保存轨迹文件的名称
    string file_name;                         // 保存轨迹文件的名称
    if (bFileName) {
        file_name = string(argv[argc - 1]);
        cout << "file name: " << file_name << endl;
    }

    // 加载所有数据集序列
    int seq;
    vector<vector<string>> vstrImageLeft;  // 存储左目图片路径
    vector<vector<string>> vstrImageRight; // 存储右目图片路径
    vector<vector<double>> vTimestampsCam; // 存储图片的时间戳(s)

    vector<vector<cv::Point3f>> vAcc, vGyro; // 存储加速度、角速度
    vector<vector<double>> vTimestampsImu;   // 存储IMU时间戳

    vector<int> nImages; // 每个数据集的图片个数

    vector<int> nImu;                  // 每个数据集IMU数据个数
    vector<int> first_imu(num_seq, 0); // 每个数据集 IMU时间戳最接近第一张图片时间戳的 索引

    vstrImageLeft.resize(num_seq);
    vstrImageRight.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq < num_seq; seq++) {
        cout << "Loading images for sequence " << seq << "...";

        string pathSeq(argv[(2 * seq) + 3]);              // 数据集路径
        string pathTimeStamps(argv[(2 * seq) + 4]);       // 图片时间戳文件路径
        string pathCam0 = pathSeq + "/mav0/cam0/data";    // 左目图片文件夹路径
        string pathCam1 = pathSeq + "/mav0/cam1/data";    // 右目图片文件夹路径
        string pathImu = pathSeq + "/mav0/imu0/data.csv"; // IMU数据文件路径

        // 加载根据所有图片的时间戳 获取 所有图片的路径
        LoadImages(pathCam0, pathCam1, pathTimeStamps, vstrImageLeft[seq], vstrImageRight[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        // 加载IMU数据
        cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(pathImu, vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageLeft[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if ((nImages[seq] <= 0) || (nImu[seq] <= 0)) {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // 找到每个数据集中 IMU时间戳 与第一张图片时间戳最接近的 索引 （假设IMU数据先于图片）
        while (vTimestampsImu[seq][first_imu[seq]] <= vTimestampsCam[seq][0]) // IMU时间戳 <= 第一张图片时间戳
            first_imu[seq]++;                                                 // 索引+1
        first_imu[seq]--;
    }

    // 检查配置文件是否能读取成功
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    vector<float> vTimesTrack; // 记录跟踪时间
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // 创建SLAM系统：读取传感器参数配置，初始化所有线程，并准备处理帧
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, false);

    cv::Mat imLeft, imRight;
    for (seq = 0; seq < num_seq; seq++) {

        vector<ORB_SLAM3::IMU::Point> vImuMeas; // 存储上一帧到当前帧的IMU数据
        double t_rect = 0.f;
        double t_resize = 0.f;
        double t_track = 0.f;
        int num_rect = 0;
        int proccIm = 0;

        // 遍历每个图片
        for (int ni = 0; ni < nImages[seq]; ni++, proccIm++) {
            // 获取左右目图片
            imLeft = cv::imread(vstrImageLeft[seq][ni], cv::IMREAD_UNCHANGED);
            imRight = cv::imread(vstrImageRight[seq][ni], cv::IMREAD_UNCHANGED);

            if (imLeft.empty()) {
                cerr << endl << "Failed to load image at: " << string(vstrImageLeft[seq][ni]) << endl;
                return 1;
            }
            if (imRight.empty()) {
                cerr << endl << "Failed to load image at: " << string(vstrImageRight[seq][ni]) << endl;
                return 1;
            }

            // 当前帧的图片时间戳
            double tframe = vTimestampsCam[seq][ni];

            // 加载上一帧至当前帧的IMU测量数据
            vImuMeas.clear();
            if (ni > 0)
                while (vTimestampsImu[seq][first_imu[seq]] <= vTimestampsCam[seq][ni]) {
                    // 构造IMU数据 (加速度x, y, z, 角速度x, y, z, IMU时间戳(s))
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x, vAcc[seq][first_imu[seq]].y, vAcc[seq][first_imu[seq]].z, vGyro[seq][first_imu[seq]].x,
                                                             vGyro[seq][first_imu[seq]].y, vGyro[seq][first_imu[seq]].z, vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

            // 跟踪当前帧
            SLAM.TrackStereo(imLeft, imRight, tframe, vImuMeas);

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
            t_track = t_rect + t_resize + std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            // 跟踪耗时，单位是秒
            // 使用std::chrono::duration_cast 函数将 t2 - t1 的结果转换为 std::chrono::duration<double> 类型，时间间隔转换为具有双精度浮点数的持续时间。.cout()返回以秒为单位的持续时间
            double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

            vTimesTrack[ni] = ttrack; // 记录每一帧的跟踪时间

            // 等待加载下一帧
            double T = 0;
            if (ni < nImages[seq] - 1)                    // 不是最后一帧
                T = vTimestampsCam[seq][ni + 1] - tframe; // 后一帧的时间戳 - 当前帧的时间戳
            else if (ni > 0)                              // 是最后一帧 且 不是第一帧
                T = tframe - vTimestampsCam[seq][ni - 1]; // 当前帧的时间戳 - 前一帧的时间戳

            // 如果跟踪时间 < 两帧之间时间戳的差值，将程序暂停执行 (T - ttrack) * 1e6 微秒
            if (ttrack < T)
                usleep((T - ttrack) * 1e6); // 1e6
        }

        // 不是最后一个数据集序列
        if (seq < num_seq - 1) {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }

    // 停止所有线程
    SLAM.Shutdown();

    // 保存相机的轨迹
    if (bFileName) {
        const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
        const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    } else {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    return 0;
}

/**
 * 加载图片路径
 * @param strPathLeft   左目图片文件夹路径
 * @param strPathRight  右目图片文件夹路径
 * @param strPathTimes  图片时间戳文件路径
 * @param vstrImageLeft     存储 左目图片路径
 * @param vstrImageRight    存储 右目图片路径
 * @param vTimeStamps       存储时间戳
 */
void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes, vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps) {
    // 打开图片时间戳文件
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());

    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);

    while (!fTimes.eof()) {
        string s;
        getline(fTimes, s); // 读取一行数据，并存储到字符串变量s中

        if (!s.empty()) {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t / 1e9); // 转换为秒
        }
    }
}

/**
 * 加载IMU数据
 * @param strImuPath    IMU数据文件路径
 * @param vTimeStamps   存储IMU时间戳 (s)
 * @param vAcc  存储加速度
 * @param vGyro 存储角速度
 */
void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro) {
    // 打开IMU数据文件，存储格式：timestamp, w_x, w_y, w_z, a_x, a_y, a_z
    ifstream fImu;
    fImu.open(strImuPath.c_str());

    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    while (!fImu.eof()) {
        string s;
        getline(fImu, s); // 获取每行数据

        // 跳过非数据行
        if (s[0] == '#')
            continue;

        if (!s.empty()) {
            string item;
            size_t pos = 0;
            double data[7]; // 存储每行数据
            int count = 0;

            // find 函数查找第一个‘,’，找到返回其位置；找不到则返回string::nops
            while ((pos = s.find(',')) != string::npos) {
                item = s.substr(0, pos);

                data[count++] = stod(item); // 转换为double

                s.erase(0, pos + 1);
            }
            // 加入最后一个数据
            item = s.substr(0, pos);
            data[6] = stod(item);

            vTimeStamps.push_back(data[0] / 1e9);                    // 时间戳转换为s
            vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));  // 加速度
            vGyro.push_back(cv::Point3f(data[1], data[2], data[3])); // 角速度
        }
    }
}
