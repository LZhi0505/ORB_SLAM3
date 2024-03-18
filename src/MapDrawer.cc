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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM3
{

// 构造函数
MapDrawer::MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings):mpAtlas(pAtlas)
{
    // 如果Settings不为空
    if(settings){
        newParameterLoader(settings);
    }
    // settings 为空
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        bool is_correct = ParseViewerParamFile(fSettings);      // 读取可视化配置文件

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }
}

// 读取配置文件
void MapDrawer::newParameterLoader(Settings *settings) {
    mKeyFrameSize = settings->keyFrameSize();           // 关键帧的 大小
    mKeyFrameLineWidth = settings->keyFrameLineWidth(); // 关键帧的 线宽
    mGraphLineWidth = settings->graphLineWidth();       // 图的 线宽
    mPointSize = settings->pointSize();                 // 点 的大小
    mCameraSize = settings->cameraSize();               // 相机的 大小
    mCameraLineWidth  = settings->cameraLineWidth();    // 相机的 线宽
}

// 读取可视化配置文件
bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false; // 是否缺少参数 标志，下面会更新

    // 关键帧的 大小
    cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
    if(!node.empty())
    {
        mKeyFrameSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    // 关键帧的 线宽
    node = fSettings["Viewer.KeyFrameLineWidth"];
    if(!node.empty())
    {
        mKeyFrameLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    // 图的 线宽
    node = fSettings["Viewer.GraphLineWidth"];
    if(!node.empty())
    {
        mGraphLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    // 点 的大小
    node = fSettings["Viewer.PointSize"];
    if(!node.empty())
    {
        mPointSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    // 相机的 大小
    node = fSettings["Viewer.CameraSize"];
    if(!node.empty())
    {
        mCameraSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    // 相机的 线宽
    node = fSettings["Viewer.CameraLineWidth"];
    if(!node.empty())
    {
        mCameraLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}


 /**
  * @brief 在地图中显示 地图点
  */
void MapDrawer::DrawMapPoints()
{
    // 获取当前地图
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap)
        return;

    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();             // 获取 当前地图的所有 地图点
    const vector<MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();    // 获取 参考地图点

    // 将 vpRefMPs 从vector容器类型转化为 set容器类型，便于使用set::count快速统计
    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    // 如果地图点为空
    if(vpMPs.empty())
        return;

    // 显示所有的地图点（不包括局部地图点），大小为2个像素，黑色
    glPointSize(mPointSize);    // 设置点大小
    glBegin(GL_POINTS);         // 开始绘制点
    glColor3f(0.0,0.0,0.0); // 设置黑色

    // 遍历所有地图点
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        // 剔除 坏的地图点 和 参考地图点ReferenceMapPoints（局部地图点）
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        // 获取地图点的 世界坐标
        Eigen::Matrix<float,3,1> pos = vpMPs[i]->GetWorldPos();
        // 绘制点
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();    // 结束绘制点

    // 显示局部地图点，大小为2个像素，红色
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    // 遍历 局部地图点
    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        Eigen::Matrix<float,3,1> pos = (*sit)->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));

    }

    glEnd();
}

/** @brief 在地图中绘制关键帧
 *
 * @param bDrawKF           为true时绘制关键帧
 * @param bDrawGraph        为true时绘制关键帧之间的连线
 * @param bDrawInertialGraph    为true时绘制IMU预积分连线
 * @param bDrawOptLba       为true时绘制优化后的关键帧
 */
void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba)
{
    // 历史关键帧图标
    const float &w = mKeyFrameSize; // 设置关键帧图标的 大小
    const float h = w*0.75;         // 设置关键帧图标的 高度
    const float z = w*0.6;          // 设置关键帧图标的 深度

    // step 1：取出所有的关键帧
    Map* pActiveMap = mpAtlas->GetCurrentMap();     // 获取当前地图
    // DEBUG LBA
    std::set<long unsigned int> sOptKFs = pActiveMap->msOptKFs;     // 优化后的关键帧
    std::set<long unsigned int> sFixedKFs = pActiveMap->msFixedKFs; // 固定的关键帧

    // 如果当前地图为空
    if(!pActiveMap)
        return;

    // 获取当前地图的 所有关键帧
    const vector<KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();

    // step 2：显示所有关键帧图标
    if(bDrawKF)
    {
        // 遍历所有关键帧
        for(size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];       // 取出关键帧
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();       // 获取关键帧的位姿
            unsigned int index_color = pKF->mnOriginMapId;

            glPushMatrix();         // 保存当前矩阵
            // 因为OpenGL中的矩阵为列优先存储，因此实际为Tcw，即相机在世界坐标下的位姿
            glMultMatrixf((GLfloat*)Twc.data());    // 设置当前矩阵为Twc,Twc为关键帧的位姿

            // 如果该关键帧 是当前地图的第一帧
            if(!pKF->GetParent()) // It is the first KF in the map
            {
                glLineWidth(mKeyFrameLineWidth*5);    // 设置绘制图形时线的 宽度
                glColor3f(1.0f,0.0f,0.0f);  // 红色
                // 用线将下面的顶点两两相连
                glBegin(GL_LINES);
            }
            // 不是第一帧
            else
            {
                //cout << "Child KF: " << vpKFs[i]->mnId << endl;
                glLineWidth(mKeyFrameLineWidth);    // 设置绘制图形时线的宽度
                // 如果绘制优化后的关键帧开关为true
                if (bDrawOptLba) {
                    // 如果该关键帧 是 优化后的关键帧
                    if(sOptKFs.find(pKF->mnId) != sOptKFs.end())
                    {
                        glColor3f(0.0f,1.0f,0.0f); // Green -> Opt KFs，绿色
                    }
                    // 如果该关键帧 是 固定的关键帧
                    else if(sFixedKFs.find(pKF->mnId) != sFixedKFs.end())
                    {
                        glColor3f(1.0f,0.0f,0.0f); // Red -> Fixed KFs，红色
                    }
                    else
                    {
                        glColor3f(0.0f,0.0f,1.0f); // Basic color，蓝色
                    }
                }
                // 不绘制优化后的关键帧
                else
                {
                    glColor3f(0.0f,0.0f,1.0f); // Basic color，蓝色
                }
                glBegin(GL_LINES);
            }

            // 下面的这些代码就是为了绘制关键帧的图标
            glVertex3f(0,0,0);  // 设置顶点是关键帧的中心
            glVertex3f(w,h,z);      // 设置顶点是关键帧图标的右上角
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();    // 结束绘制

            glPopMatrix();  // 恢复之前的矩阵

            glEnd();
        }
    }// 绘制完了所有关键帧

    // step 3：显示所有关键帧的Essential Graph (本征图)，通过显示界面选择是否显示关键帧连接关系。
    // 已知共视图中存储了所有关键帧的共视关系，本征图中对边进行了优化，保存了所有节点，只存储了具有较多共视点的边,用于进行优化，
    // 而生成树则进一步进行了优化,保存了所有节点,但是值保存具有最多共视地图点的关键帧的边
    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);   // 设置线的宽度
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        // cout << "-----------------Draw graph-----------------" << endl;
        // 遍历每一个关键帧
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            // step 3.1 共视程度比较高的共视关键帧用线连接
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);     // 得到和它共视程度比较高的关键帧
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();       // 得到 它在世界坐标系下的相机坐标
            // 如果找到共视的关键帧
            if(!vCovKFs.empty())
            {
                // 循环所有的共视信息
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    // 单向绘制
                    if((*vit)->mnId < vpKFs[i]->mnId)
                        continue;
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();        // 得到共视关键帧在世界坐标系下的相机坐标
                    glVertex3f(Ow(0),Ow(1),Ow(2));      // 设置顶点是关键帧的中心
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));   // 设置顶点是关键帧图标的右上角
                }
            }

            // Spanning tree
            // step 3.2 连接最小生成树
            KeyFrame* pParent = vpKFs[i]->GetParent();      // 得到该关键帧的父节点
            // 如果父节点存在
            if(pParent)
            {
                Eigen::Vector3f Owp = pParent->GetCameraCenter();       // 得到父节点在世界坐标系下的相机坐标
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }

            // Loops
            // step 3.3 连接闭环时形成的连接关系
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();         // 得到该关键帧形成的闭环关系
            // 遍历所有的闭环关系
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                // 单向绘制
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
    }// 绘制关键帧图结束

    // 如果显示惯性图
    if(bDrawInertialGraph && pActiveMap->isImuInitialized())
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(1.0f,0.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        //Draw inertial links
        // 遍历所有的关键帧
        for(size_t i=0; i < vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();       // 得到关键帧在世界坐标系下的相机坐标
            KeyFrame* pNext = pKFi->mNextKF;        // 得到关键帧的下一个关键帧
            // 如果下一个关键帧存在
            if(pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();     // 得到下一个关键帧在世界坐标系下的相机坐标
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }
        }

        glEnd();
    }

    // 得到所有的地图
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();

    // 如果显示关键帧
    if(bDrawKF)
    {
        // 遍历所有的地图
        for(Map* pMap : vpMaps)
        {
            // 如果是当前地图
            if(pMap == pActiveMap)
                continue;

            vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();  // 得到所有的关键帧

            // 遍历所有的关键帧
            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                unsigned int index_color = pKF->mnOriginMapId;      // 关键帧的颜色

                glPushMatrix();

                glMultMatrixf((GLfloat*)Twc.data());

                // 如果是地图的第一个关键帧
                if(!vpKFs[i]->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth*5);
                    glColor3f(1.0f,0.0f,0.0f);
                    glBegin(GL_LINES);
                }
                else
                {
                    glLineWidth(mKeyFrameLineWidth);
                    glColor3f(mfFrameColors[index_color][0],mfFrameColors[index_color][1],mfFrameColors[index_color][2]);
                    glBegin(GL_LINES);
                }

                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }
    }
}

// 画当前相机的位置
void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

// 设置当前相机的位姿
void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.inverse();
}

// 得到当前OpenGL的相机矩阵
void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    Eigen::Matrix4f Twc;
    {
        unique_lock<mutex> lock(mMutexCamera);
        Twc = mCameraPose.matrix();
    }

    for (int i = 0; i<4; i++) {
        M.m[4*i] = Twc(0,i);
        M.m[4*i+1] = Twc(1,i);
        M.m[4*i+2] = Twc(2,i);
        M.m[4*i+3] = Twc(3,i);
    }

    MOw.SetIdentity();
    MOw.m[12] = Twc(0,3);
    MOw.m[13] = Twc(1,3);
    MOw.m[14] = Twc(2,3);
}
} //namespace ORB_SLAM
