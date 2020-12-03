#pragma once
/* ********************************* FILE ************************************/
/** \file    feature.h
 *
 * \brief    This header contains a imo capable feature
 *
 *
 * \remark
 *
 *
 * \author   Mikael Persson
 * \date     2017-01-01
 * \note BSD licence
 *
 *
 ******************************************************************************/
#include <ceres/ceres.h>
#include <mlib/utils/cvl/pose.h>
#include <sfm/calibration.h>
namespace cvl{

class StereoTriangulationError{
public:
    StereoTriangulationError(Matrix4d P,
                             Vector3d obs):
        P(P),obs(obs){}


    template <typename T>
    bool operator()(const T* const x,
                    T* residuals) const {

        Vector3<T> X=Vector3<T>::copy_from(x);
        Matrix4<T> PP(P);
        //Vector3<T> xr=(PP*X.homogeneous()).dehom();
        // technically as long as last row of p is 0 0 0 1, then I can do drop instead!
        Vector3<T> xr=(PP*X.homogeneous()).drop_last();
        Vector3<T> yr=HirCalibration::common().stereo_project<T>(xr);
        // The error is the difference between the predicted and the observed position.
        Vector3<T> diff=Vector3<T>(obs) - yr;
        //diff[2]*=T(0.5);
        /*
        if(HirCalibration::common().behind_either(xr)) {
            for(int i=0;i<3;++i)              residuals[i] =T(0);
            return true;
        }
        */
        for(int i=0;i<3;++i) residuals[i]=diff[i];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(Matrix4d P,Vector3d obs) {
        // resid,first param,second param, third param
        return (new ceres::AutoDiffCostFunction<StereoTriangulationError, 3, 3>(
                    new StereoTriangulationError(P,obs)));
    }
    Matrix4d P;
    Vector3d obs;
    double baseline;
};

class StereoReprojectionError {
public:
    StereoReprojectionError(Vector3d obs):obs(obs){}

    double getError(PoseD P, Vector3d X);

    template <typename T>
    bool operator()(const T* const rotation,
                    const T* const translation,
                    const T* const x,
                    T* residuals) const {



        Pose<T> P(rotation,translation,true);
        Vector3<T> xr=P*Vector3<T>::copy_from(x);
        Vector3<T> yr=HirCalibration::common().stereo_project(xr);
        Vector3<T> diff=Vector3<T>(obs) - yr;
        //diff[2]*=T(0.5);


        for(int i=0;i<3;++i)
            residuals[i]=diff[i];
        return true;
    }
    Vector3d obs; // row,col,disparity

    static ceres::CostFunction* Create(Vector3d obs){
        // resid,first param,second param, third param
        return (new ceres::AutoDiffCostFunction<StereoReprojectionError, 3, 4, 3, 3 >(
                    new StereoReprojectionError(obs)));
    }
};

} // end namespace cvl
