#pragma once
/* ********************************* FILE ************************************/
/** \file    measurement.h
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
#include <memory>
#include <vector>
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/bounding_box.h>


namespace  cvl {
class Feature;
class TimePoint;


class Measurement{
public:
    Measurement()=default;
    Measurement(Vector2d y, // left image
                double disparity, // corresp in right
                uint frameid, int label,
                std::shared_ptr<Feature>& f,
                std::weak_ptr<TimePoint>& tp):y(y),
        disparity(disparity),
        frameid(frameid),label(label), feature(f), tp(tp){
        assert(f!=nullptr);
    }

    Vector2d yl() const;
    Vector2d yr() const;// where it was observed in right based on disparity
    Vector2d ynl() const;
    Vector2d ynr() const;
    Vector3d obs()const{return Vector3d(y[0],y[1],disparity);}
    PoseD get_pose();

    Vector3d world_velocity();



    bool sameFeature(const std::shared_ptr<Measurement>& ms){
        if(feature==nullptr) return false;
        return(feature==ms->feature);
    }
    /// in the current frame coordinates!
    Vector3d triangulate() const;
    Vector3d triangulateWorld() const;
    //bool infront(const PoseD& Plw, const PoseD& Prw);
    /*
    void reset(){
        feature->imo=nullptr;
    }
    */
    double reprojectionErrorIMO();
    double reprojectionErrorWorld();

    double error();
    double reprojectionError(PoseD Plw);
    double reprojectionError(PoseD P, Vector3d x);


    std::shared_ptr<Measurement> getPrevious();

    double motionDisparity();

    Vector2d y;
    double disparity;
    uint frameid=-1;
    int label=0;
    bool is_car(){return label>0;}

    std::shared_ptr<Feature> feature=nullptr;
    std::weak_ptr<TimePoint> tp;
    static double weighted_reprojection_error(Vector3d obsr);

    // adds another measurement to the timepoint as a copy of this, with the new feature

private:

};

BoundingBox get_bounding_box(std::vector<std::shared_ptr<Measurement>>& ms, bool conservative=false);
} // end namespace cvl

