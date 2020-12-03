#pragma once
#include <opencv2/core.hpp>
#include <mlib/utils/cvl/matrix.h>

namespace cvl {

class HirSample{
public:
    HirSample(std::vector<cv::Mat1w> images,
              cv::Mat1f disparity,
              cv::Mat1b labels,
              bool training,
              int sequenceid,
              int frameid):images(images),
        disparity(disparity), labels(labels),training_(training),
        sequenceid_(sequenceid), frameid_(frameid){}
    HirSample(std::vector<cv::Mat1w> images,
              cv::Mat1f disparity,
              cv::Mat1b labels,
              int sequenceid,
              int frameid):images(images),
        disparity(disparity), labels(labels),training_(true),
        sequenceid_(sequenceid), frameid_(frameid){}
    float getDim(double row, double col);
    float getDim(Vector2d rowcol);
    Vector3d get_3d_point(double row, double col);
    bool is_car(double row, double col);
    bool is_car(Vector2d rowcol);


    cv::Mat1b disparity_image_grey(); // rescaled for visualization, new clone
    cv::Mat3b disparity_image_rgb(); // for visualization, new clone
    cv::Mat3b rgb(uint id); // for visualization, new clone
    cv::Mat1b gray(uint id); // for visualization, new clone
    cv::Mat3b show_labels();// for visualization, new clone
    uint rows();
    uint cols();
    int training(){return training_;}
    int frameid();
    int sequenceid();
    void set_use_labels(bool v){
        use_labels=v;
    }
    std::vector<cv::Mat1w> images;    
    cv::Mat1f disparity; // holds floating point disparities
private:


    cv::Mat1b labels;
    bool training_;
    int sequenceid_;
    int frameid_;
    bool use_labels=false;
};

class DaimlerSample;
std::shared_ptr<HirSample> convert_2_hir_sample(std::shared_ptr<DaimlerSample> sd);
namespace kitti{
class KittiOdometrySample;
}
std::shared_ptr<HirSample> convert_2_hir_sample(std::shared_ptr<kitti::KittiOdometrySample> sd);

}
