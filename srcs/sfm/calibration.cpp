#include <sfm/calibration.h>

namespace cvl{
HirCalibration HirCalibration::common_cal;


void HirCalibration::set_common_calibration(  double fy_,
double fx_,
double py_,
double px_,
                                double baseline_, double rows_, double cols_){
    common_cal.fy_=fy_;
    common_cal.fx_=fx_;
    common_cal.py_=py_;
    common_cal.px_=px_;
    common_cal.baseline_=baseline_;
    common_cal.rows_=rows_;
    common_cal.cols_=cols_;
}
void HirCalibration::set_common_calibration_pose(PoseD P){
    common_cal.Pcam_vehicle=P;
}
}

