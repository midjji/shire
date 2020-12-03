#include <reprojection_error.h>
namespace cvl {

double StereoReprojectionError::getError(PoseD P, Vector3d X){
    Vector3d errs;

    (*this)(P.getRRef(),P.getTRef(),&X[0],errs.begin());

    return errs.norm();
}
}


