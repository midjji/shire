#include <utils/show_pool.h>
#include <mlib/opencv_util/draw_arrow.h>
#include <mlib/opencv_util/cv.h>
namespace cvl{


cv::Mat3b draw_feature_pool(CFeaturePool& pool,
                            cv::Mat3b rgb){
    //cout<<"drawRawTracks"<<endl;

    cv::Mat3b im=rgb.clone();
    for(auto f:pool){
        if(f.found()){
            mlib::drawCircle(rgb,f.rc<Vector2d>(),mlib::Color::green());
        }
        if(f.tracked()){
        mlib::drawArrow(im,f.rc<Vector2d>(),f.previous_rc<Vector2d>(), mlib::Color::blue());
        }
    }
    //  cout<<"drawRawTracks - done"<<endl;
    return im;
}

cv::Mat3b draw_feature_pool_prediction(CFeaturePool& pool,
                                       cv::Mat3b rgb){


    //cout<<"drawRawTracks"<<endl;
    SFeature_t* arr=pool.getArray();



    for(uint i=0;i<pool.getSize();++i)
    {
        if(!(arr[i].state==TS_FOUND ||arr[i].state==TS_TRACKED)) continue;
        // draw predicted for both found and tracked!

        Vector2d y(arr[i].u_d,arr[i].v_d);
        y=y.reverse();
        Vector2d prev_y(arr[i].lastU_d,arr[i].lastV_d);
        prev_y=prev_y.reverse();
        mlib::drawArrow(rgb,y,prev_y, mlib::Color::blue());

        if(arr[i].state==TS_FOUND){
            Vector2d y(arr[i].u_d,arr[i].v_d);
            y=y.reverse();
            mlib::drawCircle(rgb,y, mlib::Color::green());
        }

    }

    //cout<<"drawRawTracks - done"<<endl;

    return rgb;

}
}// end namespace cvl
