#include <sstream>
#include <iostream>
#include <experimental/filesystem>

#include "paramHandling/h/paramseteditor_Qt4.h"
#include <plain_hir_stream.h>
#include <ip_utils.h>
#include <mlib/utils/buffered_dataset_stream.h>
#include <kitti/mots/dataset.h>
#include <sfm/sample.h>
#include <opencv2/highgui.hpp>
#include <sfm/calibration.h>
#include <mlib/cuda/mbm.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/opencv_util/stereo.h>
#include <mlib/utils/argparser.h>

namespace fs = std::experimental::filesystem;
using std::cout;using std::endl;

cvl::PlainHirStream hir;

void kitti_mots_lhir(int seq, int index){

    cout<<"starting lhir instance!"<<endl;

    // this is as fast as the parallel one assuming its parallelized internally...
    // the test is chocking on disk

    cvl::KittiMotsDataset dm("/storage/datasets/kitti/mots/");
    cvl::KittiMotsStereoCalibration s=dm.calibration(true,0);
    double rows=dm.rows(true,0);
    double cols=dm.cols(true,0);
    cvl::MBMStereoStream mbm;
    mbm.init(64,rows,cols);
    cout<<rows<< " "<<cols<<endl;

    cvl::HirCalibration::set_common_calibration(s.fy,s.fx,s.py,s.px,s.baseline,rows,cols);

    mlib::Timer timer("Total time: ");
    mlib::sleep_ms(2000);



    //int n=0;
    while(true){ // breaks on nullptr

        //if(n++ % 2>0) continue;
        timer.tic();
        std::shared_ptr<cvl::KittiMotsSample> sd=dm.get_sample(true,seq,index++);
        cout<<"read kittisample"<<endl;
        if(!sd) continue;
        /*
        cv::imshow("rgb0", sd->rgb(0));
        cout<<sd->rgb(0).rows<<endl;
        cout<<sd->rgb(0).cols<<endl;
        cv::imshow("rgb1", sd->rgb(1));
        cv::imshow("greyb0",sd->greyb(0));
        cv::imshow("greyb1",sd->greyb(1));
        cout<<sd->greyb(0).rows<<endl;
        cout<<sd->greyb(0).cols<<endl;
        cv::imshow("greyw0",sd->greyw(0));
        cv::imshow("greyw1",sd->greyw(1));
        cout<<sd->greyw(0).rows<<", "<<sd->greyw(0).cols<<endl;
        cv::waitKey(0);
        */

        std::vector<cv::Mat1w> images;
        images.push_back(sd->greyw(0));
        images.push_back(sd->greyw(1));
        cv::Mat1b labels(rows,cols,1);
        //cv::Mat1f disparity = mbm.disparity(sd->greyb(0),sd->greyb(1));
        cv::Mat1f disparity= cvl::stereo(sd->rgb(0),sd->rgb(1),128);
        std::shared_ptr<cvl::HirSample> sh=std::make_shared<cvl::HirSample>(images, disparity, labels,sd->training,sd->sequence,sd->frameid);
        hir(sh);
        timer.toc();
        cout<<timer<<endl;
        //cv::waitKey(0);
    }
}

int main(int argc, char** argv){

    mlib::ArgParser args;
    args.add_parameter("sequence","which sequence to use", "0");
    args.add_parameter("index","which index to start at", "0");




    QApplication                  myApp(argc,argv);
    CParameterSetEditorQt4        editor;
    editor.addRootParameterSet( hir.getParamSet() );
    cout<<"stargin ip config loop"<<endl;
    std::string parampath="/home/mikael/co/imo/parameters/kitti_imo.dat";

    if(!fs::exists(parampath)){
        cout<<"parameter path not found... "<<endl;
        exit(1);
    }
    editor.load(parampath);
    std::thread thr=std::thread(kitti_mots_lhir, args.param_double(),args.param_double()); // must wait untill ip_config has started!
    mlib::sleep(1);
    editor.show();
    myApp.exec();
    thr.join();
    return 0;
}

