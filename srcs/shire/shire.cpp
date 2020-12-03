#include <string>
#include <chrono>
#include <thread>
#include <future>
#include <experimental/filesystem>
#include <fstream>

#include <shire/shire.h>


#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/highgui.hpp>

#include <mlib/vis/flow_viewer.h>
#include <mlib/utils/vector.h>


#include <mlib/utils/cvl/triangulate.h>

#include <mlib/utils/vector.h>
#include <mlib/utils/string_helpers.h>


#include <mlib/opencv_util/imshow.h>

#include <mlib/sfm/p3p/pnp_ransac.h>
#include <mlib/sfm/anms/grid.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/vis/mlib_simple_point_cloud_viewer.h>

#include <utils/vis_helpers.h>
#include <utils/flowfield_helpers.h>



#include <sfm/map.h>
#include <sfm/imo.h>
#include <sfm/reprojection_error.h>
#include <sfm/measurement.h>
#include <mlib/opencv_util/draw_arrow.h>
#include <mlib/opencv_util/cv.h>





namespace fs = std::experimental::filesystem;


using std::endl;
using std::cout;
using std::cerr;
namespace cvl{


Shire::Shire():params("HirStream"){

    tracker=std::make_shared<KLT_CUDA_Tracker::CKLT_Tracker<unsigned short>>("tracker");
    retracker=std::make_shared<KLT_CUDA_Tracker::CKLT_Tracker<unsigned short>>("retracker");

    params.add( PH_BEGIN_GROUP( "Shire", "plain hir stream parameters!" ) );
    {
        params.add(tracker->getParamSet());
        params.add(retracker->getParamSet());
        //params.add(ekf.getParamSet());
    }
    params.add( PH_END_GROUP );
    //auto tmp=mlib::pc_viewer("bundled trajectories and imo trajectories");

}



std::string Shire::get_output_directory(){
    return "/home/mikael/daimler_output/"+ mlib::getNospaceIsoDateTimeofStart()+"/";
}

void Shire::save_image(cv::Mat3b im,
                                std::string name,
                                int frameid){
    if(save_images){
        std::string path=get_output_directory()+name+"/"+mlib::toZstring(frameid)+".png";
        safe_save_image(im,path);
    }
}

std::string nospace_date(){
    std::string datetime=mlib::getIsoDateTime();
    for(uint i=0;i<datetime.size();++i)
        if(datetime[i]==' ')datetime[i]='-';
    return datetime;
}

void makedir(std::string path){
    std::string cmd="mkdir -p \"" + path +  "\"";
    int err=system(cmd.c_str());
    if(err>0) std::cout<<cmd<<" went wrong"<<endl;
}

void Shire::set_min_creation_count(int count){
    min_creation_count=count;
}
void Shire::init(int rows, int cols)
{
    cout<<"ran init!"<<endl;
    if(inited) return;
    tracker->setImageFormat(cols,rows);
    retracker->setImageFormat(cols,rows);

    inited=true;
    map=Map::create(15);
}
uint Shire::min_inlier_count(){if(min_creation_count<10) return 5; return min_creation_count-5;}
std::shared_ptr<PoseImo> Shire::hir(std::vector<std::shared_ptr<Measurement>>& ms){
    cout<<"Hir: "<<ms.size()<<endl;
    ms=not_nullptr_and_has_previous_filter(ms);

    if(ms.size()<min_creation_count) return nullptr;





    std::vector<std::shared_ptr<Measurement>> tracks;
    tracks.reserve(ms.size());

    //PoseD Pwc=ms[0]->get_pose().inverse();
    //PoseD Pwp=ms[0]->getPrevious()->get_pose().inverse();
    cout<<"prelim"<<endl;
    for(uint i=0;i<ms.size();++i){
        if(ms[i]->label==0) continue;
        if(ms[i]->feature->size()<2) continue;

        // prefilter the measurements for world movement!
        // not they are size 2 at this point, so world velocity is easy...
        //auto& m=ms[i];
        //Vector3d x=Pwc*m->triangulate();
        //auto p=m->getPrevious();
        //Vector3d px=Pwp*p->triangulate();
        // approximate velocity for the track in m/s
        // the difference must be atleast 1.5 in disparity?
        // should be atleast 5 km/h
        // if((x-px).norm()*20*3.6<5) continue;
        tracks.push_back(ms[i]);
    }
    if(tracks.size()<min_creation_count) return nullptr;
    auto imo=PoseImo::create(tracks, min_creation_count);
    if(imo!=nullptr){
        map->add_imo(imo);
    }
    cout<<"Hir Done"<<endl;
    return imo;
}









Vector3d toVector3d(mlib::Color color){
    return Vector3d(color[2],color[1],color[0])/255.0;
}
Vector3d imo2world(std::shared_ptr<Measurement>& m){
    PoseD p_c_world=*(m->tp.lock()->Pnw); // current world time
    auto imo=m->feature->get_imo();
    PoseD p_c_imo=*imo->getPose(m->frameid); // imo at current world time
    Vector3d x_world=p_c_world.inverse()*p_c_imo*m->feature->X;
    return x_world;
}





std::vector<bool> anms_pool(CFeaturePool& pool,
                            std::vector<std::shared_ptr<Measurement>>& previous,
                            double radius){

    std::vector<bool> out;
    anms::GridSolver grid;
    std::vector<anms::Data> datas;
    datas.reserve(pool.getSize());

    out.resize(pool.getSize(),false);
    for(uint i=0;i<pool.getSize();++i){
        if(pool.getArray()[i].state!=TS_TRACKED) continue;
        float str=0;
        if(previous[i])
            str=previous[i]->feature->size();
        datas.push_back(anms::Data(str,float(pool.getArray()[i].u_d),float(pool.getArray()[i].v_d),i));
    }



    std::vector<anms::Data> locked;
    grid.init(datas,locked,0,0,1024*2,2*1024);

    grid.compute(radius,0);

    auto tmp=grid.filtered;

    for(auto t:tmp){
        assert(t.id<out.size());
        out[t.id]=true;
    }
    cout<<"done anmsing"<<endl;
    return out;
}


ParamSet* Shire::getParamSet(){
    return &params;
}
PnpParams Shire::get_pnp_params()
{
    PnpParams params;
    params.max_iterations=2000;
    params.min_iterations=200;
    params.threshold=reprojection_threshold/HirCalibration::common().fx();
    //params.reference=*(map->previous()->Pnw);
    //params.max_angle=30*3.1415/180.0;
    return params;
}

void refine_all(std::vector<std::shared_ptr<Measurement>>& ms){
    std::vector<std::future<void>> fs;fs.reserve(ms.size());
    for(auto& m:ms){
        if(!m->feature->has_imo())
            fs.push_back(std::async(std::launch::async,[&](){m->feature->refine(true);}));
    }
    for(auto& f:fs){
        f.wait();
    }
}

bool in_forbidden_zone(Vector2d y){
    double row=y[0];
    double col=y[1];
    return row>842;
}



std::map<std::shared_ptr<PoseImo>, std::tuple<Vector3d,BoundingBox>>
get_centers(std::vector<std::shared_ptr<Measurement>>& ms)
{
    mlog()<<"get centers\n";
    std::map<std::shared_ptr<PoseImo>, std::vector<Vector3d>> centers;
    std::map<std::shared_ptr<PoseImo>,BoundingBox> bs;
    for(auto& m:ms){
        auto& f=m->feature;
        std::shared_ptr<PoseImo> imo=f->get_imo();
        if(imo==nullptr) continue;
        if(imo->getPose(m->frameid)==nullptr) continue;
        centers[imo].reserve(100);
        centers[imo].push_back(f->X);
        auto y=m->y;
        if(bs.find(imo)==bs.end())
            bs[imo]=BoundingBox(imo->id,y[0],y[1],y[0],y[1]);
        bs[imo].include(m->y);
    }
    std::map<std::shared_ptr<PoseImo>, std::tuple<Vector3d,BoundingBox>> cs;
    for(auto [imo, c]:centers){
        cs[imo]=std::make_tuple(median_centroidvs(c),bs[imo]);
    }
    return cs;
}




void Shire::operator()(std::shared_ptr<HirSample>& sd){


    sd->set_use_labels(use_labels);

    //cout<<"\n\n\n"<<endl;


    if(sd==nullptr) {        return ;    }
    init(HirCalibration::common().rows(),
         HirCalibration::common().cols());

    if(first)
    {
        cout<<"firt frame proc"<<endl;
        // just detect new features,
        tracker->track(sd->images[0].data);raw_pool=tracker->getFeaturePool();

        if(retracker) retracker->track(sd->images[0].data);
        // read all the measurements
        std::shared_ptr<TimePoint> tp=map->createTimePoint( sd->frameid());
        map->push(tp);
        tp->ms.reserve(tracker->getFeaturePool().getSize());
        auto& ms=tp->ms;
        for(SFeature_t f:tracker->getFeaturePool()){
            if(!f.found()) continue;
            auto ykl=f.rc<Vector2d>();

            double disp=sd->getDim(ykl);
            if(disp<0.1) continue;
            // creates measurement and feature
            auto m=tp->createMeasurement(ykl,disp,sd->is_car(ykl));
            m->feature->X=m->triangulateWorld();
            ms.push_back(m);
        }
        map->push(tp);
        first=false;
        if(run_ekf) ekf.init();
        return;
    }



    auto previous=map->previous();
    // track previous
    auto& previous_ms=previous->ms;
    std::shared_ptr<TimePoint> tp=map->createTimePoint( sd->frameid());
    map->push(tp);




    {
        timers.tic("track");
        // track
        cout<<"tracking : everything assumes tracker and retracker has the same max track count!"<<endl;
        std::vector<Vector2d> ys;ys.reserve(previous_ms.size());
        for(auto& m:previous_ms) ys.push_back(m->yl());

        tracker->getFeaturePool().set_points_to_track(ys);
        tracker->track(sd->images[0].data); raw_pool=tracker->getFeaturePool();// tracks all, both imo and not

        timers.toc("track");
        timers.tic("total");
        timers.tic("pnp");

        { // compute world pose
            cout<<"compute world pose"<<endl;
            auto& pool=tracker->getFeaturePool();
            std::vector<Vector3d> xs;xs.reserve(previous_ms.size());
            std::vector<Vector2d> yns;yns.reserve(previous_ms.size());

            cout<<"previous_ms.size(): "<<previous_ms.size()<<endl;
            for(uint i=0;i<pool.getSize() && i< previous_ms.size();++i) {
                auto f=pool.getArray()[i];
                if(!f.tracked())  continue;
                auto& p=previous_ms[i];
                auto ykl=f.rc<Vector2d>();
                auto& ff=p->feature;
                if(ff->has_imo()) continue;
                // should be a world inlier, previously

                if(reprojection_threshold<ff->error()) continue;
                // should I use history or all?
                // tempting to use all without history!

                xs.push_back(ff->X);
                //xs.push_back(previous_world_pose.inverse()*p->triangulate());
                yns.push_back(HirCalibration::common().undistort(ykl));
            }


            // now we have all xs, yns, so compute world pnp
            // compute pnp
            {
                cout<<"computing pnp"<<endl;
                // filter forth the ms
                timers.tic("pnp ransac"); // about 0.5 + 1.5 ms
                cout<<"pnp_datacount: "<<xs.size()<<endl;
                *(tp->Pnw)=pnp_ransac(xs, yns, get_pnp_params()); // 1.5ms
                timers.toc("pnp ransac");
                // now lets consider the inlier count!?
            }
            timers.toc("pnp");
        }
    }




    if(retrack_world)
    {
        timers.tic("retrack");
        // predict the positions of all old
        // then track...
        // if the retracked fails to hit world, keep original track destination
        cout<<"retracking "<<endl;
        std::vector<Vector2d> ys;ys.reserve(previous_ms.size());
        std::vector<Vector2d> yps;yps.reserve(previous_ms.size());
        PoseD Pnw=*(tp->Pnw);
        for(auto& m:previous_ms)
        {
            ys.push_back(m->yl());
            yps.push_back(HirCalibration::common().project(Pnw*m->feature->X));
        }
        retracker->getFeaturePool().set_points_to_track(ys,yps);
        retracker->track(sd->images[0].data); // tracks all, both imo and not
        // oki everything has been tracked, and retracked.

        // now see which if any of pool should have their answers replaced
        // same for both tracker and retracker!

        cout<<"update according to retrack"<<endl;
        auto& pool=tracker->getFeaturePool();
        auto& repool=retracker->getFeaturePool();

        int n=0;
        for(uint i=0;i<previous_ms.size();++i)
        {           auto& p=previous_ms[i];

            // if track is not imo
            if(p->feature->has_imo()) continue;
            // if track failed to track or new position has bad disparity
            auto f=repool.getArray()[i];
            if(!f.tracked()) continue;
            double disp=sd->getDim(f.rc<Vector2d>());;
            if(disp<0.1) continue;

            // if old track was on world and retrack is on world, keep it.
            if((p->feature->error()<reprojection_threshold) &&
                    (f.rc<Vector2d>() - yps[i]).norm()<reprojection_threshold){// we are really just checking if the right coordinates were good

                if((pool.getArray()[i].rc<Vector2d>() -yps[i]).norm()>3) n++;
                pool.getArray()[i]=f; // use the retracked one!
            }
        }
        cout<<"replaced "<<n<<" tracks"<<endl;
        timers.toc("retrack");

        if(retrack_world_redo_pnp){
            timers.tic("pnp2");

            // compute world pose
            cout<<"compute world pose again"<<endl;
            auto& pool=tracker->getFeaturePool();
            std::vector<Vector3d> xs;xs.reserve(previous_ms.size());
            std::vector<Vector2d> yns;yns.reserve(previous_ms.size());

            cout<<"previous_ms.size(): "<<previous_ms.size()<<endl;
            for(uint i=0;i<pool.getSize() && i< previous_ms.size();++i) {
                auto f=pool.getArray()[i];
                if(!f.tracked())  continue;
                auto& p=previous_ms[i];
                auto ykl=f.rc<Vector2d>();
                auto& ff=p->feature;
                if(ff->has_imo()) continue;
                // should be a world inlier, previously

                if(reprojection_threshold<ff->error()) continue;
                // should I use history or all?
                // tempting to use all without history!

                xs.push_back(ff->X);
                //xs.push_back(previous_world_pose.inverse()*p->triangulate());
                yns.push_back(HirCalibration::common().undistort(ykl));
            }


            // now we have all xs, yns, so compute world pnp
            // compute pnp
            {
                cout<<"computing pnp"<<endl;
                // filter forth the ms
                timers.tic("pnp ransac2"); // about 0.5 + 1.5 ms
                cout<<"pnp_datacount: "<<xs.size()<<endl;
                *(tp->Pnw)=pnp_ransac(xs, yns, get_pnp_params()); // 1.5ms
                timers.toc("pnp ransac2");
                // now lets consider the inlier count!?
            }
            timers.toc("pnp2");
        }
    }

    {
        cout<<"filtering broken disparity"<<endl;
        // lets filter away all the ones with broken disparity!
        auto& pool=tracker->getFeaturePool();
        for(uint i=0;i<pool.getSize();++i){
            auto f=pool.getArray()[i];
            double disp=sd->getDim(f.rc<Vector2d>());
            if(f.found()||f.tracked())
                if(disp<1) f.set_invalid();
        }
    }


    { // now we need to remove merged tracks,
        //we do this by imo status, then age,
        timers.tic("anms");
        cout<<"doing anms"<<endl;
        // get all the positions and their strength

        auto& pool=tracker->getFeaturePool();
        std::vector<ETrackingState_t> states;
        states.reserve(pool.getSize());
        std::vector<anms::Data> datas;
        for(uint i=0;i<pool.getSize();++i) {
            auto& f=pool.getArray()[i];
            states.push_back(f.state);

            if(!(f.found()||f.tracked())) continue;
            f.set_invalid();
            if(f.row()<50) continue;

            double str=0;
            if(i<previous_ms.size())
            {
                auto& m= previous_ms[i];
                if(m->feature->has_imo()) str+=1000;
                str+=m->feature->size();
            }
            datas.push_back(anms::Data(str,f.rc<Vector2d>(),i));
        }
        // do anms
        // set all bad ones to invalid
        anms::GridSolver gs;
        std::vector<anms::Data> locked;
        gs.init(datas,locked,0,0,HirCalibration::common().rows(),
                HirCalibration::common().cols());
        gs.compute(8,500);
        for(anms::Data data:gs.filtered){
            pool.getArray()[data.id].state = states[data.id];
        }
        timers.toc("anms");
    }


    // oki, now the pool contains all the relevant tracks,
    // and retrack or not no longer matters.
    {
        timers.tic("parse pool");
        PoseD Pwn=(*(tp->Pnw)).inverse();
        auto& pool=tracker->getFeaturePool();
        cout<<"update tracks and add tracks"<<endl;

        // first filter the tracks and detected tracks using anms?
        tp->ms.reserve(pool.getSize());

        // same for both tracker and retracker!
        for(uint i=0;i<pool.getSize();++i) {

            auto f=pool.getArray()[i];
            if(!(f.tracked()||f.found())) continue;
            auto ykl=f.rc<Vector2d>();
            double disp=sd->getDim(ykl);
            if(disp<1) continue;
            if(in_forbidden_zone(ykl)) continue;

            if(f.tracked() && i<previous_ms.size()) {
                tp->ms.push_back(
                            tp->createMeasurement(ykl, disp,
                                                  sd->is_car(ykl),
                                                  previous_ms[i]));
            }
            if(f.found()){
                // creates measurement and feature
                auto m=tp->createMeasurement(ykl,disp,sd->is_car(ykl));
                m->feature->X=Pwn*m->triangulate();
                tp->ms.push_back(m);
            }
            // ignore the rest
        }
        timers.toc("parse pool");
    }


    // adaptive anms for less dense on world! // hmm may look wierd...
    {
        // lets see if I can get the new pnp working as a step 0
        timers.tic("feature refine");
        std::vector<std::shared_ptr<Measurement>> world_ms;
        world_ms.reserve(tp->ms.size());
        //for(auto& m:tp->ms) m->feature->refine(false); // so this is as good as it gets
        refine_all(tp->ms);
        timers.toc("feature refine");
    }

    if(pnp_only){
        display(sd,nullptr);
        //bundle_egomotion();
        return;
    }


    // now I have tracked and computed the 3d coordinates of all in world.
    // perform pnp and BA for each imo!
    std::vector<std::shared_ptr<PoseImo>> imos; // imos found in this one...
    {
        cout<<"Perform pnp for each imo:"<<endl;
        timers.tic("imo pnp");
        // get the imo measurements
        std::map<std::shared_ptr<PoseImo>,std::vector<std::shared_ptr<Measurement>>> imoms =
                sort_by_imo(tp->ms);
        // update the estimate for each imo given its measurements
        for(auto& e:imoms)
        {
            std::shared_ptr<PoseImo> imo=e.first;
            auto& ms=e.second;
            cout<<"imo pnp: "<<imo->id<<" "<<ms.size()<<endl;
            if(imo->estimate(ms,min_inlier_count(),reprojection_threshold)){
                imos.push_back(imo);
                cout<<"found imo!"<<endl;
            }
            else{
                mlog()<<"lost imo\n";
            }
        }
        timers.toc("imo pnp");
    }

    // assign new tracks to imos if they are in its sphere?
    auto centers=get_centers(tp->ms);
    // identify bad tracks, see if shortening them and reseting them works, if not,
    // they are new imo material


    timers.tic("test features");
    std::vector<std::shared_ptr<Measurement>> imo_candidates;
    imo_candidates.reserve(tp->ms.size());

    cout<<"identify outliers"<<endl;
    // note if it was an outlier to the imo its in, it has already been split and refined
    // not so for world
    for(std::shared_ptr<Measurement>& m:tp->ms){
        if(m->feature->size()<2) continue;
        //if(m->feature->has_imo())            //cout<<"error:"<<m->feature->error()<<endl;
        if(m->feature->error()<reprojection_threshold) continue;
        m->feature->split_track();
        //m->feature->clear_imo();
        if(m->is_car()) imo_candidates.push_back(m);
    }

    std::vector<std::shared_ptr<Measurement>> new_imo_candidates;
    new_imo_candidates.reserve(imo_candidates.size());
    // assign outliers to existing imo!
    bool assign_outliers_to_imo=true;
    if(assign_outliers_to_imo)
    {
        cout<<"reassign outliers"<<endl;
        // compute some stats for the imos

        // assign outliers to existing imo
        // identify imos, once imo always the same imo!
        // try to assign each candidate to a existing imo in order
        // none of these fit the world and none of them have an imo

        // lets use a minimum difference to second best or else toss them...
        std::set<std::shared_ptr<PoseImo>> changed_imos;
        for(auto& imo_candidate:imo_candidates)
        {
            // find best fit
            std::shared_ptr<PoseImo> best_imo=nullptr;
            double besterr=50;
            double second_best_err=besterr+50;

            Vector3d best_X;
            for(std::shared_ptr<PoseImo>& imo:imos)
            {
                // updates the feature coordinates to that of the imo...
                double err=imo->getMinReprojectionDistance(imo_candidate->feature);
                // dont compare distance, compare disparity of center vs disparity of point!
                // atleast if far away, say disparity of center less than 10
                double dist=(std::get<0>(centers[imo]) - imo_candidate->feature->X).length();
                /*
                double feature_disparity=HirCalibration::common().disparity(((*imo->getPose(tp->frameid))*imo_candidate->feature->X)[2]);
                double center_disparity=HirCalibration::common().disparity(centers[imo][2]);
                if(center_disparity<10)
                dist=std::abs(center_disparity - feature_disparity);
                */
                // the distance check is a bit broken!

                if(err<besterr && dist<10) {
                    second_best_err=besterr;
                    besterr=err;
                    best_imo=imo;
                    best_X=imo_candidate->feature->X;
                }
                // compare its position to the current mean for the imo in question
            }
            //

            if(besterr<reprojection_threshold &&
                    (second_best_err>disambiguation_threshold)){
                // this is not the source of the problem!
                auto f0=imo_candidate->feature;
                auto f=imo_candidate->feature->split_track();
                f0->clear_imo();
                imo_candidate->feature->set_imo(best_imo);

                // also remove all but two, or check for inlier or the ms on result?
                imo_candidate->feature->X=best_X;
                changed_imos.insert(best_imo);
                continue;
            }
            if(besterr>candidate_threshold)
            {
                imo_candidate->feature->clear_imo();
                new_imo_candidates.push_back(imo_candidate);
                // maybe also split it here?
            }
        }
        // probably not important
        //for(auto& imo:changed_imos) imo->refine(reprojection_threshold);
    }



    timers.toc("test features");


    auto outliers=new_imo_candidates;

    // now search for new imos in the newcandidates,
    //the ones which dont fit this are plain outliers and will be removed
    timers.tic("hir");
    auto new_imo=hir(new_imo_candidates);
    timers.toc("hir");

    // lets clear up outliers, but only the really bad ones!
    for(auto& out:outliers){
        if(out->feature->has_imo()) continue; // imo got created!
        out->feature->split_track();
    }

    // split too long tracks
    if(false){
        for(auto& m:tp->ms){
            m->feature->split_long_track(); // modifies tp ms so must be prefiltered...
        }
    }
    timers.toc("total");
    //timers.toc("shire total");
    display(sd, new_imo);

    if(map->getSize()>0 && map->getSize() % 1000 ==0 && save_result)
        save_results(map->getSize()/1000);
    printTimers();
}













void Shire::bundle_egomotion(){
    if(map->getSize()<2) return;
    timers.tic("problemtimer");

    ceres::Problem problem;
    auto tps=map->get_last_n_timepoints(10);

    for(uint i=0;i<tps.size();++i)
    {
        auto tp=tps[i];
        auto ms=not_nullptr_and_has_previous_and_not_imo_filter(tp->ms);

        cout<<"Features found: "<<ms.size()<<endl;
        bool any=false;
        for(auto& m:ms)
        {

            if(m->feature->size()<2) continue;
            double err=m->reprojectionErrorWorld();
            if(err<3)
            {
                auto resid=StereoReprojectionError::Create(m->obs());
                problem.AddResidualBlock(resid,nullptr,
                                         tp->Pnw->getRRef(),
                                         tp->Pnw->getTRef(),
                                         m->feature->X.begin());
                any=true;
            }
            // what to do otherwize?
        }
        if(!any){
            cout<<"Warning missing pose in BA"<<endl;
            continue;
        }

        ceres::LocalParameterization* qp = new ceres::QuaternionParameterization;
        problem.SetParameterization(tp->Pnw->getRRef(), qp);

        if(i==0)
        { // first is always constant if present.
            problem.SetParameterBlockConstant(tps.at(0)->Pnw->getRRef());
            problem.SetParameterBlockConstant(tps.at(0)->Pnw->getTRef());
        }
    }
    timers.toc("problemtimer");
    timers.tic("batimer");

    ceres::Solver::Options options;{

        options.linear_solver_type = ceres::SPARSE_SCHUR;// default, good
        options.max_num_iterations=5;
        options.num_threads = std::thread::hardware_concurrency();
    }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout<<"Full Summary: \n"<<summary.FullReport()<<endl;
    timers.toc("batimer");
}


void Shire::compute_velocity_results(
        std::vector<std::shared_ptr<Measurement>> ms,
        std::vector<Vector7d> states)
{
#if 0
    mlog()<<"gotten states\n";
    assert(ms.size()==states.size());// same size as pool on last filter update


    // now for each imo, find measurements and state velocity pairs
    std::map<int, std::vector<std::pair<Vector3d,Vector3d>>> vs,vs2;

    for(uint i=0;i<ms.size();++i){
        auto& m=ms[i];
        auto& s=states[i];
        if(m==nullptr) continue;
        if(s[6]==0) continue;
        if(m->feature->size()<2) continue;
        auto imo=m->feature->get_imo();
        int imoid=0; // world,
        if(imo!=nullptr) imoid=imo->id;

        Vector3d imo_v=m->feature->world_velocity();
        Vector3d ekf_v(s[3],s[4],s[5]);
        if(imo_v.isnormal() && ekf_v.isnormal()){
            vs2[imoid].reserve(500);
            vs2[imoid].push_back(std::make_pair(imo_v,ekf_v));
        }
    }
    // only when we have atleast 20 of each
    for(auto v:vs2)
        if(v.second.size()>20){
            vs[v.first]=v.second;
        }
    // oki now compute the mean of each...
    std::map<int,std::pair<Vector3d,Vector3d>> means;
    for(auto v:vs){
        std::vector<std::pair<Vector3d,Vector3d>> obs=v.second;
        Vector3d mean_imo_v(0,0,0);
        Vector3d mean_ekf_v(0,0,0);
        double count=obs.size();
        for(auto [i,e]:obs){
            mean_imo_v+=i/count;
            mean_ekf_v+=e/count;
        }
        means[v.first]=std::make_pair(mean_imo_v, mean_ekf_v);
    }
    // oki now compute the variance of each...
    std::map<int, std::pair<Vector3d,Vector3d>> vars;
    for(auto v:vs)
    {
        std::vector<std::pair<Vector3d,Vector3d>> obs=v.second;
        Vector3d mean_imo_v=means[v.first].first;
        Vector3d mean_ekf_v=means[v.first].first;
        Vector3d var_imo_v(0,0,0);
        Vector3d var_ekf_v(0,0,0);

        double count=obs.size();
        for(auto [i,e]:obs)
        {
            var_imo_v+=(i-mean_imo_v).squaredNorm()/count;
            var_ekf_v+=(i-mean_ekf_v).squaredNorm()/count;
        }
        vars[v.first]=std::make_pair(var_imo_v, var_ekf_v);
    }
    // now I have a bunch of observations of the mean and variance of the imos velocity.


    // then do the same for the ages?








    for(uint i=0;i<ms.size();++i){


        auto& m=tp->ms[i];
        auto& s=states[i];
        if(m==nullptr) continue;

        if(m->feature->get_imo()==nullptr) continue;
        if(m->feature->size()<3) continue;
        if(s[6]==0) continue;


        Vector3d imo_v=m->feature->world_velocity();
        Vector3d ekf_v(s[3],s[4],s[5]);




        vdatas.reserve(1000000);
        vdatas.push_back(Vdata{int(m->feature->size()),
                               tp->frameid,
                               m->feature->get_imo()->id,
                               imo_v,ekf_v});
    }
#endif
}

void Shire::display(std::shared_ptr<HirSample> sd,
                             std::shared_ptr<PoseImo> new_imo){
    if(!draw_anything_at_all) return;
    cout<<"display0"<<endl;
    if(display_index++ % draw_increment != 0) return;
    cout<<"display"<<endl;

    auto tp=map->previous(); // this is current
    auto previous=map->previous(1); // this is current
    if(tp==nullptr) return;
    if(previous==nullptr) return;
    /// collect all drawing and dataset genereation here...
    // draw raw tracks


    if(run_ekf && draw_anything_at_all)
    {
        mlog()<<"ekf draw\n";
        PoseD Pcp=((*tp->Pnw)*previous->Pnw->inverse());
        ekf.update(Pcp, tracker->getFeaturePool(),sd->disparity);

        std::shared_ptr<FlowField> ff=ekf.get_flowfield();

        std::shared_ptr<FlowField> ff2=get_imo_flow_field(tp);
        if(!(ff==nullptr ||ff2==nullptr)){
            mlog()<<"ekf dra3w\n";
            for(Flow& f:ff2->flows)
                f.color=Vector3d(0,0,255);
            ff->append(ff2);
            show_flow(ff,"ekf and imo flows");
            mlog()<<"shown flow\n";

            // oki, one per each measurement.
            compute_velocity_results(tp->ms,ekf.get_filter_state());
        }
    }



    if(draw_anything_at_all)
    {
        mlog()<<"anything draw\n";
        timers.tic("drawtimer");
        if(draw_new_imo && new_imo!=nullptr)
        {
            cv::Mat3b rgb=sd->rgb(0);


            const auto& fs=new_imo->get_features();
            std::vector<std::shared_ptr<Measurement>> ms; ms.reserve(fs.size());
            for(const auto& f:fs)
                ms.push_back(f->getMeasurements().back());
            // draw the found imo!

            for(std::shared_ptr<Measurement>& m:ms){
                mlib::drawArrow(rgb,
                                m->getPrevious()->yl(),
                                m->yl(),
                                new_imo->get_color());
            }
            {
                int fontface = cv::FONT_HERSHEY_SIMPLEX;
                double scale = 2;
                int thickness = 2;
                cv::Vec2i origin(100,100);
                cv::putText(rgb, "frameid: "+str(ms.at(0)->frameid), origin, fontface, scale, CV_RGB(255,255,255), thickness, 8);
                //cv::waitKey(0);
            }
            imshow("found imo: "+str(new_imo->id),rgb);
        }


        if(draw_raw_tracks){

            cv::Mat3b raw_tracks=draw_feature_pool(raw_pool, sd->rgb(0),sd->disparity);
            imshow("Raw Tracks", raw_tracks);
            if(retrack_world) {
                cv::Mat3b raw_tracks=draw_feature_pool(tracker->getFeaturePool(), sd->rgb(0),sd->disparity);
                imshow("Retracked Tracks", raw_tracks);
            }
            save_image(raw_tracks, "raw_tracks",tp->frameid);

            // show the disparity, because sometimes its bonkers!
            cv::Mat1b disp=sd->disparity_image_grey();

            imshow(disp,"disparity");
            imshow(sd->rgb(0)*0.5 + (sd->disparity_image_rgb()*0.5),"combined disp and rgb");
        }
        if(draw_world_tracks_b){
            // always satisfied since we jump after tracking on the first!
            cv::Mat3b world_tracks = draw_world_tracks(tp->ms,sd->rgb(0),reprojection_threshold);
            imshow(world_tracks, "World Tracks");
            /*
            if(tp->frameid==1066)
                imshow(world_tracks, "World Tracks 1066");
            if(tp->frameid==1067)
                imshow(world_tracks, "World Tracks 1067");
            if(tp->frameid==1068)
                imshow(world_tracks, "World Tracks 1068");
              if(tp->frameid==1167)
                  imshow(world_tracks, "World Tracks 1167");
              if(tp->frameid==1168)
                  imshow(world_tracks, "World Tracks 1168");
              if(tp->frameid==1169)
                  imshow(world_tracks, "World Tracks 1169");
              if(tp->frameid==1170)
                  imshow(world_tracks, "World Tracks 1170");
              if(tp->frameid==1171)
                  imshow(world_tracks, "World Tracks 1171");
              */
            save_image(world_tracks,"world_tracks",tp->frameid);
            cout<<"done drawing world tracks"<<endl;
        }


        if(show_flow_b){
            if (flow_index++ %50==0){
                //auto ff=get_imo_mean_flow_field(tp);
                auto ff=get_imo_flow_field(tp);
                show_flow(ff);
            }
            cout<<"done drawing flow"<<endl;
        }


        if(draw_candidates_b)
        {
            cv::Mat3b candidates=draw_candidates(tp->ms, sd->rgb(0), candidate_threshold);
            imshow(candidates,"candidates");
            save_image(candidates,"candidates",tp->frameid);
            cout<<"done drawing candidates"<<endl;
        }




        if(display_index++<100 || display_index++ % 500==0){

            mlib::pc_viewer("imo trajectories in world")->setPointCloud(map->get_imo_trajectories_in_world());
            auto xss=map->get_imo_mass_centers_in_world();

            std::vector<Vector3d> xs;
            std::vector<mlib::Color> cs;
            int i=0;
            for(auto& vs:xss){
                for(auto&v:vs.xms){
                    xs.push_back(v);
                    cs.push_back(vs.color);
                }
                i++;
            }

            mlib::pc_viewer("bundled trajectories and imo trajectories")->setPointCloud(xs,cs,map->getTrajectory());
            mlib::pc_viewer("live view")->setPointCloud(xs,cs,map->getTrajectory());

            {// compute a good pose for the visualizer to center on
                // lookat
                PoseD pcw=*tp->Pnw;

                mlib::pc_viewer("live view")->set_pose(lookAt(pcw.getTinW(), pcw.getTinW() -Vector3d(0,250,0),Vector3d(1,0,0)));

            }

        }
        cout<<"done drawing mass centers"<<endl;

        timers.toc("drawtimer");

        if(start_paused){
            char key=cv::waitKey(0);
            if(key==' ')
                start_paused=false;
            if(key=='s')
                save_results(1000);
        }

        if(make_flow_dataset_b){
            timers.tic("mkf");
            make_flow_dataset( tracker->getFeaturePool(), tp->ms,
                               sd->disparity,
                               get_output_directory(), sd->frameid());
            timers.toc("mkf");
        }

        if(save_verified_depths ){
            cv::Mat1f disparities(sd->rows(),sd->cols(),-1.0f);
            int count=0;
            for(auto& m:tp->ms){
                auto f=m->feature;
                if(f->size()<3) continue;
                if(f->error()>1.5) continue;
                auto y=HirCalibration::common().stereo_project((*tp->Pnw)*f->X);
                if(f->has_imo()){
                    if(f->imo_age()<3) continue;
                    auto pose=f->get_imo()->getPose(m->frameid);
                    if(pose==nullptr) continue;
                    y=HirCalibration::common().stereo_project((*tp->Pnw)*f->X);
                }
                // now we have the observation in current frame...
                double err=(y-m->obs()).norm();
                if(err>1.5) continue;
                disparities(y[0],y[1])=y[2];
                count++;
            }

            std::string path=get_output_directory()+"verified_depths/"+mlib::toZstring(sd->frameid())+".exr";
            safe_save_image(disparities,path);
            std::string metadata_path=get_output_directory()+"verified_depths/metadata.txt";

            if(!fs::exists(metadata_path)){
                std::ofstream ofs(metadata_path);
                ofs<<"frameid disparity_count velocity(ms/s)\n";
                ofs<<"0 0 0.\n";
            }
            std::ofstream ofs(metadata_path,  std::ofstream::app);
            ofs<<sd->frameid()<< " " <<count<< " "<< map->velocity()<<"\n";
        }
    }
}

void Shire::printTimers(){
    cout<<timers<<endl;
}

}
