#include <iostream>
#include <mutex>
#include <thread>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sqlite_orm.h>


#include <daimler/dataset.h>

#include <mlib/utils/serialization.h>
#include <mlib/opencv_util/imshow.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/mlog/log.h>
#include <mlib/vis/mlib_simple_point_cloud_viewer.h>

#include <mlib/plotter/plot.h>
#include <imo.h>






namespace fs = std::experimental::filesystem;

using namespace cvl;
using std::cout;
using std::endl;
using namespace sqlite_orm;

struct RAIIThread{ // use raiithread.thr=std::thread([](){...})
    std::thread thr;
    ~RAIIThread(){
        if(thr.joinable())
            thr.join();
    }
};


template<class T> std::string str(const std::vector<T>& ts){
    if(ts.size()==0) return "[]";
    std::stringstream ss;
    ss<<"[ "<<ts[0];
    for(uint i=1;i<ts.size();++i)
        ss<<", "<<ts[i];
    ss<<" ]";
    return ss.str();
}

template<class Key,class Value> std::vector<Key> keys(const std::map<Key,Value>& map){
    std::vector<Key> keys;keys.reserve(map.size());
    for(const auto& [key,val]:map)
        keys.push_back(key);
    return keys;
}
template<class Key,class Value> std::vector<Value> vals(const std::map<Key,Value>& map){
    std::vector<Value> vals;vals.reserve(map.size());
    for(const auto& [key,val]:map)
        vals.push_back(val);
    return vals;
}
template<class Key,class Value> std::vector<uint> vals_sizes(const std::map<Key,Value>& map){
    std::vector<uint> vals;vals.reserve(map.size());
    for(const auto& [key,val]:map)
        vals.push_back(val.size());
    return vals;
}
template<class T> T sum(const std::vector<T>& ts){
    uint i=0;
    for(const auto& t:ts)
        i+=t;
    return i;
}


void set_label(cv::Mat& im, std::string label, cv::Point2i origin, mlib::Color col=mlib::Color::white())
{
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 2;
    int thickness = 2;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, origin + cv::Point(0, baseline), origin + cv::Point(text.width, -text.height), CV_RGB(0,0,0), cv::FILLED);
    cv::putText(im, label, origin, fontface, scale, col.toScalar<cv::Scalar>(), thickness, 8);
}




void draw_box(cv::Mat& im, BoundingBox box, bool gt=true){
    auto color=mlib::Color::nrlarge(box.id).toScalar<cv::Scalar>();
    int thickness=8;
    if(!gt){
        color=mlib::Color::white().toScalar<cv::Scalar>();
        thickness=4;
    }
    int r0=int(box.row_start);
    int c0=int(box.col_start);
    int r1=int(box.row_end);
    int c1=int(box.col_end);
    cv::Point2i a(c0,r0);
    cv::Point2i b(c0,r1);
    cv::Point2i c(c1,r0);
    cv::Point2i d(c1,r1);
    cv::line(im,a,b,color,thickness);
    cv::line(im,a,c,color,thickness);
    cv::line(im,d,b,color,thickness);
    cv::line(im,d,c,color,thickness);
}

struct Key{
    int gt,imo;
    double iou;
};
bool operator<(Key a, Key b){
    if(a.gt<b.gt) return true;
    if(a.imo<b.imo) return true;
    if(a.iou<b.iou) return true;
    return false;
}
// for a single frame!
// gt->imo, one to one!
std::vector<Key> greedy_associate(std::map<Key,double> map){
    std::vector<Key> ret;ret.reserve(map.size());
    while(map.size()>0)
    {
        // find the max iou for any combo
        double max_iou =-2;
        Key max_key;
        for(auto [key, iou]:map){
            if(iou>max_iou){
                max_iou=iou;
                max_key=key;
            }
        }
        if(max_iou<=0) continue;
        ret.push_back(max_key);
        std::map<Key,double> map2;
        //then remove all keys with that gt or imo
        for(auto it=map.begin();it!=map.end();++it){
            if(it->first.gt ==max_key.gt || it->first.imo==max_key.imo){
                // map.erase(it); // does not work, but it really should...
                continue;
            }
            map2[it->first]=it->second;
        }
        map=map2;
    }
    return ret;
}




class State{
public:

    int max_frame_id=7000;

    State(std::string path,
          std::string gt_path,
          std::string result_path):
        dm{DaimlerDataset(path,gt_path)},
        result_storage(
            sqlite_orm::make_storage(
                result_path,
                sqlite_orm::make_table(
                    "imo_results",
                    sqlite_orm::make_column("uid", &Imoresrow::uid,
                                            sqlite_orm::autoincrement(),
                                            sqlite_orm::primary_key()),
                    sqlite_orm::make_column("frame_id", &Imoresrow::frame_id),
                    sqlite_orm::make_column("imo_id", &Imoresrow::imo_id),
                    sqlite_orm::make_column("pose0", &Imoresrow::pose0),
                    sqlite_orm::make_column("pose1", &Imoresrow::pose1),
                    sqlite_orm::make_column("pose2", &Imoresrow::pose2),
                    sqlite_orm::make_column("pose3", &Imoresrow::pose3),
                    sqlite_orm::make_column("pose4", &Imoresrow::pose4),
                    sqlite_orm::make_column("pose5", &Imoresrow::pose5),
                    sqlite_orm::make_column("pose6", &Imoresrow::pose6),

                    sqlite_orm::make_column("fxm_x", &Imoresrow::fxm_x),
                    sqlite_orm::make_column("fxm_y", &Imoresrow::fxm_y),
                    sqlite_orm::make_column("fxm_z", &Imoresrow::fxm_z),

                    sqlite_orm::make_column("xm_x", &Imoresrow::xm_x),
                    sqlite_orm::make_column("xm_y", &Imoresrow::xm_y),
                    sqlite_orm::make_column("xm_z", &Imoresrow::xm_z),


                    sqlite_orm::make_column("row_start", &Imoresrow::row_start),
                    sqlite_orm::make_column("col_start", &Imoresrow::col_start),
                    sqlite_orm::make_column("row_end", &Imoresrow::row_end),
                    sqlite_orm::make_column("col_end", &Imoresrow::col_end)
                    ),
                sqlite_orm::make_table(
                    "egomotion",
                    sqlite_orm::make_column("uid", &Egomotionrow::uid, sqlite_orm::autoincrement(), sqlite_orm::primary_key()),
                    sqlite_orm::make_column("frame_id", &Egomotionrow::frame_id),
                    sqlite_orm::make_column("pose0", &Egomotionrow::pose0),
                    sqlite_orm::make_column("pose1", &Egomotionrow::pose1),
                    sqlite_orm::make_column("pose2", &Egomotionrow::pose2),
                    sqlite_orm::make_column("pose3", &Egomotionrow::pose3),
                    sqlite_orm::make_column("pose4", &Egomotionrow::pose4),
                    sqlite_orm::make_column("pose5", &Egomotionrow::pose5),
                    sqlite_orm::make_column("pose6", &Egomotionrow::pose6)),
                sqlite_orm::make_table(
                    "timing",
                    sqlite_orm::make_column("uid", &Timingrow::uid, sqlite_orm::autoincrement(), sqlite_orm::primary_key()),
                    sqlite_orm::make_column("frame_id", &Timingrow::frame_id),
                    sqlite_orm::make_column("time_ns", &Timingrow::time_ns)),
                sqlite_orm::make_table(
                    "velocity",
                    sqlite_orm::make_column("uid", &Vdata::uid,
                                            sqlite_orm::autoincrement(),
                                            sqlite_orm::primary_key()),
                    sqlite_orm::make_column("frameid", &Vdata::frameid),
                    sqlite_orm::make_column("imoid", &Vdata::imoid),
                    sqlite_orm::make_column("imo_vx", &Vdata::imo_vx),
                    sqlite_orm::make_column("imo_vy", &Vdata::imo_vy),
                    sqlite_orm::make_column("imo_vz", &Vdata::imo_vz),
                    sqlite_orm::make_column("ekf_vx", &Vdata::ekf_vx),
                    sqlite_orm::make_column("ekf_vy", &Vdata::ekf_vy),
                    sqlite_orm::make_column("ekf_vz", &Vdata::ekf_vz))
                )){
        ///home/mikael/daimler_output/2020-02-21_11-14-44.220/results/0008.db.sqlite
        //  fs::path p=result_path;
        //    p.parent_path()
        //      cout<<"result_path: "<<result_path<<endl;
        //        exit(1);
        // only check as far as the results go:

        // select max(frame_id) from imo_results

        //max_frame_id= ;
        // fixed towards gt!
        cout<<"gt max_frame_id: "<<max_frame_id<< "result max frame id: "<<*result_storage.max(&Imoresrow::frame_id) <<endl;

    }
    void init()
    {
        cv::namedWindow("labels", cv::WINDOW_GUI_EXPANDED);
        cv::namedWindow("left", cv::WINDOW_GUI_EXPANDED);
    }
    void update(uchar key){

        switch (key)
        {
        case 'a':{ frameid=std::max(frameid-speed,0); return; }
        case 'd':{ frameid=std::min(frameid+speed, (int)dm.samples()-1); return; }
        case 'w':{ speed++; if(speed>50) speed=50; return; }
        case 's':{ speed--; if(speed<1) speed=1; return; }
        case 'e':{
            show_errors();return;
        }
        case 'v':{
            show_velocities();return;
        }
        case 'm':{
            show_egomotion();return;
        }
        case 27:{ done=true; return; }

            // no op
        case 255: { return ; }
        default:{ cout<<"unknown key: "<<key<< " " <<uint(key)<<endl; return;}
        }
    }
    void show_egomotion()
    {
        auto res=result_storage.get_all<Egomotionrow>(where(c(&Egomotionrow::frame_id) < max_frame_id));
        std::vector<PoseD> ps;
        for(auto p:res){
            ps.push_back(p.pose());
        }
        mlib::pc_viewer("egomotion")->setPointCloud(ps);

    }
    void help(){
        cout<<"use wasd to navigate the sequence, a,d for earlier, later, and s,w as speed. \n Press e to generate the result text and plots.\n That will take a while... "<<endl;
    }
    void read(){

        if(frameid!=lastframeid ){
            sd=dm.get_sample(frameid);
            lastframeid=frameid;
        }
    }

    void paint()
    {

        read();
        if(!sd) return;
        // copy from sd to buffers
        cv::Mat3b left = sd->rgb(0);
        cv::Mat3b labels = sd->show_labels();
        cv::Mat3b dim = sd->disparity_image_rgb();


        std::vector<std::pair<cv::Mat3b,std::string>> imgs;
        imgs.push_back(std::make_pair(left,"left"));
        imgs.push_back(std::make_pair(labels,"labels"));
        imgs.push_back(std::make_pair(dim,"disparity"));


        for(auto im:imgs){
            // draw stuff:
            set_label (im.first,mlib::toZstring(sd->frameid()),cv::Point2i(50,60));set_label (im.first,mlib::toZstring(speed,0),cv::Point2i(50,120));
            set_label (im.first,mlib::toZstring(object_id),cv::Point2i(50,160), mlib::Color::red());
            // draw bounding boxes!
            for(mtable::GTRow row:dm.gt_storage.get_all<mtable::GTRow>(
                    where(c(&mtable::GTRow::frame_id) == frameid))){
                draw_box(im.first,row.bb());
            }
            for(auto irr:result_storage.get_all<Imoresrow>(where(c(&Imoresrow::frame_id) == frameid)))
                draw_box(im.first, irr.bb(), false);

            // draw the image
            imshow(im.first,im.second);
        }
    }



    std::vector<int> fids;
    std::vector<int> get_frame_ids()
    {
        if(fids.size()>0) return fids;
        // all frame ids in gt!
        auto gts=dm.gt_storage.get_all<mtable::GTRow>(where(c(&mtable::GTRow::frame_id) < max_frame_id));

        for(auto gt:gts) fids.push_back(gt.frame_id);
        return fids;
    }


    auto get_first_frameid_of(int imo_id){
        return dm.gt_storage.get_all<mtable::GTRow>(where(c(&mtable::GTRow::imo_id) == imo_id),order_by(&mtable::GTRow::frame_id),limit(1));
    }




    std::vector<mtable::GTRow> filter_gt_imos(std::vector<mtable::GTRow> in){
        std::vector<mtable::GTRow> out;
        out.reserve(in.size());

        for(mtable::GTRow gtrow: in){
            // skip small imos!
            if(dm.gt_storage.count<mtable::GTRow>(where(c(&mtable::GTRow::imo_id) == gtrow.imo_id))<5)
                continue;
            // stuff near the image edge wont work, because the tracker and stereo will fail.
            // arguably more representative
            if(gtrow.bb().near_image_edge(50, 1024, 2048)) continue;
            // skip the first frame, as its not possible to succeed there more representative?
            auto gtrs=get_first_frameid_of(gtrow.imo_id); // guaranteed to be one because of the earlier check
            //if(gtrs.at(0).frame_id == frame_id) continue;

            out.push_back(gtrow);
        }
        sort(out.begin(), out.end(), [](mtable::GTRow a, mtable::GTRow b) {return a.area() > b.area(); });
        return out;

    }




    std::atomic<bool> association_done{false};
    std::mutex ass_mtx;
    void associate_gt2imo(){
        std::unique_lock<std::mutex> ul(ass_mtx);
        if(association_done) return;

        int total_missed_detections=0;
        int total_extra_uids=0;

        std::vector<double> ious;
        std::map<int,int> missed_detections;
        std::map<int,int> id_switches;
        std::map<int,int> total_gt; // gt id, gt annotation count
        std::map<int,std::set<int>> uids;
        // gt->(frame->(imo,iou))
        std::map<int,std::map<int,std::tuple<int,double>>> associations;


        // this takes a while...
        std::atomic<bool> waiting{true};
        auto print_waiting=[&waiting](){
            std::cout<<"associating ground truth to imo: ";
            std::vector<char> cs{'-','|','/','-','\\'};
            int i=0;
            while(waiting)
            {

                std::cout<<'\b'; // backspace
                std::cout<<cs[i++%cs.size()];
                std::cout.flush();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::cout<<endl;
        };
        RAIIThread thr;        thr.thr=std::thread(print_waiting);




        ious.reserve(100000);


        for(int frame_id: get_frame_ids())
        {
            std::vector<mtable::GTRow> gtrows = dm.gt_storage.get_all<mtable::GTRow>(
                        where(c(&mtable::GTRow::frame_id) == frame_id));
            gtrows=filter_gt_imos(gtrows);
            if(gtrows.size()==0) continue;
            auto imoresrows = result_storage.get_all<Imoresrow>(
                        where(c(&Imoresrow::frame_id) == frame_id));
            //if(imoresrows.size()==0) continue; // so missed detections get counted properly!

            // so there are gt and imos, now lets compute their matching scores.
            std::map<Key,double> scores;
            for(const auto& gt:gtrows)
            {
                total_gt[gt.imo_id]++;

                associations[gt.imo_id][frame_id]=std::make_tuple(-1,0);

                for(const auto& imores:imoresrows)                {
                    double iou=gt.bb().iou(imores.bb());
                    if(iou<0.2) continue;
                    Key key{gt.imo_id,imores.imo_id,iou};
                    scores[key]=iou;
                }
            }
            //cout<<"fid:  "<<frame_id<< " scores: " <<scores.size()<<endl;
            // then lets associate, greedy will do...
            std::vector<Key> matches=greedy_associate(scores);

            // this list has gt,imo,iou, and the first two are unique.
            for(const Key& key:matches)
                associations[key.gt][frame_id]=std::make_tuple(key.imo,key.iou);
        }


        // now we have the association sequences...
        // first compute the ious, including all the zeros...
        // gt->(frame->(imo,iou))

        for(auto& [gt,ass]:associations)
            for(auto& [frame, match]:ass)
                ious.push_back(std::get<1>(match));
        // next compute the number of id switches. Or maybe the number of unique ids? or both?!

        for(auto& [gt,ass]:associations){
            id_switches[gt]=0;
            auto& ids=id_switches[gt];
            auto& us=uids[gt];
            auto& md=missed_detections[gt];

            md=0; // technically allready init as 0
            if(ass.size()==0) continue;
            int id=-1;
            for(auto& [frame, match]:ass){

                int mid=std::get<0>(match);
                double iou=std::get<1>(match);
                if(iou<0.0 ||mid<0){
                    md++;
                    continue;
                }
                if(id==mid) continue;
                id=mid;
                ids++;
                us.insert(id);
            }
        }
        // now id_switches is id_switches+1, if it found one at all.
        for(auto& [gt,ids]:id_switches) if(ids>0) ids--;






        for(auto [imo,us]:uids)
            if(us.size()>1)
                total_extra_uids+=us.size()-1;
        for(auto [gt, md]:missed_detections)
            total_missed_detections+=md;




        // compute iou fractions from ious, note they include the missed ones as zeros.
        std::map<double, double> iou_fractions;
        for(int i=0;i<100;++i) iou_fractions[i]=0; // so none is missing!

        for(int i=0;i<100;++i){
            double key=i;
            auto& ifi=iou_fractions[key];
            for(auto iou:ious)
                if(iou>=key/100.0)ifi+=1.0;
            ifi=ifi/double(ious.size());
        }

        auto xs=keys(iou_fractions);
        auto ys=vals(iou_fractions);
        std::stringstream ss;
        ss<<"total_id_switches:       "<<sum(vals(id_switches))<<endl;
        ss<<"total_extra_uids:        "<<total_extra_uids<<endl;
        ss<<"total missed detections: "<<sum(vals(missed_detections))<<endl;
        ss<<"total gtimo annotations: "<<sum(vals(total_gt))<<endl;
        ss<<"id_switches       = " + str(vals(id_switches))<<endl;
        ss<<"uids:             = " + str(vals_sizes(uids))<<endl;
        ss<<"missed_detections = " + str(vals(missed_detections))<<endl;
        //ss<<"gt_propagations   = " + str(vals(gt_id_propagations))<<endl;
        ss<<"\n";
        ss<<"iou_fractions = "+ str(vals(iou_fractions))<<endl<<"\n";
        plot(xs,ys,"iou_fractions");
        waiting=false;

        cout<<ss.str()<<endl;
                association_done=true;
    }



    std::vector<int> get_imo_ids(){
        std::set<int> ids;

        auto rrs= result_storage.get_all<Imoresrow>(where(c(&Imoresrow::frame_id) < max_frame_id));
        for(auto r:rrs)
            ids.insert(r.imo_id);
        std::vector<int> ret;
        for(auto i:ids)
            ret.push_back(i);
        return ret;
    }
    std::map<int, PoseD> egomotion_poses(){
        std::map<int, PoseD> ps;
        auto rrs= result_storage.get_all<Egomotionrow>();
        for(Egomotionrow ego:rrs)
            ps[ego.frame_id]=ego.pose();
        return ps;
    }
    void show_velocities(){
        auto ids=get_imo_ids();
        auto ps=egomotion_poses();
        std::map<int,std::vector<double>> vs;
        for(int id:ids)
        {
            std::vector<double> velocity;
            auto rrs= result_storage.get_all<Imoresrow>(where(c(&Imoresrow::frame_id) < max_frame_id and
                                                              c(&Imoresrow::imo_id) == id));
            cout<<"found rows: "<<rrs.size()<<endl;
            for(uint i=3;i<rrs.size();++i){
                Imoresrow p=rrs[i-3];
                Imoresrow c=rrs[i];

                // coordinates in world!
                Vector3d px=ps[p.frame_id].inverse()*p.fxm_in_camera();
                Vector3d cx=ps[c.frame_id].inverse()*c.fxm_in_camera();

                velocity.push_back((cx - px).norm());
            }
            if(velocity.size()>20)
                vs[id]=velocity;
        }
        // now write it to matlab friendly text...
        // find the longest, then pad the rest with nan
        uint len=0;
        for(auto [id, v]:vs)
            if(len<v.size())
                len=v.size();
        for(auto& [id,v]:vs)
            v.resize(len,NAN);
        std::stringstream ss;
        ss<<"vs = [";
        for(auto  [id, v]:vs){


            for(uint i=0;i<v.size();++i){
                ss<<v[i];
                if(i!=v.size()-1)
                    ss<<",";
            }
            ss<<";\n";

        }
        ss<<"]\n";
        cout<<ss.str()<<endl;;


    }



    std::mutex show_error_mtx;
    RAIIThread rthr;
    std::atomic<bool> in_progress{false};
    void show_errors(){ // the do unless cache available is a nice pattern, as is do in different thr unless cache...
        std::unique_lock<std::mutex> ul(show_error_mtx);
        if(in_progress) return;
        in_progress=true;
        //rthr.thr=std::thread([this](){this->associate_gt2imo();}); // gets oom, also probably trouble with the db and async
        associate_gt2imo();
    }

    bool get_done(){
        return done;
    }
private:

    int frameid=0;
    int speed=1;
    int lastframeid=-1;
    bool done=false;
    int object_id=0;
    DaimlerDataset dm;
    mtable::result_db_type result_storage;
    std::shared_ptr<DaimlerSample> sd=nullptr;
    std::string matlab_results_path;
};


void show_result(std::string sequence_path, std::string gt_path, std::string result_path){
    State state(sequence_path,gt_path, result_path);
    state.init();
    state.help();
    cout<<"state inited!"<<endl;
    while(!state.get_done()){
        state.update(uchar(cv::waitKey(1000/50)));
        state.paint();
    }
}

int main(int argc, char** argv){
    //initialize_plotter();
    //cv::Mat1b im(100,100);    cv::imshow("test",im);
    if(!(argc==4)){
        cout<< "make -j8 && ./apps/show_result dataset_path gt_path result_path"<<endl;
        return 0;
    }
    if(!fs::exists(fs::path(argv[2]))){
        cout<<"gt database file "+ std::string(argv[2])<<" does not exist";
        return 0;
    }
    if(!fs::exists(fs::path(argv[3]))){
        cout<<"result file "+ std::string(argv[3])<<" does not exist";
        return 0;
    }

    show_result(argv[1], argv[2], argv[3]);
    return 0;
}
