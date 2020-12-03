
#include <iostream>
#include <daimler/dataset.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <mlib/utils/serialization.h>
#include <mlib/opencv_util/imshow.h>
#include <mlib/utils/string_helpers.h>
#include <mlib/utils/mlog/log.h>

#include <sqlite_orm.h>
#include <mlib/utils/colormap.h>


using namespace cvl;
using std::cout;
using std::endl;
using namespace sqlite_orm;



void set_label(cv::Mat& im, std::string label, cv::Point2i origin,
               mlib::Color col=mlib::Color::white())
{

    int fontface = 0;//cv::FONT_HERSHEY_SIMPLEX;
    double scale = 2;
    int thickness = 2;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, origin + cv::Point(0, baseline), origin + cv::Point(text.width, -text.height), CV_RGB(0,0,0), cv::FILLED);
    cv::putText(im, label, origin, fontface, scale, col.toScalar<cv::Scalar>(), thickness, 8);
}


void process_mouse(int event, int row, int col, int flags, void* state_);

void draw_box(cv::Mat& im, BoundingBox box){
    auto color=mlib::Color::nrlarge(box.id).toScalar<cv::Scalar>();
    int thickness=4;
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






Vector3d get_median_depth(sDaimlerSample sd,
                          BoundingBox bb){
    std::vector<Vector3d> xs;xs.reserve(int(bb.area()));
    for(int r=int(bb.row_start);r<int(bb.row_end);++r)
        for(int c=int(bb.col_start);c<int(bb.col_end);++c)
            if(sd->is_car(Vector2d(c, r)) &&
                    sd->getDim(Vector2d(c, r))>0)
                xs.push_back(sd->get_3d_point(r,c));
    if(xs.size()==0) return Vector3d(0,0,0);
    std::vector<std::vector<double>> dss;dss.resize(3);
    for(auto x:xs)
        for(int i=0;i<3;++i)
            dss[i].push_back(x[i]);
    for(auto& ds:dss)
        std::sort(ds.begin(), ds.end());

    return Vector3d(dss[0][dss[0].size()/2],dss[1][dss[1].size()/2],dss[2][dss[2].size()/2]);
}


class State{
public:



    mtable::gt_db_type storage;
    bool play=false;
    DaimlerDataset dm;





    uint object_offset=0;
    State(std::string path,std::string gt_path):dm{DaimlerDataset(path, gt_path)},
        storage(make_storage(gt_path,
                             make_table("boundingboxes",
                                        make_column("id", &mtable::GTRow::uid, autoincrement(), primary_key()),
                                        make_column("frame_id", &mtable::GTRow::frame_id),
                                        make_column("imo_id", &mtable::GTRow::imo_id),
                                        make_column("row_start", &mtable::GTRow::row_start),
                                        make_column("col_start", &mtable::GTRow::col_start),
                                        make_column("row_end", &mtable::GTRow::row_end),
                                        make_column("col_end", &mtable::GTRow::col_end),
                                        make_column("x", &mtable::GTRow::x),
                                        make_column("y", &mtable::GTRow::y),
                                        make_column("z", &mtable::GTRow::z)))){


        storage.sync_schema();

        //storage.update_all<mtable::GTRow>(set(c(&mtable::GTRow::row_start) = 0.0f), where(c(&mtable::GTRow::row_start) < 0.0f ));
        //storage.update_all<mtable::GTRow>(set(c(&mtable::GTRow::col_start) = 0.0f), where(c(&mtable::GTRow::col_start) < 0.0f ));

    }
    void init()
    {
        cv::namedWindow("labels", cv::WINDOW_GUI_EXPANDED);
        cv::namedWindow("left", cv::WINDOW_GUI_EXPANDED);
        cv::setMouseCallback("labels", process_mouse, this);
        cv::setMouseCallback("left", process_mouse, this);
    }
    void update(uchar key){
        if(play){
            if(key==' ') play=false;
            frameid++;return;
        }
        switch (key)
        {
        case 'a':{ frameid=std::max(frameid-speed,0); return; }
        case 'd':{ frameid=std::min(frameid+speed, (int)dm.samples()-1); return; }
        case 'w':{ speed++; if(speed>50) speed=50; return; }
        case 's':{ speed--; if(speed<1) speed=1; return; }

        case 27:{ done=true; return; } // escape
        case 'c':{ clear_imobb(frameid, object_id); return;}

        case '+':{ object_id++; return;}
        case '-':{ if(object_id>0) object_id--; return;}
        case 'f':{ snap_bounding_box2_label(); return;}
        case ' ': {frameid=0; speed=1;play=true;return;}
        case 'g': {snap_bounding_box2_label();frameid=std::max(frameid-speed,0); return;}

            // no op
        case 255: { return ; }
        default:{ cout<<"unknown key: "<<key<< " " <<uint(key)<<endl; help(); return;}
        }
    }
    void help(){
        cout<<"use wasd to navigate the sequence, a,d for earlier, later, and s,w as speed."<<endl;
        cout<<"use escape to quit, use +,- to change what object id you are annotating\n";
        cout<<"use f to snap to car segmentation, use g to snap and go one step back.\n";
        cout<<"left doubleclick the object of interest to get its object id.";
        cout<<"drag middle mouse button to add or adjust bounding box of selected object id\n";
        cout<<"c to clear bounding box of current id.\n";
        cout<<"Annotations are saved online, but do use esc to quit!"<<endl;
    }
    void read(){
        if(frameid!=lastframeid ){
            sd=dm.get_sample(frameid);
            lastframeid=frameid;
        }
    }
    void clear_imobb(int frame_id, int imo_id){
        storage.remove_all<mtable::GTRow>(
                    where(c(&mtable::GTRow::frame_id) == frame_id and c(&mtable::GTRow::imo_id) == imo_id));
    }
    void snap_bounding_box2_label(){

        // compute new bb!
        std::vector<mtable::GTRow> earlier = storage.get_all<mtable::GTRow>(
                    where(c(&mtable::GTRow::frame_id) == frameid and c(&mtable::GTRow::imo_id) == object_id));
        if(earlier.size()==0) return;
        if(earlier.size()>1) {
            cout<<"WTF?!"<<endl;
            exit(1);
        }

        auto bb=earlier.at(0).bb();
        cv::Mat1b tmp=sd->labels;

        // where does the ones start along the rows?
        std::vector<bool> cs;cs.resize(tmp.cols,false);
        std::vector<bool> rs;rs.resize(tmp.rows,false);
        cout<<bb<<endl;

        for(int r=int(bb.row_start);r<int(bb.row_end);++r)
            for(int c=int(bb.col_start);c<int(bb.col_end);++c){
                if(tmp(r,c)>0){
                    cs.at(c)=true;
                    rs.at(r)=true;
                }
            }

        int csmin=0;
        int rsmin=0;
        int csmax=int(cs.size()-1);
        int rsmax=int(rs.size()-1);

        {
            for(;csmin<int(cs.size());++csmin)
                if(cs[csmin])
                    break;
            for(;rsmin<int(rs.size());++rsmin)
                if(rs[rsmin])
                    break;
            for(int i=cs.size()-1; i>csmin;--i)
                if(cs[i]) {csmax=i;break;}
            for(int i=rs.size()-1; i>rsmin;--i)
                if(rs[i]) {rsmax=i;break;}
        }

        mtable::GTRow row=earlier.at(0);
        cout<<"snap2: "<<row.row_start<<" "<<rsmin<<" "
           <<row.col_start<<" "<<csmin<<" "
          <<row.row_end<<" "<<rsmax<<" "
         <<row.col_end<<" "<<csmax<<endl;
        // needs some margin!
        float boundrary=0;
        row.row_start=std::max(float(rsmin-boundrary),row.row_start) ;
        row.col_start=std::max(float(csmin -boundrary), row.col_start);
        row.row_end=std::min(float(rsmax +boundrary), row.row_end);
        row.col_end=std::min(float(csmax +boundrary), row.col_end);

        set_or_update(row);
        // store new bb
    }

    bool is_id_in_use(int id){
        return storage.count<mtable::GTRow>(where(c(&mtable::GTRow::imo_id) == id))>0;
    }
    void set_or_update(mtable::GTRow row){
        std::vector<mtable::GTRow> earlier = storage.get_all<mtable::GTRow>(
                    where(c(&mtable::GTRow::frame_id) == row.frame_id and c(&mtable::GTRow::imo_id) == row.imo_id));
        if(earlier.size()==0)
            storage.insert(row);
        if(earlier.size()==1)
        {
            row.uid=earlier.at(0).uid;
            storage.update(row);
        }
        if(earlier.size()>1){cout<<"Database corruption, "<<endl;exit(1);}
    }

    void add_box(Vector2d a, Vector2d b){
        if(((a-b).length()>100) && (std::abs(a[0]-b[0])>50) && (std::abs(a[1]-b[1])>50))
        {
            // reshuffle the points,
            if(a[0]>b[0]) std::swap(a[0],b[0]);
            if(a[1]>b[1]) std::swap(a[1],b[1]);

            if(a[0]<0) a[0]=0;
            if(a[1]<0) a[1]=0;

            if(b[0]>=sd->dim.rows) b[0]=sd->dim.rows-1;
            if(b[1]>=sd->dim.cols) b[1]=sd->dim.cols-1;
            cout<<"imsize: "<<sd->dim.rows<< " "<<sd->dim.cols<<endl;


            auto box = BoundingBox(object_id,a[0],a[1],b[0],b[1]);
            mtable::GTRow row{-1, frameid, object_id,
                        float(box.row_start), float(box.col_start), float(box.row_end), float(box.col_end),
                        0.0f,0.0f,0.0f};
            set_or_update(row);
        }
    }


    void paint()
    {

        read();
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
            set_label (im.first,mlib::toZstring(sd->frameid()),
                       cv::Point2i(50,60));
            set_label (im.first,mlib::toZstring(speed,0),cv::Point2i(50,120));
            if(is_id_in_use(object_id))
                set_label (im.first,mlib::toZstring(object_id),cv::Point2i(50,160), mlib::Color::green());
            else
                set_label (im.first,mlib::toZstring(object_id),cv::Point2i(50,160), mlib::Color::red());





            auto rows=storage.get_all<mtable::GTRow>(where(c(&mtable::GTRow::frame_id) == frameid));
            for(mtable::GTRow row:rows){
                if(row.xm().length()<0.1){
                    //compute and set xm
                    row.set_xm(get_median_depth(sd,row.bb()));
                    storage.update(row);
                }
            }

            // draw bounding boxes!
            for(mtable::GTRow row:rows)
                draw_box(im.first,row.bb());

            if(drag_started)
                draw_box(im.first,BoundingBox(object_id,drag_start[0],drag_start[1],drag_ongoing[0],drag_ongoing[1]));
            // draw the image
            imshow(im.first,im.second);
        }
    }
    bool get_done()
    {
        return done;
    }
    void set_imo_id2(int row0, int col0)
    {
        for(mtable::GTRow row:storage.get_all<mtable::GTRow>(where(c(&mtable::GTRow::frame_id) == frameid))){
            if(row.bb().in(row0,col0))
                object_id=row.imo_id;
        }
    }
    Vector2d drag_ongoing=Vector2d(0,0);
    Vector2d drag_start=Vector2d(0,0);
    bool drag_started=false;

    void set_start(uint start){
        frameid=start;
    }
private:


    int frameid=0;
    int speed=1;
    int lastframeid=-1;
    bool done=false;
    int object_id=0;

    std::shared_ptr<DaimlerSample> sd=nullptr;
    std::vector<std::map<int,BoundingBox>> boxes;
};
void process_mouse(int event, int col, int row, [[maybe_unused]] int flags, void* state_){
    State* state=(State*)state_;


    switch(event){
    case cv::EVENT_MBUTTONDOWN:
        state->drag_started=true;
        state->drag_start=Vector2d(row,col);

        return;
    case cv::EVENT_MOUSEMOVE:
        if(state->drag_started)
            state->drag_ongoing=Vector2d(row,col);
        else
            state->drag_ongoing=Vector2d(row,col);

        return;
    case cv::EVENT_MBUTTONUP:
        // create new box
        state->add_box(state->drag_start, state->drag_ongoing);
        state->drag_started=false;
        return;
    case cv::EVENT_LBUTTONDBLCLK:
        state->set_imo_id2(row,col);
        return;
    default:{}
    }
}

void show_dataset(std::string dataset_path, std::string gt_path, int start){
    State state(dataset_path, gt_path);

    state.init();
     state.help();
    state.set_start(start);
    while(!state.get_done()){
        state.update(cv::waitKey(1000/50));
        state.paint();
    }
}


int toint(std::string str){
    std::stringstream ss;
    ss<<str;
    int i=0;
    ss>>i;
    return i;
}

int main(int argc, char** argv){

    if(!(argc==4)){
        cout<< "make -j8 && ./show_dataset dataset_path gt_path start"<<endl;
        return 0;
    }
    uint id=toint(argv[3]);
    show_dataset(argv[1], argv[2], id);
    return 0;
}
