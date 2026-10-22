#include "chassis_search.h"
#include <plan_env/body_geometry.h>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <starpath_3d_navigation/msg/chassis_path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <atomic>
#include <future>
#include <mutex>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

using Clock=std::chrono::steady_clock;
using Cloud=pcl::PointCloud<pcl::PointXYZ>;
using Parts=std::vector<scan345::BodyPart>;
static double age(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct Snapshot {
  Cloud::Ptr cloud;
  Parts parts;
  Eigen::Matrix3d tilt;
  double z;
  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  Snapshot(Cloud::Ptr c,Parts p,Eigen::Matrix3d r,double height):cloud(c),parts(p),tilt(r),z(height){tree.setInputCloud(cloud);}
  double radius()const {
    double result=0;
    for(const auto& part:parts) {
      if(part.kind==1)result=std::max(result,(tilt*part.center).head<2>().norm()+part.half.x());
      else for(int x:{-1,1})for(int y:{-1,1})for(int z:{-1,1})
        result=std::max(result,(tilt*(part.center+Eigen::Vector3d(x*part.half.x(),y*part.half.y(),z*part.half.z()))).head<2>().norm());
    }
    return result;
  }
  bool free(const chassis::Pose& p,double margin=0.) const {
    // Evaluate Eigen expressions: keeping `auto` here would retain a
    // reference to the temporary yaw matrix after it has been destroyed.
    const Eigen::Matrix3d r=scan345::yawRotation(p.yaw)*tilt;
    const Eigen::Vector3d translation(p.x,p.y,z);
    if(!r.allFinite() || !translation.allFinite())return false;
    // Same protected model as Runtime.clearance: current URDF bounds,
    // current roll/pitch and no additional collision clearance.
    for(const auto& part:parts) {
      Eigen::Vector3d center=r*part.center+translation;
      if(!center.allFinite())return false;
      std::vector<int> ids;std::vector<float> ds;
      // Broaden only the candidate query for PCL's exclusive radius boundary.
      // The exact test below still has zero geometric margin.
      tree.radiusSearch(pcl::PointXYZ(center.x(),center.y(),center.z()),
        (part.kind==1?part.half.x():part.half.norm())+std::max(0.,margin)+1e-6,ids,ds);
      for(int i:ids) {
        Eigen::Vector3d q=(*cloud)[i].getVector3fMap().cast<double>();
        Eigen::Vector3d delta=r.transpose()*(q-translation)-part.center;
        if(part.kind==1) {if(delta.norm()<=part.half.x()+margin)return false;}
        else {
          Eigen::Vector3d d=delta.cwiseAbs()-part.half;
          if(d.cwiseMax(0.).norm()+std::min(0.,d.maxCoeff())<=margin)return false;
        }
      }
    }
    return true;
  }
};

class Planner:public rclcpp::Node {
  int lock_=-1;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr alive_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr ack_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr error_;
  rclcpp::Publisher<starpath_3d_navigation::msg::ChassisPath>::SharedPtr path_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions_;
  rclcpp::TimerBase::SharedPtr timer_;
  Cloud::Ptr cloud_;
  Parts parts_;
  Eigen::Matrix3d rotation_=Eigen::Matrix3d::Identity();
  chassis::Pose body_;double z_=0;
  bool ready_=false;Clock::time_point odom_time_,cloud_time_;
  std::atomic<uint64_t> generation_{0};
  struct Job {chassis::Result result;uint64_t id,generation;double z;bool validated=false;};
  std::future<Job> future_;
  geometry_msgs::msg::PoseStamped::SharedPtr pending_;
  template<class T,class F> void sub(const std::string& name,F cb) {
    subscriptions_.push_back(create_subscription<T>(name,rclcpp::QoS(1).best_effort(),cb));
  }
  void error(uint64_t id,const std::string& code) {
    std_msgs::msg::String m;m.data="{\"command_id\":"+std::to_string(id)+",\"code\":\""+code+"\"}";
    error_->publish(m);RCLCPP_WARN(get_logger(),"Chassis goal %lu: %s",id,code.c_str());
  }
  void launch() {
    auto m=pending_;pending_.reset();
    uint64_t id=uint64_t(m->header.stamp.sec)*1000000000ull+m->header.stamp.nanosec;
    if(!ready_ || age(odom_time_)>.5 || age(cloud_time_)>.6 || !cloud_ || cloud_->empty() || parts_.empty()) {
      error(id,"inputs_stale");return;
    }
    auto target=m->pose.position;
    auto q=m->pose.orientation;
    bool heading=std::hypot(q.z,q.w)>.9;
    chassis::Pose goal{target.x,target.y,heading?2*std::atan2(q.z,q.w):0};
    auto snapshot=std::make_shared<Snapshot>(cloud_,parts_,scan345::yawRotation(-body_.yaw)*rotation_,z_);
    uint64_t gen=generation_.load();auto start=body_;double z=z_;
    future_=std::async(std::launch::async,[this,snapshot,start,goal,heading,id,gen,z](){
      if(!snapshot->free(start))return Job{chassis::Result{{},{},false,"start_occupied"},id,gen,z};
      std::vector<double> headings;
      if(std::hypot(goal.x-start.x,goal.y-start.y)<=2.) {
        bool possible=false;
        if(heading) {
          possible=snapshot->free(goal);headings.push_back(goal.yaw);
        } else {
          // Any angle is at most 1 degree from a sampled angle. The maximum
          // motion of every protected body point is bounded by this amount.
          // Reject all orientations only if collision persists even after
          // subtracting that bound; sampling alone must not prove infeasibility.
          double bound=2*snapshot->radius()*std::sin(chassis::pi/360);
          for(int i=0;i<180;++i) {
            if(generation_.load()!=gen)return Job{chassis::Result{{},{},false,"cancelled"},id,gen,z};
            chassis::Pose p=goal;p.yaw=-chassis::pi+i*chassis::pi/90;
            if(snapshot->free(p,-bound))possible=true;
            if(snapshot->free(p))headings.push_back(p.yaw);
          }
        }
        if(!possible)return Job{chassis::Result{{},{},false,"goal_occupied"},id,gen,z};
      }
      auto result=chassis::search(start,goal,heading,[snapshot](const auto& p){return snapshot->free(p);},
          [this,gen](){return generation_.load()!=gen;},3.5,headings);
      return Job{result,id,gen,z};
    });
  }
  void tick() {
    std_msgs::msg::Bool a;a.data=true;alive_->publish(a);
    if(future_.valid() && future_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
      Job j=future_.get();
      if(j.generation==generation_.load()) {
        if(!j.result.error.empty())RCLCPP_INFO(get_logger(),"Chassis search expanded %d states",j.result.expanded);
        if(!j.result.error.empty())error(j.id,j.result.error);
        else if(!ready_ || age(odom_time_)>.5 || age(cloud_time_)>.6)error(j.id,"inputs_stale");
        else if(!j.validated) {
          // Revalidate the complete route against a fresh immutable scan,
          // after search, without blocking ROS heartbeats or cancellation.
          auto snapshot=std::make_shared<Snapshot>(cloud_,parts_,scan345::yawRotation(-body_.yaw)*rotation_,j.z);
          future_=std::async(std::launch::async,[this,j,snapshot]() mutable {
            for(size_t i=0;i<j.result.edges.size();++i) {
              if(generation_.load()!=j.generation){j.result.error="cancelled";break;}
              if(!chassis::sweep(j.result.poses[i],j.result.edges[i],
                                [snapshot](const auto& p){return snapshot->free(p);})) {
                j.result.error="trajectory_collision";break;
              }
            }
            j.validated=true;return j;
          });
        } else {
          starpath_3d_navigation::msg::ChassisPath m;
          m.header.frame_id="airy_global";m.header.stamp=now();m.command_id=j.id;m.final_segment=j.result.final;
          for(auto p:j.result.poses) {
            geometry_msgs::msg::Pose v;v.position.x=p.x;v.position.y=p.y;v.position.z=j.z;
            v.orientation.z=std::sin(p.yaw/2);v.orientation.w=std::cos(p.yaw/2);m.poses.push_back(v);
          }
          for(auto e:j.result.edges){m.speeds.push_back(e.v);m.yaw_rates.push_back(e.w);m.durations.push_back(e.t);}
          path_->publish(m);
          RCLCPP_INFO(get_logger(),"Chassis goal %lu: %zu motions, %d expansions, final=%d",j.id,j.result.edges.size(),j.result.expanded,j.result.final);
        }
      }
    }
    if(pending_ && !future_.valid())launch();
  }
public:
  Planner():Node("scan345_planner") {
    const char* domain=std::getenv("ROS_DOMAIN_ID");
    std::string lockname="/tmp/scan345_planner_"+std::string(domain?domain:"0")+".lock";
    lock_=open(lockname.c_str(),O_CREAT|O_RDWR,0600);
    if(lock_<0 || flock(lock_,LOCK_EX|LOCK_NB)!=0)throw std::runtime_error("A chassis planner already owns this ROS domain");
    alive_=create_publisher<std_msgs::msg::Bool>("planner_alive",1);
    ack_=create_publisher<std_msgs::msg::Float64MultiArray>("body_envelope_applied",1);
    error_=create_publisher<std_msgs::msg::String>("planning_error",1);
    path_=create_publisher<starpath_3d_navigation::msg::ChassisPath>("planning/chassis_path",1);
    sub<nav_msgs::msg::Odometry>("body_pose",[this](nav_msgs::msg::Odometry::SharedPtr m){
      if(m->header.frame_id!="airy_global")return;
      auto p=m->pose.pose.position;auto q=m->pose.pose.orientation;
      Eigen::Quaterniond quaternion(q.w,q.x,q.y,q.z);
      if(!quaternion.coeffs().allFinite() || quaternion.norm()<.9 || !std::isfinite(p.x+p.y+p.z))return;
      rotation_=quaternion.normalized().toRotationMatrix();
      body_={p.x,p.y,std::atan2(rotation_(1,0),rotation_(0,0))};z_=p.z;odom_time_=Clock::now();
    });
    sub<sensor_msgs::msg::PointCloud2>("cloud",[this](sensor_msgs::msg::PointCloud2::SharedPtr m){
      if(m->header.frame_id!="airy_global")return;
      auto cloud=std::make_shared<Cloud>();pcl::fromROSMsg(*m,*cloud);
      Cloud::Ptr finite(new Cloud);
      finite->reserve(cloud->size());
      for(auto p:*cloud)if(std::isfinite(p.x+p.y+p.z))finite->push_back(p);
      cloud_=finite;cloud_time_=Clock::now();
    });
    sub<std_msgs::msg::Float64MultiArray>("body_envelope",[this](std_msgs::msg::Float64MultiArray::SharedPtr m){
      if(m->data.size()<8)return;
      Parts p;if(!scan345::decodeBody(std::vector<double>(m->data.begin()+1,m->data.end()),p))return;
      parts_=p;ack_->publish(*m);
    });
    sub<std_msgs::msg::Bool>("input_ready",[this](std_msgs::msg::Bool::SharedPtr m){ready_=m->data;});
    sub<std_msgs::msg::Bool>("reset",[this](std_msgs::msg::Bool::SharedPtr){++generation_;pending_.reset();});
    sub<geometry_msgs::msg::PoseStamped>("move_base_simple/goal",[this](geometry_msgs::msg::PoseStamped::SharedPtr m){
      if(m->header.frame_id!="airy_global" || !std::isfinite(m->pose.position.x+m->pose.position.y))return;
      ++generation_;pending_=m;
    });
    timer_=create_wall_timer(std::chrono::milliseconds(50),[this](){tick();});
  }
  ~Planner(){++generation_;if(future_.valid())future_.wait();if(lock_>=0)close(lock_);}
};
int main(int argc,char** argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<Planner>());rclcpp::shutdown();}
