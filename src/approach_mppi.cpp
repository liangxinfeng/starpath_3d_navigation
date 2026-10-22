// Uses the upstream Nav2 MPPI optimizer and collision critics unchanged.
// Controller composition follows nav2_mppi_controller/controller.cpp (Apache-2.0,
// Copyright 2022 Samsung Research America). No direct terminal velocity servo.
#include <atomic>
#include <chrono>
#include <mutex>
#include "nav2_core/controller.hpp"
#include "nav2_core/controller_exceptions.hpp"
#include "nav2_mppi_controller/controller.hpp"
#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_controller/plugins/stopped_goal_checker.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "std_msgs/msg/string.hpp"
#include "starpath_3d_navigation/approach_context.hpp"
#include <sstream>
#include "nav2_costmap_2d/footprint_collision_checker.hpp"

namespace scan345 {
using Steady=std::chrono::steady_clock;
thread_local const ApproachContext* approach_context=nullptr;
struct ContextScope {
  const ApproachContext* previous;
  explicit ContextScope(const ApproachContext&c):previous(approach_context){approach_context=&c;}
  ~ContextScope(){approach_context=previous;}
};
class ApproachGoalChecker:public nav2_controller::StoppedGoalChecker {
 public:
  void reset()override{StoppedGoalChecker::reset();stable_=false;fresh_=false;}
  void observe(bool hold,double d,double a){eligible_=hold&&d<=.05&&a<=.05;observed_=Steady::now();fresh_=true;}
  bool isGoalReached(const geometry_msgs::msg::Pose&p,const geometry_msgs::msg::Pose&g,const geometry_msgs::msg::Twist&v)override{
    const auto now=Steady::now();
    if(!(fresh_&&std::chrono::duration<double>(now-observed_).count()<.2&&eligible_&&StoppedGoalChecker::isGoalReached(p,g,v))){stable_=false;return false;}
    if(!stable_){stable_=true;since_=now;}
    return std::chrono::duration<double>(now-since_).count()>=.35;
  }
 private:bool stable_=false,fresh_=false,eligible_=false;Steady::time_point since_,observed_;
};
class ApproachOptimizer:public mppi::Optimizer {
 public:
  void limits(double linear,double angular,double external){
    auto& s=settings_;s.constraints=s.base_constraints;
    double ratio=std::min(std::clamp(external,0.,1.),linear/std::max(.001f,s.base_constraints.vx_max));
    s.constraints.vx_max=s.base_constraints.vx_max*ratio;
    s.constraints.vx_min=s.base_constraints.vx_min*ratio;
    s.constraints.vy=s.base_constraints.vy*ratio;
    s.constraints.wz=std::min(double(s.base_constraints.wz)*external,angular);
    // Both sampling and the vehicle model receive the independent bounds.
    motion_model_->initialize(s.constraints,s.model_dt);
  }
};
class ApproachMPPIController:public nav2_core::Controller {
 public:
  void configure(const rclcpp_lifecycle::LifecycleNode::WeakPtr&parent,std::string name,
    std::shared_ptr<tf2_ros::Buffer>tf,std::shared_ptr<nav2_costmap_2d::Costmap2DROS>costmap)override{
    parent_=parent;name_=name;tf_=tf;costmap_=costmap;
    params_=std::make_unique<mppi::ParametersHandler>(parent);
    optimizer_.initialize(parent,name,costmap,params_.get());
    path_.initialize(parent,name,costmap,tf,params_.get());
    auto n=parent.lock();vmax_=n->get_parameter(name+".vx_max").as_double();wmax_=n->get_parameter(name+".wz_max").as_double();
    auto parameter=[&](const std::string&key,double value){if(!n->has_parameter(name+key))n->declare_parameter(name+key,value);return n->get_parameter(name+key).as_double();};
    approach_distance_=parameter(".approach_max_distance",3.0);
    steps_=n->get_parameter(name+".time_steps").as_int();model_dt_=n->get_parameter(name+".model_dt").as_double();
    if(!(approach_distance_>=1.&&approach_distance_<=3.2&&vmax_>0))throw nav2_core::ControllerException("invalid approach distance");
    status_=n->create_publisher<std_msgs::msg::String>("approach_status",1);
    curve_pub_=n->create_publisher<nav_msgs::msg::Path>("approach_path",1);reset();
  }
  void cleanup()override{optimizer_.shutdown();params_.reset();}
  void activate()override{status_->on_activate();curve_pub_->on_activate();params_->start();}
  void deactivate()override{status_->on_deactivate();curve_pub_->on_deactivate();}
  void reset()override{optimizer_.reset(false);curve_.q.clear();progress_=0.;hold_.reset();has_goal_=false;last_=Steady::time_point{};linear_cap_=vmax_;angular_cap_=wmax_;}
  void setPlan(const nav_msgs::msg::Path&path)override{
    if(path.poses.empty())throw nav2_core::InvalidPath("empty path");
    auto g=path.poses.back();if(g.header.frame_id.empty())g.header.frame_id=path.header.frame_id;
    bool changed=!has_goal_||g.header.frame_id!=goal_.header.frame_id||std::hypot(g.pose.position.x-goal_.pose.position.x,g.pose.position.y-goal_.pose.position.y)>1e-5||std::abs(wrap(tf2::getYaw(g.pose.orientation)-tf2::getYaw(goal_.pose.orientation)))>1e-5;
    if(changed){curve_.q.clear();progress_=0.;hold_.reset();}
    goal_=g;has_goal_=true;route_.clear();for(auto&p:path.poses)route_.push_back({p.pose.position.x,p.pose.position.y});path_.setPath(path);
  }
  void setSpeedLimit(const double&limit,const bool&percent)override{
    external_=limit<=0?1.:std::clamp(percent?limit/100.:limit/vmax_,0.,1.);
  }
  geometry_msgs::msg::TwistStamped computeVelocityCommands(const geometry_msgs::msg::PoseStamped&pose,
    const geometry_msgs::msg::Twist&speed,nav2_core::GoalChecker*checker)override{
    auto gate=dynamic_cast<ApproachGoalChecker*>(checker);
    if(!gate||!has_goal_)throw nav2_core::ControllerException("terminal MPPI requires matching goal checker");
    std::lock_guard<std::mutex> param_lock(*params_->getLock());
    auto node=parent_.lock();geometry_msgs::msg::TransformStamped t;
    try{t=tf_->lookupTransform(goal_.header.frame_id,costmap_->getBaseFrameID(),tf2::TimePointZero);}
    catch(const tf2::TransformException&e){gate->observe(false,1,1);throw nav2_core::ControllerTFError(e.what());}
    double age=(node->now()-rclcpp::Time(t.header.stamp)).seconds();if(age>.3||age<-.1){gate->observe(false,1,1);throw nav2_core::ControllerTFError("stale arrival transform");}
    double x=t.transform.translation.x,y=t.transform.translation.y,yaw=tf2::getYaw(t.transform.rotation),goal_yaw=tf2::getYaw(goal_.pose.orientation);
    double distance=std::hypot(x-goal_.pose.position.x,y-goal_.pose.position.y),angle=std::abs(wrap(goal_yaw-yaw));
    auto local_goal=path_.getTransformedGoal(pose.header.stamp);auto transformed=path_.transformPath(pose);
    double ld=std::hypot(pose.pose.position.x-local_goal.pose.position.x,pose.pose.position.y-local_goal.pose.position.y);
    double la=std::abs(wrap(tf2::getYaw(pose.pose.orientation)-tf2::getYaw(local_goal.pose.orientation)));
    double left=remaining(route_,x,y);const auto now=Steady::now();
    double dt=last_==Steady::time_point{}?1./15.:std::clamp(std::chrono::duration<double>(now-last_).count(),.001,.2);last_=now;
    auto map=costmap_->getCostmap();std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*map->getMutex());
    const double offset=wrap(tf2::getYaw(pose.pose.orientation)-yaw),co=std::cos(offset),so=std::sin(offset);
    auto local=[&](double gx,double gy){double dx=gx-x,dy=gy-y;return Point{pose.pose.position.x+co*dx-so*dy,pose.pose.position.y+so*dx+co*dy};};
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D*> collision(map);
    const auto footprint=costmap_->getRobotFootprint();
    auto safe=[&](double gx,double gy,double a){auto p=local(gx,gy);unsigned int mx,my;
      if(!map->worldToMap(p.x,p.y,mx,my)||map->getCost(mx,my)>=nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE)return false;
      double cost=collision.footprintCostAtPose(p.x,p.y,a+offset,footprint);
      return cost>=0 && cost<nav2_costmap_2d::LETHAL_OBSTACLE;
    };
    bool invalid=false;
    if(!curve_.empty()){
      progress_=std::max(progress_-.02,curve_.project(x,y,progress_));
      auto ref=curve_.at(progress_);
      invalid=std::hypot(x-ref.x,y-ref.y)>.30||left>approach_distance_+.7;
      for(size_t i=0;!invalid&&i<curve_.q.size();i+=2){auto&q=curve_.q[i];if(q.t>=progress_-.1&&!safe(q.x,q.y,q.yaw))invalid=true;}
      if(invalid){curve_.q.clear();progress_=0.;publish_curve(pose.header.stamp);}
    }
    // Angle-dependent entry length; no fixed final-metre slowdown.
    const double entry=std::min(approach_distance_,1.8+.8*angle);
    if(curve_.empty() && !invalid && left<=entry && left<=distance*1.25+.15 && distance>.30 &&
       std::abs(speed.linear.y)<.04 && now-attempted_>std::chrono::milliseconds(700)){
      attempted_=now;
      curve_=make_curve({x,y},yaw,speed.linear.x,speed.angular.z,
        {goal_.pose.position.x,goal_.pose.position.y},goal_yaw,vmax_,route_,safe);
      if(!curve_.empty()){progress_=0.;publish_curve(pose.header.stamp);}
    }
    bool hold=hold_.hold(std::max(distance,ld),std::max(angle,la));gate->observe(hold,distance,angle);
    // Translation and yaw have separate braking envelopes. Distance-to-go must
    // never reduce angular authority when substantial heading error remains.
    double linear=curve_.empty()?std::min(vmax_,.035+1.2*distance):vmax_;
    double angular=curve_.empty()?std::min(wmax_,std::max(.08,std::sqrt(2.*.9*angle))):wmax_;
    linear_cap_=std::max(linear,linear_cap_-.6*dt);angular_cap_=std::max(angular,angular_cap_-1.8*dt);
    optimizer_.limits(linear_cap_,angular_cap_,external_.load());
    if(now-reported_>std::chrono::milliseconds(200)){
      double vref=curve_.empty()?0:curve_.at(progress_).v;
      std::ostringstream out;out<<"{\"active\":"<<(!curve_.empty()?"true":"false")<<",\"remaining\":"<<left<<",\"distance\":"<<distance<<",\"yaw_error\":"<<angle<<",\"reference_speed\":"<<vref<<",\"progress\":"<<progress_<<",\"duration\":"<<(curve_.empty()?0:curve_.q.back().t)<<",\"holding\":"<<(hold?"true":"false")<<"}";
      std_msgs::msg::String m;m.data=out.str();status_->publish(m);reported_=now;
      if(!curve_.empty())publish_curve(pose.header.stamp);
    }
    RCLCPP_INFO_THROTTLE(node->get_logger(),*node->get_clock(),2000,"Joint approach: active=%d distance=%.3f yaw=%.3f phase=%.3f hold=%d",!curve_.empty(),distance,angle,progress_,hold);
    if(hold){geometry_msgs::msg::TwistStamped zero;zero.header=pose.header;return zero;}
    ApproachContext ctx;ctx.active=!curve_.empty();
    if(ctx.active){
      // References in the continuous local frame, advancing from measured route
      // progress. Replans do not restart a wall-clock schedule or reset MPPI.
      double scale=external_.load();
      for(int j=0;j<steps_;++j){auto q=curve_.at(progress_+(j+1)*model_dt_*scale);auto p=local(q.x,q.y);q.x=p.x;q.y=p.y;q.yaw+=offset;q.v*=scale;q.w*=scale;ctx.reference.push_back(q);}
    }
    ContextScope scope(ctx);
    return optimizer_.evalControl(pose,speed,transformed,local_goal.pose,checker);
  }
 private:
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;std::string name_;
  std::shared_ptr<tf2_ros::Buffer>tf_;std::shared_ptr<nav2_costmap_2d::Costmap2DROS>costmap_;
  std::unique_ptr<mppi::ParametersHandler>params_;mppi::PathHandler path_;ApproachOptimizer optimizer_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>::SharedPtr status_;
  Curve curve_;double progress_=0.,approach_distance_=3.;ArrivalHold hold_;geometry_msgs::msg::PoseStamped goal_;bool has_goal_=false;
  std::vector<Point>route_;double vmax_=.45,wmax_=1.3,linear_cap_=.45,angular_cap_=1.3;std::atomic<double>external_{1.};
  Steady::time_point last_,reported_,attempted_;
  int steps_=40;double model_dt_=1./15.;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr curve_pub_;
  void publish_curve(const builtin_interfaces::msg::Time&stamp){
    nav_msgs::msg::Path path;path.header.frame_id=goal_.header.frame_id;path.header.stamp=stamp;
    for(auto&q:curve_.q){geometry_msgs::msg::PoseStamped p;p.header=path.header;p.pose.position.x=q.x;p.pose.position.y=q.y;p.pose.orientation.z=std::sin(q.yaw/2);p.pose.orientation.w=std::cos(q.yaw/2);path.poses.push_back(p);}
    curve_pub_->publish(path);
  }
};
}
PLUGINLIB_EXPORT_CLASS(scan345::ApproachMPPIController,nav2_core::Controller)
PLUGINLIB_EXPORT_CLASS(scan345::ApproachGoalChecker,nav2_core::GoalChecker)
