#include <path_searching/dyn_a_star.h>
#include <stdexcept>
#include <iostream>
void require(bool condition,const char* message) {
  if(!condition)throw std::runtime_error(message);
}
int main(int argc,char** argv) {
  rclcpp::init(argc,argv);
  {
    bool blocked=false,revisited=false;
    AStar search;
    search.initGridMap(nullptr,{21,21,5},[&](const Eigen::Vector3d& p,double yaw) {
      const int x=std::lround(p.x()),y=std::lround(p.y());
      if(x==1 && y==0) {
        if(std::abs(yaw-M_PI/2)>.01){blocked=true;return 1;}
        if(blocked)revisited=true;
        return 0;
      }
      if(x==2 && y==0)return std::abs(yaw)>.01 ? 1 : 0;
      return ((x==0&&y==0)||(x==0&&y==-1)||(x==1&&y==-1)) ? 0 : 1;
    });
    require(search.AstarSearch(1.,{0,0,0},{2,0,0})==ASTAR_RET::SUCCESS,
            "A cell rejected from one heading must remain discoverable from another");
    require(blocked&&revisited,"Regression must exercise blocked-then-free arrival");
    auto path=search.getPath();
    require(path.size()>=4,"Path must pass through the directional gate");
    require((path.front()-Eigen::Vector3d(0,0,0)).norm()<1e-8,"Correct start cell");
  }
  {
    AStar search;
    search.initGridMap(nullptr,{51,51,5},[](const Eigen::Vector3d&,double){return 0;});
    for(int i=0;i<20;++i) {
      Eigen::Vector3d start(-1.4,.2,-.3),end(.6,.2,-.3);
      if(i%2)std::swap(start,end);
      require(search.AstarSearch(.2,start,end)==ASTAR_RET::SUCCESS,"Repeated search succeeds");
      auto path=search.getPath();
      require((path.front()-start).norm()<1e-8,"Negative grid offsets must round symmetrically");
      require((path.back()-end).norm()<1e-8,"Repeated-search goal is not stale");
    }
  }
  require(std::isinf(inf),"Unvisited score must be infinity, not zero");
  std::cout<<"ASTAR_REGRESSION_PASS: directional revisit, negative offsets, repeated targets, infinity\n";
  rclcpp::shutdown();
}
