#pragma once
// Forward differential-drive SE(2) search. Every edge is a checked motion,
// not a line segment whose orientation is inferred after planning.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace chassis {
constexpr double pi=3.14159265358979323846;
inline double wrap(double a) { return std::remainder(a,2*pi); }
struct Pose {double x=0,y=0,yaw=0;};
struct Edge {double v=0,w=0,t=0;};
inline Pose advance(Pose s,Edge e,double t) {
  if(std::abs(e.w)<1e-10) {s.x+=e.v*t*std::cos(s.yaw);s.y+=e.v*t*std::sin(s.yaw);}
  else {s.x+=e.v/e.w*(std::sin(s.yaw+e.w*t)-std::sin(s.yaw));
        s.y-=e.v/e.w*(std::cos(s.yaw+e.w*t)-std::cos(s.yaw));}
  s.yaw=wrap(s.yaw+e.w*t);return s;
}
using Free=std::function<bool(const Pose&)>;
inline bool sweep(Pose s,Edge e,const Free& free) {
  int n=std::max(1,int(std::ceil(std::max(e.v*e.t/.02,std::abs(e.w*e.t)/(pi/180)))));
  for(int i=1;i<=n;++i)if(!free(advance(s,e,e.t*i/n)))return false;
  return true;
}
struct Result {
  std::vector<Pose> poses;
  std::vector<Edge> edges;
  bool final=false;
  std::string error;
  int expanded=0;
};
struct Key {int x,y,a;bool operator==(const Key& b)const{return x==b.x&&y==b.y&&a==b.a;}};
struct Hash {size_t operator()(const Key& k)const {
  return size_t(uint32_t(k.x))*73856093u ^ size_t(uint32_t(k.y))*19349663u ^ size_t(k.a)*83492791u;
}};
inline Key key(Pose s,Pose origin) {
  return {int(std::lround((s.x-origin.x)/.04)),int(std::lround((s.y-origin.y)/.04)),
          (int(std::lround(wrap(s.yaw-origin.yaw)/(pi/18)))+36)%36};
}
inline Edge motion(double length,double angle) {
  double t=std::max(length/.12,std::abs(angle)/.24);
  return {t>0?length/t:0,t>0?angle/t:0,t};
}
// Exact connector; never moves the goal to a free point or alters the start.
inline bool connect(Pose s,Pose goal,bool heading,const Free& free,std::vector<Edge>& edges) {
  edges.clear();double d=std::hypot(goal.x-s.x,goal.y-s.y);
  if(d>1e-5) {
    double bearing=wrap(std::atan2(goal.y-s.y,goal.x-s.x)-s.yaw);
    // Exact circular arc through the target, tangent to the current chassis
    // heading. Prefer this over gratuitous stop-turn-go near the endpoint.
    double angle=2*bearing;
    double length=std::abs(angle)<1e-8?d:d*angle/(2*std::sin(angle/2));
    Edge arc=motion(length,angle);
    if(std::abs(angle)<pi*.45 && sweep(s,arc,free)) {
      edges.push_back(arc);s=advance(s,arc,arc.t);
    } else {
      // Large orientation changes must be searched and costed, not hidden
      // inside an uncosted final connector.
      if(std::abs(bearing)>pi/6)return false;
      Edge turn=motion(0,bearing);
      if(turn.t>1e-6) {if(!sweep(s,turn,free))return false;edges.push_back(turn);s=advance(s,turn,turn.t);}
      Edge line=motion(d,0);
      if(!sweep(s,line,free))return false;
      edges.push_back(line);s=advance(s,line,line.t);
    }
  }
  if(heading) {
    Edge turn=motion(0,wrap(goal.yaw-s.yaw));
    if(turn.t>1e-6) {if(!sweep(s,turn,free))return false;edges.push_back(turn);}
  }
  return true;
}
inline Result search(Pose start,Pose goal,bool heading,const Free& free,
                     std::function<bool()> cancelled=[](){return false;},double budget=3.5,
                     const std::vector<double>& goal_headings={}) {
  Result out;
  if(!free(start)){out.error="start_occupied";return out;}
  if(cancelled()){out.error="cancelled";return out;}
  auto begun=std::chrono::steady_clock::now();
  const double initial_distance=std::hypot(goal.x-start.x,goal.y-start.y);
  const bool distant=initial_distance>2.0;
  Pose direct_goal=goal;
  if(distant) {
    direct_goal.x=start.x+(goal.x-start.x)*1.8/initial_distance;
    direct_goal.y=start.y+(goal.y-start.y)*1.8/initial_distance;
  }
  // In open space take one checked tangent arc/line; don't postpone lateral
  // correction until the end of a row of lattice cells and then turn sharply.
  if(connect(start,direct_goal,heading && !distant,free,out.edges)) {
    out.final=!distant;out.poses.push_back(start);
    for(auto e:out.edges)out.poses.push_back(advance(out.poses.back(),e,e.t));
    return out;
  }
  out.edges.clear();
  struct Node{Pose p;double g;int parent;Edge edge;};
  struct Item {double f;int id;bool operator<(const Item& o)const{return f>o.f;}};
  std::vector<Node> nodes;nodes.reserve(50000);
  std::priority_queue<Item> open;std::unordered_map<Key,double,Hash> best;
  auto distance=[&](Pose p){return std::hypot(goal.x-p.x,goal.y-p.y);};
  auto guidance=[&](Pose p) {
    double d=distance(p);
    if(distant || goal_headings.empty())return d;
    double score=std::numeric_limits<double>::infinity();
    for(double a:goal_headings) {
      const double dx=goal.x-p.x,dy=goal.y-p.y;
      const double along=dx*std::cos(a)+dy*std::sin(a);
      const double side=std::abs(-dx*std::sin(a)+dy*std::cos(a));
      // Guide toward a feasible approach side, not just the occupied near
      // face of the goal. This is search guidance, never a collision bypass.
      double approach=along>=0 && side<.15 ? d :
          std::hypot(goal.x-.25*std::cos(a)-p.x,goal.y-.25*std::sin(a)-p.y)+.25;
      score=std::min(score,approach+.14*std::abs(wrap(a-p.yaw)));
    }
    return score;
  };
  nodes.push_back({start,0,-1,{}});best[key(start,start)]=0;open.push({guidance(start),0});
  std::vector<Edge> primitives={motion(.10,0),motion(.10,pi/18),motion(.10,-pi/18),
      motion(.10,pi/9),motion(.10,-pi/9),motion(0,pi/18),motion(0,-pi/18)};
  int finish=-1;std::vector<Edge> tail;
  while(!open.empty()) {
    if(cancelled()){out.error="cancelled";return out;}
    if(++out.expanded>18000 || std::chrono::duration<double>(std::chrono::steady_clock::now()-begun).count()>budget) {
      out.error="search_budget";return out;
    }
    int id=open.top().id;open.pop();
    const Node n=nodes[id];
    if(n.g>best[key(n.p,start)]+1e-8)continue;
    if(!distant && distance(n.p)<.25 && connect(n.p,goal,heading,free,tail)) {
      finish=id;out.final=true;break;
    }
    // A local segment must make actual progress toward the unchanged remote
    // goal. Subsequent segments are requested from measured stopped poses.
    if(distant && std::hypot(n.p.x-start.x,n.p.y-start.y)>=1.8 &&
       distance(n.p)<distance(start)-.8) {finish=id;out.final=false;break;}
    for(const auto& e:primitives) {
      Pose p=advance(n.p,e,e.t);
      if(std::hypot(p.x-start.x,p.y-start.y)>3.0)continue;
      double g=n.g+e.v*e.t+.14*std::abs(e.w*e.t)+.008+(e.v==0?.015:0.);
      Key k=key(p,start);auto found=best.find(k);
      if(found!=best.end() && found->second<=g+1e-8)continue;
      if(!sweep(n.p,e,free))continue;
      best[k]=g;int next=int(nodes.size());nodes.push_back({p,g,id,e});
      // Weighted A*: bounded computation, no claim of optimality.
      open.push({g+1.25*guidance(p),next});
    }
  }
  if(finish<0){out.error="no_chassis_route";return out;}
  std::vector<Edge> reversed;
  for(int i=finish;nodes[i].parent>=0;i=nodes[i].parent)reversed.push_back(nodes[i].edge);
  std::reverse(reversed.begin(),reversed.end());reversed.insert(reversed.end(),tail.begin(),tail.end());
  // Coalesce identical controls only. Never fit a spline through corners.
  for(auto e:reversed) {
    if(!out.edges.empty() && std::abs(out.edges.back().v-e.v)<1e-8 &&
       std::abs(out.edges.back().w-e.w)<1e-8)out.edges.back().t+=e.t;
    else out.edges.push_back(e);
  }
  out.poses.push_back(start);
  for(auto e:out.edges)out.poses.push_back(advance(out.poses.back(),e,e.t));
  return out;
}
} // namespace chassis
