#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace scan345 {
constexpr double pi=3.14159265358979323846;
inline double wrap(double a){return std::remainder(a,2*pi);}
struct Point {double x=0,y=0;};
inline Point operator+(Point a,Point b){return {a.x+b.x,a.y+b.y};}
inline Point operator-(Point a,Point b){return {a.x-b.x,a.y-b.y};}
inline Point operator*(Point a,double t){return {a.x*t,a.y*t};}
inline double norm(Point a){return std::hypot(a.x,a.y);}
inline double remaining(const std::vector<Point>&p,double x,double y){
 double tail=0,best=1e100,left=1e100;
 for(size_t j=p.size();j>1;--j){auto a=p[j-2],b=p[j-1],v=b-a;double l=norm(v);
  double t=l>1e-8?std::clamp(((x-a.x)*v.x+(y-a.y)*v.y)/(l*l),0.,1.):0.;
  double d=norm(Point{x,y}-(a+v*t));if(d<best-1e-7){best=d;left=tail+(1-t)*l+d;}tail+=l;
 }return left;
}
struct Knot {double x=0,y=0,yaw=0,k=0,s=0,t=0,v=0,w=0;};
struct Curve {
 std::vector<Knot> q;
 bool empty()const{return q.empty();}
 Knot at(double t)const{
  if(t<=0)return q.front();if(t>=q.back().t)return q.back();
  auto it=std::lower_bound(q.begin(),q.end(),t,[](const Knot&a,double b){return a.t<b;});
  const auto&a=*(it-1);const auto&b=*it;double f=(t-a.t)/(b.t-a.t);
  return {a.x+f*(b.x-a.x),a.y+f*(b.y-a.y),a.yaw+f*wrap(b.yaw-a.yaw),a.k+f*(b.k-a.k),a.s+f*(b.s-a.s),t,a.v+f*(b.v-a.v),a.w+f*(b.w-a.w)};
 }
 double project(double x,double y,double lower=0.)const{
  double best=1e100,t=lower;
  for(size_t i=1;i<q.size();++i){auto&a=q[i-1];auto&b=q[i];if(b.t<lower-.2)continue;
   double dx=b.x-a.x,dy=b.y-a.y,l2=dx*dx+dy*dy;
   double f=l2>1e-10?std::clamp(((x-a.x)*dx+(y-a.y)*dy)/l2,0.,1.):0.;
   double d=std::hypot(x-a.x-f*dx,y-a.y-f*dy);
   if(d<best){best=d;t=a.t+f*(b.t-a.t);}
  }return t;
 }
};
struct ArrivalHold {bool holding=false;void reset(){holding=false;}
 bool hold(double d,double a){if(holding&&(d>.05||a>.05))holding=false;if(!holding&&d<=.045&&a<=.045)holding=true;return holding;}
};
// Quintic Bezier: position, tangent and curvature match at the entry; final
// tangent equals the requested heading and final curvature is zero.
template<class Safe>
Curve make_curve(Point start,double yaw,double velocity,double yawrate,Point goal,double gyaw,
                 double vmax,const std::vector<Point>&route,Safe safe){
 Curve best;double score=1e100;const double chord=norm(goal-start);
 if(chord<.3||chord>3.4||!std::isfinite(chord)||velocity<-.02)return best;
 const double k0=velocity>.08?yawrate/velocity:0.;
 if(std::abs(k0)>3. || (velocity<=.08&&std::abs(yawrate)>.15))return best;
 const double delta=wrap(gyaw-yaw);
 for(auto ratios:std::vector<Point>{{.12,.12},{.18,.18},{.24,.24},{.30,.30},{.36,.36},{.06,.24},{.10,.24},{.10,.30},{.14,.30},{.14,.40},{.18,.30}}){
  const double h=chord*ratios.x,hg=chord*ratios.y;Point u{std::cos(yaw),std::sin(yaw)},ug{std::cos(gyaw),std::sin(gyaw)};
  std::array<Point,6> p{start,start+u*h,start+u*(2*h)+Point{-u.y,u.x}*(1.25*h*h*k0),goal-ug*(2*hg),goal-ug*hg,goal};
  Curve c;double s=0,turn=0,dev=0;bool good=true;Point previous=start;double prevyaw=yaw;
  const int n=std::max(160,int(chord/.008));
  for(int i=0;i<=n;++i){
   double t=double(i)/n,v=1-t;
   Point a=p[0]*(v*v*v*v*v)+p[1]*(5*v*v*v*v*t)+p[2]*(10*v*v*v*t*t)+p[3]*(10*v*v*t*t*t)+p[4]*(5*v*t*t*t*t)+p[5]*(t*t*t*t*t);
   Point d=(p[1]-p[0])*(5*v*v*v*v)+(p[2]-p[1])*(20*v*v*v*t)+(p[3]-p[2])*(30*v*v*t*t)+(p[4]-p[3])*(20*v*t*t*t)+(p[5]-p[4])*(5*t*t*t*t);
   Point dd=(p[2]-p[1]*2+p[0])*(20*v*v*v)+(p[3]-p[2]*2+p[1])*(60*v*v*t)+(p[4]-p[3]*2+p[2])*(60*v*t*t)+(p[5]-p[4]*2+p[3])*(20*t*t*t);
   double speed=norm(d);if(speed<.03){good=false;break;}
   double k=(d.x*dd.y-d.y*dd.x)/(speed*speed*speed),ang=prevyaw+wrap(std::atan2(d.y,d.x)-prevyaw);
   double ds=norm(a-previous);s+=ds;turn+=std::abs(ang-prevyaw);
   // Reject cusps, loops and excessive detours; never force a U-turn connector.
   if(std::abs(k)>7. || s>chord*1.7 || turn>std::abs(delta)+1.5){good=false;break;}
   int pieces=std::max(1,int(std::ceil(std::max(ds/.008,std::abs(ang-prevyaw)/.02))));
   for(int j=1;j<=pieces;++j){double f=double(j)/pieces;auto pt=previous+(a-previous)*f;
    if(!safe(pt.x,pt.y,prevyaw+(ang-prevyaw)*f)){good=false;break;}}
   if(!good)break;
   double nearest=1e100;for(auto r:route)nearest=std::min(nearest,norm(a-r));dev=std::max(dev,nearest);
   if(dev>.75){good=false;break;}
   c.q.push_back({a.x,a.y,ang,k,s,0,0,0});previous=a;prevyaw=ang;
  }
  if(!good||c.q.size()<2||std::abs(wrap(c.q.back().yaw-gyaw))>1e-5)continue;
  // Curvature-aware speed profile. The sole stop is at the final pose.
  for(size_t i=0;i<c.q.size();++i){auto&q=c.q[i];q.v=std::min(vmax,std::min(.85/std::max(.001,std::abs(q.k)),std::sqrt(.30/std::max(.001,std::abs(q.k)))));
   if(i>0&&i+1<c.q.size()){
    double kp=(c.q[i+1].k-c.q[i-1].k)/std::max(.001,c.q[i+1].s-c.q[i-1].s);
    q.v=std::min(q.v,std::sqrt(.8/std::max(.001,std::abs(kp))));
   }
  }
  c.q.front().v=std::clamp(velocity,0.,vmax);c.q.back().v=0.;
  for(size_t i=1;i<c.q.size();++i)c.q[i].v=std::min(c.q[i].v,std::sqrt(c.q[i-1].v*c.q[i-1].v+2*.40*(c.q[i].s-c.q[i-1].s)));
  // Reserve actuator response distance in the braking envelope; the physical
  // chassis does not follow a zero-latency velocity command as the ideal model did.
  for(size_t i=0;i<c.q.size();++i){
   double remaining=c.q.back().s-c.q[i].s;
   double brake=std::sqrt(.30*.30*.16*.16+2*.30*remaining)-.30*.16;
   c.q[i].v=std::min(c.q[i].v,brake);
  }
  for(size_t i=c.q.size()-1;i>0;--i)c.q[i-1].v=std::min(c.q[i-1].v,std::sqrt(c.q[i].v*c.q[i].v+2*.30*(c.q[i].s-c.q[i-1].s)));
  if(c.q.front().v+ .025<velocity)continue; // Do not abruptly lower entry speed.
  for(size_t i=0;i<c.q.size();++i){auto&q=c.q[i];q.w=q.v*q.k;
   if(i){auto&prev=c.q[i-1];double dt=2*(q.s-prev.s)/std::max(.001,q.v+prev.v);q.t=prev.t+dt;
    if(std::abs(q.w-prev.w)/std::max(.001,dt)>1.65){good=false;break;}
   }
  }
  if(!good)continue;
  double peak_k=0.;for(auto&q:c.q)peak_k=std::max(peak_k,std::abs(q.k));
  // Prefer a broader, less abrupt turn over a slightly shorter connector that
  // demands a sharp velocity dip just before arrival.
  double value=c.q.back().t+dev*.5+peak_k*.7;if(value<score){score=value;best=std::move(c);}
 }return best;
}
}
