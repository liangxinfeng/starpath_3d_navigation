#include "nav2_mppi_controller/critic_function.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "starpath_3d_navigation/approach_context.hpp"
#include "nav2_mppi_controller/critics/goal_critic.hpp"
#include "nav2_mppi_controller/critics/goal_angle_critic.hpp"
#include "nav2_mppi_controller/critics/path_align_critic.hpp"
#include "nav2_mppi_controller/critics/path_follow_critic.hpp"
#include "nav2_mppi_controller/critics/path_angle_critic.hpp"
#include "nav2_mppi_controller/critics/prefer_forward_critic.hpp"
namespace mppi::critics {
class ApproachGoalCritic:public GoalCritic {
 public:void score(CriticData&d)override{if(!(scan345::approach_context&&scan345::approach_context->active))GoalCritic::score(d);}
};
class ApproachGoalAngleCritic:public GoalAngleCritic {
 public:void score(CriticData&d)override{if(!(scan345::approach_context&&scan345::approach_context->active))GoalAngleCritic::score(d);}
};
class ApproachPathAlignCritic:public PathAlignCritic {
 public:void score(CriticData&d)override{if(!(scan345::approach_context&&scan345::approach_context->active))PathAlignCritic::score(d);}
};
class ApproachPathFollowCritic:public PathFollowCritic {
 public:void score(CriticData&d)override{if(!(scan345::approach_context&&scan345::approach_context->active))PathFollowCritic::score(d);}
};
class ApproachPathAngleCritic:public PathAngleCritic {
 public:void score(CriticData&d)override{if(!(scan345::approach_context&&scan345::approach_context->active))PathAngleCritic::score(d);}
};
class ApproachPreferForwardCritic:public PreferForwardCritic {
 public:void score(CriticData&d)override{if(!(scan345::approach_context&&scan345::approach_context->active))PreferForwardCritic::score(d);}
};
class JointTrajectoryCritic:public CriticFunction {
 public:
 void initialize()override{auto p=parameters_handler_->getParamGetter(name_);p(position_,"position_weight",24.f);p(heading_,"heading_weight",9.f);p(velocity_,"velocity_weight",3.f);}
 void score(CriticData&d)override{
  auto c=scan345::approach_context;if(!enabled_||!c||!c->active||c->reference.empty())return;
  for(size_t i=0;i<d.trajectories.x.shape(0);++i){double cost=0;int count=0;
   for(size_t j=1;j<d.trajectories.x.shape(1);j+=2){auto&r=c->reference[std::min(j,c->reference.size()-1)];
    double dx=d.trajectories.x(i,j)-r.x,dy=d.trajectories.y(i,j)-r.y;
    double da=scan345::wrap(d.trajectories.yaws(i,j)-r.yaw);
    double dv=d.state.vx(i,j)-r.v,dw=d.state.wz(i,j)-r.w;
    cost+=position_*std::hypot(dx,dy)+heading_*std::abs(da)+velocity_*(std::abs(dv)+.25*std::abs(dw)+std::abs(d.state.vy(i,j)));++count;
   }
   d.costs(i)+=cost/std::max(1,count);
  }
 }
 private:float position_=24.f,heading_=9.f,velocity_=3.f;
};
}
PLUGINLIB_EXPORT_CLASS(mppi::critics::JointTrajectoryCritic,mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::ApproachGoalCritic,mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::ApproachGoalAngleCritic,mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::ApproachPathAlignCritic,mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::ApproachPathFollowCritic,mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::ApproachPathAngleCritic,mppi::critics::CriticFunction)
PLUGINLIB_EXPORT_CLASS(mppi::critics::ApproachPreferForwardCritic,mppi::critics::CriticFunction)
