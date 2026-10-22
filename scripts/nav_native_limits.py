"""Native navigation bounds are read from the same controller config."""
from pathlib import Path
import math,yaml
from ament_index_python.packages import get_package_share_directory

def native_limits():
    path=Path(get_package_share_directory('starpath_3d_navigation'))/'config/native_nav2.yaml'
    p=yaml.safe_load(path.read_text())['/scan345/controller_server']['ros__parameters']['FollowPath']
    out={k:float(p[k]) for k in ['vx_max','vx_min','vy_max','wz_max']}
    if not all(math.isfinite(v) for v in out.values()) or not (out['vx_max']>0 and out['vx_min']<0 and out['vy_max']>0 and out['wz_max']>0):raise ValueError('原导航速度配置无效')
    return out

def constrain_native(v,limits):
    if len(v)!=3 or not all(math.isfinite(x) for x in v):raise ValueError('MPPI输出含无效数值')
    caps=(max(limits['vx_max'],-limits['vx_min']),limits['vy_max'],limits['wz_max'])
    if any(abs(x)>2*c+.05 for x,c in zip(v,caps)):raise ValueError('MPPI输出异常超出原导航参数')
    # The native smoother can overshoot slightly. Clamp only to native limits,
    # then collision-check this actual output; do not abort a normal goal.
    return (max(limits['vx_min'],min(limits['vx_max'],v[0])),
            max(-limits['vy_max'],min(limits['vy_max'],v[1])),
            max(-limits['wz_max'],min(limits['wz_max'],v[2])))


def chassis_yaw_scale():
    path=Path(get_package_share_directory('control_center_navigation'))/'config/docking_params.yaml'
    value=float(yaml.safe_load(path.read_text())['cmd_vel_arbiter']['ros__parameters']['chassis_yaw_command_scale'])
    if not math.isfinite(value) or not .5<=value<=1.:raise ValueError('底盘转向标定无效')
    return value

def chassis_command(v,scale):
    if not all(math.isfinite(x) for x in v) or not math.isfinite(scale) or not .5<=scale<=1.:raise ValueError('无效底盘指令')
    return (float(v[0]),float(v[1]),float(v[2])*scale)
