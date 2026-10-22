from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,RegisterEventHandler,EmitEvent,OpaqueFunction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory,get_package_prefix
from pathlib import Path
import os,fcntl,yaml,sys,tempfile,json,atexit
_stack_lock=None

def setup(context):
    global _stack_lock
    _stack_lock=open('/tmp/scan345_stack_'+os.environ.get('ROS_DOMAIN_ID','0')+'.lock','w')
    fcntl.flock(_stack_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    share=Path(get_package_share_directory('starpath_3d_navigation'));config=str(share/'config/native_nav2.yaml')
    sys.path.insert(0,str(Path(get_package_prefix('starpath_3d_navigation'))/'lib/starpath_3d_navigation'))
    from nav_footprint import chassis_footprint
    footprint=chassis_footprint().tolist()
    params=yaml.safe_load(Path(config).read_text())
    for name in ['global_costmap','local_costmap']:
        cfg=params['/scan345/'+name+'/'+name]['ros__parameters']
        cfg['footprint']=json.dumps(footprint);cfg['footprint_padding']=0.0
    fd,config=tempfile.mkstemp(prefix='scan345_urdf_nav2_',suffix='.yaml')
    with os.fdopen(fd,'w') as f:yaml.safe_dump(params,f,sort_keys=False)
    atexit.register(lambda:os.path.exists(config) and os.unlink(config))
    limits=yaml.safe_load(Path(config).read_text())['/scan345/controller_server']['ros__parameters']['FollowPath']
    speed=float(limits['vx_max']);turn=float(limits['wz_max'])
    nodes=[Node(package='nav2_planner',executable='planner_server',name='planner_server',namespace='scan345',parameters=[config],output='screen'),
           Node(package='nav2_controller',executable='controller_server',name='controller_server',namespace='scan345',parameters=[config],remappings=[('cmd_vel','nav2_command')],additional_env={'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'},output='screen'),
           Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='navigation_lifecycle',namespace='scan345',parameters=[{'autostart':False,'bond_timeout':3.,'node_names':['planner_server','controller_server']}],output='screen'),
           Node(package='starpath_3d_navigation',executable='nav_global_map.py',namespace='scan345',output='screen')]
    runtime=Node(package='starpath_3d_navigation',executable='nav_nav2_runtime.py',namespace='scan345',parameters=[str(share/'config/robot.yaml'),{'max_speed':speed,'max_yaw_rate':turn,'session':LaunchConfiguration('session'),'allow_hardware':ParameterValue(LaunchConfiguration('allow_hardware'),value_type=bool),'body_model_enabled':ParameterValue(LaunchConfiguration('body_model_enabled'),value_type=bool),'fixed_sector_filter':ParameterValue(LaunchConfiguration('fixed_sector_filter'),value_type=bool),'dynamic_geometry':ParameterValue(LaunchConfiguration('dynamic_geometry'),value_type=bool),'self_extra_profile':ParameterValue(LaunchConfiguration('self_extra_profile'),value_type=str)}],condition=IfCondition(LaunchConfiguration('start_adapter')),output='screen')
    watchdog=Node(package='starpath_3d_navigation',executable='nav_watchdog.py',namespace='scan345',parameters=[{'session':LaunchConfiguration('session'),'allow_hardware':ParameterValue(LaunchConfiguration('allow_hardware'),value_type=bool)}],condition=IfCondition(LaunchConfiguration('start_adapter')),output='screen')
    chassis=Node(package='autolife_s3_support',executable='s3_navigation_odometry',name='native_chassis_odometry',namespace='scan345',parameters=[{'publish_tf':False,'output_topic':'/scan345/chassis_odom'}],remappings=[('/s3/navigation_odometry_status','/scan345/chassis_odometry_status')],condition=IfCondition(LaunchConfiguration('start_adapter')),output='screen')
    nodes += [chassis,runtime,watchdog]
    events=[RegisterEventHandler(OnProcessExit(target_action=n,on_exit=[EmitEvent(event=Shutdown(reason='navigation component stopped'))])) for n in nodes]
    return events+nodes

def generate_launch_description():
    args=[DeclareLaunchArgument(k,default_value=v) for k,v in [('session',''),('allow_hardware','false'),('body_model_enabled','true'),('dynamic_geometry','false'),('fixed_sector_filter','true'),('self_extra_profile',''),('start_adapter','true')]]
    return LaunchDescription(args+[OpaqueFunction(function=setup)])
