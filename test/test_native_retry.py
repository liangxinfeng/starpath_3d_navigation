import sys,time
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nav_obstacle_wait import ObstacleWait
from nav_nav2_runtime import NativeRuntime
from nav_safety import Safety
from nav_msgs.msg import Path as RosPath
from geometry_msgs.msg import PoseStamped

def runtime(execute=True):
    n=NativeRuntime.__new__(NativeRuntime)
    n.obstacle_wait=ObstacleWait();n.epoch=0;n.follow_serial=0
    n.plan_handle=n.follow_handle=n.plan_future=n.follow_future=None
    n.safety=Safety(time.monotonic);n.safety.armed=execute
    n.goal=dict(point=np.array([1.,0.,0.]),yaw=.7,requested_yaw=.7,execute=execute)
    n.obstacle_replans=0;n.scans=10;n.native_path=None;n.plan_started=time.monotonic()
    n.send_drive=Mock();n.reset_pub=Mock();n.path_pub=Mock()
    return n

def result():
    p=RosPath();p.header.frame_id='airy_global'
    for x in [0.,1.]:
        q=PoseStamped();q.pose.position.x=x;p.poses.append(q)
    return Mock(result=lambda:S(status=4,result=S(path=p,error_code=0,error_msg='')))

def test_unlimited_retry_retains_goal_and_stops():
    n=runtime();goal=n.goal
    for i in range(25):
        before=time.monotonic();n.retry('blocked')
        assert n.goal is goal and n.safety.armed and n.goal['yaw']==.7
        assert before+1<=n.retry_at<=time.monotonic()+1
    assert n.obstacle_replans==25 and n.epoch==25
    assert all(c.args==(0.,0.,False) for c in n.send_drive.call_args_list)

def test_clear_requires_distinct_fresh_scans():
    w=ObstacleWait();w.begin(1)
    for i in range(20):w.observe(1,True,0.)
    assert not w.ready(0.)
    for i in [2,3]:w.observe(i,True,i*.1)
    assert not w.ready(.3)
    w.observe(4,True,.4);assert w.ready(.4)
    assert not w.ready(1.1)
    w.observe(5,True,1.1);assert not w.ready(1.1)
    w.observe(6,False,1.2);assert w.clear_scans==0

def test_block_remains_waiting_across_replans():
    n=runtime();n.follow=Mock();n.retry('blocked')
    for i in range(10):
        n.obstacle_wait.observe(11+i,False,time.monotonic())
        n.planned(result(),n.epoch)
    assert n.goal and n.obstacle_replans==11 and not n.follow.called

def test_clear_path_can_resume_after_three_scans():
    n=runtime();n.follow=Mock();n.retry('blocked')
    for i in range(3):n.obstacle_wait.observe(11+i,True,time.monotonic())
    n.planned(result(),n.epoch)
    assert n.follow.call_count==1 and n.native_path is not None

def test_cancel_fences_late_plan_and_result():
    n=runtime();n.retry('blocked');epoch=n.epoch;serial=n.follow_serial
    n.abort('cancel');n.planned(result(),epoch)
    n.followed(Mock(result=lambda:S(status=4,result=S(error_code=0))),epoch,serial)
    assert n.goal is None and not n.safety.armed and not n.obstacle_wait.active
    assert n.state=='cancel'

def test_late_follow_acceptance_cancelled():
    n=runtime();future=Mock();n.follow_client=Mock()
    n.follow_client.send_goal_async.return_value=future
    path=RosPath();path.poses=[PoseStamped()]
    n.follow(path);callback=future.add_done_callback.call_args.args[0]
    n.abort('cancel');handle=Mock(accepted=True)
    callback(Mock(result=lambda:handle))
    handle.cancel_goal_async.assert_called_once()
    assert not handle.get_result_async.called and n.goal is None

def test_preview_never_arms_or_follows():
    n=runtime(False);n.follow=Mock()
    for i in range(6):n.retry('blocked')
    n.planned(result(),n.epoch)
    assert not n.safety.armed and not n.follow.called and not n.obstacle_wait.active

def test_arrival_clears_goal():
    n=runtime();n.retry('blocked')
    n.followed(Mock(result=lambda:S(status=4,result=S(error_code=0))),n.epoch,n.follow_serial)
    assert n.goal is None and not n.safety.armed and not n.obstacle_wait.active

def test_fault_still_terminates():
    n=runtime();n.retry('blocked');n.abort('实时点云中断')
    n.retry('late retry')
    assert n.goal is None and not n.safety.armed and n.state=='实时点云中断'

def test_goal_transform_uses_current_localization():
    n=runtime();n.follow_client=Mock()
    path=RosPath();path.header.stamp.sec=123
    q=PoseStamped();q.header=path.header;path.poses=[q]
    n.follow(path)
    sent=n.follow_client.send_goal_async.call_args.args[0].path
    assert sent.poses[-1].header.stamp.sec==0 and sent.poses[-1].header.stamp.nanosec==0
    assert path.header.stamp.sec==123

def test_empty_follow_retries_without_sending():
    n=runtime();n.follow_client=Mock();n.follow(RosPath())
    assert not n.follow_client.send_goal_async.called and n.obstacle_wait.active
