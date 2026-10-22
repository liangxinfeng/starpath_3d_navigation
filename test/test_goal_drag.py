import os,unittest,math
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
from types import SimpleNamespace as S
from PyQt5 import QtWidgets as W,QtCore as C,QtGui as G
from nav_view import NavView
from nav_ui import Console

app=W.QApplication.instance() or W.QApplication([])
def event(kind,x,y,button=C.Qt.LeftButton,buttons=C.Qt.LeftButton):
    return G.QMouseEvent(kind,C.QPointF(x,y),button,buttons,C.Qt.NoModifier)
class GoalDragTests(unittest.TestCase):
    def setUp(self):
        self.v=NavView();self.v.resize(900,600);self.v.begin_pick();self.results=[]
        self.v.goalSelected.connect(lambda *args:self.results.append(args))
    def tearDown(self):self.v.close()
    def drag(self,a,b):
        self.v.mousePressEvent(event(C.QEvent.MouseButtonPress,*a))
        self.v.mouseMoveEvent(event(C.QEvent.MouseMove,*b,button=C.Qt.NoButton))
        self.assertEqual(self.results,[])
        self.v.mouseReleaseEvent(event(C.QEvent.MouseButtonRelease,*b,buttons=C.Qt.NoButton))
    def test_four_world_directions(self):
        for target,angle in [((550,300),0),((450,200),math.pi/2),((350,300),math.pi),((450,400),-math.pi/2)]:
            self.v.begin_pick();self.results=[];self.drag((450,300),target)
            self.assertAlmostEqual(abs(self.results[-1][2]),abs(angle),places=8)
            if angle!=math.pi:self.assertAlmostEqual(self.results[-1][2],angle,places=8)
    def test_click_clears_old_heading(self):
        self.v.goal_yaw=1.;self.drag((450,300),(452,301));self.assertIsNone(self.results[-1][2])
    def test_map_coordinate_roundtrip_pan_zoom_rotated_view(self):
        self.v.center=np.array([3.,-4.,.8]);self.v.radius=6.;self.v.zoom=2.1;self.v.pan=np.array([27.,-42.]);self.v.yaw=.6
        p=np.array([3.6,-4.2,1.1]);uv=self.v.project([p])[0]
        np.testing.assert_allclose(self.v.screen_on_plane(C.QPointF(*uv),p[2]),p[:2],atol=1e-10)
    def test_cancel_restores_previous_target(self):
        self.v.goal=[1,2,0];self.v.goal_yaw=.7
        self.v.mousePressEvent(event(C.QEvent.MouseButtonPress,450,300))
        self.v.mousePressEvent(event(C.QEvent.MouseButtonPress,460,300,button=C.Qt.RightButton,buttons=C.Qt.RightButton))
        self.assertEqual(self.results,[]);self.assertEqual(self.v.goal,[1,2,0]);self.assertEqual(self.v.goal_yaw,.7);self.assertFalse(self.v.pick)
    def test_selection_updates_ui_and_payload_without_execution(self):
        x=W.QDoubleSpinBox();y=W.QDoubleSpinBox();heading=W.QDoubleSpinBox();check=W.QCheckBox();label=W.QLabel()
        for widget in [x,y,heading]:widget.setRange(-1000,1000)
        calls=[];obj=S(x=x,y=y,heading=heading,heading_check=check,goal_hint=label,view=self.v,body=np.eye(4),request=lambda *a:calls.append(a))
        obj.sync_goal=lambda *args:Console.sync_goal(obj)
        Console.picked(obj,1.25,-2.5,math.pi/2)
        self.assertTrue(check.isChecked());self.assertEqual(heading.value(),90);self.assertEqual(calls,[])
        Console.goal(obj,False)
        self.assertEqual(calls[-1][1]['execute'],False);self.assertAlmostEqual(calls[-1][1]['yaw'],math.pi/2)
        Console.picked(obj,1,2,None);Console.goal(obj,False);self.assertIsNone(calls[-1][1]['yaw'])
    def test_header_click_does_not_create_target(self):
        self.v.mousePressEvent(event(C.QEvent.MouseButtonPress,450,15));self.assertIsNone(self.v.goal_drag)

if __name__=='__main__':unittest.main()
