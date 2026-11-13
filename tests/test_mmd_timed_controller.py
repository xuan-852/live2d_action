import unittest
import numpy as np
from tools.control.mmd_timed_controller import make_stream


class MmdTimedInterfaceTests(unittest.TestCase):
    def test_projection_request_keeps_source_shortening_below_actuator_limit(self):
        doc=self.document()
        for frame in doc['sides']['anatomicalRight']['frames']:
            frame['target']=frame['target']|dict(palmProjectedRatio=.4)
        home=np.array([[0.,0.],[0.,-1.],[0.,-2.],[0.,-2.2]])
        old=make_stream(doc,dict(native=home,patch=home))
        new=make_stream(doc,dict(native=home,patch=home),patch_projection=True)
        for a,b in zip(old['frames'],new['frames']):
            self.assertEqual(a['targets']['patch']['wrist'],b['targets']['patch']['wrist'])
            self.assertEqual(a['targets']['patch']['elbow'],b['targets']['patch']['elbow'])
            self.assertEqual(a['headBody'],b['headBody'])
            if b['stage']=='mmd-source':
                self.assertEqual(b['targets']['patch']['projection'],[.6,.8,.4])
                self.assertNotIn('projection',b['targets']['native'])

    def document(self):
        target=dict(upperHeadingDeg=-70,forearmHeadingDeg=-50,palmHeadingDeg=-30,
                    upperProjectedRatio=.6,forearmProjectedRatio=.8)
        frames=[dict(seconds=i/30,target=target|dict(palmHeadingDeg=-30+i*2)) for i in range(3)]
        return dict(sourceFile='test.vmd',sourceSha256='test',sides={k:dict(frames=frames) for k in ('anatomicalLeft','anatomicalRight')},
                    coordinated=[dict(headLocal=dict(yaw=0,pitch=0,roll=0),body=dict(yaw=0,pitch=0,roll=0)) for _ in frames])

    def test_source_samples_clocks_modes_and_shortening_preserved(self):
        home=np.array([[0.,0.],[0.,-1.],[0.,-2.],[0.,-2.2]])
        stream=make_stream(self.document(),dict(native=home,patch=home))
        source=[f for f in stream['frames'] if f['stage']=='mmd-source']
        self.assertEqual([f['sourceTime'] for f in source],[0.,1/30,2/30])
        self.assertEqual(len(stream['frames']),93)
        for i,f in enumerate(source):
            t=f['targets']['patch']
            self.assertEqual(t['mode'],'track')
            self.assertAlmostEqual(t['time'],(31+i)/30)
            self.assertAlmostEqual(np.linalg.norm(t['elbow']),.6)
            self.assertAlmostEqual(np.linalg.norm(np.array(t['wrist'])-t['elbow']),.8)
            self.assertAlmostEqual(t['palm_heading'],-30+2*i)
            self.assertIn('velocity',t)
            if i:
                previous=source[i-1]['targets']['patch']
                np.testing.assert_allclose(t['velocity'],(np.array(t['wrist'])-previous['wrist'])*30)
            else:
                np.testing.assert_array_equal(t['velocity'],[0,0])
        self.assertEqual(stream['frames'][0]['targets']['patch']['mode'],'hold')
        self.assertEqual(stream['frames'][-1]['targets']['patch']['time'],63/30)

    def test_misaligned_arm_clock_rejected(self):
        doc=self.document()
        doc['sides']['anatomicalRight']={'frames':[dict(seconds=0,target={})]}
        with self.assertRaises(ValueError):
            make_stream(doc,dict(native=np.zeros((4,2)),patch=np.zeros((4,2))))
