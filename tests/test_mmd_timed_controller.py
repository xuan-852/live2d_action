import unittest
import numpy as np
from tools.control.mmd_timed_controller import make_stream


class MmdTimedInterfaceTests(unittest.TestCase):
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
