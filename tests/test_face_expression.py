import unittest
import numpy as np
from tools.control.face_expression import face_goal,PRESETS
from tools.control.face_control import FaceController


class FaceExpressionTests(unittest.TestCase):
    def test_blinks_changes_and_cancel_use_actual_state_without_violation(self):
        c=FaceController(PRESETS['neutral']);closed=[]
        for i in range(1,721):
            now=i/30
            affect=('warm' if i<240 else 'uncertain' if i<420 else 'firm')
            packet={'intent':dict(affect=affect,released=False,kind='say' if i<600 else 'cancel',speaking=i<600)}
            q=c.command(face_goal(packet,now,0.),now)
            report=c.observe(q.astype(np.float32))
            self.assertFalse(report['reasons'],(i,report))
            if 3.5<=now<=3.94:closed.append(q[0])
        self.assertLess(min(closed),.10)
        np.testing.assert_allclose(c.q,PRESETS['neutral'],atol=1e-5)

    def test_cancel_overrides_affect_and_speaking(self):
        p={'intent':dict(affect='warm',released=False,kind='cancel',speaking=True)}
        self.assertEqual(face_goal(p,1.,0.),PRESETS['neutral'])

    def test_legacy_six_inputs_preserved_and_dimension_changes_rejected(self):
        c=FaceController(PRESETS['neutral'][:6])
        self.assertEqual(len(c.command(PRESETS['warm'][:6],0.)),6)
        with self.assertRaises(ValueError):c.command(PRESETS['warm'],0.)
