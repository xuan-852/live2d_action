"""Whole-baseline mechanisms; no human-quality certification."""
import unittest
import numpy as np
from tools.whole.batch import sample_document


class WholeTests(unittest.TestCase):
    def document(self):
        frames=[]
        for t,angle in [(0,0),(.5,20),(1,0)]:
            frames.append(dict(t=t,shoulder=angle,elbow=angle,wrist=0,
                native={'Param94':angle},projection={'upperLength':1,'forearmLength':1,
                'palmLength':1,'palmWidth':1,'palmShear':0},
                girdle={'rootX':0,'rootY':0,'socketX':0,'socketY':0}))
        return {'frames':frames}

    def test_same_clock_all_tracks_and_neutral_end(self):
        times,curves,patch=sample_document(self.document())
        self.assertEqual(len(times),31)
        self.assertEqual(len(patch),len(curves['Param94']))
        np.testing.assert_allclose(np.diff(times),1/30)
        self.assertAlmostEqual(curves['Param94'][15],20)
        self.assertAlmostEqual(patch[15]['shoulder'],20)
        self.assertEqual(patch[-1]['shoulder'],0)
        self.assertEqual(curves['Param94'][-1],0)

    def test_short_final_interval_holds_endpoint_not_omits_it(self):
        d=self.document();d['frames'][-1]['t']=1.013
        times,curves,patch=sample_document(d)
        self.assertGreaterEqual(times[-1],1.013)
        self.assertEqual(curves['Param94'][-1],0)
        self.assertEqual(patch[-1]['elbow'],0)
