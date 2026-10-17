import unittest
from reasoning.full_length import select,CUTOFF

class FullLengthTests(unittest.TestCase):
    def features(self):
        return [dict(id=str(i),cluster=i%4,total_tokens=2000+i*200,
                     response_tokens=1800+i*200,complete_box=True,repetition=0.) for i in range(60)]
    def test_unique_input_order_independent_and_long_context_coverage(self):
        fs=self.features();a,q=select(fs,32);b,r=select(fs[::-1],32)
        self.assertEqual((a,q),(b,r));self.assertEqual(len({x['id'] for x in a}),32)
        self.assertTrue(any(x['total_tokens']>8192 for x in a))
        self.assertEqual({x['cluster'] for x in a},{0,1,2,3})
        self.assertTrue(all(set(x['quartile_quotas'])=={0,1,2,3} for x in q.values()))
    def test_incomplete_repeated_and_overlength_are_excluded(self):
        fs=self.features();fs[0]['complete_box']=False;fs[1]['repetition']=.5;fs[2]['total_tokens']=CUTOFF
        chosen,_=select(fs,32);self.assertFalse({'0','1','2'}&{x['id'] for x in chosen})
    def test_capacity_and_duplicate_identity_fail_closed(self):
        fs=self.features()
        with self.assertRaises(ValueError):select(fs,100)
        with self.assertRaises(ValueError):select(fs+[fs[0]],32)

if __name__=='__main__':unittest.main()
