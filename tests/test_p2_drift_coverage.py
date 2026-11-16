"""Protocol-boundary checks for budgets and independent head selection."""
import unittest
import torch
from collections import Counter
from research.diagnostics.p2_drift_coverage import balanced_tensors,choose


class TestCoverageBudget(unittest.TestCase):
    def test_data_sizes_have_equal_presentations_and_different_unique_windows(self):
        for n,expected_per_trajectory in ((8,80),(32,20)):
            data=[{'x0':torch.zeros(1,2),'u_seq':torch.zeros(50,1),
                   'x_seq':torch.zeros(50,2)} for _ in range(n)]
            batch,samples=balanced_tensors(data,42)
            self.assertEqual(len(batch[0]),640)
            self.assertEqual(set(Counter(i for i,start in samples).values()),{expected_per_trajectory})
            self.assertEqual(len(set(samples)),160 if n==8 else 640)

    def test_each_head_can_select_its_own_lr(self):
        rows=[]
        for general in (False,True):
            for lr in (.001,.01):
                ratio=.05 if (general and lr==.01) or (not general and lr==.001) else .2
                for seed in (42,43,44):
                    rows.append({'general':general,'cfg':{'lr':lr},'seed':seed,'status':'complete',
                                 'final_validation_or_tiny_train':{'passed':True,
                                  'response':.01,'H15_ratio_to_ignore_input':ratio}})
        self.assertEqual(choose(rows,False)['lr'],.001)
        self.assertEqual(choose(rows,True)['lr'],.01)
