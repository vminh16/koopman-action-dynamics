"""Hand-worked cases for the preregistered multi-seed selection boundary."""
import unittest
from research.diagnostics.p2_robust_flow import selection


def rows(lr, responses):
    return [{'cfg':{'lr':lr},'seed':seed,'status':'complete',
             'final_validation_or_tiny_train':{'response':response,
                 'H15_ratio_to_ignore_input':.1,'passed':response<=.05}}
            for seed,response in zip((42,43,44),responses)]


class TestMultiSeedSelection(unittest.TestCase):
    def test_one_good_seed_cannot_hide_two_failures(self):
        candidates=rows(.001,(.0001,.3,.3))+rows(.01,(.03,.04,.02))
        self.assertEqual(selection(candidates)['cfg']['lr'],.01)
        self.assertTrue(selection(candidates)['development_qualified'])

    def test_worst_seed_takes_precedence_over_mean(self):
        candidates=rows(.001,(.0001,.0001,.04))+rows(.01,(.02,.02,.03))
        self.assertEqual(selection(candidates)['cfg']['lr'],.01)
