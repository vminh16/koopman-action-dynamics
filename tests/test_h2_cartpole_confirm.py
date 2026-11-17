import unittest

from research.diagnostics import h2_cartpole_confirm as k


class C4RuleTest(unittest.TestCase):
    def out(self, fb, g5):
        hb = '[1.57,3.14)'
        runs = {f'fixedB/s{s}/{hb}': fb for s in k.SEEDS} | {f'cosG5/s{s}/{hb}': g5 for s in k.SEEDS}
        return {'floors': {hb: {'fixed_B': .771, 'scalar_gain': .908}}, 'runs': runs}

    def test_rule_passes_only_when_both_classes_sit_at_their_floor(self):
        self.assertEqual(k.rules(self.out(.80, .90))['C4'], 'PASS')
        self.assertEqual(k.rules(self.out(1.2, .90))['C4'], 'FAIL')  # fixed-B far above floor: fitting issue
        self.assertEqual(k.rules(self.out(.80, .03))['C4'], 'FAIL')  # G.5 fixes it


if __name__ == '__main__':
    unittest.main()
