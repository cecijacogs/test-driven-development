import unittest

class TestExperiment(unittest.TestCase):

    # Test 1: Verifies that add_condition() correctly stores SignalDetection objects and labels
    def test_add_condition(self):
        exp = Experiment()
        sdt = SignalDetection(40, 10, 20, 30)
        exp.add_condition(sdt, "Condition A")
        self.assertEqual(len(exp.conditions), 1)
        self.assertEqual(exp.conditions[0][1], "Condition A")
        self.assertIsInstance(exp.conditions[0][0], SignalDetection)

    # Test 2: Verifies that sorted_roc_points() correctly returns sorted false alarm rates and hit rates
    def test_sorted_roc_points(self):
        exp = Experiment()
        exp.add_condition(SignalDetection(40, 10, 20, 30), "Condition A")
        exp.add_condition(SignalDetection(30, 15, 10, 25), "Condition B")
        
        fa_rates, hit_rates = exp.sorted_roc_points()
        self.assertEqual(fa_rates, [0.4, 0.3])  # Expected sorted FA rates
        self.assertEqual(hit_rates, [0.8, 0.6666666666666666])  # Expected sorted hit rates

    # Test 3: Verifies that compute_auc() produces expected AUC = 0.5 for two conditions (0,0) and (1,1)
    def test_compute_auc_two_conditions(self):
        exp = Experiment()
        exp.add_condition(SignalDetection(40, 10, 20, 30), "Condition A")  # (0, 0)
        exp.add_condition(SignalDetection(30, 15, 10, 25), "Condition B")  # (1, 1)
        
        auc = exp.compute_auc()
        self.assertAlmostEqual(auc, 0.5, places=2)  # AUC should be 0.5

    # Test 4: Verifies that compute_auc() produces expected AUC = 1 for three conditions: (0,0), (0,1), and (1,1)
    def test_compute_auc_perfect(self):
        exp = Experiment()
        exp.add_condition(SignalDetection(40, 10, 20, 30), "Condition A")  # (0, 0)
        exp.add_condition(SignalDetection(40, 10, 10, 40), "Condition B")  # (0, 1)
        exp.add_condition(SignalDetection(30, 15, 10, 25), "Condition C")  # (1, 1)
        
        auc = exp.compute_auc()
        self.assertEqual(auc, 1.0)  # Perfect AUC when conditions are perfectly distinguishable

    # Test 5: Verifies that compute_auc() raises ValueError when there are no conditions
    def test_compute_auc_empty(self):
        exp = Experiment()
        with self.assertRaises(ValueError):
            exp.compute_auc()

    # Test 6: Verifies that sorted_roc_points() raises ValueError when there are no conditions
    def test_sorted_roc_points_empty(self):
        exp = Experiment()
        with self.assertRaises(ValueError):
            exp.sorted_roc_points()

    # Test 7: Verifies that add_condition() adds multiple conditions correctly
    def test_add_multiple_conditions(self):
        exp = Experiment()
        sdt1 = SignalDetection(40, 10, 20, 30)
        sdt2 = SignalDetection(30, 15, 10, 25)
        
        exp.add_condition(sdt1, "Condition A")
        exp.add_condition(sdt2, "Condition B")
        
        self.assertEqual(len(exp.conditions), 2)
        self.assertEqual(exp.conditions[0][1], "Condition A")
        self.assertEqual(exp.conditions[1][1], "Condition B")

    # Test 8: Verifies that the ROC curve plotting function works
    def test_plot_roc_curve(self):
        # Just verify that the plot function doesn't raise errors
        exp = Experiment()
        exp.add_condition(SignalDetection(40, 10, 20, 30), "Condition A")
        exp.add_condition(SignalDetection(30, 15, 10, 25), "Condition B")
        
        try:
            exp.plot_roc_curve(show_plot=False)  # Don't display the plot in unit test
        except Exception as e:
            self.fail(f"plot_roc_curve raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()
