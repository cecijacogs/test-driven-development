from SignalDetection import SignalDetection
import matplotlib.pyplot as plt
class Experiment():
    # init empty list to store sdt objs
    def __init__(self):
        self.conditions = []
    def add_condition(self, sdt_obj: SignalDetection, label: str = None) -> None:
        # add the new condition to the array
        # store as a tuple
        self.conditions.append((sdt_obj, label))
    def sorted_roc_points(self) -> tuple[list[float], list[float]]:
        # return sorted false alarm and hit rates 
        false_alarm_rates = []
        hit_rates = []

        if not self.conditions:
            raise ValueError("No conditions present")
        
        for std_obj, __ in self.conditions:
            false_alarm_rates.append(std_obj.false_alarm())
            hit_rates.append(std_obj.hit_rate())
        
        sorted_pairs = sorted(zip(false_alarm_rates, hit_rates)) # order elements
        sorted_false_alarm, sorted_hit_rates = zip(*sorted_pairs) # unzip into two lists

        return list(sorted_false_alarm), list(sorted_hit_rates)
    def compute_auc(self) -> float:

        sorted_false_alarm, sorted_hit_rates = self.sorted_roc_points()
        # computer area under the curve for each condition
        if not self.sorted_false_alarm or (len(sorted_false_alarm) < 2):
            raise ValueError("No conditions present")

        # Trapezoidal Rule: AUC = sum( (x_i+1 - x_i) * (y_i+1 + y_i) / 2 )
        # computer the auc between each element in the array
        auc = sum(
            (sorted_false_alarm[i + 1] - sorted_false_alarm[i]) * (sorted_hit_rates[i + 1] + sorted_hit_rates[i]) / 2
            for i in range(len(sorted_false_alarm) - 1)
        )
        
        return auc
    def plot_roc_curve(self, show_plot: bool = True):
        try:
            false_alarm_rates, hit_rates = self.sorted_roc_points()
        except ValueError as e:
            print(e)
            return

        plt.figure(figsize=(6, 6))
        plt.plot(false_alarm_rates, hit_rates, marker='o', linestyle='-', color='b', label='ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle="--", color='gray', label="Random Classifier")
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.grid(True)

        if show_plot:
            plt.show()


