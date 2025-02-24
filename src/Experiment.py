from SignalDetection import SignalDetection
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
        
