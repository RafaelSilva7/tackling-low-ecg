from torch import Tensor
from torcheval.metrics import functional as metrics


class MulticlassMetrics:

    def __init__(self, num_classes: int, 
                 average: str = "macro", 
                 compute_matrix: bool = False) -> None:
        self.num_classes = num_classes 
        self.average = average
        self.compute_matrix = compute_matrix

    def compute(self, input: Tensor, target: Tensor) -> dict[str,Tensor]:

        self.accuracy = metrics.multiclass_accuracy(input, target).item()
        self.precision = metrics.multiclass_precision(input, target, 
                                                      num_classes=self.num_classes, 
                                                      average=self.average).item()
        self.recall = metrics.multiclass_recall(input, target, 
                                                num_classes=self.num_classes, 
                                                average=self.average).item()
        self.macro_f1 = metrics.multiclass_f1_score(input, target, 
                                                    num_classes=self.num_classes, 
                                                    average=self.average).item()

        all_metrics = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'macro_f1': self.macro_f1,
        }

        if self.compute_matrix:
            self.confusion_matrix = metrics.multiclass_confusion_matrix(input, target, 
                                                                        num_classes=self.num_classes).numpy()
            all_metrics['confusion_matrix'] = self.confusion_matrix

        return all_metrics
    
