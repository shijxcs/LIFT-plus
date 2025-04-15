import torch
from collections import defaultdict


class Evaluator:
    """Evaluator for classification."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._y_true = []
        self._y_pred = []

    def process(self, logit, label):
        # logit (torch.Tensor): model output [batch, num_classes]
        # label (torch.LongTensor): ground truth [batch]
        pred = logit.argmax(dim=1)

        self._y_pred.extend(pred.tolist())
        self._y_true.extend(label.tolist())

    def evaluate(self, many_classes=None, med_classes=None, few_classes=None):
        correct = torch.tensor(self._y_pred).eq(torch.tensor(self._y_true))
        acc = correct.float().mean().mul_(100.0)
        print(f"* Overall accuracy: {acc:.2f}%")

        _per_class_correct = defaultdict(int)
        _per_class_total = defaultdict(int)
        _per_class_acc = defaultdict(float)

        for gt, pred in zip(self._y_true, self._y_pred):
            _per_class_correct[gt] += int(gt == pred)
            _per_class_total[gt] += 1

        for gt in _per_class_correct.keys():
            _per_class_acc[gt] = _per_class_correct[gt] / _per_class_total[gt]
        
        cls_accs = torch.Tensor([100.0 * _per_class_acc[gt] for gt in sorted(_per_class_correct.keys())])
        mean_acc = torch.mean(cls_accs)
        print(f"* Mean class accuracy: {mean_acc:.2f}%")

        if (many_classes is not None) and (med_classes is not None) and (few_classes is not None):
            many_acc = torch.mean(cls_accs[many_classes])
            med_acc = torch.mean(cls_accs[med_classes])
            few_acc = torch.mean(cls_accs[few_classes])
            print(f"* Many: {many_acc:.2f}%  Med: {med_acc:.2f}%  Few: {few_acc:.2f}%")
