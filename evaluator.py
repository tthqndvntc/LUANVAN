from sklearn.metrics import accuracy_score, confusion_matrix

class Evaluator:
    @staticmethod
    def get_results(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return acc, cm