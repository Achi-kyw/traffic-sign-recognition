import editdistance

def character_accuracy(gt, pred):
    correct = sum(g == p for g, p in zip(gt, pred))
    return correct / max(len(gt), len(pred)) if max(len(gt), len(pred)) > 0 else 0.0

def character_error_rate(gt, pred):
    distance = editdistance.eval(gt, pred)
    return distance / len(gt) if len(gt) > 0 else 0.0

def levenshtein_similarity(gt, pred):
    distance = editdistance.eval(gt, pred)
    max_len = max(len(gt), len(pred))
    return 1 - distance / max_len if max_len > 0 else 1.0
