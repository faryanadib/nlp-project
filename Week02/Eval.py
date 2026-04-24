y_true = [1, 0, 0, 1, 0]
y_pred = [1, 0, 0, 0, 1]
def confusion_matrix(y_true, y_pred):
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1
        else:
            TN += 1

    return TP, FP, FN, TN


def accuracy(y_true, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    total = TP + FP + FN + TN
    return (TP + TN) / total


def precision(y_true, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    return (TP / (TP + FP))    if TP + FP >0 else 0

def recall(y_true, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    return (TP / (TP + FN)) if TP + FN >0 else 0

def f1_score(y_true, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    perc1 = precision(y_true,y_pred)
    rec1 = recall(y_true,y_pred)
    return (2 * (perc1 * rec1) / (perc1 + rec1)) if perc1 + rec1 >0 else 0


# ==============================================================
# Multi-Class (3+ labels e.g. pos / neu / neg)
# ==============================================================

def multiclass_precision(y_true, y_pred, label):
    TP = 0
    FP = 0
    for true, pred in zip(y_true, y_pred):
        if pred == label and true == label:
            TP += 1
        elif pred == label and true != label:
            FP += 1
    return TP / (TP + FP) if TP + FP > 0 else 0

def multiclass_recall(y_true, y_pred, label):
    TP = 0
    FN = 0
    for true, pred in zip(y_true, y_pred):
        if true == label and pred == label:
            TP += 1
        elif true == label and pred != label:
            FN += 1
    return TP / (TP + FN) if TP + FN > 0 else 0

def multiclass_f1(y_true, y_pred, label):
    p = multiclass_precision(y_true, y_pred, label)
    r = multiclass_recall(y_true, y_pred, label)
    return 2 * p * r / (p + r) if p + r > 0 else 0

def macro_precision(y_true, y_pred):
    labels = set(y_true)
    total = 0
    for label in labels:
        total += multiclass_precision(y_true, y_pred, label)
    return total / len(labels)

def macro_recall(y_true, y_pred):
    labels = set(y_true)
    total = 0
    for label in labels:
        total += multiclass_recall(y_true, y_pred, label)
    return total / len(labels)

def macro_f1(y_true, y_pred):
    labels = set(y_true)
    total = 0
    for label in labels:
        total += multiclass_f1(y_true, y_pred, label)
    return total / len(labels)


# ==============================================================
# Test Cases
# ==============================================================

def test_case_1():
    # perfect classification
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    print("Test 1 - Perfect Classification")
    print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"  accuracy={accuracy(y_true, y_pred)}, precision={precision(y_true, y_pred)}, recall={recall(y_true, y_pred)}, f1={f1_score(y_true, y_pred)}")

def test_case_2():
    # all wrong
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 1, 1]
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    print("Test 2 - All Wrong")
    print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"  accuracy={accuracy(y_true, y_pred)}, precision={precision(y_true, y_pred)}, recall={recall(y_true, y_pred)}, f1={f1_score(y_true, y_pred)}")

def test_case_3():
    # no predicted positives
    y_true = [1, 0, 1, 0]
    y_pred = [0, 0, 0, 0]
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    print("Test 3 - No Predicted Positives")
    print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"  precision={precision(y_true, y_pred)}, recall={recall(y_true, y_pred)}, f1={f1_score(y_true, y_pred)}")

def test_case_4():
    # no actual positives
    y_true = [0, 0, 0, 0]
    y_pred = [1, 0, 1, 0]
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    print("Test 4 - No Actual Positives")
    print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"  precision={precision(y_true, y_pred)}, recall={recall(y_true, y_pred)}, f1={f1_score(y_true, y_pred)}")

def test_case_5():
    # high recall, low precision
    # classifier predicts almost everything as positive
    # catches all real positives (recall=1) but many false alarms (low precision)
    y_true = [1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 1, 0]
    print("Test 5 - High Recall, Low Precision")
    print(f"  precision={precision(y_true, y_pred):.2f}, recall={recall(y_true, y_pred):.2f}")

def test_case_6():
    # high precision, low recall
    # classifier only predicts positive when very sure
    # almost no false alarms (precision=1) but misses many positives (low recall)
    y_true = [1, 1, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 0, 0, 0]
    print("Test 6 - High Precision, Low Recall")
    print(f"  precision={precision(y_true, y_pred):.2f}, recall={recall(y_true, y_pred):.2f}")

def test_case_7():
    # imbalanced data, classifier always predicts negative
    # accuracy looks high (0.9) but precision/recall/f1 are all 0
    y_true = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print("Test 7 - Imbalanced Data, Always Predict Negative")
    print(f"  accuracy={accuracy(y_true, y_pred)}, precision={precision(y_true, y_pred)}, recall={recall(y_true, y_pred)}, f1={f1_score(y_true, y_pred)}")

def test_multiclass():
    y_true = ["pos", "pos", "pos", "neu", "neu", "neg", "neg", "neg"]
    y_pred = ["pos", "pos", "neu", "pos", "neu", "neg", "neg", "neg"]
    print("Test Multi-Class")
    for label in ["pos", "neu", "neg"]:
        p = multiclass_precision(y_true, y_pred, label)
        r = multiclass_recall(y_true, y_pred, label)
        f = multiclass_f1(y_true, y_pred, label)
        print(f"  {label}: precision={p:.2f}, recall={r:.2f}, f1={f:.2f}")
    print(f"  Macro Precision={macro_precision(y_true, y_pred):.2f}, Macro Recall={macro_recall(y_true, y_pred):.2f}, Macro F1={macro_f1(y_true, y_pred):.2f}")


if __name__ == "__main__":
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()
    test_case_7()
    test_multiclass()
