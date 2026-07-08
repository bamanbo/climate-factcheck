from factcheck.evaluate import macro_f1

def test_perfect_prediction_is_one():
    y = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"] 
    assert macro_f1(y, y) == 1.0 
    #if a prediction doesn't score exactly 1.0, something is wired wrong
    #might be arguments swapped, wrong metric, wrong averaging)

def test_hand_computed_case():
    """Check macro_F1 against a value derived by hand, independently of the code.
    Setup: one A and one B; the model predicts A for both. 
        - Class A: precision = 1/2 (two A-predictions, one correct), 
                    recall = 1/1 -> F1 = 2/3 
        - Class B: never predicted -> F1 = 0
        - Macro F1 = (2/3 + 0) / 2 = 1/3

    Note the misclassified B lowers A's precision, so the intuitive answer (1+0) / 2 = 0.5 is wrong.
    """
    
    y_true = ["A", "B"]
    y_pred = ["A", "A"]
    assert abs(macro_f1(y_true, y_pred) - (1/3)) < 1e-9