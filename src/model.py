import joblib
import pandas as pd


MODEL_PATH = "models/final_resume_job_ranker.pkl"
FEATURES_PATH = "models/final_features.pkl"


def load_model():
    """Load the trained ranking model."""
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, features


def predict_match_score(model, features, feature_values):
    """
    Predict the candidate-job matching score.

    Parameters
    ----------
    model : trained ranking model
    features : list
        Feature names used during training.
    feature_values : dict
        Feature values for one candidate-job pair.

    Returns
    -------
    float
        Matching probability/score.
    """

    X = pd.DataFrame(
        [[feature_values[feature] for feature in features]],
        columns=features
    )

    score = model.predict_proba(X)[0, 1]

    return float(score)
