from xgboost import XGBClassifier


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Create the XGBoost classifier
model = XGBClassifier()


# Load the trained model from the JSON file
model.load_model("xgboost_model.json")


# Map the numerical prediction returned by the model
# to the corresponding Rock-Paper-Scissors gesture.
#
# 0 -> Paper
# 1 -> Rock
# 2 -> Scissors
RPS_LABELS = {
    0: "paper",
    1: "rock",
    2: "scissors"
}


# ============================================================
# ROCK-PAPER-SCISSORS PREDICTION
# ============================================================

def predict_rps(hand_value):
    """
    Predict the Rock-Paper-Scissors gesture from
    the hand features.

    Parameters:
        hand_value : Input features extracted from the hand

    Returns:
        "paper", "rock", or "scissors"
        depending on the model prediction.
    """

    # Use the trained XGBoost model to make a prediction
    prediction = model.predict(hand_value)


    # Check that the model returned at least one prediction
    if len(prediction) >= 1:

        # Convert the numerical prediction into
        # its corresponding gesture name
        return RPS_LABELS[prediction[0]]


    # Return "None" if no prediction is available
    return "None"
