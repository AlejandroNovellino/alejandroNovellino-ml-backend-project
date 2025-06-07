"""
Outputs of the wrapper
"""

class WrapperOutput:
    """
    Output DTO for one prediction.
    """

    prediction: str

    def __init__(self, prediction: str):
        self.prediction = prediction