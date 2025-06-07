"""
Mappers for transforming the data.
"""

from api.packages.dtos import OnePredictionOutputDto
from api.packages.wrapper_outputs import WrapperOutput


def map_to_output_dto(result: WrapperOutput) -> OnePredictionOutputDto:
    """
    Map the result of a prediction to one prediction output.

    Args:
        result (WrapperOutput): Result of a prediction.

    Returns:
        OnePredictionOutputDto: One prediction output.
    """

    # create a OnePredictionOutputDto
    output: OnePredictionOutputDto = OnePredictionOutputDto(prediction=result.prediction)

    return output
