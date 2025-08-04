from enum import Enum
from src.model.movie_predictor import movie_predictor

class custom_enum(Enum):
    @classmethod
    def names(cls) -> list:
        return [member.name for member in list(cls)]

    @classmethod
    def validation(cls, name: str) -> bool:
        names = [name.lower() for name in cls.names()]
        if name.lower() in names:
            return True
        else:
            raise ValueError(f"Invalid argument. Must be one of {cls.names()}")

class model_types(custom_enum):
    MOVIE_PREDICTOR = movie_predictor

