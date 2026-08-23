from cold_model.pipeline import run

class ColdModel:

    def infer(self, readings: list[dict]) -> list:
        infered = run(readings)
        return infered