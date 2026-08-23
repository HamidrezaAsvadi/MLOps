from hot_model.pipeline import run_pipeline

class HotModel:

    def infer(self, readings: list[dict]) -> list:
        infered = run_pipeline(readings)
        return infered