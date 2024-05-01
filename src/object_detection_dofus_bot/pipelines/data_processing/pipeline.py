from kedro.pipeline import Pipeline, node, pipeline

from .nodes import infer


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=infer,
                inputs=["model", "dofus"],
                outputs=None,
                name="infer_node",
            ),
        ]
    )
