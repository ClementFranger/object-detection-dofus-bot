from kedro.pipeline import Pipeline, node, pipeline

from .nodes import infer, bot


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=infer,
                inputs=["model", "dofus"],
                outputs=None,
                name="infer_node",
            ),
            node(
                func=bot,
                inputs=["model", "dofus", "params:agent"],
                outputs=None,
                name="bot_node",
            ),
        ]
    )
