"""Register the GQLA model class with vLLM.

Two entry points to the same ``register()`` call:

* **Import side-effect** — ``import src.vllm_register`` registers
  in the current process. Sufficient for ``tensor_parallel_size=1``.
* **vLLM plugin entry point** — ``pyproject.toml`` exposes ``register`` under
  ``vllm.general_plugins``. After ``pip install -e .`` vLLM calls it in every
  process (main + each TP worker), enabling ``tensor_parallel_size >= 2``.

The ``"<module>:<class>"`` lazy form keeps the import light (no CUDA pulled in).
"""

from vllm import ModelRegistry

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    ModelRegistry.register_model(
        "Glm4MoeLiteGQLAForCausalLM",
        "src.vllm_model:Glm4MoeLiteGQLAForCausalLM",
    )
    ModelRegistry.register_model(
        "Glm4MoeLiteGQLAAbsorbForCausalLM",
        "src.vllm_model:Glm4MoeLiteGQLAAbsorbForCausalLM",
    )
    _REGISTERED = True


register()
