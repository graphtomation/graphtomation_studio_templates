from typing_extensions import TypedDict
from pydantic import create_model, Field, BaseModel
from typing import Literal, Optional, Union, List, Type, Any


class PydanticGeneratorConfig(TypedDict, total=False):
    field_name: str
    field_type: Literal["str", "int", "float", "bool", "list", "dict"]
    required: Optional[bool]
    description: Optional[str]
    default: Optional[Union[str, int, float, bool, list, dict]]


def pydantic_model_generator(
    output_json_configs: List[PydanticGeneratorConfig],
) -> Type[BaseModel]:
    """
    Generate a dynamic Pydantic model from a list of PydanticGeneratorConfig.
    Fields marked as 'required' will be mandatory in the Pydantic model.
    """
    if not isinstance(output_json_configs, list):
        return output_json_configs

    fields = {}
    type_mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    for field_conf in output_json_configs:
        field_name = field_conf.get("field_name")
        if not field_name:
            return None

        field_type_str = field_conf.get("field_type", "str")
        field_type = type_mapping.get(field_type_str, str)
        required = field_conf.get("required", False)
        default_value = field_conf.get("default", ... if required else None)
        description = field_conf.get("description", None)

        fields[field_name] = (
            field_type,
            Field(default_value, description=description),
        )

    model = create_model("PydanticModel", **fields)
    return model
