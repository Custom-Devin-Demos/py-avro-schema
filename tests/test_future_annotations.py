# Copyright 2022 J.P. Morgan Chase & Co.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

import decimal
from typing import Annotated, List, Optional, Union

import pydantic

import py_avro_schema as pas
from tests.test_pydantic import assert_schema


def test_future_annotations_self_reference():
    """Test that self-references work correctly with future annotations"""

    class PyType(pydantic.BaseModel):
        field_a: str
        field_self: Optional[PyType] = None

    expected = {
        "type": "record",
        "name": "PyType",
        "fields": [
            {
                "name": "field_a",
                "type": "string",
            },
            {
                "name": "field_self",
                "type": ["null", "PyType"],
                "default": None,
            },
        ],
    }
    assert_schema(PyType, expected)


def test_future_annotations_nested_models():
    """Test nested models with future annotations"""

    class ChildModel(pydantic.BaseModel):
        child_field: str

    class ParentModel(pydantic.BaseModel):
        parent_field: str
        child: ChildModel
        children: List[ChildModel]

    expected = {
        "type": "record",
        "name": "ParentModel",
        "fields": [
            {
                "name": "parent_field",
                "type": "string",
            },
            {
                "name": "child",
                "type": {
                    "type": "record",
                    "name": "ChildModel",
                    "fields": [
                        {
                            "name": "child_field",
                            "type": "string",
                        }
                    ],
                },
            },
            {
                "name": "children",
                "type": {
                    "type": "array",
                    "items": "ChildModel",
                },
            },
        ],
    }
    assert_schema(ParentModel, expected)


def test_future_annotations_complex_types():
    """Test complex type annotations with future annotations"""

    class PyType(pydantic.BaseModel):
        optional_str: Optional[str] = None
        union_field: Union[str, int]
        list_field: List[str]
        optional_list: Optional[List[str]] = None

    expected = {
        "type": "record",
        "name": "PyType",
        "fields": [
            {
                "name": "optional_str",
                "type": ["null", "string"],
                "default": None,
            },
            {
                "name": "union_field",
                "type": ["string", "long"],
            },
            {
                "name": "list_field",
                "type": {
                    "type": "array",
                    "items": "string",
                },
            },
            {
                "name": "optional_list",
                "type": ["null", {"type": "array", "items": "string"}],
                "default": None,
            },
        ],
    }
    assert_schema(PyType, expected)


def test_future_annotations_inheritance():
    """Test model inheritance with future annotations"""

    class BaseModel(pydantic.BaseModel):
        base_field: str
        base_self_ref: Optional[BaseModel] = None

    class DerivedModel(BaseModel):
        derived_field: int
        derived_self_ref: Optional[DerivedModel] = None

    expected = {
        "type": "record",
        "name": "DerivedModel",
        "fields": [
            {
                "name": "base_field",
                "type": "string",
            },
            {
                "name": "base_self_ref",
                "type": ["null", "BaseModel"],
                "default": None,
            },
            {
                "name": "derived_field",
                "type": "long",
            },
            {
                "name": "derived_self_ref",
                "type": ["null", "DerivedModel"],
                "default": None,
            },
        ],
    }
    assert_schema(DerivedModel, expected)


def test_future_annotations_cross_references():
    """Test cross-references between models with future annotations"""

    class ModelA(pydantic.BaseModel):
        field_a: str
        ref_to_b: Optional[ModelB] = None

    class ModelB(pydantic.BaseModel):
        field_b: str
        ref_to_a: Optional[ModelA] = None

    expected_a = {
        "type": "record",
        "name": "ModelA",
        "fields": [
            {
                "name": "field_a",
                "type": "string",
            },
            {
                "name": "ref_to_b",
                "type": ["null", "ModelB"],
                "default": None,
            },
        ],
    }
    assert_schema(ModelA, expected_a)

    expected_b = {
        "type": "record",
        "name": "ModelB",
        "fields": [
            {
                "name": "field_b",
                "type": "string",
            },
            {
                "name": "ref_to_a",
                "type": ["null", "ModelA"],
                "default": None,
            },
        ],
    }
    assert_schema(ModelB, expected_b)


def test_future_annotations_annotated_types():
    """Test Annotated types with future annotations"""

    class PyType(pydantic.BaseModel):
        decimal_field: Annotated[decimal.Decimal, pas.DecimalMeta(precision=10, scale=2)]
        annotated_str: Annotated[str, "some metadata"]

    expected = {
        "type": "record",
        "name": "PyType",
        "fields": [
            {
                "name": "decimal_field",
                "type": {
                    "type": "bytes",
                    "logicalType": "decimal",
                    "precision": 10,
                    "scale": 2,
                },
            },
            {
                "name": "annotated_str",
                "type": "string",
            },
        ],
    }
    assert_schema(PyType, expected)


def test_future_annotations_string_forward_ref():
    """Test explicit string forward references with future annotations"""

    class PyType(pydantic.BaseModel):
        field_a: str
        field_self: "PyType"

    expected = {
        "type": "record",
        "name": "PyType",
        "fields": [
            {
                "name": "field_a",
                "type": "string",
            },
            {
                "name": "field_self",
                "type": "PyType",
            },
        ],
    }
    assert_schema(PyType, expected)


def test_future_annotations_list_forward_ref():
    """Test list of forward references with future annotations"""

    class PyType(pydantic.BaseModel):
        field_a: str
        field_list: List[PyType]

    expected = {
        "type": "record",
        "name": "PyType",
        "fields": [
            {
                "name": "field_a",
                "type": "string",
            },
            {
                "name": "field_list",
                "type": {
                    "type": "array",
                    "items": "PyType",
                },
            },
        ],
    }
    assert_schema(PyType, expected)


def test_future_annotations_union_forward_ref():
    """Test union with forward references with future annotations"""

    class PyType(pydantic.BaseModel):
        field_a: str
        field_union: Union[str, PyType]

    expected = {
        "type": "record",
        "name": "PyType",
        "fields": [
            {
                "name": "field_a",
                "type": "string",
            },
            {
                "name": "field_union",
                "type": ["string", "PyType"],
            },
        ],
    }
    assert_schema(PyType, expected)
