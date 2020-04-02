from datastructures.machine.coldquanta_circuit_schema import circuit_schema

from jsonschema import validate
from jsonschema.exceptions import ValidationError
import pytest
import json

def test_valid_schema():
    circuit_instance = {
        "moments": [{
            "gates": [
                {
                    "id": {
                        "qubit": 0,
                    }
                },
                {
                    "r": {
                        "qubit": 0,
                        "theta": 0.5,
                        "phi": 0.5
                    }
                },
                {
                    "rz": {
                        "qubit": 0,
                        "phi": 0.5
                    }
                },
                {
                    "gr": {
                        "theta": 0.5,
                        "phi": 0.5
                    }
                },
                {
                    "cz": {
                        "control": 0,
                        "target": 1
                    }
                }
            ]
        }]
    }

    validate(circuit_instance, circuit_schema)

def test_gate_type_not_exist():
    circuit_instance = {
        "moments": [{
            "gates": [
                {
                    "not_exist": {
                        "qubit": 0,
                    }
                }
            ]
        }]
    }

    with pytest.raises(ValidationError) as exc:
        validate(circuit_instance, circuit_schema)

    assert "'id' is a required property" in exc.value.args[0]


def test_no_gates():
    circuit_instance = {
        "moments": []
    }
    with pytest.raises(ValidationError) as exc:
        validate(circuit_instance, circuit_schema)

    assert "is too short" in exc.value.args[0]

def test_too_many_gates():
    moment = {"gates": [{
                            "cz": {
                                "control": 0,
                                "target": 1
                            }
                        }
                      ]}

    circuit_instance = {
        "moments": [moment] * 101
    }

    with pytest.raises(ValidationError) as exc:
        validate(circuit_instance, circuit_schema)

    assert "is too long" in exc.value.args[0]