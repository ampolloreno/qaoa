qubit_id = {
    "type": "integer",
    "minimum": 0,
    "maximum": 1000
}

rotation_angle = {
    "type": "number"
}

gate_cz = {
    "type": "object",
    "properties": {
        "cz": {
            "type": "object",
            "properties": {
                "qubit_a": qubit_id,
                "qubit_b": qubit_id
            }
        }
    },
    "required": ["cz"]
}


gate_r = {
    "type": "object",
    "properties": {
        "r": {
            "type": "object",
            "properties": {
                "qubit": qubit_id,
                "theta": rotation_angle,
                "phi": rotation_angle
            },
            "required": ["qubit", "theta", "phi"]
        }
    },
    "required": ["r"]
}


gate_rz = {
    "type": "object",
    "properties": {
        "rz": {
            "type": "object",
            "properties": {
                "qubit": qubit_id,
                "phi": rotation_angle
            },
            "required": ["qubit", "phi"]
        }
    },
    "required": ["rz"]
}


gate_gr = {
    "type": "object",
    "properties": {
        "gr": {
            "type": "object",
            "properties": {
                "theta": rotation_angle,
                "phi": rotation_angle
            },
            "required": ["theta", "phi"]
        }
    },
    "required": ["gr"]
}

gate_id = {
    "type": "object",
    "properties": {
        "id": {
            "type": "object",
            "properties": {
                "qubit": qubit_id
            },
            "required": ["qubit"]
        }
    },
    "required": ["id"]
}

gate = {
    "oneOf": [
        gate_id,
        gate_cz,
        gate_r,
        gate_rz,
        gate_gr
    ]
}

moment= {
    "type": "object",
    "properties": {
        "gates": {
            "type": "array",
            "items": gate,
            "minItems": 1,
            "maxItems": 1000
        }
    },
    "required": ["gates"]
}

circuit_schema = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "items": moment,
            "minItems": 1,
            "maxItems": 100
        }
    },
    "required": ["moments"]
}