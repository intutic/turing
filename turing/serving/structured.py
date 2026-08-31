"""
Structured Output & JSON Schema Validation Engine for Turing Engine.
Enforces valid JSON object generation, JSONSchema adherence, and auto-bracket repair.
"""

import json
import re
from typing import Dict, Any, Optional, Tuple, Union, List


class StructuredOutputParser:
    """
    Handles prompt schema injection, JSON parsing, validation, and truncation repair.
    """

    @staticmethod
    def inject_json_instruction(prompt: str, schema: Optional[Dict[str, Any]] = None, schema_name: Optional[str] = None) -> str:
        """
        Injects JSON formatting instructions or schema constraints into the prompt.
        """
        instruction_parts = []
        if schema:
            schema_str = json.dumps(schema, indent=2)
            name_str = f" for '{schema_name}'" if schema_name else ""
            instruction_parts.append(
                f"\n\nIMPORTANT: You must respond ONLY with a valid JSON object matching the following JSON Schema{name_str}:\n```json\n{schema_str}\n```\nDo not include any conversational filler, explanation, or markdown formatting outside the JSON object."
            )
        else:
            instruction_parts.append(
                "\n\nIMPORTANT: You must respond ONLY with a valid JSON object. Do not include markdown codeblocks or conversational text outside the JSON."
            )
        return prompt + "".join(instruction_parts)

    @staticmethod
    def extract_json(text: str) -> Tuple[bool, Optional[Any], str]:
        """
        Extracts JSON from raw text or markdown codeblocks.
        Returns (is_valid, parsed_obj, cleaned_text).
        """
        raw = text.strip()

        # Check for ```json ... ``` or ``` ... ``` code blocks
        json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        candidate = json_block_match.group(1).strip() if json_block_match else raw

        # Direct JSON parse attempt
        try:
            parsed = json.loads(candidate)
            return True, parsed, candidate
        except Exception:
            pass

        # Try to find outermost { ... } or [ ... ]
        first_brace = candidate.find("{")
        first_bracket = candidate.find("[")
        
        start_idx = -1
        if first_brace != -1 and first_bracket != -1:
            start_idx = min(first_brace, first_bracket)
        elif first_brace != -1:
            start_idx = first_brace
        elif first_bracket != -1:
            start_idx = first_bracket

        if start_idx != -1:
            sub = candidate[start_idx:]
            try:
                parsed = json.loads(sub)
                return True, parsed, sub
            except Exception:
                # Try auto-repairing truncated JSON
                repaired = StructuredOutputParser.repair_truncated_json(sub)
                try:
                    parsed = json.loads(repaired)
                    return True, parsed, repaired
                except Exception:
                    pass

        return False, None, candidate

    @staticmethod
    def repair_truncated_json(text: str) -> str:
        """
        Repairs truncated JSON strings caused by max_tokens limits by closing open quotes,
        brackets, and braces.
        """
        s = text.strip()
        if not s:
            return "{}"

        try:
            import turing.turing_csrc as turing_csrc
            res = turing_csrc.scan_json_structure_fast(s)
            suffix = res.get("repair_suffix", "")
            if suffix:
                s_repaired = re.sub(r",\s*$", "", s)
                s_repaired = re.sub(r":\s*$", ': null', s_repaired)
                return s_repaired + suffix
        except Exception:
            pass

        # State tracking
        in_string = False
        escape = False
        stack: List[str] = []

        for char in s:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if char in ("{", "["):
                stack.append(char)
            elif char == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif char == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

        # If inside an open string, close it
        if in_string:
            s += '"'

        # Remove trailing dangling commas or colons before closing
        s = re.sub(r",\s*$", "", s)
        s = re.sub(r":\s*$", ': null', s)

        # Close all remaining unclosed containers in reverse order
        while stack:
            opener = stack.pop()
            if opener == "{":
                s += "}"
            elif opener == "[":
                s += "]"

        return s

    @staticmethod
    def validate_schema(obj: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates a parsed JSON object against a JSON schema.
        Supports 'type', 'properties', 'required', and 'items'.
        """
        expected_type = schema.get("type")
        if expected_type == "object":
            if not isinstance(obj, dict):
                return False, f"Expected object, got {type(obj).__name__}"
            
            # Check required fields
            for req_key in schema.get("required", []):
                if req_key not in obj:
                    return False, f"Missing required property '{req_key}'"
            
            # Check property types
            properties = schema.get("properties", {})
            for key, prop_schema in properties.items():
                if key in obj:
                    val = obj[key]
                    val_valid, val_err = StructuredOutputParser.validate_schema(val, prop_schema)
                    if not val_valid:
                        return False, f"Invalid property '{key}': {val_err}"

        elif expected_type == "array":
            if not isinstance(obj, list):
                return False, f"Expected array, got {type(obj).__name__}"
            item_schema = schema.get("items")
            if item_schema:
                for idx, item in enumerate(obj):
                    item_valid, item_err = StructuredOutputParser.validate_schema(item, item_schema)
                    if not item_valid:
                        return False, f"Invalid item at index {idx}: {item_err}"

        elif expected_type == "string":
            if not isinstance(obj, str):
                return False, f"Expected string, got {type(obj).__name__}"

        elif expected_type == "number":
            if not isinstance(obj, (int, float)):
                return False, f"Expected number, got {type(obj).__name__}"

        elif expected_type == "integer":
            if not isinstance(obj, int) or isinstance(obj, bool):
                return False, f"Expected integer, got {type(obj).__name__}"

        elif expected_type == "boolean":
            if not isinstance(obj, bool):
                return False, f"Expected boolean, got {type(obj).__name__}"

        return True, None
