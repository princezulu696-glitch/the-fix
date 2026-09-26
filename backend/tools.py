import ast
import operator
import re

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate(expression: str):
    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)
        return result
    except Exception:
        raise ValueError("Invalid mathematical expression.")


def evaluate(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        if type(node.op) not in OPERATORS:
            raise ValueError("Operation not allowed.")

        left = evaluate(node.left)
        right = evaluate(node.right)

        return OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in OPERATORS:
            raise ValueError("Operation not allowed.")

        value = evaluate(node.operand)

        return OPERATORS[type(node.op)](value)

    raise ValueError("Unsupported expression.")


def extract_math_expression(message: str):
    text = message.lower().strip()

    # Remove common question wording
    prefixes = [
        "what is ",
        "what's ",
        "calculate ",
        "solve ",
        "compute ",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    # Remove question marks and periods
    text = text.rstrip("?.!")

    # Convert common multiplication symbols
    text = text.replace("×", "*")
    text = text.replace("÷", "/")

    # Remove spaces
    text = text.replace(" ", "")

    # Allow only mathematical characters
    if re.fullmatch(r"[0-9+\-*/%().]+", text):
        if re.search(r"\d", text):
            return text

    return None


def use_calculator(message: str):
    expression = extract_math_expression(message)

    if expression is None:
        return None

    try:
        result = calculate(expression)

        return {
            "tool": "calculator",
            "expression": expression,
            "result": result
        }

    except ValueError:
        return None