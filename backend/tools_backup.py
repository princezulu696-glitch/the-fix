import ast
import operator


# Allowed mathematical operations
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
    """
    Safely calculate a mathematical expression.

    Examples:
        10 + 5
        20 * 3
        100 / 4
        2 ** 8
        (10 + 5) * 2
    """

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)
        return result

    except Exception:
        raise ValueError("Invalid mathematical expression.")


def evaluate(node):

    # Numbers
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    # Binary operations
    if isinstance(node, ast.BinOp):

        if type(node.op) not in OPERATORS:
            raise ValueError("This mathematical operation is not allowed.")

        left = evaluate(node.left)
        right = evaluate(node.right)

        return OPERATORS[type(node.op)](left, right)

    # Positive/negative numbers
    if isinstance(node, ast.UnaryOp):

        if type(node.op) not in OPERATORS:
            raise ValueError("This operation is not allowed.")

        value = evaluate(node.operand)

        return OPERATORS[type(node.op)](value)

    raise ValueError("Unsupported expression.")