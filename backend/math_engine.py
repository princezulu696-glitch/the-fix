import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

VERSION = "3.1.0"

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

x, y, z, t, a, b, c, n = sp.symbols("x y z t a b c n")


# ============================================================
# DISPLAY FORMATTER
# ============================================================

SUPERSCRIPTS = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
}


def superscript(text):
    return "".join(SUPERSCRIPTS.get(ch, ch) for ch in str(text))


def format_expression(expr):
    """
    Convert SymPy expressions into readable student-friendly text.
    """

    if isinstance(expr, str):
        text = expr
    else:
        expr = sp.simplify(expr)

        # Exact integers
        if expr.is_Integer:
            return str(expr)

        # Exact rational numbers
        if expr.is_Rational:
            return f"{expr.p}/{expr.q}" if expr.q != 1 else str(expr.p)

        text = sp.sstr(expr)

    # Powers: x**2 -> x²
    pattern = r"([A-Za-z0-9_\)\]])\*\*(-?\d+)"

    def power_replace(match):
        base = match.group(1)
        exponent = match.group(2)

        if all(ch in SUPERSCRIPTS for ch in exponent):
            return base + superscript(exponent)

        return f"{base}^{exponent}"

    text = re.sub(pattern, power_replace, text)

    # Multiplication
        # Make multiplication easier to read
    text = text.replace(" × ", "*")

    # 3*x -> 3x
    text = re.sub(r"(\d+)\*([A-Za-z])", r"\1\2", text)

    # x*y -> xy
    text = re.sub(r"([A-Za-z])\*([A-Za-z])", r"\1\2", text)

    # x*2 -> 2x
    text = re.sub(r"([A-Za-z])\*(\d+)", r"\2\1", text)

    # Restore multiplication where it should remain visible
    text = text.replace("*", " × ")

    text = re.sub(r"\s+", " ", text).strip()

    # Make common functions look natural
    text = text.replace("sqrt", "√")

    return text


def format_number(value):
    """
    Prevent ugly outputs such as:
    37.0000000000000
    """

    value = sp.simplify(value)

    if value.is_Integer:
        return str(value)

    if value.is_Rational:
        return format_expression(value)

    if value.is_Float:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return f"{number:.10g}"

    return format_expression(value)


def format_solution(solution, variable=x):
    """
    Convert:
        [2, 3]

    into:
        x = 2
        x = 3
    """

    if isinstance(solution, list):
        if len(solution) == 0:
            return "No solution."

        if len(solution) == 1:
            return f"{variable} = {format_expression(solution[0])}"

        lines = []

        for value in solution:
            lines.append(
                f"{variable} = {format_expression(value)}"
            )

        return "\n".join(lines)

    return format_expression(solution)


# ============================================================
# PARSER
# ============================================================

def clean_input(text):
    text = text.strip()

    replacements = {
        "−": "-",
        "×": "*",
        "÷": "/",
        "π": "pi",
        "∞": "oo",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def parse_math_expression(text):
    text = clean_input(text)

    local_dict = {
        "x": x,
        "y": y,
        "z": z,
        "t": t,
        "a": a,
        "b": b,
        "c": c,
        "n": n,

        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "cot": sp.cot,
        "sec": sp.sec,
        "csc": sp.csc,

        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,

        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,

        "log": sp.log,
        "ln": sp.log,
        "exp": sp.exp,

        "sqrt": sp.sqrt,

        "pi": sp.pi,
        "E": sp.E,
        "e": sp.E,
        "oo": sp.oo,
    }

    return parse_expr(
        text,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


# ============================================================
# EQUATIONS
# ============================================================

def solve_equation(text):

    text = clean_input(text)

    text = re.sub(
        r"^\s*(solve|find|calculate)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    if "=" not in text:
        return None

    left, right = text.split("=", 1)

    left_expr = parse_math_expression(left)
    right_expr = parse_math_expression(right)

    equation = sp.Eq(left_expr, right_expr)

    symbols = equation.free_symbols

    if not symbols:
        return {
            "success": True,
            "type": "equation",
            "answer": "True" if sp.simplify(
                left_expr - right_expr
            ) == 0 else "False",
            "raw_answer": equation,
        }

    variable = sorted(
        symbols,
        key=lambda s: str(s)
    )[0]

    solutions = sp.solve(equation, variable)

    return {
        "success": True,
        "type": "equation",
        "answer": format_solution(solutions, variable),
        "raw_answer": solutions,
        "variable": str(variable),
    }


# ============================================================
# DERIVATIVE
# ============================================================

def solve_derivative(text):

    text = clean_input(text)

    text = re.sub(
        r"^\s*(differentiate|derivative|diff|derive)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    expr = parse_math_expression(text)

    variable = sorted(
        expr.free_symbols,
        key=lambda s: str(s)
    )[0] if expr.free_symbols else x

    result = sp.diff(expr, variable)

    return {
        "success": True,
        "type": "derivative",
        "answer": format_expression(result),
        "raw_answer": result,
        "variable": str(variable),
    }


# ============================================================
# INTEGRATION
# ============================================================

def solve_integral(text):

    text = clean_input(text)

    text = re.sub(
        r"^\s*(integrate|integral|∫)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Definite integral:
    # integrate x^2 from 0 to 2
    match = re.match(
        r"(.+?)\s+from\s+(.+?)\s+to\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        expr_text = match.group(1)
        lower_text = match.group(2)
        upper_text = match.group(3)

        expr = parse_math_expression(expr_text)
        lower = parse_math_expression(lower_text)
        upper = parse_math_expression(upper_text)

        variable = sorted(
            expr.free_symbols,
            key=lambda s: str(s)
        )[0] if expr.free_symbols else x

        result = sp.integrate(
            expr,
            (variable, lower, upper)
        )

        return {
            "success": True,
            "type": "definite_integral",
            "answer": format_expression(result),
            "raw_answer": result,
            "variable": str(variable),
        }

    expr = parse_math_expression(text)

    variable = sorted(
        expr.free_symbols,
        key=lambda s: str(s)
    )[0] if expr.free_symbols else x

    result = sp.integrate(expr, variable)

    return {
        "success": True,
        "type": "integral",
        "answer": format_expression(result),
        "raw_answer": result,
        "variable": str(variable),
    }


# ============================================================
# LIMIT
# ============================================================

def solve_limit(text):

    text = clean_input(text)

    text = re.sub(
        r"^\s*limit\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Supports:
    # x->0 sin(x)/x
    # x -> 0 sin(x)/x
    # sin(x)/x as x->0
    # sin(x)/x as x -> 0

    match = re.match(
        r"^\s*([A-Za-z]+)\s*[-=]>\s*(.+?)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        variable_text = match.group(1)
        point_text = match.group(2)
        expression_text = match.group(3)

    else:

        match = re.match(
            r"^\s*(.+?)\s+as\s+([A-Za-z]+)\s*[-=]>\s*(.+)$",
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            return {
                "success": False,
                "type": "limit",
                "answer": "Limit format not recognized.",
            }

        expression_text = match.group(1)
        variable_text = match.group(2)
        point_text = match.group(3)

    variable = sp.Symbol(variable_text)

    expr = parse_math_expression(expression_text)
    point = parse_math_expression(point_text)

    result = sp.limit(
        expr,
        variable,
        point
    )

    return {
        "success": True,
        "type": "limit",
        "answer": format_expression(result),
        "raw_answer": result,
        "variable": variable_text,
    }
# ============================================================
# FACTOR
# ============================================================

def solve_factor(text):

    text = re.sub(
        r"^\s*factor\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    expr = parse_math_expression(text)

    result = sp.factor(expr)

    return {
        "success": True,
        "type": "factor",
        "answer": format_expression(result),
        "raw_answer": result,
    }
# ============================================================
# EXPAND
# ============================================================

def solve_expand(text):

    text = re.sub(
        r"^\s*expand\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    expr = parse_math_expression(text)

    result = sp.expand(expr)

    return {
        "success": True,
        "type": "expand",
        "answer": format_expression(result),
        "raw_answer": result,
    }


# ============================================================
# SIMPLIFY
# ============================================================

def solve_simplify(text):

    text = re.sub(
        r"^\s*simplify\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    expr = parse_math_expression(text)

    result = sp.simplify(expr)

    return {
        "success": True,
        "type": "simplify",
        "answer": format_expression(result),
        "raw_answer": result,
    }


# ============================================================
# CALCULATION
# ============================================================

def solve_calculation(text):

    text = re.sub(
        r"^\s*(calculate|calc)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    expr = parse_math_expression(text)

    exact = sp.simplify(expr)

    return {
        "success": True,
        "type": "calculation",
        "answer": format_number(exact),
        "raw_answer": exact,
    }


# ============================================================
# MAIN ENGINE
# ============================================================

def solve_math(text):

    if not text or not text.strip():
        return {
            "success": False,
            "type": "error",
            "answer": "Please provide a mathematical problem.",
        }

    text = text.strip()

    try:

        # EQUATIONS
        if "=" in text and not re.search(
            r"(differentiate|integrate|factor|expand|simplify)",
            text,
            re.IGNORECASE,
        ):
            result = solve_equation(text)

            if result:
                return result

        # DERIVATIVE
        if re.match(
            r"^\s*(differentiate|derivative|diff|derive)\b",
            text,
            re.IGNORECASE,
        ):
            return solve_derivative(text)

        # INTEGRAL
        if re.match(
            r"^\s*(integrate|integral|∫)\b",
            text,
            re.IGNORECASE,
        ):
            return solve_integral(text)

        # LIMIT
        if re.match(
            r"^\s*limit\b",
            text,
            re.IGNORECASE,
        ):
            return solve_limit(text)

        # FACTOR
        if re.match(
            r"^\s*factor\b",
            text,
            re.IGNORECASE,
        ):
            return solve_factor(text)

        # EXPAND
        if re.match(
            r"^\s*expand\b",
            text,
            re.IGNORECASE,
        ):
            return solve_expand(text)

        # SIMPLIFY
        if re.match(
            r"^\s*simplify\b",
            text,
            re.IGNORECASE,
        ):
            return solve_simplify(text)

        # DEFAULT CALCULATION
        return solve_calculation(text)

    except Exception as e:

        return {
            "success": False,
            "type": "error",
            "answer": f"Could not solve this expression: {str(e)}",
            "error": str(e),
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("THE FIX - POWERFUL MATHEMATICS ENGINE")
    print(f"Engine: {VERSION}")
    print(f"SymPy: {sp.__version__}")
    print("=" * 60)

    tests = [
        "solve 2*x + 5 = 15",
        "solve x^2 - 5*x + 6 = 0",
        "differentiate x^3 + 2*x^2 - 5*x",
        "differentiate sin(x) + x^3",
        "integrate x^2",
        "integrate sin(x)",
        "integrate x^2 from 0 to 2",
        "factor x^2 - 9",
        "expand (x+2)*(x+3)",
        "simplify (x^2 - 9)/(x - 3)",
        "limit x->0 sin(x)/x",
        "calculate sin(pi/2)",
        "calculate log(100,10)",
        "2 + 5 * 10",
        "sqrt(144) + 5^2",
        "3^4",
    ]

    for test in tests:

        print()
        print(test)

        result = solve_math(test)

        print("SUCCESS:", result["success"])
        print("TYPE:", result["type"])
        print("ANSWER:", result["answer"])