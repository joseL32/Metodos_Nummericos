from flask import Flask, render_template, request, jsonify
import sympy as sp
import numpy as np

app = Flask(__name__)

# Función para evaluar
def evaluate_function(func_str, x):
    x_sym = sp.Symbol('x')
    func = sp.sympify(func_str)
    func_lambdified = sp.lambdify(x_sym, func, 'numpy')
    return func_lambdified(x)

# Métodos numéricos

def bisection_method(func_str, a, b, tol):
    func = lambda x: evaluate_function(func_str, x)
    iterations = []
    if func(a) * func(b) >= 0:
        return "El método de bisección no es aplicable. Verifica los límites a y b.", []

    c = a
    while (b - a) / 2.0 > tol:
        c = (a + b) / 2.0
        iterations.append({"a": a, "b": b, "c": c, "f(c)": func(c)})
        if func(c) == 0.0:
            break
        elif func(a) * func(c) < 0:
            b = c
        else:
            a = c

    iterations.append({"a": a, "b": b, "c": c, "f(c)": func(c)})
    return c, iterations

def false_position_method(func_str, a, b, tol):
    func = lambda x: evaluate_function(func_str, x)
    iterations = []
    if func(a) * func(b) >= 0:
        return "El método de falsa posición no es aplicable. Verifica los límites a y b.", []

    c = a
    while abs(func(c)) > tol:
        c = (a * func(b) - b * func(a)) / (func(b) - func(a))
        iterations.append({"a": a, "b": b, "c": c, "f(c)": func(c)})
        if func(c) == 0.0:
            break
        elif func(a) * func(c) < 0:
            b = c
        else:
            a = c

    iterations.append({"a": a, "b": b, "c": c, "f(c)": func(c)})
    return c, iterations

def secant_method(func_str, x0, x1, tol):
    func = lambda x: evaluate_function(func_str, x)
    iterations = []
    while abs(x1 - x0) > tol:
        x2 = x1 - func(x1) * (x1 - x0) / (func(x1) - func(x0))
        iterations.append({"x0": x0, "x1": x1, "x2": x2, "f(x2)": func(x2)})
        x0, x1 = x1, x2
        if func(x2) == 0.0:
            break

    iterations.append({"x0": x0, "x1": x1, "x2": x2, "f(x2)": func(x2)})
    return x2, iterations

def newton_raphson_method(func_str, x0, tol):
    x = sp.Symbol('x')
    func = sp.sympify(func_str)
    func_prime = sp.diff(func, x)
    func_lambdified = sp.lambdify(x, func, 'numpy')
    func_prime_lambdified = sp.lambdify(x, func_prime, 'numpy')
    iterations = []
    while True:
        x1 = x0 - func_lambdified(x0) / func_prime_lambdified(x0)
        iterations.append({"x0": x0, "x1": x1, "f(x1)": func_lambdified(x1)})
        if abs(x1 - x0) < tol:
            break
        x0 = x1

    return x1, iterations

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.json
    method = data.get("method")
    func_str = data.get("func")
    tol = float(data.get("tol"))

    if method == "bisection":
        a = float(data.get("a"))
        b = float(data.get("b"))
        result, iterations = bisection_method(func_str, a, b, tol)
    elif method == "false_position":
        a = float(data.get("a"))
        b = float(data.get("b"))
        result, iterations = false_position_method(func_str, a, b, tol)
    elif method == "secant":
        x0 = float(data.get("x0"))
        x1 = float(data.get("x1"))
        result, iterations = secant_method(func_str, x0, x1, tol)
    elif method == "newton_raphson":
        x0 = float(data.get("x0"))
        result, iterations = newton_raphson_method(func_str, x0, tol)
    else:
        result = "Método no válido."
        iterations = []

    return jsonify({"result": result, "iterations": iterations})

if __name__ == "__main__":
    app.run(debug=True)
