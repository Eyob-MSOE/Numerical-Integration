import numpy as np
import sympy as sp

def safe_eval(f, x_val):
    """Safely evaluate function f at x_val, avoiding division by zero."""
    try:
        result = f(x_val)
        if isinstance(result, complex):  # Extract real part if complex
            result = result.real
        if np.isinf(result) or np.isnan(result):
            return np.nan  # Return NaN instead of 0
        return float(result)
    except (ValueError, OverflowError, ZeroDivisionError, TypeError):
        return np.nan  # Return NaN instead of 0

def trapezoidal_rule(f, a, b, n):
    """Compute the integral of f from a to b using the trapezoidal rule with n subintervals."""
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = np.array([safe_eval(f, val) for val in x])
    return (h / 2) * (y[0] + 2 * np.sum(y[1:n]) + y[n])

def midpoint_rule(f, a, b, n):
    """Compute the integral of f from a to b using the midpoint rule with n subintervals."""
    h = (b - a) / n
    midpoints = np.linspace(a + h/2, b - h/2, n)
    y = np.array([safe_eval(f, val) for val in midpoints])
    return h * np.sum(y)

def simpsons_rule(f, a, b, n):
    """Compute the integral of f from a to b using Simpson's rule with n subintervals."""
    if n % 2 == 1:  # Simpson's rule requires even n
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = np.array([safe_eval(f, val) for val in x])
    return (h / 3) * (y[0] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]) + y[n])

def get_float_input(prompt):
    """Get a numeric input from the user, handling special cases for pi and e, and evaluating expressions."""
    while True:
        try:
            value = input(prompt).strip().lower()
            if value == "pi":
                return np.pi
            elif value == "e":
                return np.e
            else:
                # Evaluate the expression using sympy
                expr = sp.sympify(value)
                return float(expr.evalf())
        except (ValueError, sp.SympifyError):
            print("❌ Invalid input! Please enter a valid number or expression (e.g., 'pi/2').")

def get_int_input(prompt):
    """Get an integer input from the user."""
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("❌ Invalid input! Please enter an integer.")

def compute_max_derivative(f_expr, x, a, b, order):
    """Compute the maximum of the absolute value of the nth derivative of f_expr on [a, b]."""
    derivative = sp.diff(f_expr, x, order)  # Compute the nth derivative
    f_deriv = sp.lambdify(x, derivative, "numpy")  # Convert to numerical function
    x_vals = np.linspace(a, b, 1000)  # Sample points in the interval
    try:
        values = f_deriv(x_vals)
        if np.any(np.iscomplex(values)):  # Ensure only real values
            values = np.real(values)
        max_value = np.nanmax(np.abs(values))
        if np.isnan(max_value):
            raise ValueError("Derivative computation returned NaN")
    except Exception as e:
        print(f"⚠️ Warning: Unable to compute max |f^{order}(x)| reliably. Error: {e}")
        max_value = np.nan  # Return NaN to indicate failure
    return max_value

def compute_required_n(f_expr, x, a, b, tolerance=1e-4):
    """Compute the required number of subintervals n for each method to achieve the desired tolerance."""
    K2 = compute_max_derivative(f_expr, x, a, b, 2)  # Max |f''(x)|
    K4 = compute_max_derivative(f_expr, x, a, b, 4)  # Max |f''''(x)|

    # Compute required "n" for each method separately
    n_trap = 1
    while ((b - a) ** 3 / (12 * n_trap**2)) * K2 >= tolerance:
        n_trap += 1

    n_mid = 1
    while ((b - a) ** 3 / (24 * n_mid**2)) * K2 >= tolerance:
        n_mid += 1

    n_simp = 1
    while ((b - a) ** 5 / (180 * n_simp**4)) * K4 >= tolerance:
        n_simp += 1
        if n_simp % 2 == 1:  # Ensure even n for Simpson’s rule
            n_simp += 1

    return n_trap, n_mid, n_simp

def main():
    """Main program loop."""
    while True:
        # Get user input for function and interval
        func_str = input("\nEnter function of x (e.g., x**2 + 2*x + 1, sin(pi*x)): ")
        a = get_float_input("Enter the start of interval (a): ")
        b = get_float_input("Enter the end of interval (b): ")
        n = get_int_input("Enter the number of subintervals (n): ")

        # Get user input for decimal places
        decimal_places_approx = get_int_input("Enter the number of decimal places for approximations: ")
        decimal_places_error = get_int_input("Enter the number of decimal places for error bounds: ")

        # Convert user input function to a callable function
        x = sp.symbols('x')
        func_str = func_str.replace("^", "**").replace("pi", str(np.pi)).replace("e^", "exp")  # Fix pi, e and ^
        func_expr = sp.sympify(func_str, evaluate=True)  # Convert string to symbolic expression
        func_expr = sp.simplify(func_expr)  # Simplify expression to avoid issues
        func = sp.lambdify(x, func_expr, modules=["numpy"])

        # Compute exact integral using SymPy
        exact_integral = float(sp.integrate(func_expr, (x, a, b)))

        # Compute approximations
        trap = trapezoidal_rule(func, a, b, n)
        mid = midpoint_rule(func, a, b, n)
        simp = simpsons_rule(func, a, b, n)

        # Compute absolute errors
        if exact_integral is not None and not np.isnan(exact_integral):
            trap_error = exact_integral - trap
            mid_error = exact_integral - mid
            simp_error = exact_integral - simp
        else:
            trap_error = mid_error = simp_error = np.nan

        # Compute theoretical error bounds
        K2 = compute_max_derivative(func_expr, x, a, b, 2)  # Max |f''(x)|
        K4 = compute_max_derivative(func_expr, x, a, b, 4)  # Max |f''''(x)|

        # Ensure derivatives are valid before computing error bounds
        if np.isnan(K2) or np.isinf(K2):
            K2 = 1  # Default fallback value

        if np.isnan(K4) or np.isinf(K4):
            K4 = 1  # Default fallback value

        trap_bound = ((b - a) ** 3 / (12 * n**2)) * K2
        mid_bound = ((b - a) ** 3 / (24 * n**2)) * K2
        simp_bound = ((b - a) ** 5 / (180 * n**4)) * K4

        # Compute the required "n" for each method
        n_trap, n_mid, n_simp = compute_required_n(func_expr, x, a, b)

        # Display results
        print("\nUser-Entered Subintervals (n):", n)
        print("\nApproximations:")
        print(f"Trapezoidal Rule: {trap:.{decimal_places_approx}f}")
        print(f"Midpoint Rule: {mid:.{decimal_places_approx}f}")
        print(f"Simpson’s Rule: {simp:.{decimal_places_approx}f}")

        # Display errors
        print("\nActual Errors (Absolute Difference from Exact Integral):")
        print(f"Trapezoidal Rule Error: {trap_error:.{decimal_places_approx}f}")
        print(f"Midpoint Rule Error: {mid_error:.{decimal_places_approx}f}")
        print(f"Simpson’s Rule Error: {simp_error:.{decimal_places_approx}f}")

        # Display error bounds
        print("\nTheoretical Error Bounds:")
        print(f"Trapezoidal Rule Error Bound: {trap_bound:.{decimal_places_error}f}")
        print(f"Midpoint Rule Error Bound: {mid_bound:.{decimal_places_error}f}")
        print(f"Simpson’s Rule Error Bound: {simp_bound:.{decimal_places_error}f}")

        # Compare user's n with required n
        print("\n🔹 Recommended n for accuracy within 0.00001:")
        print(f"Trapezoidal Rule: {n_trap}")
        print(f"Midpoint Rule: {n_mid}")
        print(f"Simpson’s Rule: {n_simp}")

        if n >= n_trap:
            print("✅ Your chosen n is sufficient for Trapezoidal Rule.")
        else:
            print(f"⚠️ Increase n to at least {n_trap} for Trapezoidal Rule.")

        if n >= n_mid:
            print("✅ Your chosen n is sufficient for Midpoint Rule.")
        else:
            print(f"⚠️ Increase n to at least {n_mid} for Midpoint Rule.")

        if n >= n_simp:
            print("✅ Your chosen n is sufficient for Simpson’s Rule.")
        else:
            print(f"⚠️ Increase n to at least {n_simp} for Simpson’s Rule.")

        # Ask if user wants another round
        another = input("\nWould you like to calculate again? (yes/no): ").strip().lower()
        if another != 'yes':
            print("Exiting program. Goodbye!")
            break

# Run the program
if __name__ == "__main__":
    main()
