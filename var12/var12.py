import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
x, T0, q, c, L = sp.symbols('x T0 q c L')
T = T0 + q * sp.sin(sp.pi * x / L) + c * x * (L - x)
T_prime = sp.diff(T, x)
T_double_prime = sp.diff(T_prime, x)
print("Первая производная T'(x):", T_prime)
print("Вторая производная T''(x):", T_double_prime)
T_at_0 = T.subs(x, 0)
T_at_L = T.subs(x, L)
print("\nT(0) =", T_at_0)
print("T(L) =", T_at_L)

print("Условие T(0) == T0 выполняется:", sp.simplify(T_at_0 - T0) == 0)
print("Условие T(L) == T0 выполняется:", sp.simplify(T_at_L - T0) == 0)
print("\n" + "="*40 + "\n")

T0_val = 300
q_val = 50
c_val = 10
L_val = 2
kappa = 1

x_vals = np.linspace(0, L_val, 400)

T_vals = T0_val + q_val * np.sin(np.pi * x_vals / L_val) + c_val * x_vals * (L_val - x_vals)

T_prime_vals = c_val * (L_val - 2 * x_vals) + (np.pi * q_val / L_val) * np.cos(np.pi * x_vals / L_val)

J_vals = -kappa * T_prime_vals

print(f"Расчет завершен. Количество точек: {len(x_vals)}")
print(f"Максимальная температура: {np.max(T_vals):.2f}")
print(f"Минимальный тепловой поток: {np.min(J_vals):.2f}")
print(f"Максимальный тепловой поток: {np.max(J_vals):.2f}")
print("\n" + "="*40 + "\n")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))


ax1.plot(x_vals, T_vals, color='red', label='T(x)')
ax1.set_title('Распределение температуры T(x)')
ax1.set_xlabel('x')
ax1.set_ylabel('T')
ax1.grid(True)
ax1.legend()

ax2.plot(x_vals, J_vals, color='blue', label='J(x)')
ax2.set_title('Плотность теплового потока J(x)')
ax2.set_xlabel('x')
ax2.set_ylabel('J')
ax2.grid(True)
ax2.legend()


plt.tight_layout()
plt.show()