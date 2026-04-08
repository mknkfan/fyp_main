import pandas as pd
import matplotlib.pyplot as plt

data = [
    (100, 100, 31.4895, 37.54),
    (100, 200, 30.4775, 72.07),
    (100, 500, 29.0107, 173.18),
    (100, 1000, 28.7898, 341.59),
    (100, 2000, 27.7439, 680.18),
    (100, 5000, 27.0394, 1688.11),

    (200, 100, 34.9338, 72.34),
    (200, 200, 31.4429, 141.09),
    (200, 500, 31.0543, 351.95),
    (200, 1000, 31.0387, 694.41),
    (200, 2000, 28.1241, 1383.70),
    (200, 5000, 25.0459, 3455.24),

    (500, 100, 32.1223, 201.17),
    (500, 200, 29.2483, 394.91),
    (500, 500, 29.1074, 973.11),
    (500, 1000, 29.1007, 1951.31),
    (500, 2000, 29.0947, 3877.30),
    (500, 5000, 28.4799, 9596.48),

    (1000, 100, 30.0171, 475.63),
    (1000, 200, 28.7917, 949.76),
    (1000, 500, 28.6363, 2355.97),
    (1000, 1000, 28.0491, 4685.99),
]

df = pd.DataFrame(data, columns=["pop", "gen", "fitness", "time"])

plt.figure(figsize=(9,6))
sc = plt.scatter(
    df["time"],
    df["fitness"],
    c=df["pop"],
    s=df["gen"] / 8,
    cmap="viridis",
    alpha=0.85
)

plt.xscale("log")
plt.xlabel("Execution Time (s, log scale)")
plt.ylabel("Best Fitness")
plt.title("Fitness vs Computation Time (Population & Generation Scaling)")
plt.colorbar(sc, label="Population Size")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
