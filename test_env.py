import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    values = np.linspace(0, 2 * np.pi, 100)
    sine = np.sin(values)

    print(f"numpy version: {np.__version__}")
    print(f"matplotlib version: {matplotlib.__version__}")
    print(f"sample mean: {np.mean(sine):.6f}")

    fig, ax = plt.subplots()
    ax.plot(values, sine)
    ax.set_title("Sine Wave")
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    fig.savefig("test_plot.png")
    print("saved plot: test_plot.png")


if __name__ == "__main__":
    main()
