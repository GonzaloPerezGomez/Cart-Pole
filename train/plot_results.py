import matplotlib.pyplot as plt

def init_live_plot():
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_title("Entrenamiento en tiempo real")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Score")
    return fig, ax

def update_live_plot(ax, scores, window=50):
    ax.clear()
    ax.plot(scores, label="Score por episodio")
    if len(scores) >= window:
        avg = [sum(scores[max(0, i - window):i+1]) / (i - max(0, i - window) + 1) for i in range(len(scores))]
        ax.plot(avg, label=f"Media móvil ({window})", color="orange")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Score")
    ax.set_title("Progreso del entrenamiento")
    ax.legend()
    ax.grid(True)
    plt.pause(0.01)
    
    
def plot_training(scores, epsilons, losses=None, window=50):
    plt.figure(figsize=(12, 5))

    # Plot del puntaje por episodio
    plt.subplot(1, 3, 1)
    plt.plot(scores, label="Score por episodio")
    rolling_avg = [sum(scores[max(0, i - window):i+1]) / (i - max(0, i - window) + 1) for i in range(len(scores))]
    plt.plot(rolling_avg, label=f"Media móvil ({window})")
    plt.xlabel("Episodio")
    plt.ylabel("Puntaje")
    plt.legend()
    plt.grid()

    # Plot de epsilon
    plt.subplot(1, 3, 2)
    plt.plot(epsilons, color="orange")
    plt.xlabel("Episodio")
    plt.ylabel("Epsilon")
    plt.title("Exploración")
    plt.grid()

    # Plot de pérdidas (si se proporcionan)
    if losses:
        plt.subplot(1, 3, 3)
        plt.plot(losses, color="red")
        plt.xlabel("Episodio")
        plt.ylabel("Loss promedio")
        plt.title("Pérdida por episodio")
        plt.grid()

    plt.tight_layout()
    plt.show()
    plt.savefig("grafico.png")
