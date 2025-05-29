# Cart-Pole

```bash
cartpole-rl/
│
├── agent/                     # Código del agente RL
│   ├── model.py               # Red neuronal (por ejemplo, una MLP)
│   ├── dqn_agent.py           # Agente DQN u otra variante
│   └── replay_buffer.py       # Memoria de experiencias
│
├── train/                     # Entrenamiento y evaluación
│   ├── train.py               # Script de entrenamiento
│   ├── evaluate.py            # Script para evaluar el agente
│   └── plot_results.py        # Script para visualizar recompensas, etc.
│
├── tests/                     # Pruebas unitarias
│   ├── test_agent.py
│   ├── test_model.py
│   └── test_env.py
│
├── notebooks/                 # Notebooks para pruebas y visualización
│   └── cartpole_analysis.ipynb
│
├── config/                    # Configuraciones en YAML o JSON
│   └── dqn_config.yaml
│
├── logs/                      # Registros de entrenamiento, tensorboard, etc.
│   └── tensorboard/
│
├── saved_models/              # Modelos entrenados
│   └── dqn_cartpole.pth
│
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Descripción general del proyecto
└── .gitignore                 # Ignorar carpetas como logs, modelos, etc.

```
