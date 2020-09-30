import optuna
from functools import partial
from train import objective, run_training

DEVICE = "cuda"
EPOCHS = 100


if __name__ == "__main__":

    partial_obj = partial(objective)
    study = optuna.create_study(direction="minimize")
    study.optimize(partial_obj, n_trials=150)

    print("Best trial:")
    trial_ = study.best_trial

    print(f"Value: {trial_.value}")
    print("Params: ")
    best_params = trial_.params
    print(best_params)

    scores = 0
    for j in range(5):
        score = run_training(fold=j, params=best_params, save_model=True)
        scores += score
    print(f"OOF Score {scores/5}")
