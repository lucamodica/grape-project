import itertools
import matplotlib.pyplot as plt

def plot_results(results_dict, num_epochs=10):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.flatten()
    epochs = range(num_epochs)

    # Define a color cycle to assign consistent colors per model
    color_cycle = itertools.cycle(
        plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Dictionary to store the color for each model
    model_colors = {}

    # Plot training and validation losses in the first subplot
    axs[0].set_title('Training Loss')
    for model_name, result in results_dict.items():
        # Assign a color to the model if it hasn't been assigned yet
        if model_name not in model_colors:
            model_colors[model_name] = next(color_cycle)

        color = model_colors[model_name]
        axs[0].plot(epochs, result['train_losses'],
                    label=f'{model_name} (train)', color=color)
        axs[0].plot(epochs, result['val_losses'],
                    label=f'{model_name} (val)', color=color, linestyle='dashed')

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_xticks(ticks=epochs, labels=[str(i+1) for i in epochs])
    axs[0].legend()

    # Now plot the metrics in the remaining 3 subplots
    metric_names = list(next(iter(results_dict.values()))[
                        'train_metrics_history'].keys())  # Get metric names from the first model

    for i, metric_name in enumerate(metric_names):
        axs[i+1].set_title(metric_name)
        for model_name, result in results_dict.items():
            # Use the same color as for the loss plot
            color = model_colors[model_name]
            axs[i+1].plot(epochs, result['train_metrics_history']
                          [metric_name], label=f'{model_name} (train)', color=color)
            axs[i+1].plot(epochs, result['val_metrics_history'][metric_name],
                          label=f'{model_name} (val)', color=color, linestyle='dashed')

        axs[i+1].set_xlabel('Epoch')
        axs[i+1].set_ylabel(metric_name)
        axs[i+1].set_xticks(ticks=epochs, labels=[str(i+1) for i in epochs])
        axs[i+1].legend()

    plt.tight_layout()
    plt.show()


def plot_losses_and_f1(results_dict, num_epochs=10):
    fig, axs = plt.subplots(1, 2, figsize=(11, 7))
    epochs = range(num_epochs)

    # Define a color cycle to assign consistent colors per model
    color_cycle = itertools.cycle(
        plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Dictionary to store the color for each model
    model_colors = {}

    # Plot training and validation losses in the first subplot
    axs[0].set_title('Training Loss')
    for model_name, result in results_dict.items():
        # Assign a color to the model if it hasn't been assigned yet
        if model_name not in model_colors:
            model_colors[model_name] = next(color_cycle)

        color = model_colors[model_name]
        axs[0].plot(epochs, result['train_losses'],
                    label=f'{model_name} (train)', color=color)
        axs[0].plot(epochs, result['val_losses'],
                    label=f'{model_name} (val)', color=color, linestyle='dashed')

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_xticks(ticks=epochs, labels=[str(i+1) for i in epochs])
    axs[0].legend()

    # Plot F1 scores in the second subplot
    axs[1].set_title('F1 Score')
    for model_name, result in results_dict.items():
        # Use the same color as for the loss plot
        color = model_colors[model_name]
        axs[1].plot(epochs, result['train_metrics_history']['f1'],
                    label=f'{model_name} (train)', color=color)
        axs[1].plot(epochs, result['val_metrics_history']['f1'],
                    label=f'{model_name} (val)', color=color, linestyle='dashed')

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('F1 Score')
    axs[1].set_xticks(ticks=epochs, labels=[str(i+1) for i in epochs])
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def plot_f1(results_dict):
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    epochs = range(10)

    # Define a color cycle to assign consistent colors per model
    color_cycle = itertools.cycle(
        plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Dictionary to store the color for each model
    model_colors = {}

    # Plot F1 scores in the second subplot
    axs.set_title('F1 Score')
    for model_name, result in results_dict.items():
        # make model name capitalized
        model_name = model_name.upper()
        # Assign a color to the model if it hasn't been assigned yet
        if model_name not in model_colors:
            model_colors[model_name] = next(color_cycle)

        color = model_colors[model_name]
        axs.plot(epochs, result['val_metrics_history']['f1'],
                 label=f'{model_name} (val)', color=color, linestyle='dashed')
        axs.plot(epochs, result['train_metrics_history']['f1'],
                 label=f'{model_name} (val)', color=color)

    axs.set_xlabel('Epoch')
    axs.set_ylabel('F1 Score')
    axs.set_xticks(ticks=epochs, labels=[str(i+1) for i in epochs])
    axs.legend()

    plt.tight_layout()
    plt.show()
