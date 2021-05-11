import random

import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import lib.model
from lib import determ, data, model
from lib.datasets.retinopathyv2a import RetinopathyV2a
from lib.experiment import DataParams, dataset_defaults, execute_experiment

SPLITS_NAMES = ["Train", "Validation", "Test"]
PROJECT_ROOT = '.'

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

base_model = lib.model.BaseModel.RESNET50_v2
n_layers = len(base_model.value[0]().layers)
max_train_layers = int(n_layers / 2)

dropout_options = [0.1, 0.2, 0.3, 0.4, 0.5]
tl_learning_rate_options = [1e-2, 1e-3, 1e-4]
fine_learning_rate_options = [1e-3, 1e-4, 1e-5, 1e-6]

# Attribute generator
toolbox.register("attr_dropout", lambda: random.sample(dropout_options, 1)[0])
toolbox.register("attr_train_layers", random.randint, 1, max_train_layers)
toolbox.register("attr_tl_learning_rate", lambda: random.sample(tl_learning_rate_options, 1)[0])
toolbox.register("attr_fine_learning_rate", lambda: random.sample(fine_learning_rate_options, 1)[0])

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual, [
    toolbox.attr_dropout,
    toolbox.attr_train_layers,
    toolbox.attr_tl_learning_rate,
    toolbox.attr_fine_learning_rate,
], n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval_one_max(individual):
    print(individual)
    dropout, train_layers, tl_learning_rate, fine_learning_rate = individual
    datasets, class_names, images_df = toolbox.data()

    model_params = model.ModelParams(
        base_model=model.BaseModel.RESNET50_v2,
        image_size=160,
        num_classes=len(class_names),
        dropout=dropout,
        global_pooling=model.GlobalPooling.AVG_POOLING,
        use_data_augmentation=True
    )

    training_params = model.TrainingParams(
        tl_learning_rate=tl_learning_rate,
        tl_epochs=10,
        fine_learning_rate=fine_learning_rate,
        fine_epochs=5,
        fine_layers=train_layers
    )

    metrics, reports, retinopathy_model = execute_experiment(datasets, class_names, model_params, training_params)
    fine_metrics = metrics['fine']
    print(metrics)
    return [fine_metrics['accuracy']]


toolbox.register("evaluate", eval_one_max)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    determ.set_global_determinism(42)

    data_params = DataParams(
        dataset=RetinopathyV2a.name,
        remap=RetinopathyV2a.mapping.c2.name,
        image_size=160,
        batch_size=32,
        splits=[0.7, 0.2, 0.1]
    )

    datasets, class_names, images_df = dataset_defaults(PROJECT_ROOT, data_params)
    summary_df = data.summarize_batched_datasets(datasets, SPLITS_NAMES, class_names)
    print(summary_df)

    toolbox.register("data", lambda: [datasets, class_names, images_df])
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=2,
                                   stats=stats, halloffame=hof, verbose=False)

    return pop, log, hof


if __name__ == "__main__":
    results = main()
