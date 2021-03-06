import os
import pickle
import random

import numpy
import tensorflow as tf
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import lib.model
from evo_experiment import last_checkpoint, checkpoint_file, restore_datalog, save_datalog
from lib import determ, data, model
from lib.datasets.retinopathyv3 import RetinopathyV3
from lib.experiment import DataParams, dataset_defaults, execute_experiment, stop_criteria

SPLITS_NAMES = ["Train", "Validation", "Test"]
PROJECT_ROOT = '.'

DATASET = RetinopathyV3.name
DATASET_REMAP = RetinopathyV3.mapping.c3.name
BASE_MODEL = lib.model.BaseModel.RESNET50_v2

EXPERIMENT_ROOT = f'deap-{DATASET}-{DATASET_REMAP}-{BASE_MODEL}'

os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

n_layers = len(BASE_MODEL.value[0]().layers)
max_train_layers = int(n_layers / 2)

dropout_options = [0.1, 0.2, 0.3, 0.4, 0.5]
train_layers_options = range(1, max_train_layers, 1)
tl_learning_rate_options = [1e-2, 1e-3, 1e-4]
fine_learning_rate_options = [1e-3, 1e-4, 1e-5, 1e-6]

# Attribute generator
toolbox.register("attr_dropout", lambda: random.sample(dropout_options, 1)[0])
toolbox.register("attr_train_layers", lambda: random.sample(train_layers_options, 1)[0])
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
    dropout, train_layers, tl_learning_rate, fine_learning_rate = individual
    datalog = toolbox.datalog()
    datasets, class_names, images_df = toolbox.data()

    key = f"{EXPERIMENT_ROOT}-" + "-".join(list(map(lambda x: str(x), individual)))
    if key in datalog:
        return [datalog[key]['fine']['accuracy']]

    model_params = model.ModelParams(
        base_model=BASE_MODEL,
        image_size=160,
        num_classes=len(class_names),
        dropout=dropout,
        global_pooling=model.GlobalPooling.AVG_POOLING,
        use_data_augmentation=True
    )

    training_params = model.TrainingParams(
        tl_learning_rate=tl_learning_rate,
        tl_epochs=20,
        fine_learning_rate=fine_learning_rate,
        fine_epochs=10,
        fine_layers=train_layers
    )
    print(key)
    metrics, reports, retinopathy_model = execute_experiment(datasets,
                                                             class_names,
                                                             model_params,
                                                             training_params,
                                                             verbose=2)
    print(metrics)
    datalog[key] = metrics
    fine_metrics = metrics['fine']
    tf.keras.backend.clear_session()
    save_datalog(EXPERIMENT_ROOT, datalog)
    return [fine_metrics['accuracy']]


toolbox.register("evaluate", eval_one_max)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def register_data(run_dir):
    determ.set_global_determinism(42)

    data_params = DataParams(
        dataset=DATASET,
        remap=DATASET_REMAP,
        image_size=160,
        batch_size=32,
        splits=[0.7, 0.2, 0.1]
    )

    datasets, class_names, images_df = dataset_defaults(PROJECT_ROOT, data_params)
    summary_df = data.summarize_batched_datasets(datasets, SPLITS_NAMES, class_names)
    summary_df.to_csv(os.path.join(run_dir, "data_summary.csv"))
    toolbox.register("data", lambda: [datasets, class_names, images_df])


def start_population(npop):
    dropout, train_layers, tl_learning_rate, fine_learning_rate = [0.2, 30, 0.0001, 0.00001]
    population = toolbox.population(n=npop - 1)
    default_params = [dropout, train_layers, tl_learning_rate, fine_learning_rate]
    population.insert(0, creator.Individual(default_params))

    return population


def restore_checkpoint(checkpoint_dir, npop):
    checkpoint_path = last_checkpoint(checkpoint_dir)
    population = start_population(npop)
    start_gen = 0
    halloffame = tools.HallOfFame(maxsize=5)
    logbook = tools.Logbook()

    try:
        with open(checkpoint_path, "rb") as cp_file:
            print(f"Restore checkpoint {checkpoint_path}")
            cp = pickle.load(cp_file)
            population = cp['population']
            start_gen = cp['generation'] + 1
            halloffame = cp['halloffame']
            logbook = cp['logbook']
            random.setstate(cp['rndstate'])
            return population, start_gen, halloffame, logbook
    except:
        return population, start_gen, halloffame, logbook


def register_stats():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    return stats


DATALOG = restore_datalog(EXPERIMENT_ROOT)


def main(run_number, ngen, npop, datalog, patience=10, cxpb=0.75, start_mutpb=0.5, end_mutpb=0.05, checkfreq=1):
    run_dir = os.path.join(EXPERIMENT_ROOT, f"run-{run_number:03d}")
    os.makedirs(run_dir, exist_ok=True)

    population, start_gen, halloffame, logbook = restore_checkpoint(run_dir, npop=npop)

    stats = register_stats()

    if start_gen > ngen:
        print(f"Start gen {start_gen}> ngen{ngen}")
        return

    register_data(run_dir)
    toolbox.register("datalog", lambda: datalog)

    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for generation in range(start_gen, ngen + 1):
        mutpb = start_mutpb
        if generation > 0:
            mutpb = (start_mutpb - end_mutpb) / generation

        if stop_criteria([entry["max"] for entry in logbook], patience):
            max_fitness = logbook[-1]["max"]
            print(f"Converged max fitness {max_fitness}")
            break

        print(f"Generation {generation}, mutpb={mutpb}", flush=True)
        population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=generation, evals=len(invalid_ind), **record)

        population = toolbox.select(population, k=len(population))

        if generation % checkfreq == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=generation, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())

            with open(checkpoint_file(run_dir, generation), "wb") as cp_file:
                pickle.dump(cp, cp_file)
    best = tools.selBest(population, k=10)

    print(logbook)

    for ind in best:
        print(f"{ind}, fitness: {ind.fitness}")
    return population, logbook, halloffame


if __name__ == "__main__":
    for run in range(1, 6):
        results = main(run_number=run, ngen=30, npop=100, datalog=DATALOG)
