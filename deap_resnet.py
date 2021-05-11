import os
import pickle
import random

import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import lib.model
from evo_experiment import all_equal, last_checkpoint, checkpoint_file, restore_datalog, save_datalog
from lib import determ, data, model
from lib.datasets.retinopathyv2a import RetinopathyV2a
from lib.experiment import DataParams, dataset_defaults, execute_experiment

SPLITS_NAMES = ["Train", "Validation", "Test"]
PROJECT_ROOT = '.'
EXPERIMENT_ROOT = 'deap_resnet'
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

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
    dropout, train_layers, tl_learning_rate, fine_learning_rate = individual
    datalog = toolbox.datalog()
    datasets, class_names, images_df = toolbox.data()

    key = "RESNET50v2-" + "-".join(list(map(lambda x: str(x), individual)))
    if key in datalog:
        return [datalog[key]['fine']['accuracy']]

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

    metrics, reports, retinopathy_model = execute_experiment(datasets,
                                                             class_names,
                                                             model_params,
                                                             training_params,
                                                             verbose=0)

    datalog[key] = metrics
    fine_metrics = metrics['fine']
    return [fine_metrics['accuracy']]


toolbox.register("evaluate", eval_one_max)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def register_data(run_dir):
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
    summary_df.to_csv(os.path.join(run_dir, "data_summary.csv"))
    toolbox.register("data", lambda: [datasets, class_names, images_df])


def restore_checkpoint(checkpoint_dir, npop):
    checkpoint_file = last_checkpoint(checkpoint_dir)
    population = toolbox.population(n=npop)
    start_gen = 0
    halloffame = tools.HallOfFame(maxsize=5)
    logbook = tools.Logbook()

    if checkpoint_file is None:
        return population, start_gen, halloffame, logbook
    else:
        with open(checkpoint_file, "rb") as cp_file:
            print(f"Restore checkpoint {checkpoint_file}")
            cp = pickle.load(cp_file)
            population = cp['population']
            start_gen = cp['generation'] + 1
            halloffame = cp['halloffame']
            logbook = cp['logbook']
            random.setstate(cp['rndstate'])
            return population, start_gen, halloffame, logbook


def register_stats():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    return stats


DATALOG = restore_datalog(EXPERIMENT_ROOT)


def main(run_number, ngen, npop, datalog, patience=5, cxpb=0.75, mutpb=0.05, checkfreq=1):
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

    for gen in range(start_gen, ngen + 1):
        if gen > patience and all_equal([entry["max"] for entry in logbook[-patience:]]):
            print(f"Converged max fitness {max}")
            break

        print(f"Gen {gen}", flush=True)
        population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        population = toolbox.select(population, k=len(population))

        if gen % checkfreq == 0:
            save_datalog(EXPERIMENT_ROOT, datalog)
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())

            with open(checkpoint_file(run_dir, gen), "wb") as cp_file:
                pickle.dump(cp, cp_file)
    best = tools.selBest(population, k=10)

    print(logbook)

    for ind in best:
        print(f"{ind}, fitness: {ind.fitness}")
    return population, logbook, halloffame


if __name__ == "__main__":
    results = main(run_number=1, ngen=3, npop=2, datalog=DATALOG)
