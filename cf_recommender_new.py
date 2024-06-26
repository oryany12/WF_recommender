"""
Module containing the core system of the collaborative filtering recommendations system.
"""

import os
import random
import numpy as np
import pandas as pd
from surprise import (
    dump,
    SVD,
    BaselineOnly,
    CoClustering,
    Dataset,
    KNNWithMeans,
    KNNBaseline,
    Reader,
    KNNBasic,
    KNNWithZScore,
    accuracy,
)
from surprise.model_selection import cross_validate, train_test_split
from tqdm import tqdm

from configuration import load_configuration
from water_footprint_utils import WaterFootprintUtils


class CFRecommender_new:
    """
    Class that represents the collaborative filtering algorithm.
    The datasets are the ones provided onto the configuration file.
    The algorithm used is a KNN Baseline from Surprise toolkit
    fine-tuned with different parameters.
    This class also provides a benchmark that compares 7 different
    algorithms and an evaluation method for the chosen algorithm.

    :param orders: the dataset containing the user reviews.
    :param recipes: the dataset containing the recipes.
    :param n_recommendations: the number of recommendations to be returned.
    :param disable_filter_wf: a bool representing the possibility to
        turn off water footprint search.
    """

    def __init__(
        self, orders=None, recipes=None, n_recommendations=10, disable_filter_wf=False,
            norm=None, alpha=1, beta=1
    ):
        """
        Constructor method for the class.
        If param orders is not provided it will be taken the default one from config.
        If param recipes is not provided it will be taken the default one from config.
        """
        config = load_configuration()
        self.orders = (
            orders if orders is not None else pd.read_pickle(config["path_orders"])
        )
        self.recipes = (
            recipes if recipes is not None else pd.read_pickle(config["path_recipes"])
        )
        self.n_recommendations = n_recommendations
        self.disable_filter_wf = disable_filter_wf
        self.model_path = config["path_cf_model"]
        self.reader = Reader(rating_scale=(0, 5))

        self.norm = norm
        self.alpha = alpha
        self.beta = beta

    def get_data(self):
        """
        Get the data for the collaborative filtering algorithm in a compatible form
        for the surprise toolkit.

        :return: the dataset composed by orders and reader.
        """
        return Dataset.load_from_df(self.orders, self.reader)

    def get_benchmark(self, verbose=True):
        """
        Return collaborative filtering benchmark with 7 different algorithm on the
        provided data. Results are ranked and sorted by evaluating the test RSME of
        all the algorithm on cross validation.

        :param verbose: possibility to print the results on the console. Default is true.
        :return: a dataframe containing the benchmark results.
        """
        benchmark = []
        data = self.get_data()
        algorithms = [
            BaselineOnly(verbose=False),
            SVD(n_epochs=5, verbose=False, lr_all=0.01, reg_all=0.06),
            KNNBasic(n_epochs=10, lr_all=0.010, n_factors=90, verbose=False),
            KNNWithMeans(sim_options={"name": "cosine", "user_based": "False", "min_k": 2}, verbose=False),
            KNNWithZScore(sim_options={"name": "msd", "user_based": "False", "min_support": 5}, verbose=False),
            KNNBaseline(sim_options={"name": "msd", "user_based": "False", "min_support": 5}, verbose=False),
            CoClustering(verbose=False)
        ]
        for algorithm in tqdm(algorithms, desc="Computing benchmark"):
            results = cross_validate(
                algorithm, data, measures=["RMSE"], cv=3, verbose=False
            )
            tmp = pd.DataFrame.from_dict(results).mean(axis=0)
            tmp = tmp.append(
                pd.Series(
                    [str(algorithm).split(" ")[0].split(".")[-1]], index=["Algorithm"]
                )
            )
            benchmark.append(tmp)

        results = (
            pd.DataFrame(benchmark).set_index("Algorithm").sort_values("test_rmse")
        )
        if verbose:
            print(results)
        return results, algorithms

    def save_cf_model(self, algo):
        """
        Save the collaborative filtering model provided as a pickle file in the
        directory provided onto the configuration file.

        :param algo: the model that must be saved.
        :return: a boolean indication if the model is saved successfully or not.
        """
        print(">> Saving the model <<")
        dump.dump(self.model_path, algo=algo)
        return os.path.exists(self.model_path)

    def load_cf_model(self):
        """
        Load the collaborative filtering model saved in the directory provided
        into the configuration file.

        :return: the collaborative filtering model.
        """
        print(">> Loading the model <<")
        _, model = dump.load(self.model_path)
        return model

    def get_algorithm(self):
        """
        Instantiate the collaborative filtering algorithm used with
        fine-tuned parameters. Algorithm used is kNN Baseline with
        Mean Squared Difference for measuring similarity, item-item
        similarity and a minimum number of users.

        :return: a KNNBaseline algorithm.
        """
        options = {"name": "pearson_baseline", "user_based": "False", "min_support": 5}
        return KNNBaseline(sim_options=options, verbose=False)

    def create_cf_model(self, save=False):
        """
        Create the collaborative filtering model from the data provided, model
        is trained on train set and validated on test set. Test size is 25%.
        The algorithm used is the one provided by the previous method.

        :param save: the possibility to directly save the model.
        :return: the created collaborative filtering model.
        """
        print(">> Creating the model <<")
        data = self.get_data()
        algo = self.get_algorithm()
        train = data.build_full_trainset()
        algo.fit(train)
        if save:
            self.save_cf_model(algo)
        return algo

    def get_model_evaluation(self, test_size=0.25, algo=None):
        """
        Compute the evaluation for the model in terms of RMSE.
        Algorithm used is the one provided from the method above
        on the data. Model is evaluated on test set.
        The default test data is 25% percent of all data.

        :param test_size: percentage of test size. Default is 0.25.
        :return: the RMSE of the model.
        """
        data = self.get_data()
        train, test = train_test_split(data, test_size=test_size)
        algo = self.get_algorithm() if algo is None else algo
        algo = algo.fit(train)
        predictions = algo.test(test)
        return accuracy.rmse(predictions, verbose=False)

    def __get_recipe_from_id(self, recipe_id):
        """
        Return the recipe row from the dataframe based on the recipe id.

        :param recipe_id: the id of the recipe.
        :return: a dataframe row containing the recipe at the provided id.
        """
        return self.recipes.query(f"id == {recipe_id}")[["name", "wf", "category"]]

    def __get_prediction(self, user_id, recipe_id, model):
        """
        Return a tuple composed by the id of the recipe and the
        predictions based on the id of the user.

        :param user_id: the id of the user.
        :param recipe_id: the id of the recipe.
        :param model: the model used to recommend.
        :return: a tuple composed by id of recipe and rating (from 0 to 5)
            (raw item id, rating estimation)
        """
        prediction = model.predict(uid=user_id, iid=recipe_id)
        return recipe_id, prediction.est

    def __get_recipes_average_wf(self, recipes):
        """
        Return the average water footprint based on the recipes
        id provided in input. 

        :recipes: a list containg the recipes id.
        :return: an int indicating the average water footprint of the recipes
        """
        summon = 0
        for id in recipes:
            ls = self.recipes.query(f"id == {id}")["wf"].to_list()
            summon = summon + ls[0] if ls else summon
        return summon / len(recipes)

    def __get_recipes_sum_wf(self, recipes):
        """
        Return the sum water footprint based on the recipes'
        id provided in input. 

        :recipes: a list containing the recipes' id.
        :return: an int indicating the average water footprint of the recipes
        """
        summon = 0
        for id in recipes:
            ls = self.recipes.query(f"id == {id}")["wf"].to_list()
            summon = summon + ls[0] if ls else summon
        return summon

    def get_cf_hit_ratio_and_wf(self, model=None, filter_wf=False):
        """
        Computes the HitRatio@10 score of the collaborative
        filtering algorithm on all users present into the dataset.
        If the test is in the top 10 element recommended to the
        user, it is considered as a hit. The HitRatio is the
        difference between all hits and all users.

        :return: the HitRatio@10 score.
        """
        wf = WaterFootprintUtils()
        model = model if model is not None else self.load_cf_model()
        users = self.orders["user_id"].unique()
        hit = 0
        wfs = []

        wf = self.recipes[['id','wf']]
        # convert wf to dictionary
        wf = wf.set_index('id').T.to_dict('records')[0]
        all_recipes = self.orders["id"].unique()
        for user_id in tqdm(users):
            user_orders = self.orders.query(f"user_id == {user_id}")
            test_recipe = user_orders.tail(1)["id"].tolist()[0]
            user_recipes = user_orders["id"].unique()
            random_recipes = random.sample(list(set(all_recipes) - set(user_recipes)), 99)
            random_recipes.append(test_recipe)
            recommendations = [self.__get_prediction(user_id, recipe_id, model) for recipe_id in random_recipes]
            recipes_wf = np.array([wf.get(recipe_id,1000) for recipe_id, _ in recommendations])
            if self.norm is None:
                recipes_wf_norm = recipes_wf
            elif self.norm == 'inverse':
                recipes_wf_norm = (1 / recipes_wf)
            elif self.norm == 'minmax':
                recipes_wf_norm = (1 - ((recipes_wf - recipes_wf.min()) / (recipes_wf.max() - recipes_wf.min())))
            elif self.norm == 'zscore':
                recipes_wf_norm = (-1 * ((recipes_wf - recipes_wf.mean()) / recipes_wf.std()))

            #add wf_score to the recommendations tuples
            recommendations = [(recipe_id, rating, wf) for (recipe_id, rating), wf in zip(recommendations, recipes_wf_norm)]
            recommendations = sorted(recommendations, key=lambda tup: self.alpha*tup[1] + self.beta*tup[2], reverse=True)
            recommendations = [recipe_id for recipe_id, _, _ in recommendations]
            recommendations = recommendations[:10]
            wfs.append(self.__get_recipes_sum_wf(recommendations))
            hit = hit + 1 if test_recipe in recommendations else hit + 0
        wfs = sum(wfs)/len(wfs)

        return round(hit/len(users), 2), wfs/10

    def get_user_recommendations(self, user_id, model=None):
        """
        Get the best n recommendations for the provided user id.
        If the water footprint is not disable it filter the best
        recommendations in order to lower to user water consumptions.

        :param user_id: the id of the user that needs recommendations.
        :param model: the collaborative filtering model.
        :return: a dataframe containing the recommendations for the user.
        """
        wf = WaterFootprintUtils()
        model = model if model is not None else self.load_cf_model()
        recipes = self.orders["id"].unique()
        orders = self.orders.query(f"user_id == {user_id}")["id"].tolist()
        not_orders = list(set(recipes) - set(orders))
        recommendations = [self.__get_prediction(user_id, recipe_id, model) for recipe_id in not_orders]
        recommendations = sorted(recommendations, key=lambda tup: tup[1])
        recommendations = [recipe_id for recipe_id, _ in recommendations][:500]
        recommendations = (
            wf.get_recommendations_correct(recommendations, user_id, "cf")
            if not self.disable_filter_wf
            else recommendations
        )
        return pd.concat(
            [self.__get_recipe_from_id(recipe_id) for recipe_id in recommendations]
        ).head(self.n_recommendations)


if __name__ == "__main__":
    rec = CFRecommender_new()
    mod = rec.create_cf_model(save=False)
    print(rec.get_cf_hit_ratio_and_wf(model=mod, filter_wf=False))
    print(rec.get_cf_hit_ratio_and_wf(model=mod, filter_wf=True))
    

