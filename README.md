# A Food Recommender System for reducing the water footprint 🌎

### An health and planet aware recommender system for reducing water footprint of users' diet

This repository contains a recommender system that takes into account water footprint to suggest recipes to users. 
It provides a command line utility to explore the recommender system from terminal, and also, provides a Streamlit application to explore and configure all the possibility inserted into the recommender system. 
Before running the system it is possible to configure it via the application, you can choose around a content based or a collaborative filtering algorithm, the number of recommendations and the choice to activate or deactivate the water footprint filter. 

- The full article is available [here](https://www.mdpi.com/2071-1050/14/7/3833).
- Data can be downloaded [here](https://www.kaggle.com/turconiandrea/water-footprint-recommender-system-data).

## Setup
1. Clone the application in a local folder

2. Download the dataset from the following links 
   * embbedding (ready to use): [embedding-data folder](https://www.kaggle.com/turconiandrea/water-footprint-recommender-system-data)

3. Paste the downloaded folder into ` data/ ` folder
4. Choose the configuration file: under the ` configuration folder `, it is possible to choose from which data run the system (Planeat.eco or Food.com). In order to change the data it is necessary to rename the selected file as ` config.json `. From the ` config.json ` file the system will gather all the data it needs without any further configuration. 

## Execution 
* In order to run the streamlit web application:
```bash
streamlit run app.py
```
* In order to run the application from command line (it is necesary the user id)
```bash
python main.py --user-id 52543 --algo cb
```
Arguments:
* user-id: the id of the user.
* algo: the algorithm type (cf or cb)
* no-filter-wf: to run the recommendation disabling the water footprint filter

## Citation
```
  @Article{su14073833,
      AUTHOR = {Gallo, Ignazio and Landro, Nicola and La Grassa, Riccardo and Turconi, Andrea},
      TITLE = {Food Recommendations for Reducing Water Footprint},
      JOURNAL = {Sustainability},
      VOLUME = {14},
      YEAR = {2022},
      NUMBER = {7},
      ARTICLE-NUMBER = {3833},
      URL = {https://www.mdpi.com/2071-1050/14/7/3833},
      ISSN = {2071-1050},
      ABSTRACT = {Most existing food-related research efforts focus on recipe retrieval, user preference-based food recommendation, kitchen assistance, or nutritional and caloric estimation of dishes, ignoring personalized and conscious food recommendations resources of the planet. Therefore, in this work, we present a personalized food recommendation scheme, mapping the ingredients to the most resource-friendly dishes on the planet and in particular, selecting recipes that contain ingredients that consume as little water as possible for their production. The system proposed here is able to understand the user&rsquo;s behavior and to suggest tailor-made recipes with lower water quantity used in production. By continuously using the system, the user can gradually reduce their water footprint and benefit from a healthier diet. The proposed recommendation system was compared with the results of two papers available in the literature that represent the state of the art, obtaining similar results. Therefore, the results of the presented recommendation system can be considered reliable.},
      DOI = {10.3390/su14073833}
}
```

## References
[1] - [The paper: Food Recommendations for Reducing Water Footprint](https://www.mdpi.com/2071-1050/14/7/3833)
