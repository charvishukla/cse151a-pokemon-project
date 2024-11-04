# cse151a-pokemon-project
Building Machine learning models around Pokemon card prices and attributes

## Dataset Retrieval
Our data was scraped using the [Price charting API](https://www.pricecharting.com/api-documentation). Initially, we had $60296$ observations with $27$ features. The second dataset we used was the [Pokemon Cards dataset from Kaggle](https://www.kaggle.com/datasets/adampq/pokemon-tcg-all-cards-1999-2023/data). We combined both of these datasets, and the resulting dataset has $30300$ observations with $56$ features. We dropped columns with a really high number of null values, and also irrelevant colunns that we thought were un-informative to our study. The remaining features are listed below. The code for the same can be found in our `cse151a_pokemon_project_exploration.ipynb` file

## Dataset download and Environmment set-up instructions:
**Dataset Download** -
1. The Pokemon Cards Data can be downloaded form Kaggle by following this link: https://www.kaggle.com/datasets/adampq/pokemon-tcg-all-cards-1999-2023/data
2. The Price Charting Data was extracted using a paid API, which is stored in a private google drive found here: https://ucsdcloud-my.sharepoint.com/:f:/g/personal/cshukla_ucsd_edu/EiksPRBMWu1AsIEoCJaYNPcBaycdRswLyPCJO9FwLywatQ?e=5TOFP8

For this project, please download and add these `.csv` files in the root directory of the project. 

**Environment Set-up**:
We have used the following Python Libraries:
- `pandas`
- `seaborn`
- `numpy`
- `scipy`
- `matplotlib`

This repository can be cloned using the following command:

```
https://github.com/charvishukla/cse151a-pokemon-project.git
```
---

## Feature Description

The code can either be run locally using Jupyter Notebook (`cd cse151a-pokemon-project; jupyter-notebook`) or can be uploaded and run on Google Collab. 

We have the following features in `price-guide.csv` file:

1. `id` - The unique identifier for the specific Pokémon Card labeled by Price Charting       
2. `console-name` - The name of the collection (set) that the Pokémon Card was released in
3. `product-name` - The name of the Pokémon Card
4. `loose-price` - The price of the Pokémon Card without any grade
5. `graded-price` - The price of the Pokémon Card graded at a 9
6. `box-only-price` - The price of the Pokémon Card graded at a 9.5
7. `manual-only-price` - The price of the Pokémon Card graded at a 10 by PSA
8. `bgs-10-price` - The price of the Pokémon Card graded at a 10 by BGS
9. `condition-17-price` - The price of the card when graded at a 10 by CGC
10. `condition-18-price` - The price of the card when graded at a 10 by SGC
11. `retail-loose-buy` - Price recommended by Price Charting to buy the Pokémon Card at without any grade
12. `retail-loose-sell` - Price recommended by Price Charting to sell the Pokémon Card at without any grade   
13. `sales-volume` - Yearly units sold
14. `tcg-id` - ID of the Pokémon Card in its specific set
15. `release-date` - Date of when the Pokémon Card was released
16. `tcg_id` - Pokémon TCG API English Card ID
17. `set` - card sequence num in the current set
18. `series` - card's series name
19. `publisher` - card's publisher (For example, WOTC = Wizards of the Coast, TPCI = The Pokémon Company International)
20. `generation` - card's generation (numerical equivalent of series)
21. `release_date` - card release date
22. `artist` - The card's artist 
23. `name` - The name of the Pokémon Card (according to the TCG dataset)
24. `set_num` - card sequence num in the current set
25. `types` - card type(s) (Example: colorless, psychic, lightning, metal etc)
26. `supertype` - card supertype (Pokémon, Trainer, Energy)
27. `subtypes` - ard's subtype(s) 
28. `hp` -  card's hit points
29. `weaknesses` - card's weaknesses (i.e. what type is our current pokemon the weakest against)
30. `rarity` - The rarity of the Pokemon Card (not pokemon)
31. `legalities` - The legalities of the pokemon card
32. `resistances`-  The resistances listed on the Pokemon Card 

---
## Milestone 3 TODO List:

- **Renaming Columns**
  - For the pre-processing step, we aim to re0name the features to be more intuitive and in-line with what we are testing while keeping them true to their original values.

- **Z-score Normalization and Outlier detection**
  - We have the following standard deviation in the prices present in our dataset at the moment. This high standard deviation is a result of extreme outliers which are visible in the pair-plots plotted in our EDA Jupyter notebook. 
  ```
  loose-price             79.443088
  graded-price           323.930300
  box-only-price         486.569103
  manual-only-price     3465.324301
  bgs-10-price          5542.460171
  sales-volume           118.630244
  condition-17-price    1133.810455
  condition-18-price    2079.091298
  hp                      52.895938
  dtype: float64
  ```
  - The summary statistics indicate that the standard deviations across multiple columns are extremely high relative to their means, suggesting a wide spread of values within each variable. For instance, `loose-price` has a standard deviation of 869.98, while its mean is only 36.69. Similarly, `graded-price`, `box-only-price`, and `manual-only-price` all display substantial variability with standard deviations much larger than their respective means.
  - This high variability indicates the presence of outliers. This can also be seen from our pairplots in the previous sections where there are some scatter-points really away from the clusters. Therefore, for the pre-processing stage, we will apply z-score normalization and drop extreme values (i.e. when the z-score is greater than 3 for some value). This will allow us to reveal the underlying shape of the distribution. 

- **Categorical Variable Encoding**
  - For categorical data such as `artist`, `type`, `super type`, `publisher`,  `console-name`, etc. we will be using encoding techniques such as one-hot encoding and oridnal encoding to be able to use these variables in our ML models. 
  
