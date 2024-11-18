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
# Milestone 3:

(Note: all preprocessing code and model code can be found in our github repository: https://github.com/charvishukla/cse151a-pokemon-project/blob/Milestone3/Milestone3-Preprocessing-Model1.ipynb)

## Preprocessing:

We began Milestone 3 by finishing the major preprocessing we had yet to do from milestone 2.
We did the following:
- Dropped null rows from the merged dataset from MS2
- Dropping categories with very few observations for certian categories:
  - We used a threshold to fix the minimum number of observations each category needs to have. We used the following theresholds:
       - `types`: 1000     
       - `rarity`: 1000  
       - `generation`:1500
- One-Hot Encoding for categorical variables `type` and `generation`


---
## Model 1: Random Forest
In this model, we are using a **Random Forest Classifier** to predict the **rarity** of Pokemon cards based on a combination of numerical as well as categorical features.

#### Features :
1. **Categorical Features**:
   - `types`: Represents the type/category of the item.
   - `generation`: Refers to the generation or version of the item.

2. **Numerical Features**:
   - `bgs-10-price`: The graded price of the item in mint condition.
   - `graded-price`: The general graded price of the item.
   - `hp`: Represents the item's hit points (a measure of power or health).
   - `sales-volume`: Indicates the volume of sales for the item.

3.  Target Variable :
- **rarity**: The classification label indicating how rare the item is.
 
### Steps 
#### Address class imbalance using `SMOTE (Synthetic Minority Oversampling Technique)`

#### Cross Validation using `K-fold` from `sklearn`:
- Split the data into 5 folds for training and validation.
- We train and evaluate model on different subsets of the data to prevent overfitting
- Following are the training accuracies for each fold:
``` [0.77840344 0.77475985 0.76507621 0.75878065 0.77269715]```

The trend in the training accuracy across folds can be seen below:




## Training vs Testing Error:

Our training accuracy was at 99% while our testing accuracy was at 67%. This leaves us with a gap of about 32% between the two sets. This can indicate that our model is overfitting based off of our training data, and we may need to adjust our training and testing splits as well as consider adding a validation set to ensure that we have an accurate accuracy for both of our data sets. The training set had near perfect precision and recall which definitely does indicate overfitting and we will aim to address this in future models.


## Testing Set Evaluation:

The accuracy of our testing set was 66.97% indicating that our model was accurately able to predict a card's rarity based on the price of the card at about 67% accuracy. For the common cards, we had a balanced performance with high precision and recall at 83% and 85% respectively. For our rare and uncommon cards, the model struggled a bit more with precision and recall for rare cards being 54% and 52% respectively and for uncommon cards being 57% and 58%. Since our recall and precision are lower for these card rarities, this can indicate that there might be a class imbalance or that the features we chose are not relevant enough. We believe that it might not be because of class imbalance though because we chose to resample the data before fitting it into our model.



## Interpreting The Fitting Graph:

After plotting our model predictions for our training and testing splits, we can see that there is a large gap between the testing curve and training curve. The training curve does not move while our testing curve has an upward trend as our training set size increases. Due to this large gap, this definitely means that we have a problem with overfitting our data.

## Next Model Considerations:

For the next models we want to consider, we want to create more classification models for our data in the form of logistic regression, and we'd also like to use a linear regression model to predict the price based on our features as well. Before we start on these next models though, we would like to adjust some of the data for our random forest model as we believe we can prevent overfitting by adjusting the training and testing splits while also adding a validation set. We would like to revisit this random forest model and improve upon it as we continue to create new models for our data.

## Conclusion

For our first model, we have concluded that we are overfitting our model based on the results we have gathered. With an accuracy of 67% on our testing predictions, we are happy with the result as it is our first model. However, we cannot take this as necessarily the correct accuracy because of our overfitting problem and we will explore this in the future. We will improve this model by introducing a validation set and adjusting the splits between the sets as well. We would also like to take a look at the features to ensure we are including relevant features that will help us have a higher accuracy with our predictions. We may also take a look at class imbalance and improve our issue with it further by using class weights.
