# CSE151A Pokemon Card Project
Building a machine learning model around Pokemon card prices and rarity

# Introduction

The Pokemon card market has experienced a dramatic surge in popularity, especially following the COVID-19 pandemic. What started as a children’s trading card game has transformed into a high-stakes collectors’ market, where rare and legendary cards can sell for hundreds of thousands of dollars. The rise of professional grading companies, which certify card quality and condition, has further fueled the demand, creating a competitive marketplace with unprecedented prices. 

This topic was chosen because of the group’s shared passion for Pokemon and the need for reliable, data-driven tools in this rapidly growing, multi-million-dollar industry. Pokemon universally appeals to generations of people, making this an exciting and relatable subject, as even enthusiasts like Professor Solares would agree (we think). While traditional methods—such as relying on anecdotal evidence, historical data, or recent transactions of a similar card and quality—can provide some guidance, they often result in inefficiencies and inaccuracies, leading to undervalued or overpriced cards. As the market continues to expand, a more sophisticated and structured approach to evaluating card rarity and market value is essential.

To address this need, this study incorporates several advanced machine learning techniques to predict the rarity of Pokemon cards based on features such as grade quality, statistics (i.e. Health Points, Evolution state, Move Damage), and visual features (i.e. regular, holographic, reverse holographic). Predictive models include random forest classifiers and neural networks, designed to identify relationships between card characteristics, rarity, and pricing trends. This study also explores broader market trends from 2021 onward, using methods like linear regression and neural networks to identify key factors influencing Pokemon card prices.

This topic is particularly cool because it combines a popular hobby and machine learning techniques. Pokemon card collecting has evolved into a sophisticated and lucrative global market, and by enabling accurate predictions of card rarity and value, this research enhances the collecting experience while promoting fair pricing and informed decision-making. Data-driven predictions have the potential to stabilize and bring more transparency to the Pokemon card marketplace.
The broader impacts of this research extend beyond the Pokemon card market, providing a foundation for improving valuation in other collectible industries. By replacing reliance on anecdotal evidence and word-of-mouth pricing with machine learning models, this study ensures greater transparency and efficiency in high-value markets. As these models evolve, they may soon become integral to future pricing techniques, influencing buyers and sellers with reliable data to make informed decisions. This approach ensures market stability and promotes a sustainable, equitable framework for collectibles industries worldwide. 


# Methods
![](imgs/compare.png)

# Results
![](imgs/ms4-training-loss.png)

# Discussion

# Conclusion
In the beginning, we started with a dataset containing about 30300 observations and 56. This dataset resulted from merging two separate datasets, of which one was scraped using an API. Throughout this project, we were only able to utilize a small subset of this dataset’s features as many features proved irrelevant after processing. We built a Random Forest and a simple Neural Network to predict the rarity of a Pokemon Card, based on a combination of categorical and numerical variables. The results from our Random Forest and Neural Network was as follows:

![](imgs/ms3-learning-curve.png)
![](imgs/ms4-trainingacc.png)

The models of our choice — although powerful and simple — were only able to handle a small subset of our features, given the amount of data. Thus, in the future, to tackle such complexity, we aim to work with Ensemble Models. These models would employ a combination of Machine Learning algorithms instead of just one, leading to greater predictive power and generalizability.

There are many different possible routes we wanted our project to extend to. We realized in the time that we were allotted for this project that we would only be able to predict Pokemon Card rarity based on our selected features. In the future, we would like to create models that can accomplish other tasks within the Pokemon Card market space. This includes models that could predict the price of a card or the possible grade a card may receive based on a provided image of said card. We also worked a lot with models focused on classification for this project. Though classification was productive in our case, there is a case that can be made that choosing a different model for finance related data could help improve our accuracies further. However, we were able to build a high accuracy model with classification for the task we aimed for.

We are definitely not the first group of people to ever build a functional machine learning model to predict characteristics of a Pokemon Card. Many researchers and students have created successful models that accurately predict factors such as price and rarity. In fact, research has shown that there are some models that are more accurate at grading cards than trained professionals hired at grading companies. Despite this, the adoption of machine learning in the Pokemon Card market is still something that is heavily debated. Sellers and buyers in the market prefer the opinion of authorized graders over machine learning models that are created by researchers. Even if the grading of a card is more subjective overall with a human due to bias, many individuals still choose to accept the opinion of the elite rather than algorithms trained with statistical data. As a result, machine learning may need more of an introduction to the community as a tool for our benefit. As models become more accurate and further research is done on Pokemon Cards, we believe the legitimacy of machine learning in the market will be extremely beneficial. Therefore, the research we have done serves a purpose greater than just obtaining a grade for the class. It provides a means of objectivity in a market where everyone is trying to prove the worth of their own Pokemon Card collections.

# Statement of Collaboration

Charvi Shukla:  
Pedro Arias: 
David Kim:  
Kenneth Song: Team member/writer: Introduction writeup portion, organized writeup figures to use, set up meetings, double checked work before submissions, communicated often with group
George Li:
Anthony Georgis: 
