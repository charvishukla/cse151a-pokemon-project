# cse151a-pokemon-project
Building Machine learning models around Pokemon card prices and attributes
## Milstone 2 Todos
* Setup Notebook for data exploration (similar to prof's dataset notebooks)
* Figure out what features and observations to drop
* Data cleaning (strings)
* Think of data encoding schemes
* Distribute roles/jobs
* FINALIZE FEATURES FOR DATASET
* Updates on merging


## Dataset Retrieval
Our data was scraped using the [Price charting API](https://www.pricecharting.com/api-documentation). Initially, we had $60296$ observations with $27$ features. 


We have the following features in `price-guide.csv` file:
1. `id`                     
2. `console-name`        
3. `product-name`          
4. `loose-price`           
5. `graded-price`          
6. `box-only-price`        
7. `manual-only-price`     
8. `bgs-10-price`          
9. `condition-17-price`   
10. `condition-18-price`    
11. `retail-loose-buy`      
12. `retail-loose-sell`     
13. `sales-volume`           
14. `tcg-id`                
15. `release-date`          

We also have image data (?)