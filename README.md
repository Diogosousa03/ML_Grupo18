Repeat Purchase Prediction

Project to predict whether a customer will make another purchase (repeat purchase), starting from a customer table and a table of purchases.

Project goal

Build a model that predicts, for each customer, whether they will buy again. The work is organised in four steps:

Data ingestion and exploratory data analysis (EDA): inspect the customer table and understand its quality and structure.
Data cleaning: handle the problems found in the EDA (empty variables, missing values, identifier-like variables, rare categories).
Feature evaluation: study the relationship between the customer features and repeat purchase and select the features for the model.
Modelling: train and evaluate the model.
Data
File	Description
clientes.csv	Customer table: 1143 customers and 13 variables (identification, location, company name and customer_since_year).
train.csv	Purchases, linked to the customer table through customer_id (columns used so far: customer_id, purchase_date).

The data files are expected in a data/ folder, as the notebook reads them from ../../data/. They are not included in this repository. <!-- TODO: say where the data comes from, or remove this sentence -->

Customer table (clientes.csv)
Variable	Description
customer_id	Unique customer identifier (join key)
country_or_region, commercial_region, island, municipality, city, postal_area, postal_code, street, address	Location of the customer, from country down to address
customer_type	Empty in the whole table
company_name	Name of the customer's company
customer_since_year	Year the customer joined (2015–2026)
Notebooks
Notebook	Content
1_DataIngestion_InitialInspection.ipynb	Data ingestion, duplicates, data quality (missing values and inconsistencies), descriptive statistics, univariate and bivariate analysis, summary of findings and next steps
Main findings of the EDA
Data quality: customer_type is completely empty. The geographic variables have many missing values (street 52.3%, island 50.6%, city 47.8%), and they are mostly structural: island, municipality and postal_area are only filled for CT10 customers.
Identifier-like variables: company_name, address, street and postal_code are (almost) unique per customer, so they do not work as categories.
Imbalance and high cardinality: 94% of the customers are from CT10 and 50% of the customers with a commercial region are in RG17. city, postal_area and municipality have many categories with very few customers.
Redundant geography: the geographic variables are hierarchically nested and strongly associated (Cramér's V between 0.91 and 0.99).
Customer seniority: 58% of the customers have customer_since_year = 2015, which probably means "2015 or earlier" and not the real starting year.
Next steps
Remove customer_type, exclude the identifier-like variables and group rare categories.
Keep the structural missing values as their own category instead of imputing them.
Build the repeat-purchase target from train.csv and study its relationship with the customer features.
Select the features, train and evaluate the model.
Requirements
Python 3 <!-- TODO: add the exact version -->
pandas, numpy, matplotlib, seaborn, scipy
jupyter to run the notebooks
bash
pip install pandas numpy matplotlib seaborn scipy jupyter
How to run
Put clientes.csv and train.csv in the data/ folder.
Open the notebook with Jupyter and run it from top to bottom.
Authors
<!-- TODO: names, course / institution and date -->
Nova IMS 