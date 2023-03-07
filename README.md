## Quality of content-based information in datasets used for RecSys

Exploratory project for *Introduction to RecSys and User Preferences* class meant to analyze individual differences between varying content-based information provided by selected datasets. For each dataset, there exists a limited set of information that can be exploited/used for recommendation purposes. Project tries to evaluate Content-based recommender systems using subsets of available information from individual datasets and compare key offline evaluation metrics between used subsets of information.

#### Selected Datasets:
- **MovieLens-100K** (also ML-Latest for feature extraction and analysis)
  - *CB Information*:
    - Genres
    - Tags
    - Release Years
- **Book Crossing Dataset**
  - Using pre-filtered version with only 3 event types
  - This dataset consists of:
    - 92,490 interactions from 3,431 users on 8,885 items.
    - History: 3,423 users and 8,878 items (78,372 accesses) 
    - Purchase: 824 users and 3,077 items (5,089 interactions)
    - Add to cart: 1,557 users and 4,447 items (9,029 interactions) 
    - Content: Implicit feedback by Event type (View, Add to cart, Purchase)

- **RetailRocket**
  - Using pre-filtered version
  - 272,679 interactions (explicit / implicit) from 2,946 users on 17,384 books.
  - Ratings: 1,295 users and 14,684 books (62,657 ratings applied)
  - Ratings are between 1 - 10.
  - *CB Information*: Simple demographic info for the users (age, gender, occupation, zip)
