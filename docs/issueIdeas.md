# Datathon	
**April 24, 2015**

## Some Ideas for Potential Issues
- Maybe a simple example for number 3: count the number of restaurants around a school, and decay the score for each restaurant if another type of that restaurant is also around the school.  This will measure culinary diversity. (#3: Come up with ratings, for 30 institutions (including RPI!), rating how good the culinary/food scene around each institution is.)

- I was thinking based on the office hours of businesses, we could rank Universities which have the best night life? We could use the positivity in the reviews of the said businesses to help rank the Universities.
	- Python has an excellent natural language processing toolkit.  I've played with it to predict sentiment on IMDB review with good results based on a Kaggle tutorial.  I wonder if we could use that on the review data
	- If we do that, then we can separate quality from quantity. Coming back to the nightlife question, if the Uni has a lot of places that are open late but *bad* reviews then it may not be favorable to another Uni that has less businesses open, but *good* reviews

- Some "double-barrell" evaluation, like "I like golf and craft beer"

## Some Python Modules
- `NLTK`
- `sklearn`'s `RandomForests`