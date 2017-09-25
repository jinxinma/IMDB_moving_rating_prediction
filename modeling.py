from user_defined_functions.eda_functions import *
from user_defined_functions.modeling_functions import *


### 1. TRAINING DATA PREPARATION
imdb = pd.read_csv('movie_metadata.csv.zip', compression='zip', delimiter=',')
data, target = imdb.drop('imdb_score', axis=1), imdb['imdb_score']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True), \
                                   y_train.reset_index(drop=True), y_test.reset_index(drop=True)
data_with_y = pd.concat([y_train, X_train], axis=1)


# feature engineering
data_with_y['budget'] = data_with_y['budget'].apply(budget_adjust)
director_avg_df = avg_rating(data_with_y, 'director_name')
actor1_avg_df = avg_rating(data_with_y, 'actor_1_name')

data_with_y = pd.merge(data_with_y, director_avg_df, how='left', on=['director_name'])
data_with_y = pd.merge(data_with_y, actor1_avg_df, how='left', on=['actor_1_name'])

data_with_y['year_cat'] = data_with_y['title_year'].apply(year_category)
data_with_y['num_genre'] = data_with_y['genres'].apply(num_genres)
data_with_y['content_cat'] = data_with_y['content_rating'].apply(content_rating_cat)


### 2. MODELING
cat_features = ['year_cat', 'content_cat']
num_features = ['num_critic_for_reviews', 'duration', 'director_facebook_likes',
                'gross', 'num_voted_users', 'cast_total_facebook_likes',
                'num_user_for_reviews', 'budget', 'avg_director_name_score', 
                'avg_actor_1_name_score', 'num_genre']

# numerical feature pipeline
num_pipeline = Pipeline([('feature_selector', FeatureSelector(num_features)),
                         ('imputer', Imputer(strategy="median")),
                         ('feature_scaler', StandardScaler())])

# sklearn doesn't have a function that fill missing values for categorical features
# we need to use the fillna function from pandas here
data_with_y[cat_features] = data_with_y[cat_features].fillna(value='missing')

# categorical feature pipeline
cat_pipeline = Pipeline([('feature_selector', FeatureSelector(cat_features)),
                         ('one_hot_encoder', CategoricalEncoder(encoding='onehot-dense'))]) 

full_preprocess_pipeline = FeatureUnion(transformer_list =[('num_pipeline', num_pipeline), 
                                                           ('cat_pipeline', cat_pipeline)])

X_train_full = full_preprocess_pipeline.fit_transform(data_with_y)
xgboost_full_reg, xgboost_full_rmse = train_model(X_train_full, y_train, XGBRegressor())
xgb_feature_importance = xgboost_full_reg.booster().get_score(importance_type='gain').values()

# final pipeline to train the model
final_pipeline = Pipeline([
    ('preparation', full_preprocess_pipeline),
    ('feature_selection', TopFeatureSelector(xgb_feature_importance, 12))
])

X_train_final = final_pipeline.fit_transform(data_with_y)
xgboost_reg, xgboost_rmse = train_model(X_train_final, y_train, xgboost_full_reg)

print "XGBoost with most important features RMSE: "
cross_validation_rmse(xgboost_reg, X_train_final, y_train)

# hyperparamter tuning
print "Tuning paramers will take about 15 seconds..."
param_distribs = {
        'learning_rate': [0.001, 0.1, 0.3],
        'n_estimators': [100, 300],
        'reg_alpha': [0, 0.1]
    }

clf = GridSearchCV(XGBRegressor(), param_distribs, cv=5,
                   scoring='neg_mean_squared_error')
clf.fit(X_train_final, y_train)


### 3. PREDICT

# feature engineering on test set
X_test = pd.merge(X_test, director_avg_df, how='left', on=['director_name'])
X_test_prepared = pd.merge(X_test, actor1_avg_df, how='left', on=['actor_1_name'])
X_test_prepared['year_cat'] = X_test_prepared['title_year'].apply(year_category)
X_test_prepared['num_genre'] = X_test_prepared['genres'].apply(num_genres)
X_test_prepared['content_cat'] = X_test_prepared['content_rating'].apply(content_rating_cat)

# make and save predictions
X_test_final = final_pipeline.transform(X_test_prepared)
y_test_pred = clf.predict(X_test_final)
print 'RMSE on test set is: ' + str(np.sqrt(mean_squared_error(y_test, y_test_pred)))
predict_df = pd.DataFrame({'movie_title': X_test['movie_title'], 'moving_rating':y_test, 'rating_prediction':y_test_pred})
predict_df.to_csv('IMDB_movie_rating_prediction.csv', index=None)


### 4. SAVE MODEL
joblib.dump(clf, "imdb_model.pkl")
imdb_model = joblib.load("imdb_model.pkl")
imdb_model