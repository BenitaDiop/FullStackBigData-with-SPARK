
%%info
sc.install_pypi_package("matplotlib==3.2.1")
sc.install_pypi_package("pandas==1.0.3")
sc.install_pypi_package("seaborn==0.10.0")
sc.install_pypi_package("numpy==1.18.4")
sc.install_pypi_package("pyspark==2.4.5")
sc.install_pypi_package("ipython==7.14.0")
sc.install_pypi_package("sklearn==0.0")


from IPython import display 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
from pyspark import *
import pyspark.sql.functions as f
from pyspark.sql import functions as f
from pyspark.sql.functions import rank, sum, col 



from pyspark.sql import Window 
window = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

usr = spark.read.json("s3://my-little-pony/yelp/yelp_academic_dataset_user.json")
view = spark.read.load('s3://my-little-pony/yelp/reviews.json', format='json')
business = spark.read.json("s3://my-little-pony/yelp/yelp_academic_dataset_business.json")



usr.createOrReplaceTempView("usr")
query = """
SELECT 
name,
user_id as ID,
explode(split(elite, ',')),
elite
FROM usr"""
usr1 = spark.sql(query).withColumnRenamed("col", "year").drop("elite")



usr.createOrReplaceTempView("csv")
query = """
SELECT *, 
explode(split(elite, ',')),
elite
FROM csv"""
csv =spark.sql(query)



usr = usr1.groupBy("name", "ID").count().withColumnRenamed('count', 'YRSelite').sort("YRSelite", ascending=False)


usr = usr.withColumn('eliteSTAT', 
              f.when(usr.YRSelite > 1, 1).otherwise(0))
              
              
 sub = csv.select("user_id", 
                 "fans", 
                 "cool", 
                 "useful", 
                 "average_stars",
                "compliment_cool",
                "compliment_funny", 
                "compliment_hot",
                "compliment_more",
                "compliment_note",
                "compliment_plain", 
                "compliment_photos",
                "compliment_profile")
usr.createOrReplaceTempView("us")
us = usr.join(sub, usr.ID == sub.user_id, "full")
us = us.withColumnRenamed('compliment_cool', 'cc')\
       .withColumnRenamed('compliment_funny', 'cf')\
       .withColumnRenamed('compliment_profile', 'cpr')\
       .withColumnRenamed('compliment_note', 'cn')\
       .withColumnRenamed('compliment_more', 'cm')\
       .withColumnRenamed('compliment_photos', 'cph')\
       .withColumnRenamed('compliment_plain', 'cpl')\
       .withColumnRenamed('compliment_hot', 'ch').drop("user_id")

us.columns



view.createOrReplaceTempView("tmp2")
query = """
SELECT
user_id,
stars as ReviewRating,
business_id,
review_id,
text, 
date 
FROM tmp2
"""



tmp2 = spark.sql(query)
tmp2.show()
expo = tmp2.join(us, us.ID == tmp2.user_id, "full")


expo.createOrReplaceTempView("df")
query = """
(SELECT * FROM df)"""

df = spark.sql(query)
df = df.distinct()
df = df.dropDuplicates()


%timeit
train_df,test_df=df.randomSplit([0.9,0.1])
val, tdf = test_df.randomSplit([0.99, .01])

print("""
Number of Rows in FULL Data {} \n

Number of Rows in Test Data {} \n 

Number of rows in Sub Test Data {} 
""".format(df.count(), train_df.count(), tdf.count()))


from pyspark.ml.feature import Tokenizer

tokenization=Tokenizer(inputCol='text',outputCol='tokens')

tokenized_df=tokenization.transform(expo)

tokenized_df.select("text", "tokens").show(5, True, True)

from pyspark.ml.feature import StopWordsRemover

stopword_removal=StopWordsRemover(inputCol='tokens',outputCol='refined_tokens')

refined_df=stopword_removal.transform(tokenized_df)

rdf = refined_df.select(['user_id','tokens','refined_tokens'])
rdf.show(2, True, True)




from pyspark.ml.feature import CountVectorizer

count_vec=CountVectorizer(inputCol='refined_tokens',outputCol='features')

cv_df=count_vec.fit(refined_df).transform(refined_df)

cv_df.select(['user_id',"business_id", "review_id", 'refined_tokens','features']).show(1,True, True)

count_vec.fit(refined_df).vocabulary


from pyspark.ml.feature import HashingTF,IDF

hashing_vec=HashingTF(inputCol='refined_tokens',outputCol='tf_features')

hashing_df=hashing_vec.transform(refined_df)

hashing_df.select(['user_id','refined_tokens','tf_features']).show(4,True, True)

tf_idf_vec=IDF(inputCol='tf_features',outputCol='tf_idf_features')

tf_idf_df=tf_idf_vec.fit(hashing_df).transform(hashing_df)

tf_idf_df.select('tf_idf_features').show(1,True, True)
tf_idf_df.show(1, True, True)

def get_dummy(df, indexCol, categoricalCols,
              continuousCols, labelCol, dropLast=False):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorAssembler
    from pyspark.sql.functions import col
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                for c in categoricalCols ]
 
    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
                               outputCol="{0}_encoded".format(indexer.getOutputCol()),
                               dropLast=dropLast)
                for indexer in indexers ]
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol()
                                           for encoder in encoders]
                                + continuousCols, outputCol="features")
    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    model=pipeline.fit(df)
    data = model.transform(df)
    if indexCol and labelCol:
    
        data = data.withColumn('label',col(labelCol))
        return data.select(indexCol,'features','label')
    elif not indexCol and labelCol:   # for supervised learning
        data = data.withColumn('label',col(labelCol))
        return data.select('features','label')
    elif indexCol and not labelCol:
   
        return data.select(indexCol,'features')
    elif not indexCol and not labelCol:
        # for unsupervised learning
        return data.select('features')


indexCol = "user_id", "business_id", "review_id", "name"
categoricalCols = "ReviewRating", "average_stars", "eliteSTAT"
continuousCols = []
labelCol = []

mat = get_dummy(cv_df,indexCol,categoricalCols,continuousCols,labelCol)
mat.show()

def get_dummy(df,indexCol,categoricalCols,continuousCols,labelCol):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.functions import col
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
for c in categoricalCols ]
# default setting: dropLast=True
encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="{0}_encoded".format(indexer.getOutputCol()))
for indexer in indexers ]
(assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders] + continuousCols, outputCol="features")
pipeline = Pipeline(stages=indexers + encoders + [assembler])
model=pipeline.fit(df)
data = model.transform(df)
data = data.withColumn('label',col(labelCol))
if indexCol:
return data.select(indexCol,'features','label')
else:
return data.select('features','label')


import sklearn
from sklearn import *
from sklearn import metrics, linear_model, tree, externals
from sklearn.externals import *

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
# from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

rdf['tk'] = map(lambda x: '|'.join(x), rdf['refined_tokens'])
tok_df = data.tk.str.get_dummies(sep='|')
data = data.merge(tok_df, left_index=True, right_index=True)
data.drop(['refined_clean', 'tk'], axis=1, inplace=True)

model_df = select_df[attribute_cols]  # use just attributes not basic data or check-in columns
model_df['is_open'] = business_df['is_open']  # add one basic data column; not sure why this causes error

feature_cols = model_df.columns
X = model_df
y = select_df.stars

lr = LinearRegression()
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
cross_val_scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=kfold)
print '10-fold RMSEs:'
print [np.sqrt(-x) for x in cross_val_scores]
print 'CV RMSE:'
print np.sqrt(-np.mean(cross_val_scores))  # RMSE is the sqrt of the avg of MSEs
print 'Std of CV RMSE:'
print np.std(cross_val_scores)























              
              
              






