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
)




t = usr.select(["YRSelite", "eliteSTAT"])\
            .groupBy('eliteSTAT')\
            .agg(f.count('YRSelite')\
                 .alias('Years_Num'), 
                 f.mean('YRSelite')\
                 .alias('Years_Avg'),
                 f.min('YRSelite')\
                 .alias('Years_Min'),
                 f.max('YRSelite')\
                 .alias('Years_Max'))\
.withColumn('total', sum(col('Years_Num')).over(window))\
.withColumn('Percent %', f.format_string("%5.0f%%\n", col('Years_Num')*100/col('total')))
tab = t.drop('Percent %').withColumn('Percent', col('Years_Num')*100/col('total'))
t.drop(col('total'))


tab = tab.toPandas()




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

train_df,test_df=df.randomSplit([0.9,0.1])
val, tdf = test_df.randomSplit([0.99, .01])
tdf = tdf.toPandas()

tdf.show()
























