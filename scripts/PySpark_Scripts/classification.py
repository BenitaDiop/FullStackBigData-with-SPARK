
text_df=spark.read.csv('reviews.csv',inferSchema=True,header=True,sep=',')
text_df.printSchema()
text_df.count()

from pyspark.sql.functions import rand 
text_df.orderBy(rand()).show(10,False)
text_df=text_df.filter(((text_df.Sentiment =='1') | (text_df.Sentiment =='0')))

text_df.count()
text_df.groupBy('Sentiment').count().show()


text_df.printSchema()
text_df = text_df.withColumn("Label", text_df.Sentiment.cast('float')).drop('Sentiment')
text_df.orderBy(rand()).show(10,False)
text_df.groupBy('label').count().show()

