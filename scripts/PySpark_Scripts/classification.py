
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




# Add length to the dataframe
from pyspark.sql.functions import length
text_df=text_df.withColumn('length',length(text_df['Review']))

text_df.orderBy(rand()).show(10,False)
text_df.groupBy('Label').agg({'Length':'mean'}).show()






# Data Cleaning
tokenization=Tokenizer(inputCol='Review',outputCol='tokens')
tokenized_df=tokenization.transform(text_df)

tokenized_df.show()

stopword_removal=StopWordsRemover(inputCol='tokens',outputCol='refined_tokens')
refined_text_df=stopword_removal.transform(tokenized_df)

refined_text_df.show()



from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import *

len_udf = udf(lambda s: len(s), IntegerType())

refined_text_df = refined_text_df.withColumn("token_count", len_udf(col('refined_tokens')))

refined_text_df.orderBy(rand()).show(10)
count_vec=CountVectorizer(inputCol='refined_tokens',outputCol='features')
cv_text_df=count_vec.fit(refined_text_df).transform(refined_text_df)
cv_text_df.select(['refined_tokens','token_count','features','Label']).show(10)

