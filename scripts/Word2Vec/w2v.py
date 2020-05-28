from pyspark.ml.feature import Tokenizer

tokenization=Tokenizer(inputCol='text',outputCol='tokens')

tokenized_df=tokenization.transform(expo)

tokenized_df.select("text", "tokens").show(5, False)

from pyspark.ml.feature import StopWordsRemover

stopword_removal=StopWordsRemover(inputCol='tokens',outputCol='refined_tokens')

refined_df=stopword_removal.transform(tokenized_df)

refined_df.select(['user_id','tokens','refined_tokens']).show(2,False)

from pyspark.ml.feature import CountVectorizer

count_vec=CountVectorizer(inputCol='refined_tokens',outputCol='features')

cv_df=count_vec.fit(refined_df).transform(refined_df)

cv_df.select(['user_id','refined_tokens','features']).show(4,False)

count_vec.fit(refined_df).vocabulary

from pyspark.ml.feature import HashingTF,IDF

hashing_vec=HashingTF(inputCol='refined_tokens',outputCol='tf_features')

