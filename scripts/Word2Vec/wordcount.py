import nltk
nltk.download('stopwords', download_dir='.')

from nltk.corpus import stopwords
nltk.data.path.append('.')
stop_words = stopwords.words('english')

from textblob.utils import strip_punc
tokenized = sc.textFile('wasb:///example/data/RomeoAndJuliet.txt')\
              .map(lambda line: strip_punc(line, all=True).lower())\
              .flatMap(lambda line: line.split())

filtered = tokenized.filter(lambda word: word not in stop_words)

from operator import add
word_counts = filtered.map(lambda word: (word, 1)).reduceByKey(add)

filtered_counts = word_counts.filter(lambda item: item[1] >= 60)

from operator import itemgetter
sorted_items = sorted(filtered_counts.collect(), 
                      key=itemgetter(1), reverse=True)

max_len = max([len(word) for word, count in sorted_items])
for word, count in sorted_items:
    print('{:>{width}}: {}'.format(word, count, width=max_len))
