from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer
from cleantext import sanitize

def split_arr_to_word(arr):
    new_arr = []
    for i in range(len(arr)):
        temp_arr = arr[i].split()
        for w in temp_arr:
            new_arr.append(w)
    return new_arr

def main(context):
    """Main function takes a Spark SQL context."""
    # Task 1: load data
    comments = context.read.json("comments-minimal.json.bz2")
    submissions = context.read.json("submissions.json.bz2")
    labels = context.read.csv("labeled_data.csv", header=True)

    # Task 2: join labeled_data with comments_minimal
    table = labels.join(comments, labels.Input_id == comments.id)
    
    # Task 4 & 5: Generate unigrams, bigrams, and trigrams for each comment in the labeled data,
    # store all of them into one column and split them by words.
    sanitize_udf = udf(sanitize, ArrayType(StringType()))
    split_udf = udf(split_arr_to_word, ArrayType(StringType()))
    sanitized_table = table.select("Input_id", "id", "labeldem", "labelgop", "labeldjt", \
            split_udf(sanitize_udf("body")).alias("sanitized_text"))
    
    # Task 6A: Turn raw features into a sparse feature vector. Only use tokens that appear more than 10 times.
    cv = CountVectorizer(minDF=10.0, inputCol="sanitized_text", outputCol="vectors")
    cv_table = cv.fit(sanitized_table)
    result = cv_table.transform(sanitized_table)
    result.show()
    

if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
