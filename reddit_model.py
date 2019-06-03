from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer
from cleantext import sanitize
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def split_arr_to_word(arr):
    new_arr = []
    for i in range(len(arr)):
        temp_arr = arr[i].split()
        for w in temp_arr:
            new_arr.append(w)
    return new_arr

def remove_first_three(id):
    new_id = id[3:]
    return new_id

# This function was written by StackOverflow user zero323
# https://stackoverflow.com/questions/39555864/how-to-access-element-of-a-vectorudt-column-in-a-spark-dataframe
def ith_(v, i):
    try:
        return float(v[i])
    except ValueError:
        return None

def main(context):
    """Main function takes a Spark SQL context."""
    # Task 1: load data
    # Run this if loading for the first time
    ''' 
    comments = context.read.json("comments-minimal.json.bz2")
    submissions = context.read.json("submissions.json.bz2")
    labels = context.read.csv("labeled_data.csv", header=True)
    
    comments.write.parquet("comments.parquet")
    submissions.write.parquet("submissions.parquet")
    labels.write.parquet("labels.parquet")
    '''

    comments = context.read.parquet("comments.parquet")
    submissions = context.read.parquet("submissions.parquet")
    labels= context.read.parquet("labels.parquet")

    # Task 2: join labeled_data with comments_minimal
    table = labels.join(comments, labels.Input_id == comments.id)
    
    # Task 4 & 5: Generate unigrams, bigrams, and trigrams for each comment in the labeled data,
    # store all of them into one column and split them by words.
    sanitize_udf = udf(sanitize, ArrayType(StringType()))
    split_udf = udf(split_arr_to_word, ArrayType(StringType()))
    sanitized_table = table.select("id", "labeldjt", \
            split_udf(sanitize_udf("body")).alias("sanitized_text"))
    
    # Task 6A: Turn raw features into a sparse feature vector. Only use tokens that appear more than 10 times.
    cv = CountVectorizer(minDF=10.0, inputCol="sanitized_text", outputCol="vectors")
    cv_table = cv.fit(sanitized_table)
    result = cv_table.transform(sanitized_table)
    
    # Task 6B: Add columns for positive and negative labels
    final = result.withColumn("positive", F.when(result.labeldjt == 1, 1).otherwise(0))\
            .withColumn("negative", F.when(result.labeldjt == -1, 1).otherwise(0))
    pos = final.select(col("id"), col("vectors").alias("features"), col("positive").alias("label"))
    neg = final.select(col("id"), col("vectors").alias("features"), col("negative").alias("label"))

    # Task 7: Initialize two logistic regression models.
    # Code to generate the models:
    '''
    # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    poslr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
    neglr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator()
    negEvaluator = BinaryClassificationEvaluator()
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,
        numFolds=5)
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=5)
    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50
    posTrain, posTest = pos.randomSplit([0.5, 0.5])
    negTrain, negTest = neg.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)

    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.save("project2/pos.model")
    negModel.save("project2/neg.model")
    '''

    # To load saved models:
    posModel = CrossValidatorModel.load("project2/pos.model")
    negModel = CrossValidatorModel.load("project2/neg.model")


    # Task 8: read more parts of comments
    fix_link_udf = udf(remove_first_three, StringType())
    comments_fixed = comments.select(col("id").alias("comment_id"), fix_link_udf("link_id").alias("link_id"),"created_utc","body",col("author_flair_text").alias("state"))
    submissions_limited = submissions.select("id", "title")
    new_table = submissions_limited.join(comments_fixed, comments_fixed.link_id == submissions_limited.id)

    # Task 9:
    # remove any comments that contain '\s' or '&gt;'
    new_table = new_table.filter(~new_table.body.contains("&gt;") & ~new_table.body.contains("\s"))
    # repeat task 4 and 5 and 6A
    # [link_id, state, comment_id, body, created_utc, title, id]
    sanitized_new_table = new_table.select("link_id", "state", "comment_id", "body", "created_utc", \
            "title", "id", split_udf(sanitize_udf("body")).alias("sanitized_text"))
    final_table = cv_table.transform(sanitized_new_table)

    # run the models
    ith = udf(ith_, FloatType())
    task9_table = final_table.select("link_id", "state", "comment_id", "body", "created_utc", "title", "id", "sanitized_text", col("vectors").alias("features"))
    task9_table = posModel.transform(task9_table)
    task9_table = task9_table.withColumn("pos", F.when(ith(task9_table.probability, lit(1)) > 0.2, 1).otherwise(0)).select("link_id", "state", "comment_id", "body", "created_utc", "title", "id", "features", "pos")
    task9_table = negModel.transform(task9_table)
    task9_table = task9_table.withColumn("neg", F.when(ith(task9_table.probability, lit(1)) > 0.25, 1).otherwise(0)).select("link_id", "state", "comment_id", "body", "created_utc", "title", "id", "pos", "neg")
    task9_table.show(n=100)



if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
