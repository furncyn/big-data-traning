from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, col, lit, from_unixtime
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
    # TASK 1: load data
        
    # Read from original source files
    comments = context.read.json("comments-minimal.json.bz2")
    submissions = context.read.json("submissions.json.bz2")
    labels = context.read.csv("labeled_data.csv", header=True)
    '''
    # Write to parquet files
    comments.write.parquet("comments.parquet")
    submissions.write.parquet("submissions.parquet")
    labels.write.parquet("labels.parquet")
    # Read from parquet files
    comments = context.read.parquet("comments.parquet")
    submissions = context.read.parquet("submissions.parquet")
    labels= context.read.parquet("labels.parquet")
    '''

    comments = comments.select("id", "body", "created_utc", "author_flair_text", "link_id", col("score").alias("c_score"))
    submissions = submissions.select("title", "id", col("score").alias("s_score"))
    labels = labels.select("Input_id", "labeldjt")

    # TASK 2: join labeled_data with comments_minimal
    comments_labels = labels.join(comments, labels.Input_id == comments.id).select("id", "body", "created_utc", "author_flair_text", "link_id", "labeldjt")
    
    # TASK 4, 5: Generate unigrams, bigrams, and trigrams for each comment in the labeled data,
    # store all of them into one column and split them by words.
    sanitize_udf = udf(sanitize, ArrayType(StringType()))
    split_udf = udf(split_arr_to_word, ArrayType(StringType()))
    sanitized_table = comments_labels.select("id", "labeldjt", \
            split_udf(sanitize_udf("body")).alias("sanitized_text"))
    
    # TASK 6A: Turn raw features into a sparse feature vector. Only use tokens that appear more than 10 times.
    cv = CountVectorizer(minDF=10.0, inputCol="sanitized_text", outputCol="vectors")
    cv_table = cv.fit(sanitized_table)
    vec_table = cv_table.transform(sanitized_table)
    
    # TASK 6B: Add columns for positive and negative labels
    final = vec_table.withColumn("positive", F.when(vec_table.labeldjt == 1, 1).otherwise(0))\
            .withColumn("negative", F.when(vec_table.labeldjt == -1, 1).otherwise(0))
    pos = final.select(col("id"), col("vectors").alias("features"), col("positive").alias("label"))
    neg = final.select(col("id"), col("vectors").alias("features"), col("negative").alias("label"))

    # TASK 7: Initialize two logistic regression models.

    # Code to generate the models:
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
    
    '''
    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.save("project2/pos.model")
    negModel.save("project2/neg.model")
    
    # To load saved models:
    posModel = CrossValidatorModel.load("project2/pos.model")
    negModel = CrossValidatorModel.load("project2/neg.model")
    '''

    # TASK 8: read more parts of comments
    fix_link_udf = udf(remove_first_three, StringType())
    comments_fixed = comments.select(col("id").alias("comment_id"), fix_link_udf("link_id").alias("link_id"),"created_utc","body",col("author_flair_text").alias("state"), "c_score")
    new_table, = submissions.join(comments_fixed, comments_fixed.link_id == submissions.id).randomSplit([0.2])

    # TASK 9:
    # remove any comments that contain '\s' or '&gt;'
    new_table = new_table.filter(~new_table.body.contains("&gt;") & ~new_table.body.contains("\s"))
    # Repeat task 4 and 5 and 6A
    sanitized_new_table = new_table.select("link_id", "state", "comment_id", "body", "created_utc", \
            "title", split_udf(sanitize_udf("body")).alias("sanitized_text"), "s_score", "c_score")
    final_table = cv_table.transform(sanitized_new_table)

    # Run the models
    ith = udf(ith_, FloatType())
    task9_table = final_table.select("link_id", "state", "comment_id", "body", "created_utc", "title", "c_score", "s_score", "sanitized_text", col("vectors").alias("features"))
    task9_table = posModel.transform(task9_table)
    task9_table = task9_table.withColumn("pos", F.when(ith(task9_table.probability, lit(1)) > 0.2, 1).otherwise(0)).select("link_id", "state", "comment_id", "body", "c_score", "s_score", "created_utc", "title", "features", "pos")
    task9_table = negModel.transform(task9_table)
    task9_table = task9_table.withColumn("neg", F.when(ith(task9_table.probability, lit(1)) > 0.25, 1).otherwise(0)).select("link_id", "state", "comment_id", "body", "c_score", "s_score", "created_utc", "title", "pos", "neg")

    # TASK 10: calculate statistics
    # Part 1:
    part1 = task9_table.groupBy().agg(F.avg("pos").alias("pos"), F.avg("neg").alias("neg"))
    # Part 2:
    part2 = task9_table.groupBy(from_unixtime("created_utc","yyyy-MM-dd")).agg(F.avg("pos").alias("pos"), F.avg("neg").alias("neg"))
    # Part 3: 
    states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    part3 = task9_table.where(col("state").isin(states)).groupBy("state").agg(F.avg("pos").alias("pos"), F.avg("neg").alias("neg"))
    # Part 4:
    part4 = task9_table.groupBy("title").agg(F.avg("pos").alias("pos"), F.avg("neg").alias("neg"))
    by_c_score = task9_table.groupBy("c_score").agg(F.avg("pos").alias("pos"), F.avg("neg").alias("neg"))
    by_s_score = task9_table.groupBy("s_score").agg(F.avg("pos").alias("pos"), F.avg("neg").alias("neg"))
   
    # Save the data
    part1.repartition(1).write.format("com.databricks.spark.csv").save("part1.csv")
    part2.repartition(1).write.format("com.databricks.spark.csv").save("part2.csv")
    part3.repartition(1).write.format("com.databricks.spark.csv").save("part3.csv")
    by_c_score.repartition(1).write.format("com.databricks.spark.csv").save("c_score.csv")
    by_s_score.repartition(1).write.format("com.databricks.spark.csv").save("s_score.csv")
    part4.repartition(1).write.format("com.databricks.spark.csv").save("part4.csv")
    part5.repartition(1).write.format("com.databricks.spark.csv").save("part5.csv")
    
    # Part 5: calculate percentage of positive and negative comments by month
    time = context.read.csv("part2.csv")
    time = time.withColumn("pos", time["_c1"].cast(FloatType()))\
            .withColumn("neg", time["_c2"].cast(FloatType()))\
            .drop("_c1").drop("_c2").na.drop()
    part5 = time.groupBy(F.month(time._c0), F.year(time._c0)).agg(F.avg("pos").alias("pos"), F.avg("neg").alias("neg"))
    
    # for plot 4
    part4.sort(col("pos").desc()).limit(10).repartition(1).write.format("com.databricks.spark.csv").save("plot4_pos.csv")
    part4.sort(col("neg").desc()).limit(10).repartition(1).write.format("com.databricks.spark.csv").save("plot4_neg.csv")



if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
