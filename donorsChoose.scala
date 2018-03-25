// Databricks notebook source
import org.apache.spark.sql.SparkSession
import spark.implicits._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier, LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoderEstimator, Bucketizer, VectorAssembler}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}


// COMMAND ----------

val spark = SparkSession
  .builder()
  .getOrCreate()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Read in sample data
// MAGIC Uploaded into databricks filestore

// COMMAND ----------

val rawData = spark.read.format("csv")
  .option("header", "true")
  .option("quote", "\"")
  .option("escape", "\"")
  .load("/FileStore/tables/train.csv")

// COMMAND ----------

display(rawData)

// COMMAND ----------

rawData.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Data Exploration

// COMMAND ----------

// MAGIC %md
// MAGIC Our dataset contains rows of project applications, and we want to predict approval -- the project_is_approved column.
// MAGIC 
// MAGIC Hopefully, a rich feature will be the contents of the project essays. There's a lot of text featurizing we could do with the essay columns, but I will try to keep it simple first to see what we can get with the other columns and basic information about the essays before digging deeper.
// MAGIC 
// MAGIC Let's start by getting an idea of our average approval rate. Notice that everything read in as a string, so we should cast to numeric types first.

// COMMAND ----------

val data = rawData
  .withColumn("project_is_approved2", rawData.col("project_is_approved").cast(DoubleType))
  .withColumn("teacher_number_of_previously_posted_projects2", rawData.col("teacher_number_of_previously_posted_projects").cast(DoubleType))
  .withColumn("project_submitted_datetime2", unix_timestamp($"project_submitted_datetime", "yyyy-MM-dd HH:mm:ss"))
  .drop("project_is_approved", "project_submitted_datetime", "teacher_number_of_previously_posted_projects")
  .withColumnRenamed("project_is_approved2", "project_is_approved")
  .withColumnRenamed("project_submitted_datetime2", "project_submitted_datetime")
  .withColumnRenamed("teacher_number_of_previously_posted_projects2", "teacher_number_of_previously_posted_projects")

data.printSchema()

// COMMAND ----------

data.select(avg($"project_is_approved")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC Approval rate is much higher than I expected on the training dataset -- 85% of projects get approved. So a naive solution could get us 85% accuracy. This is an imbalanced class, but not so imbalanced that we should be worried about identifying the minority class if we can extract useful features.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Feature Extraction
// MAGIC We have a few opportunities for features.
// MAGIC 1. Parse out the project subject categories and subcategories and one hot encode them.
// MAGIC 2. Length of each essay (wordcount)
// MAGIC 3. Look for certain words in the project title
// MAGIC 
// MAGIC Not going to do anything more intense with the text columns yet.

// COMMAND ----------

display(data)

// COMMAND ----------

val fullData = data
  .withColumn("tod", hour(from_unixtime($"project_submitted_datetime")))
  .withColumn("dow", dayofmonth(from_unixtime($"project_submitted_datetime")))
        

// COMMAND ----------

display(fullData)

// COMMAND ----------

fullData.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Modeling

// COMMAND ----------

// MAGIC %md
// MAGIC With an even class and no problems with high dimensionality, logistic regression, random forest, and gradient boosted trees are all good model candidates to predict whether the invite will result in a quote.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Split data into training and test

// COMMAND ----------

val Array(trainingData, testingData) = fullData.randomSplit(Array(0.8, 0.2), seed=15)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Transform features

// COMMAND ----------

val colsToStringIndex: Array[String] = Array("teacher_prefix", "school_state", "project_grade_category", "project_subject_categories")
var colsToOneHotEncode: Array[String] = Array()
var oheOutputCols: Array[String] = Array()
// val colsToBucketize: Array[String] = Array("location_size_proxy")
// var bucketizerOutputCols: Array[String] = Array()

// COMMAND ----------

var transformerStages: Array[PipelineStage] = Array()

for(col <- colsToStringIndex) {
  val indexerStage = new StringIndexer()
    .setInputCol(col)
    .setOutputCol(col + "_indexed")
    .setHandleInvalid("keep")
  transformerStages = transformerStages ++ Array(indexerStage)
  colsToOneHotEncode = colsToOneHotEncode ++ Array(col + "_indexed")
  oheOutputCols = oheOutputCols ++ Array(col + "_ohe")
}

val oneHotEncodeStage = new OneHotEncoderEstimator()
  .setInputCols(colsToOneHotEncode)
  .setOutputCols(oheOutputCols)
transformerStages = transformerStages ++ Array(oneHotEncodeStage)

/*
for(col <- colsToBucketize) {
  val bucketizerStage = new Bucketizer()
    .setInputCol(col)
    .setOutputCol(col+ "_bucketized")
    .setSplits(Array(Double.NegativeInfinity, 10, 20, 30, 40, 50, Double.PositiveInfinity))
  
  transformerStages = transformerStages ++ Array(bucketizerStage)
  bucketizerOutputCols = bucketizerOutputCols ++ Array(col + "_bucketized")
}
*/

// COMMAND ----------

// var assemblerCols: Array[String] = oheOutputCols ++ bucketizerOutputCols
var assemblerCols: Array[String] = oheOutputCols
val assembler = new VectorAssembler()
  .setInputCols(assemblerCols)
  .setOutputCol("features")
val allNonModelStages = transformerStages ++ Array(assembler)

// COMMAND ----------

val labelCol = "project_is_approved"

// COMMAND ----------

// MAGIC %md
// MAGIC ### Build logistic regression model pipeline

// COMMAND ----------

val lrModel = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)
  .setLabelCol(labelCol)
  .setRawPredictionCol("rawPrediction")

val allLrStages = allNonModelStages ++ Array(lrModel)

// COMMAND ----------

val lrPipeline = new Pipeline()
  .setStages(allLrStages)

val lrParamGrid = new ParamGridBuilder()
  .addGrid(lrModel.regParam, Array(0.1, 0.01))
  .build()

val lrEvaluator = new BinaryClassificationEvaluator()
  .setLabelCol(labelCol)

val lrCV = new CrossValidator()
  .setEstimator(lrPipeline)
  .setEvaluator(lrEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(3)
  .setParallelism(2)


// COMMAND ----------

// MAGIC %md
// MAGIC ### Build random forest model pipeline

// COMMAND ----------

val rfModel = new RandomForestClassifier()
  .setLabelCol(labelCol)
  .setRawPredictionCol("rawPrediction")
val allRfStages = allNonModelStages ++ Array(rfModel)
val rfPipeline = new Pipeline().setStages(allRfStages)

val rfParamGrid = new ParamGridBuilder()
  .addGrid(rfModel.numTrees, Array(100, 250))
  .addGrid(rfModel.maxDepth, Array(3, 5, 7))
  .build()

val rfCV = new CrossValidator()
  .setEstimator(rfPipeline)
  .setEvaluator(new BinaryClassificationEvaluator().setLabelCol(labelCol))
  .setEstimatorParamMaps(rfParamGrid)
  .setNumFolds(3)
  .setParallelism(2)

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Build GBT model pipeline

// COMMAND ----------

val gbtModel = new GBTClassifier()
  .setLabelCol(labelCol)
  .setRawPredictionCol("rawPrediction")
  .setFeatureSubsetStrategy("auto")

val allGbtStages = allNonModelStages ++ Array(gbtModel)
val gbtPipeline = new Pipeline().setStages(allGbtStages)

val gbtParamGrid = new ParamGridBuilder()
  .addGrid(gbtModel.stepSize, Array(0.01, 0.05, 0.1))
  .addGrid(gbtModel.maxDepth, Array(3, 5, 7))
  .build()

val gbtCV = new CrossValidator()
  .setEstimator(gbtPipeline)
  .setEvaluator(new BinaryClassificationEvaluator().setLabelCol(labelCol))
  .setEstimatorParamMaps(gbtParamGrid)
  .setNumFolds(3)
  .setParallelism(2)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Evaluate Models

// COMMAND ----------

val lrCvModel = lrCV.fit(trainingData)
val rfCvModel = rfCV.fit(trainingData)
val gbtCvModel = gbtCV.fit(trainingData)


// COMMAND ----------

// MAGIC %md
// MAGIC ### Make predictions on the testing data

// COMMAND ----------

val transformedLrTestingData = lrCvModel.transform(testingData)
val transformedRfTestingData = rfCvModel.transform(testingData)
val transformedGbtTestingData = gbtCvModel.transform(testingData)

// COMMAND ----------

transformedLrTestingData.select("probability").printSchema()


// COMMAND ----------

val probArray = transformedLrTestingData.select("probability").rdd.map(r => (r.getLong(0).toDouble)).collect().toArray


// COMMAND ----------

val probs = new DenseVector(probArray)

// COMMAND ----------

display(transformedLrTestingData)

// COMMAND ----------

import org.apache.spark.ml.linalg.DenseVector
    val prob = new DenseVector(Array(0.37, 0.63))
    val first = prob.toDense(0)
  

// COMMAND ----------

// MAGIC %md
// MAGIC ### Look at our best models

// COMMAND ----------

val bestLrModel = lrCvModel.bestModel.asInstanceOf[PipelineModel]
val lrStage = bestLrModel.stages(8).asInstanceOf[LogisticRegressionModel]

val bestRfModel = rfCvModel.bestModel.asInstanceOf[PipelineModel]
val rfStage = bestRfModel.stages(8).asInstanceOf[RandomForestClassificationModel]

val bestGbtModel = gbtCvModel.bestModel.asInstanceOf[PipelineModel]
val gbtStage = bestGbtModel.stages(8).asInstanceOf[GBTClassificationModel]

// COMMAND ----------

val meta: org.apache.spark.sql.types.Metadata = transformedLrTestingData
  .schema(transformedLrTestingData.schema.fieldIndex("features"))
  .metadata

val featureMeta = meta.getMetadata("ml_attr").getMetadata("attrs") 
val coefs = lrStage.coefficients
coefs.size
toJsonValue(featureMeta)

// COMMAND ----------

def printEvaluationStatistics(model: LogisticRegressionModel): Unit = {
  val trainingSummary = model.summary
  val objectiveHistory = trainingSummary.objectiveHistory
  println("objectiveHistory:")
  objectiveHistory.foreach(loss => println(loss))
  val accuracy = trainingSummary.accuracy
  val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
  val truePositiveRate = trainingSummary.weightedTruePositiveRate
  val fMeasure = trainingSummary.weightedFMeasure
  val precision = trainingSummary.weightedPrecision
  val recall = trainingSummary.weightedRecall
  println(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +
    s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")
}

// COMMAND ----------

// MAGIC %md
// MAGIC Random forest feature importance

// COMMAND ----------

val featureImportances = rfStage.featureImportances
val assemblerStage = bestRfModel.stages(7).asInstanceOf[VectorAssembler]
val assembledCols = assemblerStage.getInputCols // This should match assemblerCols that we input but just to be safe

var featureImportanceArray: Array[(String, Double)] = Array()
var i = 0
for(col <- assembledCols) {
  var importanceScore = featureImportances(i)
  featureImportanceArray = featureImportanceArray ++ Array((col, importanceScore))
  i = i + 1
}

val importanceDf = spark.sqlContext.sparkContext.parallelize(featureImportanceArray).toDF("Feature", "Importance").orderBy(col("Importance").desc)
display(importanceDf)

// COMMAND ----------

printEvaluationStatistics(lrStage)

// COMMAND ----------



// COMMAND ----------


