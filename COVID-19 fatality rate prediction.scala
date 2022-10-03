// ------Final Project- done by Mahmoud and Pragathi-----

// first let's import all the required libraries

import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, CountVectorizer, NGram, IDF, VectorAssembler, StringIndexer}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.sql.types.{IntegerType, DoubleType}

import org.apache.spark.sql.functions.{expr, col, column, min, max, desc, avg, when, count}
import org.apache.spark.sql.types._

// let's define a schema for our dataset

val covid_schema_MP = StructType(Array(
		
		StructField("_id", IntegerType, true),
		StructField("Assigned_ID", IntegerType, true),
		StructField("Outbreak Associated", StringType, true),
		StructField("Age Group", StringType, true),
		StructField("Neighbourhood Name", StringType, true),
		StructField("FSA", StringType, true),
		StructField("Source of Infection", StringType,true),
		StructField("Classification", StringType,true),
		StructField("Episode Date", StringType,true),
		StructField("Reported Date", StringType,true),
		StructField("Client Gender", StringType,true),
		StructField("Outcome", StringType,true),
		StructField("Currently Hospitalized", StringType,true),
		StructField("Currently in ICU", StringType,true),
		StructField("Currently Intubated", StringType,true),
		StructField("Ever Hospitalized", StringType,true),
		StructField("Ever in ICU", StringType,true),
		StructField("Ever Intubated", StringType,true)));


// let load the dataset

val covid_raw_MP = spark
  .read.format("csv")
  .option("header", "true")
  .schema(covid_schema_MP)
  .load("hdfs://10.128.0.4:8020/Mahmoud_Pragathi/COVID19_Cases.csv")

// next let's print the first 10 records to check the dataset


covid_raw_MP.show(10)



// let's check the schema of the dataset and make sure that we have the right type for each column

covid_raw_MP.printSchema



// now we will keep only the needed features in our project

val covid_raw_MP_drop_cols = covid_raw_MP.drop("_id","Assigned_ID","Neighbourhood Name", "FSA", "Classification", "Episode Date", "Reported Date", "Currently Hospitalized","Currently in ICU", "Currently Intubated")

covid_raw_MP_drop_cols.show(10)



// first let's drop the Active outcome from the dataset and keep only Fatal, and Resolved
val covid_raw_MP_no_Active = covid_raw_MP_drop_cols.filter($"Outcome" =!= "ACTIVE")

// let's check the number of null values in the data

import org.apache.spark.sql.Column

def countMissingValues_Mahmoud(columns:Array[String]):Array[Column]={
    columns.map(columnName=>{
      count(when(col(columnName).isNull, columnName)).alias(columnName)
    })
}
covid_raw_MP_no_Active.select(countMissingValues_Mahmoud(covid_raw_MP_no_Active.columns):_*).show()



// let's check if the 362 from the Fatal outcome data or the resolved

covid_raw_MP_no_Active.filter($"Outcome" === "FATAL").select(countMissingValues_Mahmoud(covid_raw_MP_no_Active.filter($"Outcome" === "FATAL").columns):_*).show()



// since the 362 null rows in the age_group from teh resolved outcome data, then we will drop it from the dataset
val covid_raw_MP_no_nulls = covid_raw_MP_no_Active.na.drop()

// now since we got rid off the null values we can start wokring on the unbalanced data issue

// first let's check the length of the whole dataset (both resolved, and fatal)

println(covid_raw_MP_no_nulls.count)

// the length is: 346273

val fatal_df_MP = covid_raw_MP_no_nulls.filter($"Outcome" === "FATAL")
println(fatal_df_MP.count)
// lenght of Fatal data: 4364

val resolved_df_MP = covid_raw_MP_no_nulls.filter($"Outcome" === "RESOLVED")
println(resolved_df_MP.count)
// lenght of Resolved data: 341909

// based on the One-vs-all rule, taken by the ref: https://www.linkedin.com/pulse/multi-class-classification-imbalanced-data-using-random-burak-ozen/
// "While training a model for class Fatal, use all class Fatal data (%1.26) and randomly selecting the same amount of data from the resolved data (%98.74)"
// also the %1.26 is basically the number of rows of Fatal 4364 / total number of rows 346273 * 100

// now let's sample the resoved dataset, and take about 4364 of rows from it

val guessedFraction = 0.0127 // increased the fraction to get the number ot be around 4364 (the number of fatal data)
val newSample_Resolved = resolved_df_MP.sample(true, guessedFraction).limit(4364) 
println(newSample_Resolved.count) // the new sample lenght is: 4364

// now we combine the newSample_Resolved and the fatal_df_MP, to have the full dataset
val covid_union_MP_1 = fatal_df_MP.union(newSample_Resolved)

// next I will be shuffelling the dataset 
import org.apache.spark.sql.functions.rand
val covid_union_MP = covid_union_MP_1.orderBy(rand())
covid_union_MP.show(20)


// to make sure that we have done the right thing here, let's check the count of each
println(covid_union_MP.filter($"Outcome" === "FATAL").count) // lenght is 4364
println(covid_union_MP.filter($"Outcome" === "RESOLVED").count) // lenght is 4364

// let's change the values of the outcome column so that we have resolved as 1, and fatal as 0

val covid_outcome_labled_MP = covid_union_MP.withColumn("Outcome_Labeled", when($"Outcome" === "RESOLVED", 1).otherwise(0))

// now we will start building the Random Forest machine learning model


// let's cretae an array that has all the features we have in the dataset
val cols_MP = Array("Outbreak Associated","Age Group", "Source of Infection", "Client Gender" , "Ever Hospitalized", "Ever in ICU", "Ever Intubated")

// the following is to index (binning) for the features, to convert it form string to numeric values
val indexer_MP = cols_MP.map { colName =>
   new StringIndexer()
    .setInputCol(colName)
    .setOutputCol(colName + "_I")
    .fit(covid_outcome_labled_MP)
}
// the following pipeline is used to apply the indexer above
val indexer_pipeline = new Pipeline().setStages(indexer_MP)

// now to we will add all the indexed features in a new dataframe
val covid_indexed_MP = indexer_pipeline.fit(covid_outcome_labled_MP).transform(covid_outcome_labled_MP)

// let's cretae an array that has all the indexed features we have in the dataset
val col_I_MP =  Array("Outbreak Associated_I","Age Group_I", "Source of Infection_I", "Client Gender_I" , "Ever Hospitalized_I", "Ever in ICU_I", "Ever Intubated_I")

// now let's do the VectorAssembler to add feature column - we are going to name the output of it as features which will contain all the features values at once to be used in the machine learning model
val vAssembler_MP = new VectorAssembler()
  .setInputCols(col_I_MP)
  .setOutputCol("features")

// now adding the random forest model
val rf_MP = new RandomForestClassifier()
 .setFeaturesCol("features")// this is the assembeled dataset features
 .setLabelCol("Outcome_Labeled") // that's our target
 .setSeed(500) //  seed number

// creating pipline: the pipline is used to set stages and tell it where to start with and end
val pipeline_MP = new Pipeline()
 .setStages(Array(vAssembler_MP, rf_MP))

/*
next up is hyper parameter tunning, the hyper parameter of the random forest are going to be as follows:
maxDepth: this indecates the depth of the tree model, the deeper the model the stronger, but we have to watch this carfully as it might lead to overfitting in our model if it's too deep
maxBins: this indecates the max number of bins which is used for desritizing continious features like Age
impurity: which measures the homogeneity of the labels at the node, we are going to use two impurity features (both for calsiification):
  1- Gini: measures how well the split was in the model, and it helps in building a pure decision tree by determinning which splitter is best
  2- Entropy: it measures the purity of the sub-splits
*/

val paramGrid_MP = new ParamGridBuilder()  
  .addGrid(rf_MP.maxDepth, Array(3, 5, 8))
  .addGrid(rf_MP.impurity, Array("entropy","gini"))
  .addGrid(rf_MP.maxBins, Array(32, 50, 55))
  .build()

// let's build the evaluater: here we are using the MulticlassClassificationEvaluator with the accurecy metric which calculates the percentage of the accurate results between the actuals and predicted 
val evaluator_MP = new MulticlassClassificationEvaluator()
  .setLabelCol("Outcome_Labeled")
  .setPredictionCol("prediction") // our predictions will be stored in this column 
  .setMetricName("accuracy")

// here is the CrossValidator  which will combine everything together in one object, 
//where the pipline has the stages we did above, the evaluter, and the hyper parameter from the paramGrid
// note that we gave the cros validater a number of folds of 3, meaning it will try the hyper parameter 3 times against each other
// ex. maxDepth 3 wiht impurity gini, then max depth 3 with entropy, and so on...
  val cross_validator_MP = new CrossValidator()
  .setEstimator(pipeline_MP)
  .setEvaluator(evaluator_MP)
  .setEstimatorParamMaps(paramGrid_MP)
  .setNumFolds(3)

// splitting teh data into tarining and testing 80% train and 20% testing
val Array(train_data, test_data) = covid_indexed_MP.randomSplit(Array(0.8, 0.2), 500) // the seed number is 500


// next feeding the trainting data into the cross validater object to train the model

val cvModel_MP = cross_validator_MP.fit(train_data)

// follows making the model to predict the test_data

val predictions_MP = cvModel_MP.transform(test_data)

// finally evaluating the model using the evaluator object, to get the accuracy (percentage of trues) 

val accuracy_MP = evaluator_MP.evaluate(predictions_MP)

println("accuracy on test data = " + accuracy_MP)

// after evaluating the model, we ended up with the following accuarcy: 94%

println(predictions_MP.filter($"prediction" === 0 && $"Outcome_Labeled"=== 0).count) // 834
/*
834 / 870 (the number of outcomes 0 ) = 95.86%

The fatality Accuracy is: 834 / 870 (the number of outcomes 0 ) = 95.86%
*/

// ================================================================================================================================================
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
// now we will build the Logistec Regression model:

// adding teh logistec regression model
val lr_MP = new LogisticRegression()
 .setFeaturesCol("features")// this is the assembeled dataset features
 .setLabelCol("Outcome_Labeled") // that's our target


// creating pipline: the pipline is used to set stages and tell it where to start wiht and end, in our example (assembler, then logistec regresssion model)

val pipeline_MP_lr = new Pipeline()
  .setStages(Array(vAssembler_MP, lr_MP))

// We use a ParamGridBuilder to construct a grid of parameters to search over.
//// elasticNet: The elastic net method overcomes the limitations of the LASSO (least absolute shrinkage and selection operator) method
/// The elastic net method improves lasso’s limitations, i.e., where lasso takes a few samples for high dimensional data
val paramGrid_MP_lr = new ParamGridBuilder()
  .addGrid(lr_MP.elasticNetParam, Array(0.0, 0.5, 1.0)) // explained above
  .addGrid(lr_MP.regParam, Array(0.0, 0.1, 0.01))// Set the regularization parameter -- make the data a little bit wider - to avoid overfitting
  .build()

// let's build the evaluater: here we are using the BinaryClassificationEvaluator
// the Receiver Operator Characteristic (ROC) is a metric used to evaluate binary classifications, since the logistec predicts binary events, 
//we are going to use this one. Also notting that this metric measures the area under teh curve to get the  probability.
val evaluator_MP_lr = new BinaryClassificationEvaluator()
  .setLabelCol("Outcome_Labeled")
  .setMetricName("areaUnderROC")

// here is the CrossValidator  which will combine everything together in one object, 
//where the pipline has the stages we did above, the evaluter, and the hyper parameter from the paramGrid
// note that we gave the cros validater a number of folds of 3, meaning it will try the hyper parameter 3 times against each other
val cross_validator_MP_lr = new CrossValidator()
  .setEstimator(pipeline_MP_lr)
  .setEvaluator(evaluator_MP_lr)
  .setEstimatorParamMaps(paramGrid_MP_lr)
  .setNumFolds(3)  // Use 3 folds
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

// next feeding the trainting data into the cross validater object to train teh model

val cvModel_MP_lr = cross_validator_MP_lr.fit(train_data)

// follows making teh model to predict the test_data

val predictions_MP_lr = cvModel_MP_lr.transform(test_data)

// finally evaluating the model usin gthe evaluator object, to get teh areaUnderROC

val accuracy_MP_lr = evaluator_MP_lr.evaluate(predictions_MP_lr)

println("areaUnderROC on test data = " + accuracy_MP_lr)

// after evaluating the model, we ended up with the following areaUnderROC: %95.9

println(predictions_MP_lr.filter($"prediction" === 0 && $"Outcome_Labeled"=== 0).count) // 768

// =======================================================================================================================
import org.apache.spark.ml.classification.{LinearSVC}
// now building the LinearSVC model

val lsvc_MP = new LinearSVC()
 .setFeaturesCol("features")
 .setLabelCol("Outcome_Labeled")

 // creating pipline: the pipline is used to set stages and tell it where to start with and end
val pipeline_MP_SVC = new Pipeline()
 .setStages(Array(vAssembler_MP, lsvc_MP))

 /*
next up is hyper parameter tunning, the hyper parameter of the random forest are going to be as follows:
*/

val paramGrid_MP_SVC = new ParamGridBuilder()
  .addGrid(lsvc_MP.maxIter, Array(10,20,30)) // maximum iterations the model will run on
  .addGrid(lsvc_MP.regParam, Array(0,0.1,0.01)) // Set the regularization parameter -- make the data a little bit wider - to avoid overfitting
  .build()

// here is the CrossValidator  which will combine everything together in one object.
// note that we gave the cros validater a number of folds of 3, meaning it will try the hyper parameter 3 times against each other

 val cross_validator_MP_SVC = new CrossValidator()
 .setEstimator(pipeline_MP_SVC)
 .setEvaluator(evaluator_MP)
 .setEstimatorParamMaps(paramGrid_MP_SVC)
 .setNumFolds(3)

// Train the model on training data
val model_MP_svc = cross_validator_MP_SVC.fit(train_data)

//Test the model on test data
val predictions_MP_svc = model_MP_svc.transform(test_data)

//Evaluate accuracy
val accuracy_MP_svc = evaluator_MP.evaluate(predictions_MP_svc)
println("accuracy on test data = " + accuracy_MP_svc)

// accuracy on test data = %91.9
println(predictions_MP_svc.filter($"prediction" === 0 && $"Outcome_Labeled"=== 0).count) // 765



//====================================================================================================================

val unseenData = Seq(("Outbreak Associated", "30 to 39 Years", "Close Contact", "MALE", "RESOLVED", "Yes", "No", "Yes", 1 ,1.0, 4.0, 4.0, 0.0, 1.0, 0.0, 1.0),
                      ("Sporadic", "20 to 29 Years", "Household Contact", "FEMALE", "FATAL", "No", "No", "No", 0 ,0.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0),
                      ("Outbreak Associated", "20 to 29 Years", "Close Contact", "MALE", "FATAL", "No", "No", "No", 0 ,1.0, 3.0, 4.0, 0.0, 1.0, 1.0, 1.0)
                   ).toDF("Outbreak Associated","Age Group", "Source of Infection","Client Gender", "Outcome",
                   "Ever Hospitalized","Ever in ICU","Ever Intubated","Outcome_Labeled","Outbreak Associated_I",
                   "Age Group_I","Source of Infection_I","Client Gender_I","Ever Hospitalized_I","Ever in ICU_I","Ever Intubated_I")

unseenData.show()



// println(evaluator_MP.evaluate(cvModel_MP.transform(unseenData)))