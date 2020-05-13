
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.Vectors

object LinearRegressionDataFrame {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession
      .builder
      .appName("LinearRegressionDF")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "file:///C:/temp") // Necessary to work around a Windows bug in Spark 2.0.0; omit if you're not on Windows.
      .getOrCreate()
    val inputLines = spark.sparkContext.textFile("../regression.txt")
    val data = inputLines.map(_.split(",")).map(x => (x(0).toDouble, Vectors.dense(x(1).toDouble)))
    import spark.implicits._
    val colNames = Seq("label", "features")
    val df = data.toDF(colNames: _*)
    val trainTest = df.randomSplit(Array(0.5, 0.5))
    val trainingDF = trainTest(0)
    val testDF = trainTest(1)
    val lir = new LinearRegression()
      .setRegParam(0.3) // regularization 
      .setElasticNetParam(0.8) // elastic net mixing
      .setMaxIter(100) // max iterations
      .setTol(1E-6) // convergence tolerance
    val model = lir.fit(trainingDF)
    
    val fullPredictions = model.transform(testDF).cache()
   
    val predictionAndLabel = fullPredictions.select("prediction", "label").rdd.map(x => (x.getDouble(0), x.getDouble(1)))
    for (prediction <- predictionAndLabel) {
      println(prediction)
    }

    spark.stop()

  }
}
