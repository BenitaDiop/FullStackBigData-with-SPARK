
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.optimization.SquaredL2Updater

object LinearRegressionRDD {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val sc = new SparkContext("local[*]", "LinearRegression")
    val trainingLines = sc.textFile("../regression.txt")

    val testingLines = sc.textFile("../regression.txt")
    val trainingData = trainingLines.map(LabeledPoint.parse).cache()
    val testData = testingLines.map(LabeledPoint.parse)
    
    val algorithm = new LinearRegressionWithSGD()
    algorithm.optimizer
      .setNumIterations(100)
      .setStepSize(1.0)
      .setUpdater(new SquaredL2Updater())
      .setRegParam(0.01)
      
      
    val model = algorithm.run(trainingData)
   
    val predictions = model.predict(testData.map(_.features))
    val predictionAndLabel = predictions.zip(testData.map(_.label))

    for (prediction <- predictionAndLabel) {
      println(prediction)
    }
  }
}
