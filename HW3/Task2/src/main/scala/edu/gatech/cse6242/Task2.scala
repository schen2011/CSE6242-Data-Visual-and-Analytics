package edu.gatech.cse6242

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Task2 {
  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("Task2"))
    // read the file
    val file = sc.textFile("hdfs://localhost:8020" + args(0)) // RDD[String]
    // construct two key/value pairs
    val outputSrc = file.map(x => x.split("\t")).map(line => (line(0), line(2).toInt)).filter(_._2 != 1).mapValues(v => v * 0.8).reduceByKey(_+_) // [(String, Int)]

    val outputTgt = file.map(x => x.split("\t")).map(line => (line(1), line(2).toInt)).filter(_._2 != 1).mapValues(v => v * 0.2).reduceByKey(_+_) // [(String, Int)]

    val result = (outputSrc union outputTgt).reduceByKey(_ + _).map(line=>Array(line._1, line._2.toString).mkString("\t")) //[(String, String)]
    result.saveAsTextFile("hdfs://localhost:8020" + args(1))
  }
}
