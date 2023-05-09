import configparser

from pyspark import SparkConf

def load_survey_df(spark, data_file):
    return spark.read\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .csv(data_file)


def count_by_country(survey_df):
    return survey_df.filter("Age < 40")\
        .select("Age", "Gender", "Country", "state")\
        .groupby("Country")\
        .count()


def get_spark_app_config():
    spark_conf = SparkConf()
    config = configparser.ConfigParser()
    config.read("spark.conf")
    
    for (key, value) in config.items("SPARK_APP_COFIG"):
        spark_conf.set(key, value)
    return spark_conf


