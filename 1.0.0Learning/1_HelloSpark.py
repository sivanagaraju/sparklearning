from pyspark.sql import *
import sys
from lib.logger import Log4j
from lib.utils import *

if __name__ ==  "__main__":
    conf = get_spark_app_config
    
    spark = SparkSession.builder\
        .appName("Hello Spark")\
        .master("local[2]")\
        .getOrCreate()
        
    """
    data_list = [("Ravi", 28),
                 ("David", 45),
                 ("About", 37)]
    
    df = spark.createDataFrame(data_list).toDF("Name", "Age")
    df.show()
    """
    
    logger = Log4j(spark)
    
    if len(sys.argv) != 2:
        logger.error("Usage: HelloSpark <filename>")
        sys.exit(-1)
        
    logger.info("Startig HelloSpark")
    
    survey_raw_df =  load_survey_df(spark, sys.argv[1])
    partitioned_survey_df = survey_raw_df.repartition(2)
    count_df = count_by_country(partitioned_survey_df)
    count_df.show()
    
    logger.info("Finished HallSpark")
    # breakpoint()
    spark.stop()
    