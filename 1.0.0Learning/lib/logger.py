class Log4j(object):
    def __init__(self, spark) -> None:
        root_class = "guru.learningjournal.spark.examples"
        conf = spark.sparkContext.getConf()
        app_name = conf.get("spark.app.name")
        log4j = spark._jvm.org.apache.log4j  # import log4j library  provided by spark
        self.logger = log4j.LogManager.getLogger(root_class + '.' + app_name)
    
    def warn(self, message):
        self.logger.warn(message)
    
    def info(self, message):
        self.logger.info(message)
        
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)