# -*- coding:utf-8 -*-
import os,sys
os.environ['PYSPARK_PYTHON']="xieliming.zip/bin/python"
os.environ['PYSPARK_DRIVER_PYTHON']="xieliming.zip/bin/python"
os.environ['PYSPARK_SUBMIT_ARGS']="--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=xieliming.zip/bin/python " + \
                                  "--archives hdfs://ns1018/user/jd_ad/ads_sz/tmp/xieliming.zip " + \
                                  "pyspark-shell"

# 加载pyspark的环境，在spark_home路径下。保证版本是一致的。
spark_path = r"/software/servers/tyrande/jd_ad/spark" # spark installed folder
os.environ['SPARK_HOME'] = spark_path
sys.path.insert(0, spark_path + "/bin")
sys.path.insert(0, spark_path + "/python/pyspark/")
sys.path.insert(0, spark_path + "/python/lib/pyspark.zip")
sys.path.insert(0, spark_path + "/python/lib/py4j-0.10.6-src.zip")


from pyspark.sql import SparkSession

spark = (SparkSession
    .builder
    .appName("mytest")
    .enableHiveSupport()
    .config("spark.executor.instances", "20")
    .config("spark.executor.memory","10g")
    .config("spark.executor.cores","5")
    .config("spark.driver.memory","4g")
    .config("spark.sql.shuffle.partitions","500")
    .getOrCreate())

#
df=spark.sql("show databases")
df.show(20)
#
df=spark.sql("select model_id,count(1) from app.app_szad_m_dyrec_rerank group by model_id")
df.show(20)

