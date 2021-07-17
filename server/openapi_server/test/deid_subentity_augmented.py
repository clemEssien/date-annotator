import os
import json
import sparknlp
import sparknlp_jsl
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.base import *
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import monotonically_increasing_id
import time


params = {"spark.driver.memory":"12G",
"spark.kryoserializer.buffer.max":"2000M",
"spark.driver.maxResultSize":"2000M"}

spark = sparknlp_jsl.start(os.environ['JSL_SECRET'],params=params)


input_df = [""" 
  My name is Wolfgang and I live at 1000 Ashland Street 65202 Columbia Missouri United States. My phone number is 572401245
alternate no is 5731111111. Tom's email is wolfgang_02@yahoo.com. My favorite website is
http://www.goal.com. My social security is 886-12-1234 or simpley written as 886121234. I was born on 1st of January 1900. 
Today is 07/09/2021. I saw the doctor at Boone Medical Center.  
"""]
spark_df = spark.createDataFrame([input_df],["text"])



documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

  # Sentence Detector annotator, processes various sentences per line

sentenceDetector = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentence")

# Tokenizer splits words in a relevant format for NLP

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")


word_embeddings = WordEmbeddingsModel.load("nlp_models/embeddings_clinical_en")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

deid_ner = MedicalNerModel.load("nlp_models/ner_deid_subentity_augmented") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")


nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, deid_ner, ner_converter])

input_df = [("""My name is Wolfgang and I live in Columbia Missouri . My phone number is
alternate no is 5731111111. Tom's email is wolfgang_00000@yahoo.com. My website is""" \
"""www.stilt.com. My social security is 886067534. I was born on 1st of January 1900. Today is 07/09/2021.""")]
spark_df = spark.createDataFrame([input_df],["text"])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform((spark_df).toDF("text"))

df = results.toPandas()

json_var = df.to_dict(orient="records")

for record in json_var:
      for chunk in record["ner_chunk"]:
            annotator = chunk.asDict()
            print(annotator['result'], annotator['begin'], annotator['end'],  annotator['metadata']['entity'],annotator['metadata']['confidence'] )