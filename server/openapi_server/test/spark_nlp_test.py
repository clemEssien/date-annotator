from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from pyspark.ml import Pipeline
from sparknlp.base import LightPipeline

from sparknlp import Finisher
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline

spark = sparknlp.start()

finisher = Finisher().setInputCols(["token", "pos"])
explain_pipeline_model = PretrainedPipeline("explain_document_ml").model

pipeline = Pipeline() \
    .setStages([
        explain_pipeline_model,
        finisher
        ])


sentences = [
    ['Hello, this is an example sentence'],
    ['And this is a second sentence.']
]
data = spark.createDataFrame(sentences).toDF("text")

model = pipeline.fit(data)
annotations_finished_df = model.transform(data)


print(annotations_finished_df.select('finished_token').show(truncate=False))

