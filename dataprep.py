import bisect
import hashlib
import random
import os
from datetime import datetime
import apache_beam as beam
import shutil
from typing import Any, Dict, List, Text
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import bigquery
import tensorflow as tf


PROJECT_ID = 'skyuk-uk-decis-models-01-dev'
REGION = 'europe-west1'
OUTPUT_DIR = '/home/jupyter/output'
DATASET_NAME = 'uk_stg_product_propensity_models_ic'
TABLE_NAME = 'KIDS_TRAINING_BASE'
QUERY_SAMPLE_RATE = 0.0001
MAX_INT64 = '0x7FFFFFFFFFFFFFFF'


@beam.ptransform_fn
def _ReadFromBigQuery(pipeline, query):
    """Simple pipeline to read the table from BigQuery via the specified query"""
    return (
        pipeline | 'QueryTable' >> beam.io.Read(beam.io.BigQuerySource(
            query=query, use_standard_sql=True)))


@beam.ptransform_fn
def _SplitData(pcoll, test=0.3):
    """Pipeline to split the input bigquery query into two PCollections: train and test.
    The train and test ratio is passed in as explicit side inputs, the output is a tuple of the
    two corresponding PCollections
    """
    if test < 0 or test > 1:
        raise ValueError('Invalid test size')
    
    class _Split(beam.DoFn):
        def process(self, element):
            if element['rand_num'] < test:
                yield beam.pvalue.TaggedOutput('test', element)
            else:
                yield beam.pvalue.TaggedOutput('train', element)
        
    split = (
        pcoll | 'Split' >> beam.ParDo(_Split()).with_outputs(
            'train',
            'test'))
    
    result = {}
    result['train'] = split['train']
    result['test'] = split['test']

    return result


@beam.ptransform_fn
def _SeperateAndUndersample(pcoll, want_ratio=0.1):
    """Undersample the majority class"""
        
    percentage = (pcoll
        | 'ReduceToClass' >> beam.Map(lambda x: 1.0 * x['Target'])
        | beam.CombineGlobally(beam.combiners.MeanCombineFn()))
    
    class _Seperate(beam.DoFn):
        """DoFn that seperates positive from negative"""
        def process(self, element):
            if element['Target'] == 1:
                yield beam.pvalue.TaggedOutput('minority', element)
            else:
                yield beam.pvalue.TaggedOutput('majority', element)
                
    class _Undersample(beam.DoFn):
        """DoFn that undersamples the input pcollection"""
        def process(self, element, orig_ratio):
            sample_ratio = orig_ratio/want_ratio
            r = random.random()
            
            if r <= sample_ratio:
                yield element
            else:
                return
            
    seperate = (pcoll
            | 'SeperateData' >> beam.ParDo(_Seperate()).with_outputs(
                'majority',
                'minority'))
    
    majority, minority = seperate['majority'], seperate['minority']
    
    undersampled_majority = (majority
            | 'UndersampleMajority' >> beam.ParDo(
                _Undersample(),
                orig_ratio=beam.pvalue.AsSingleton(percentage)))
    
    merged = ((undersampled_majority, minority)
            | 'MergePCollections' >> beam.Flatten())
    
    return merged


@beam.ptransform_fn
def _WriteSplit(example_split, split_name):

    table_schema = {'fields' : [
        {'name': 'h_age_fine', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'DTV_contract_segment', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'SGE_Active', 'type': 'INTEGER', 'mode': 'NULLABLE'},
        {'name': 'Target', 'type': 'INTEGER', 'mode': 'NULLABLE'},
        {'name': 'rand_num', 'type': 'FLOAT', 'mode': 'NULLABLE'}
    ]}

    table_spec = '{}:{}.NNM01_beam_{}'.format(PROJECT_ID, DATASET_NAME, split_name)

    return (example_split
        | 'Shuffle' >> beam.transforms.Reshuffle()
        | 'Write' >> beam.io.WriteToBigQuery(
            table_spec,
            schema = table_schema,
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )


def GenerateExamplesByBeam(pipeline, query):
    """
    Reads, splits, oversamples, and serializes the input.
    """

    example_splits = (pipeline
        | 'ReadFromBigQuery' >> _ReadFromBigQuery(query)
        | 'SplitData' >> _SplitData())

    train = example_splits['train']

    undersampled_train = (train
        | 'UndersampleTrain' >> _SeperateAndUndersample())

    example_splits['train'] = undersampled_train

#     serialized_train = (example_splits['train']
#         | 'ToSerializedTFTrain' >> _InputToSerializedExample(
#           input_to_example,
#           query))

#     serialized_test = (example_splits['test']
#         | 'ToSerializedTFTest' >> _InputToSerializedExample(
#           input_to_example,
#           query))

    return example_splits


def main():
    query = """
        SELECT h_age_fine,
               DTV_contract_segment,
               SGE_Active,
               Target,
               rand_num
        FROM `{project}.{dataset}.{table}`
        WHERE (ABS(FARM_FINGERPRINT(account_number)) / {max_int64})
        < {query_sample_rate}
    """.format(project=PROJECT_ID,
            dataset=DATASET_NAME,
            table=TABLE_NAME,
            max_int64=MAX_INT64,
            query_sample_rate=QUERY_SAMPLE_RATE)
    
    options = {
        "staging_location": os.path.join(OUTPUT_DIR, "tmp", "staging"),
        "temp_location": os.path.join(OUTPUT_DIR, "tmp"),
        "job_name": 'local_job_{}'.format(datetime.now().strftime('%Y%m%d%H%M%S')),
        "project": PROJECT_ID,
        "teardown_policy": "TEARDOWN_ALWAYS",
        "save_main_session": False,
        'region': REGION,
        "requirements_file": "requirements.txt"
    }

    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)
    
    
    with beam.Pipeline('DirectRunner', options=pipeline_options) as p:

        example_splits = GenerateExamplesByBeam(p, query)

        for k, v in example_splits.items():
            _ = (v | k >> _WriteSplit(k))
    
    
if __name__ == "__main__":
    main()
    


