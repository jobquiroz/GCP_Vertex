?	K?*nT(@K?*nT(@!K?*nT(@	???S@???S@!???S@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$K?*nT(@??s?????Aj?t??@Y?L?n #@*	??C???@2]
&Iterator::Model::BatchV2::Shuffle::Mapd ????@!???+AK@)????Y@1??M??G@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd?@gҦ?@!?~???@@)?@gҦ?@1?~???@@:Preprocessing2O
Iterator::Model::BatchV2Z-??D?"@!???ޑ!P@)ur???w??1?<?KԌ@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::PrefetchdI,)w????! ?x??b@)I,)w????1 ?x??b@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled__?R? @!??e4?QL@)k{?????15;8?@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved}??AѼ@!N??]??A@)?_??ME??1???????:Preprocessing2F
Iterator::Model)?k{??"@!?	#Q?$P@)hyܝ?{?1?z?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 78.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9???S@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??s???????s?????!??s?????      ??!       "      ??!       *      ??!       2	j?t??@j?t??@!j?t??@:      ??!       B      ??!       J	?L?n #@?L?n #@!?L?n #@R      ??!       Z	?L?n #@?L?n #@!?L?n #@JCPU_ONLYY???S@b Y      Y@q??P?iV@"?
host?Your program is HIGHLY input-bound because 78.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?88.2721% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 