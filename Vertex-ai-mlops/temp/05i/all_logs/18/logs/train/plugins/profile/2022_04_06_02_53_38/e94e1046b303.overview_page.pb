?	??U+3%@??U+3%@!??U+3%@	??(??#T@??(??#T@!??(??#T@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??U+3%@z6?>W[??A??ne???Y?F??!@*	?rh?Mx?@2]
&Iterator::Model::BatchV2::Shuffle::Mapd?-?s?@!!????M@)4?V@1??|?PI@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd8?0C??
@!.&n???;@)8?0C??
@1.&n???;@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd?jH?c???!7䐁?T!@)?jH?c???17䐁?T!@:Preprocessing2O
Iterator::Model::BatchV2z?蹅? @!?c???Q@)??N???1?ӷ/?f @:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled?.n?@!tҤoAO@)? Uܸ??1)e??s?	@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved??7?-@!?<侔;=@)?zk`???1?faW???:Preprocessing2F
Iterator::Model????!@!??F??Q@)??ky?z{?1HZi?㕬?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 80.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9??(??#T@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	z6?>W[??z6?>W[??!z6?>W[??      ??!       "      ??!       *      ??!       2	??ne?????ne???!??ne???:      ??!       B      ??!       J	?F??!@?F??!@!?F??!@R      ??!       Z	?F??!@?F??!@!?F??!@JCPU_ONLYY??(??#T@b Y      Y@q(?i??V@"?
host?Your program is HIGHLY input-bound because 80.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?90.7615% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 