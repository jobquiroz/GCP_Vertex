?	_b,?/)7@_b,?/)7@!_b,?/)7@	,???>?E@,???>?E@!,???>?E@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$_b,?/)7@t34???A???@??)@Y?k%t??#@*	0?$??@2]
&Iterator::Model::BatchV2::Shuffle::Mapd????M@!?]????H@)????q@1f???_C@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd?%?/?@!?ư??fA@)?%?/?@1?ư??fA@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchdճ ??q??!?pN??%@)ճ ??q??1?pN??%@:Preprocessing2O
Iterator::Model::BatchV2ʌ??^?#@!o??k?mO@)S#?3????16?VE??#@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled_9??? @!??ڢzJ@)}@?3i???1X??yn?
@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved?j?a@!n?6V??B@)U1?~????1?bH?@:Preprocessing2F
Iterator::Model??Y?h?#@!?ɩnwO@)????)??1bG??{&??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 43.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9,???>?E@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	t34???t34???!t34???      ??!       "      ??!       *      ??!       2	???@??)@???@??)@!???@??)@:      ??!       B      ??!       J	?k%t??#@?k%t??#@!?k%t??#@R      ??!       Z	?k%t??#@?k%t??#@!?k%t??#@JCPU_ONLYY,???>?E@b Y      Y@q*1s@T@"?
host?Your program is HIGHLY input-bound because 43.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?80.4571% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 