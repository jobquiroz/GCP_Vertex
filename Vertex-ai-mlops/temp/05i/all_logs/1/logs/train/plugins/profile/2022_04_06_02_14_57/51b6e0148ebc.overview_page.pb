?	?X S?%@?X S?%@!?X S?%@	~ٕ()T@~ٕ()T@!~ٕ()T@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?X S?%@:??KTo??Af???~3??Y
???%b!@*	4^?IL?@2]
&Iterator::Model::BatchV2::Shuffle::Mapd?m?2?@!?|.???I@)??G???@1?(?lzD@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetdK ?)UR@!)???,B@)K ?)UR@1)???,B@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd?#?@:??!??0?$@)?#?@:??1??0?$@:Preprocessing2O
Iterator::Model::BatchV20L?
FM!@!w?????N@)S%??RN??1??"???@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled
?]?F@!ZE,?GK@)7???-??1օ?_??@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved8??_?F@!?c?|C@)?]0?????1?q??T??:Preprocessing2F
Iterator::Model??ImP!@!?p?R??N@)%xC8y?1u>?*???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 80.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9~ٕ()T@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	:??KTo??:??KTo??!:??KTo??      ??!       "      ??!       *      ??!       2	f???~3??f???~3??!f???~3??:      ??!       B      ??!       J	
???%b!@
???%b!@!
???%b!@R      ??!       Z	
???%b!@
???%b!@!
???%b!@JCPU_ONLYY~ٕ()T@b Y      Y@qe??:ɂV@"?
host?Your program is HIGHLY input-bound because 80.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?90.0435% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 