?	???r>&@???r>&@!???r>&@	3ֽ?S@3ֽ?S@!3ֽ?S@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???r>&@8???LM??A??B?] @Yu?????!@*	 ?x?F??@2]
&Iterator::Model::BatchV2::Shuffle::Mapd?? ?m?@!?@?<J@)N^d~?@1O????F@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd???x?@!????0A@)???x?@1????0A@:Preprocessing2O
Iterator::Model::BatchV2?"?-??!@!?I??"?O@)rl=Cx??1'??E"?!@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::PrefetchdHm??~??!VۏD:@)Hm??~??1VۏD:@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffledo)狽G@!W? ZtK@)??g?????1?Y1?1@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved_
?]?@!1S?+`B@)??_vO??1+u??}:??:Preprocessing2F
Iterator::ModelC??g?!@!ά<ԟ?O@)6!?1脀?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 79.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no93ֽ?S@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	8???LM??8???LM??!8???LM??      ??!       "      ??!       *      ??!       2	??B?] @??B?] @!??B?] @:      ??!       B      ??!       J	u?????!@u?????!@!u?????!@R      ??!       Z	u?????!@u?????!@!u?????!@JCPU_ONLYY3ֽ?S@b Y      Y@q??4??)W@"?
host?Your program is HIGHLY input-bound because 79.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?92.6493% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 