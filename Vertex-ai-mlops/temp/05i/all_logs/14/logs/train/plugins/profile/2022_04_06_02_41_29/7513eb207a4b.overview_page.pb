?	F$a??6@F$a??6@!F$a??6@	_?+??U@_?+??U@!_?+??U@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$F$a??6@??|@?3??AB?v???@Y?6S!	4@*	?Z?J?@2]
&Iterator::Model::BatchV2::Shuffle::Mapds.?UeW2@!_??nR@)+?m?1@1??uS#Q@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd??0a?@!??e??2@)??0a?@1??e??2@:Preprocessing2O
Iterator::Model::BatchV2??};?3@!??(+?T@)?d??~???17?6߽@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd75?|Ν??!1	`'?@)75?|Ν??11	`'?@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled??)??2@!?xE9??R@)??2????1??V????:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved{??@!Jo???3@)???????1??+?;,??:Preprocessing2F
Iterator::Model'?|??3@!n8䚟T@)????oa}?1#8=?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 87.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9_?+??U@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??|@?3????|@?3??!??|@?3??      ??!       "      ??!       *      ??!       2	B?v???@B?v???@!B?v???@:      ??!       B      ??!       J	?6S!	4@?6S!	4@!?6S!	4@R      ??!       Z	?6S!	4@?6S!	4@!?6S!	4@JCPU_ONLYY_?+??U@b Y      Y@q??.??T@"?
host?Your program is HIGHLY input-bound because 87.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?83.7301% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 