?	k??)@k??)@!k??)@	?mo?+T@?mo?+T@!?mo?+T@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$k??)@??[<????A?o?DI?@Y1C?? .$@*	?v????@2]
&Iterator::Model::BatchV2::Shuffle::Mapd/????% @!???7}-K@)?P???j@11J*v??G@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetdHj?drz@!?V?*??@)Hj?drz@1?V?*??@:Preprocessing2O
Iterator::Model::BatchV2T?? $@!??+???P@)!yv9??1/??VY?#@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd?+f????!m?@)?+f????1m?@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled1???*!@!튟?L@)#???R??1??y,?x@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved0???s;@!?RU I/@@)?<??- ??1I?d`?M??:Preprocessing2F
Iterator::Model?0DN_$@!?V?o[?P@)Ot]?????1?{?ަ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 80.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?mo?+T@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??[<??????[<????!??[<????      ??!       "      ??!       *      ??!       2	?o?DI?@?o?DI?@!?o?DI?@:      ??!       B      ??!       J	1C?? .$@1C?? .$@!1C?? .$@R      ??!       Z	1C?? .$@1C?? .$@!1C?? .$@JCPU_ONLYY?mo?+T@b Y      Y@q???m8jV@"?
host?Your program is HIGHLY input-bound because 80.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?89.6597% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 