	y ?HW-@y ?HW-@!y ?HW-@	A??7hP@A??7hP@!A??7hP@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$y ?HW-@4?27?@A?ю~?@Y?^'?eA#@*	L7?A???@2]
&Iterator::Model::BatchV2::Shuffle::Mapd?+??yP@!?`?V??L@)Uj?@+P@1a[?%H@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetdӥI*?@!?efݹI<@)ӥI*?@1?efݹI<@:Preprocessing2O
Iterator::Model::BatchV2??H?(#@!??fx?Q@)??%?"??1aH??_e#@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::PrefetchdS??:??!????["@)S??:??1????["@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled?B?Y?? @!R???ON@)?U?p??1??c?-	@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved?	F?"@!?l7?}?=@)??f*?#??1?m=><??:Preprocessing2F
Iterator::Modelj?WV?,#@!?$????Q@)dyW=`??1???ɣ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 65.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t16.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9A??7hP@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	4?27?@4?27?@!4?27?@      ??!       "      ??!       *      ??!       2	?ю~?@?ю~?@!?ю~?@:      ??!       B      ??!       J	?^'?eA#@?^'?eA#@!?^'?eA#@R      ??!       Z	?^'?eA#@?^'?eA#@!?^'?eA#@JCPU_ONLYYA??7hP@b 