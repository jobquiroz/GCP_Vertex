	;ŪA?c&@;ŪA?c&@!;ŪA?c&@	`?+?dS@`?+?dS@!`?+?dS@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$;ŪA?c&@>Y1\ ??A???qa@Y?Zd;!@*	 ??Q???@2]
&Iterator::Model::BatchV2::Shuffle::Mapdq;4,F?@!??^]Y?L@)??֪?@1?y±DH@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd???jdw@!.???9<@)???jdw@1.???9<@:Preprocessing2O
Iterator::Model::BatchV2(}!??!@!???/?Q@)?^??x???1T?L@?,"@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd???Xm???!@k?k?v!@)???Xm???1@k?k?v!@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled?"????@!???<uN@)?r.?Ue??1??F+.@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::InterleavedJ?o	?@!~-????=@)z?蹅??1?$NOA??:Preprocessing2F
Iterator::Model_9??!@!?t? ??Q@)?,^,??1?g̩????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 76.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9a?+?dS@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	>Y1\ ??>Y1\ ??!>Y1\ ??      ??!       "      ??!       *      ??!       2	???qa@???qa@!???qa@:      ??!       B      ??!       J	?Zd;!@?Zd;!@!?Zd;!@R      ??!       Z	?Zd;!@?Zd;!@!?Zd;!@JCPU_ONLYYa?+?dS@b 