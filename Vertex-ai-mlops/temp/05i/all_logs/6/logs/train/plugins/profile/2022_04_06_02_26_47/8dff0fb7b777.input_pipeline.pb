	L?$z}$@L?$z}$@!L?$z}$@	?????OT@?????OT@!?????OT@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$L?$z}$@?2??(??A??9Ϙ??Y?-?R\? @*	?rh???@2]
&Iterator::Model::BatchV2::Shuffle::Mapd??4?@!^??J@)F???j?@1?9+PE@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetdKO?l@!?2?kP+B@)KO?l@1?2?kP+B@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd?A`??b??!jG|&#@)?A`??b??1jG|&#@:Preprocessing2O
Iterator::Model::BatchV2W횐֐ @!?M?
??N@)ʇ?j?j??1?OR?d@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled????N4@!???RK@)???z?2??1W?}??@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved%??CK@!x???B@)3?<Fy???1?t????:Preprocessing2F
Iterator::Model?ȳ? @!?,4??O@)?v????v?1?jz?Uo??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 81.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?????OT@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?2??(???2??(??!?2??(??      ??!       "      ??!       *      ??!       2	??9Ϙ????9Ϙ??!??9Ϙ??:      ??!       B      ??!       J	?-?R\? @?-?R\? @!?-?R\? @R      ??!       Z	?-?R\? @?-?R\? @!?-?R\? @JCPU_ONLYY?????OT@b 