	ő"??=@ő"??=@!ő"??=@	?~????U@?~????U@!?~????U@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ő"??=@????je??Ac??Ր?@Y??1??9@*	?S???G?@2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd?2??((@!U????B@)?2??((@1U????B@:Preprocessing2]
&Iterator::Model::BatchV2::Shuffle::Mapd1%??e?7@!???aR@)v()??&@1?v5}??A@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd?6????@!?>~?V?2@)?6????@1?>~?V?2@:Preprocessing2O
Iterator::Model::BatchV2v()??9@!-s?Y?T@)?*?w????1`??SD@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled??q6	8@!ׂ?v?R@)T?{F"4??1`p?єZ??:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved-x?W?v@!?[?^??3@)?D??]??1?!$]??:Preprocessing2F
Iterator::Model???p?9@!0?O?T@)?????{?1/6`??ؕ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 86.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?~????U@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????je??????je??!????je??      ??!       "      ??!       *      ??!       2	c??Ր?@c??Ր?@!c??Ր?@:      ??!       B      ??!       J	??1??9@??1??9@!??1??9@R      ??!       Z	??1??9@??1??9@!??1??9@JCPU_ONLYY?~????U@b 