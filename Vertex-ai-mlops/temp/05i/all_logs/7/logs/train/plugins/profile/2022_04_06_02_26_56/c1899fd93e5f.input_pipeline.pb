	:z?ަ?*@:z?ަ?*@!:z?ަ?*@	?#?m??R@?#?m??R@!?#?m??R@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$:z?ަ?*@|'f????A? ?=?@Y???ƹ#@*	e;?O???@2]
&Iterator::Model::BatchV2::Shuffle::MapdP@??/@!c???kI@)N?q??@1?+W1?E@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd??AA)*@!"?@@A@)??AA)*@1"?@@A@:Preprocessing2O
Iterator::Model::BatchV2?E_A??#@!?e?r??O@)??f???1????4%@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd:?6???!?fK@):?6???1?fK@:Preprocessing2X
!Iterator::Model::BatchV2::ShuffledN?t"Y @!???w?J@)???R%??1???&K?@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleavedɐc?@!?p?B@)%??ID???1??ۉ?m??:Preprocessing2F
Iterator::Model?l\??#@!???R?O@)u/3l???1^}ݖ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 75.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?#?m??R@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|'f????|'f????!|'f????      ??!       "      ??!       *      ??!       2	? ?=?@? ?=?@!? ?=?@:      ??!       B      ??!       J	???ƹ#@???ƹ#@!???ƹ#@R      ??!       Z	???ƹ#@???ƹ#@!???ƹ#@JCPU_ONLYY?#?m??R@b 