	?+????'@?+????'@!?+????'@	???1[T@???1[T@!???1[T@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?+????'@,?????A??F!?, @Y???2?#@*	i??|?J?@2]
&Iterator::Model::BatchV2::Shuffle::Mapd??N?B @!??\"'K@)?M?M?G@1?J???F@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd??v<@!?/??@@)??v<@1?/??@@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd(?N>=???!1?Gx?x!@)(?N>=???11?Gx?x!@:Preprocessing2O
Iterator::Model::BatchV2?Q??k#@!䙭?0P@)?M?t???1.|?/?2!@:Preprocessing2X
!Iterator::Model::BatchV2::ShuffledY?oC?? @!??e?JL@)ʌ??^???1wc!y??:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved+???+@!i?P??A@)f1???6??1?1?F???:Preprocessing2F
Iterator::Model?}?po#@!K?W?2P@)??-?|?1?:?O}???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 81.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9???1[T@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	,?????,?????!,?????      ??!       "      ??!       *      ??!       2	??F!?, @??F!?, @!??F!?, @:      ??!       B      ??!       J	???2?#@???2?#@!???2?#@R      ??!       Z	???2?#@???2?#@!???2?#@JCPU_ONLYY???1[T@b 