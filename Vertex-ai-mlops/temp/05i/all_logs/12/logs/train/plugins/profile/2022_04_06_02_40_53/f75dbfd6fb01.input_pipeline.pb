	jj?Z&@jj?Z&@!jj?Z&@	?G?A?oS@?G?A?oS@!?G?A?oS@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$jj?Z&@f??Os??AϽ?K??@YD?|?&!@*㥛?p??@)      ?=2]
&Iterator::Model::BatchV2::Shuffle::Mapd??*lX@!??Ou?J@)?????@1??????F@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetdG9?M?!@!$??Ӑ<@)G9?M?!@1$??Ӑ<@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd?@??h??!?@$o? @)?@??h??1?@$o? @:Preprocessing2O
Iterator::Model::BatchV2?P??!@!c???.P@)H??Q???1:-r?X?@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved????/?@!]?g.?A@)UO?}??1?Oh?#?@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffledޯ|??@!??&kL@)?E~???1?#???@:Preprocessing2F
Iterator::Modelnڌ?!@!x???1P@)?Z'.?+??1"???z???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 77.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?G?A?oS@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	f??Os??f??Os??!f??Os??      ??!       "      ??!       *      ??!       2	Ͻ?K??@Ͻ?K??@!Ͻ?K??@:      ??!       B      ??!       J	D?|?&!@D?|?&!@!D?|?&!@R      ??!       Z	D?|?&!@D?|?&!@!D?|?&!@JCPU_ONLYY?G?A?oS@b 