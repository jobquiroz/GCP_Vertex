	??qo&'@??qo&'@!??qo&'@	?a^?5?S@?a^?5?S@!?a^?5?S@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??qo&'@; ??^E??AˡE???@Y?j???4"@*	?G?z??@2]
&Iterator::Model::BatchV2::Shuffle::Mapd????;@!???t?M@)?_#I^@10?A??G@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd8h??@!?d)Jp;@)8h??@1?d)Jp;@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchdj?֍w??!?[8?}'@)j?֍w??1?[8?}'@:Preprocessing2O
Iterator::Model::BatchV2?'c|?"@!????Q@)L7?A`%??1Nj???? @:Preprocessing2X
!Iterator::Model::BatchV2::Shuffledd???@!??Α?fO@)?F??R^??1G??J?
@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved??s??2@!Qn????<@)?v???1D?զ?9??:Preprocessing2F
Iterator::Model??ڧ?!"@!l??Q@)}<?ݭ,??1R0ɲ????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 78.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?a^?5?S@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	; ??^E??; ??^E??!; ??^E??      ??!       "      ??!       *      ??!       2	ˡE???@ˡE???@!ˡE???@:      ??!       B      ??!       J	?j???4"@?j???4"@!?j???4"@R      ??!       Z	?j???4"@?j???4"@!?j???4"@JCPU_ONLYY?a^?5?S@b 