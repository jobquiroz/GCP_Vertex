	&?\R??1@&?\R??1@!&?\R??1@	?????IT@?????IT@!?????IT@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$&?\R??1@??!????A?$Ί??@Y?b?=?,@*	x?&1??@2]
&Iterator::Model::BatchV2::Shuffle::Mapd?
?H?d(@!??XD?M@)hA(??&@1w?:j?J@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd??ǘ?V@![_?Fs?<@)??ǘ?V@1[_?Fs?<@:Preprocessing2O
Iterator::Model::BatchV2.=??Ɍ,@!?\F?}}Q@)?l??p_??1??????@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd?IF?????!?"	??@)?IF?????1?"	??@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled?O??ۀ)@!?ob?O@)]Ot]????1W?s??@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleaved?^???x@!/+???=@)7?7M???1K??-2??:Preprocessing2F
Iterator::Model_??x??,@!45z?Q@) ?Ȓ9???1??6>???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 81.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?????IT@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??!??????!????!??!????      ??!       "      ??!       *      ??!       2	?$Ί??@?$Ί??@!?$Ί??@:      ??!       B      ??!       J	?b?=?,@?b?=?,@!?b?=?,@R      ??!       Z	?b?=?,@?b?=?,@!?b?=?,@JCPU_ONLYY?????IT@b 