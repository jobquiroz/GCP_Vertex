	?D??d'@?D??d'@!?D??d'@	F4??S@F4??S@!F4??S@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?D??d'@+?`??ARb???@YƈD?}"@*	?G?z4??@2]
&Iterator::Model::BatchV2::Shuffle::Mapd.9(af@!?o?M?)I@)?cZ??F@1=0??&?C@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd?Mָ@!96??p?A@)?Mָ@196??p?A@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::PrefetchdЀz3j~??!t?D???$@)Ѐz3j~??1t?D???$@:Preprocessing2O
Iterator::Model::BatchV2?eO?c"@!gj???zO@)g?????1?nb?M#@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffledk?) ?#@!?Nxi?J@)??M???18???q?@:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleavedy;?i??@!J?t?~B@)??e?c]??1/?W?G??:Preprocessing2F
Iterator::Models?]??g"@!???V?O@))??q??1=??'7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 79.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9F4??S@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	+?`??+?`??!+?`??      ??!       "      ??!       *      ??!       2	Rb???@Rb???@!Rb???@:      ??!       B      ??!       J	ƈD?}"@ƈD?}"@!ƈD?}"@R      ??!       Z	ƈD?}"@ƈD?}"@!ƈD?}"@JCPU_ONLYYF4??S@b 