	???26*@???26*@!???26*@	Wiz?p9S@Wiz?p9S@!Wiz?p9S@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???26*@'?E'K???A???^@Y???'$@*	MbX	??@2]
&Iterator::Model::BatchV2::Shuffle::Mapd????Q!@!L\B?tL@)???@1?lT??H@:Preprocessing2?
TIterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleave[0]::BigQueryAvroDatasetd?Ѫ?t?@!:?ފNJ@@)?Ѫ?t?@1:?ފNJ@@:Preprocessing2g
0Iterator::Model::BatchV2::Shuffle::Map::Prefetchd??????!?cm?@)??????1?cm?@:Preprocessing2O
Iterator::Model::BatchV2C??$@!T??D?yP@)\?O????1? ?ף@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled?
E???!@!??!??~M@)s?m?B<??1?$? ?? @:Preprocessing2s
<Iterator::Model::BatchV2::Shuffle::Map::Prefetch::Interleavedc?ZB>?@!ih+5oA@)\?v5y??1?%?Id??:Preprocessing2F
Iterator::Model?????$@!?KjeH}P@)+hZbe4??1??S;???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 76.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9Viz?p9S@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	'?E'K???'?E'K???!'?E'K???      ??!       "      ??!       *      ??!       2	???^@???^@!???^@:      ??!       B      ??!       J	???'$@???'$@!???'$@R      ??!       Z	???'$@???'$@!???'$@JCPU_ONLYYViz?p9S@b 