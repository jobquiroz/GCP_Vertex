  *	?Q?Ֆ?@2f
/Iterator::Root::BatchV2::Shuffle::ParallelMapV2d?a?A
?0@!7up]H@)?a?A
?0@17up]H@:Preprocessing2p
9Iterator::Root::BatchV2::Shuffle::ParallelMapV2::Prefetchd&??| @!XϢ?B8@)&??| @1XϢ?B8@:Preprocessing2?
OIterator::Root::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelInterleaveV4dF??}??@!^?????+@)F??}??@1^?????+@:Preprocessing2?
gIterator::Root::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelInterleaveV4[0]::BigQueryAvroDatasetdut\?@!?h?@)ut\?@1?h?@:Preprocessing2N
Iterator::Root::BatchV2\ A?c?2@!?֝???K@)5B?S???1???6@:Preprocessing2W
 Iterator::Root::BatchV2::Shuffled8,??j1@!???V??I@)'OYMד??1j??k?J@:Preprocessing2E
Iterator::Root ~?{??2@!?[???K@) ??Ud??1qR(????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.