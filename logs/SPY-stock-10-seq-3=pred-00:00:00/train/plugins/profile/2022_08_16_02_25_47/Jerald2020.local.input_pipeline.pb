	???<??g@???<??g@!???<??g@	|W?/6???|W?/6???!|W?/6???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???<??g@؟??N0??A?#???g@Yi?????rEagerKernelExecute 0*	?p=
׿d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatev?+.?ʹ?!?sh?XN@)??y0H??1h$!??>@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"?A?L??!??&??=@)"?A?L??1??&??=@:Preprocessing2U
Iterator::Model::ParallelMapV2i??TN??!???x0@)i??TN??1???x0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat~?N?Z???!˝??),@)????$???1=??M?X(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip ??c??!ݢ1?T	T@)Xo?
??z?1????'?@:Preprocessing2F
Iterator::Model.:Yj?ߠ?!?t9???3@)?k%t??y?1?E??Q@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?T?-??i?!j?e????)?T?-??i?1j?e????:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapl&?lsc??!LAԦ?O@)?>s֧c?1h???|??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9|W?/6???I?V??-?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	؟??N0??؟??N0??!؟??N0??      ?!       "      ?!       *      ?!       2	?#???g@?#???g@!?#???g@:      ?!       B      ?!       J	i?????i?????!i?????R      ?!       Z	i?????i?????!i?????b      ?!       JCPU_ONLYY|W?/6???b q?V??-?X@