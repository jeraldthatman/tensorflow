	????g@????g@!????g@	o????H??o????H??!o????H??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:????g@?p?a?F@A/??(g@Yv??$????rEagerKernelExecute 0*	*??Οv@2U
Iterator::Model::ParallelMapV2.?_x%???!`D??p&Q@).?_x%???1`D??p&Q@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateE???V	??!7a?40@)*???P???1ϙ 'y*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?%:?,B??!1r?Ȳ?"@)????켝?1??/?? @:Preprocessing2F
Iterator::Model˟o???!J?f??Q@)?? n/??1M}y?\?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicej?t???!??5??@)j?t???1??5??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?4S??!ؾ?e2h<@)l???C6??1ĺ???~@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJ?E?s?!D?ؠ??)J?E?s?1D?ؠ??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?P??dV??!h?{
??0@)ܸ????d?1?-wnv??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9o????H??I7?Rn?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?p?a?F@?p?a?F@!?p?a?F@      ?!       "      ?!       *      ?!       2	/??(g@/??(g@!/??(g@:      ?!       B      ?!       J	v??$????v??$????!v??$????R      ?!       Z	v??$????v??$????!v??$????b      ?!       JCPU_ONLYYo????H??b q7?Rn?X@