# test_dataset
I have a problem with this simple code based on distributed tensorflow.
Given the directory located to the path "/home/simoneneurone/Documents/images/train/", i want to create a Dataset tensorflow
object (tf.Data.Dataset), apply preprocessing operations (subtract a mean and a resize on all the images in the folder),
then create an iterator and consume it. When i run the code, i get the following error:

0
Loading filenames...
Build dataset ...
WARNING:tensorflow:From /home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/ops/control_flow_ops.py:3632: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
2019-05-06 17:23:30.862746: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-05-06 17:23:30.884302: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 4008000000 Hz
2019-05-06 17:23:30.884599: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x562e54a0cfe0 executing computations on platform Host. Devices:
2019-05-06 17:23:30.884613: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-05-06 17:23:30.885528: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:252] Initialize GrpcChannelCache for job ps -> {0 -> 172.16.69.190:2222}
2019-05-06 17:23:30.885540: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:252] Initialize GrpcChannelCache for job worker -> {0 -> localhost:2223}
2019-05-06 17:23:30.886122: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:391] Started server with target: grpc://localhost:2223
2019-05-06 17:23:30.897513: I tensorflow/core/distributed_runtime/master_session.cc:1192] Start master session 74b1f33369350af9 with config: 
Coming......
Traceback (most recent call last):
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1334, in _do_call
    return fn(*args)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1319, in _run_fn
    options, feed_dict, fetch_list, target_list, run_metadata)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1407, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.NotFoundError: /home/simoneneurone/Documents/images/train/goku/goku1.jpeg; No such file or directory
	 [[{{node ReadFile}}]]
	 [[{{node IteratorGetNext}}]]
	 [[{{node IteratorGetNext}}]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "test.py", line 149, in <module>
    main()
  File "test.py", line 123, in main
    im, ll = sess.run([images, labels])
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 676, in run
    run_metadata=run_metadata)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1171, in run
    run_metadata=run_metadata)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1270, in run
    raise six.reraise(*original_exc_info)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/six.py", line 693, in reraise
    raise value
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1255, in run
    return self._sess.run(*args, **kwargs)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1327, in run
    run_metadata=run_metadata)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1091, in run
    return self._sess.run(*args, **kwargs)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 929, in run
    run_metadata_ptr)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1152, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1328, in _do_run
    run_metadata)
  File "/home/simoneneurone/anaconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1348, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.NotFoundError: /home/simoneneurone/Documents/images/train/goku/goku1.jpeg; No such file or directory
	 [[{{node ReadFile}}]]
	 [[node IteratorGetNext (defined at test.py:89) ]]
	 [[node IteratorGetNext (defined at test.py:89) ]]

In order to run the ps:
python distribution.py --job_name ps --task_index 0

In order to run the worker:
python distribution.py --job_name worker --task_index 0

I guess that the problem is related to MonitoredTrainingSession, that is usefull in order to restore training from checkpoints, when we want to distribute the training on a cluster. In fact, if we run the same code replacing MonitoredTrainingSession with a simple session, the code works fine. 
Can you help me, please? i will really appreciate and thanks in advance.
