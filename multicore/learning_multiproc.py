
'''
Learning this from http://broadcast.oreilly.com/2009/04/pymotw-multiprocessing-part-1.html
'''


'''
Simple example of just starting processes
'''
# import multiprocessing

# def worker(num):
# 	"""thread worker function"""
# 	print 'Worker:', num
# 	return

# if __name__ == '__main__':
# 	jobs = []
# 	for i in range(5):
# 		p = multiprocessing.Process(target=worker, args=(i,))
# 		jobs.append(p)
# 		p.start()


'''
Giving names to processes and making them sleep
'''
# import multiprocessing
# import time

# def worker():
# 	name = multiprocessing.current_process().name
# 	print name, 'Starting'
# 	time.sleep(4)
# 	print name, 'Exiting'

# def my_service():
# 	name = multiprocessing.current_process().name
# 	print name, 'Starting'
# 	time.sleep(3)
# 	print name, 'Exiting'

# if __name__ == '__main__':
# 	service = multiprocessing.Process(name='my_service', target=my_service)
# 	worker_1 = multiprocessing.Process(name='worker 1', target=worker)
# 	worker_2 = multiprocessing.Process(target=worker) # use default name

# 	worker_1.start()
# 	worker_2.start()
# 	service.start()

'''
Daemon process. Main doesnt wait for daemon process to finish.
The daemon process is terminated before the main program exits, 
to avoid leaving orphaned processes running.
'''
# import multiprocessing
# import time
# import sys

# def daemon():
#     print 'Starting:', multiprocessing.current_process().name
#     time.sleep(2)
#     print 'Exiting :', multiprocessing.current_process().name

# def non_daemon():
#     print 'Starting:', multiprocessing.current_process().name
#     print 'Exiting :', multiprocessing.current_process().name

# if __name__ == '__main__':
#     d = multiprocessing.Process(name='daemon', target=daemon)
#     d.daemon = True

#     n = multiprocessing.Process(name='non-daemon', target=non_daemon)
#     n.daemon = False

#     d.start()
#     time.sleep(1)
#     n.start()

'''
To wait until a process has completed its work and exited, 
use the join() method.
Since we wait for the daemon to exit using join(), 
we do see its Exiting message.
'''
# import multiprocessing
# import time
# import sys

# def daemon():
#     print 'Starting:', multiprocessing.current_process().name
#     time.sleep(2)
#     print 'Exiting :', multiprocessing.current_process().name

# def non_daemon():
#     print 'Starting:', multiprocessing.current_process().name
#     print 'Exiting :', multiprocessing.current_process().name

# if __name__ == '__main__':
#     d = multiprocessing.Process(name='daemon', target=daemon)
#     d.daemon = True

#     n = multiprocessing.Process(name='non-daemon', target=non_daemon)
#     n.daemon = False

#     d.start()
#     time.sleep(1)
#     n.start()

#     d.join()
#     n.join()

'''
 Pass a timeout argument (a float representing the number of seconds 
 to wait for the process to become inactive). 
 If the process does not complete within the timeout period, join() returns anyway.
'''
# import multiprocessing
# import time
# import sys

# def daemon():
#     print 'Starting:', multiprocessing.current_process().name
#     time.sleep(2)
#     print 'Exiting :', multiprocessing.current_process().name

# def non_daemon():
#     print 'Starting:', multiprocessing.current_process().name
#     print 'Exiting :', multiprocessing.current_process().name

# if __name__ == '__main__':
#     d = multiprocessing.Process(name='daemon', target=daemon)
#     d.daemon = True

#     n = multiprocessing.Process(name='non-daemon', target=non_daemon)
#     n.daemon = False

#     d.start()
#     n.start()

#     d.join(1)
#     print 'd.is_alive()', d.is_alive()
#     n.join()

'''
Calling terminate() on a process object kills the child process.
join() the process after terminating it in order to give the 
background machinery time to update the status of the object to 
reflect the termination.
'''
# import multiprocessing
# import time

# def slow_worker():
#     print 'Starting worker'
#     time.sleep(0.1)
#     print 'Finished worker'

# if __name__ == '__main__':
#     p = multiprocessing.Process(target=slow_worker)
#     print 'BEFORE:', p, p.is_alive()
	
#     p.start()
#     print 'DURING:', p, p.is_alive()
	
#     p.terminate()
#     print 'TERMINATED:', p, p.is_alive()

#     p.join()
#     print 'JOINED:', p, p.is_alive()


'''
The status code produced when the 
process exits can be accessed via the exitcode attribute.
'''
# import multiprocessing
# import sys
# import time

# def exit_error():
#     sys.exit(1)

# def exit_ok():
#     return

# def return_value():
#     return 1

# def raises():
#     raise RuntimeError('There was an error!')

# def terminated():
#     time.sleep(3)

# if __name__ == '__main__':
#     jobs = []
#     for f in [exit_error, exit_ok, return_value, raises, terminated]:
#         print 'Starting process for', f.func_name
#         j = multiprocessing.Process(target=f, name=f.func_name)
#         jobs.append(j)
#         j.start()
		
#     jobs[-1].terminate()

#     for j in jobs:
#         j.join()
#         print '%s.exitcode = %s' % (j.name, j.exitcode)

'''
Logging
'''
# import multiprocessing
# import logging
# import sys

# def worker():
#     print 'Doing some work'
#     sys.stdout.flush()

# if __name__ == '__main__':
#     multiprocessing.log_to_stderr(logging.DEBUG)
#     p = multiprocessing.Process(target=worker)
#     p.start()
#     p.join()


'''
Subclass process
'''
# import multiprocessing

# class Worker(multiprocessing.Process):

#     def run(self):
#         print 'In %s' % self.name
#         return

# if __name__ == '__main__':
#     jobs = []
#     for i in range(5):
#         p = Worker()
#         jobs.append(p)
#         p.start()
#     for j in jobs:
#         j.join()

'''
Passing Messages to Processes using a queue
i found that it waits until there is something in the queue
'''
# import multiprocessing
# import time

# class MyFancyClass(object):
	
# 	def __init__(self, name):
# 		self.name = name
	
# 	def do_something(self):
# 		proc_name = multiprocessing.current_process().name
# 		print 'Doing something fancy in %s for %s!' % (proc_name, self.name)


# def worker(q):
# 	print '2'
# 	#waits here until there is something in the queue
# 	obj = q.get()
# 	print '3'
# 	obj.do_something()
# 	print '4'


# if __name__ == '__main__':
# 	queue = multiprocessing.Queue()

# 	p = multiprocessing.Process(target=worker, args=(queue,))
# 	p.start()

# 	print '1'
	
# 	time.sleep(2)

# 	print '6'

# 	queue.put(MyFancyClass('Fancy Dan'))

# 	print '5'
	
# 	# Wait for the worker to finish
# 	queue.close()
# 	queue.join_thread()
# 	p.join()


'''
The poison pill technique is used to stop the workers. 
After setting up the real tasks, the main program adds one stop
value per worker to the job queue. 
When a worker encounters the special value, it breaks out of 
its processing loop. 
'''

# import multiprocessing
# import time

# class Consumer(multiprocessing.Process):
	
#     def __init__(self, task_queue, result_queue):
#         multiprocessing.Process.__init__(self)
#         self.task_queue = task_queue
#         self.result_queue = result_queue

#     def run(self):
#         proc_name = self.name
#         while True:
#             next_task = self.task_queue.get()
#             if next_task is None:
#                 # Poison pill means we should exit
#                 print '%s: Exiting' % proc_name
#                 break
#             print '%s: %s' % (proc_name, next_task)
#             answer = next_task()
#             self.result_queue.put(answer)
#         return


# class Task(object):
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#     def __call__(self):
#         time.sleep(0.1) # pretend to take some time to do our work
#         return '%s * %s = %s' % (self.a, self.b, self.a * self.b)
#     def __str__(self):
#         return '%s * %s' % (self.a, self.b)


# if __name__ == '__main__':
#     # Establish communication queues
#     tasks = multiprocessing.Queue()
#     results = multiprocessing.Queue()
	
#     # Start consumers
#     num_consumers = multiprocessing.cpu_count()
#     print 'Creating %d consumers' % num_consumers
#     consumers = [ Consumer(tasks, results)
#                   for i in xrange(num_consumers) ]
#     for w in consumers:
#         w.start()
	
#     # Enqueue jobs
#     num_jobs = 100
#     num_jobs_dont_change = 100
#     for i in xrange(num_jobs):
#         tasks.put(Task(i, i))
	
#     # Add a poison pill for each consumer
#     for i in xrange(num_consumers):
#         tasks.put(None)
	
#     # Start printing results
#     while num_jobs:
#         result = results.get()
#         print 'Result:', result
#         print str(num_jobs_dont_change - num_jobs) + ' complete'
#         num_jobs -= 1

'''
events, wait till their set
'''

# import multiprocessing
# import time

# def wait_for_event(e):
#     """Wait for the event to be set before doing anything"""
#     print 'wait_for_event: starting'
#     e.wait()
#     print 'wait_for_event: e.is_set()->', e.is_set()

# def wait_for_event_timeout(e, t):
#     """Wait t seconds and then timeout"""
#     print 'wait_for_event_timeout: starting'
#     e.wait(t)
#     print 'wait_for_event_timeout: e.is_set()->', e.is_set()


# if __name__ == '__main__':
#     e = multiprocessing.Event()
#     w1 = multiprocessing.Process(name='block', 
#                                  target=wait_for_event,
#                                  args=(e,))
#     w1.start()

#     w2 = multiprocessing.Process(name='non-block', 
#                                  target=wait_for_event_timeout, 
#                                  args=(e, 2))
#     w2.start()

#     print 'main: waiting before calling Event.set()'
#     time.sleep(3)
#     e.set()
#     print 'main: event is set'

'''
Lock
'''

# import multiprocessing
# import sys

# def worker_with(lock, stream):
#     with lock:
#         stream.write('Lock acquired via with\n')
		
# def worker_no_with(lock, stream):
#     lock.acquire()
#     try:
#         stream.write('Lock acquired directly\n')
#     finally:
#         lock.release()

# lock = multiprocessing.Lock()
# w = multiprocessing.Process(target=worker_with, args=(lock, sys.stdout))
# nw = multiprocessing.Process(target=worker_no_with, args=(lock, sys.stdout))

# w.start()
# nw.start()

# w.join()
# nw.join()


'''
Condition objects let you synchronize parts of a workflow 
so that some run in parallel but others run sequentially, 
even if they are in separate processes.
'''

# import multiprocessing
# import time

# def stage_1(cond):
#     """perform first stage of work, then notify stage_2 to continue"""
#     name = multiprocessing.current_process().name
#     print 'Starting', name
#     with cond:
#         print '%s done and ready for stage 2' % name
#         cond.notify_all()

# def stage_2(cond):
#     """wait for the condition telling us stage_1 is done"""
#     name = multiprocessing.current_process().name
#     print 'Starting', name
#     with cond:
#         cond.wait()
#         print '%s running' % name

# if __name__ == '__main__':
#     condition = multiprocessing.Condition()
#     s1 = multiprocessing.Process(name='s1', target=stage_1, args=(condition,))
#     s2_clients = [
#         multiprocessing.Process(name='stage_2[%d]' % i, target=stage_2, args=(condition,))
#         for i in range(1, 3)
#         ]

#     for c in s2_clients:
#         c.start()
#         time.sleep(1)
#     s1.start()

#     s1.join()
#     for c in s2_clients:
#         c.join()


'''
Semaphore
Sometimes it is useful to allow more than one worker access 
to a resource at a time, while still limiting the overall number. 
For example, a connection pool might support a fixed number of 
simultaneous connections, or a network application might support 
a fixed number of concurrent downloads. A Semaphore is one way to
manage those connections.
'''

# import random
# import multiprocessing
# import time

# class ActivePool(object):
#     def __init__(self):
#         super(ActivePool, self).__init__()
#         self.mgr = multiprocessing.Manager()
#         self.active = self.mgr.list()
#         self.lock = multiprocessing.Lock()
#     def makeActive(self, name):
#         with self.lock:
#             self.active.append(name)
#     def makeInactive(self, name):
#         with self.lock:
#             self.active.remove(name)
#     def __str__(self):
#         with self.lock:
#             return str(self.active)

# def worker(s, pool):
#     name = multiprocessing.current_process().name
#     with s:
#         pool.makeActive(name)
#         print 'Now running: %s' % str(pool)
#         time.sleep(random.random())
#         pool.makeInactive(name)

# if __name__ == '__main__':
#     pool = ActivePool()
#     s = multiprocessing.Semaphore(3)
#     jobs = [
#         multiprocessing.Process(target=worker, name=str(i), args=(s, pool))
#         for i in range(10)
#         ]

#     for j in jobs:
#         j.start()

#     for j in jobs:
#         j.join()
#         print 'Now running: %s' % str(pool)


'''
Manager
In the previous example, the list of active processes
is maintained centrally in the ActivePool instance 
via a special type of list object created by a Manager. 
The Manager is responsible for coordinating shared 
information state between all of its users. By 
creating the list through the manager, the list is 
updated in all processes when anyone modifies it. In 
addition to lists, dictionaries are also supported.

What if i just used a regular dict instead of a manager dict?
ANS: I think they all make their own dicts, because the dict at 
the end is empty.

What does JOIN do??
ANS: Without join, the result is onlt the first 3 arguments, why??
Because it prints before they could finish. join makes it wait till theyre done

oh so daemon process will not stop the main from finishing. When the main
finished so does the daemon, whereas normal process will stop the main.
but in this case the print comes before the end. so I bet if we wait then print
again, they will have finished. so join makes it wait till they are finished instead of 
doing it at the end of the main

MY BET WAS CORRECT
'''

# import multiprocessing
# import time

# def worker(d, key, value):
#     d[key] = value

# if __name__ == '__main__':
#     mgr = multiprocessing.Manager()
#     d = mgr.dict()
#     #d = {}
#     jobs = [ multiprocessing.Process(target=worker, args=(d, i, i*2))
#              for i in range(10) 
#              ]
#     for j in jobs:
#         j.start()
#     # for j in jobs:
#     #     j.join()
#     print 'Results:', d

#     time.sleep(3)

#     print 'Results:', d


'''
manager with namespace
'''

# import multiprocessing

# def producer(ns, event):
#     ns.value = 'This is the value'
#     event.set()

# def consumer(ns, event):
#     try:
#         value = ns.value
#     except Exception, err:
#         print 'Before event, consumer got:', str(err)
#     event.wait()
#     print 'After event, consumer got:', ns.value

# if __name__ == '__main__':
#     mgr = multiprocessing.Manager()
#     namespace = mgr.Namespace()
#     event = multiprocessing.Event()
#     p = multiprocessing.Process(target=producer, args=(namespace, event))
#     c = multiprocessing.Process(target=consumer, args=(namespace, event))
	
#     c.start()
#     p.start()
	
#     c.join()
#     p.join()

'''
updates to mutable values in the namespace are not propagated.
'''

# import multiprocessing

# def producer(ns, event):
#     ns.my_list.append('This is the value') # DOES NOT UPDATE GLOBAL VALUE!
#     event.set()

# def consumer(ns, event):
#     print 'Before event, consumer got:', ns.my_list
#     event.wait()
#     print 'After event, consumer got:', ns.my_list

# if __name__ == '__main__':
#     mgr = multiprocessing.Manager()
#     namespace = mgr.Namespace()
#     namespace.my_list = []
	
#     event = multiprocessing.Event()
#     p = multiprocessing.Process(target=producer, args=(namespace, event))
#     c = multiprocessing.Process(target=consumer, args=(namespace, event))
	
#     c.start()
#     p.start()
	
#     c.join()
#     p.join()

'''
pool
'''

import multiprocessing
import time

def do_calculation(data):
	time.sleep(1)
	return data * 2

if __name__ == '__main__':
	pool_size = multiprocessing.cpu_count() * 2
	pool = multiprocessing.Pool(processes=pool_size)
	
	inputs = list(range(10))
	print 'Input   :', inputs
	
	start = time.time()
	builtin_outputs = map(do_calculation, inputs)
	print 'Built-in:', builtin_outputs
	end = time.time()
	print end - start
	
	start = time.time()
	pool_outputs = pool.map(do_calculation, inputs)
	print 'Pool    :', pool_outputs
	end = time.time()
	print end - start
