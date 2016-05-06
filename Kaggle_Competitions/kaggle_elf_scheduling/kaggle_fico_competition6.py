
import csv
import math
import datetime

#kraggle fico competition



#1 = print , 0 = output to file aka run in terminal vs submitjob
printing = 0

data_file = 'toys_rev2.csv'

if printing == 0:
	output = 'output26.csv'
	f2 = open(output, 'w')
	f2.write('ToyId,ElfId,StartTime,Duration')

numb_of_toys = 10000000
numb_of_elves = 900

numb_of_toys_completed = 0

class elf:

	elf_id = 0
	productivity = 1.0
	work_time_remaining = 0
	resting_time_remaining = 0
	sanctioned = 0.0
	unsanctioned = 0.0

	def __init__(self, elf_id, prod=1.0):
		self.elf_id = elf_id
		self.productivity = prod

class toy:

	toy_id = 0
	duration = 0

	def __init__(self, toy_id, duration):
		self.toy_id = toy_id
		self.duration = duration

def convert_minutes_to_date(time):

	tt = ref_time + datetime.timedelta(seconds=60*time)
	return " ".join([str(tt.year), str(tt.month), str(tt.day), str(tt.hour), str(tt.minute)])

def convert_date_to_minutes(date):

	time = date.split(' ')
	dd = datetime.datetime(int(time[0]), int(time[1]), int(time[2]), int(time[3]), int(time[4]))
	age = dd - datetime.datetime(2014, 1, 1, 0, 0)
	return int((age.microseconds + (age.seconds + age.days * 24 * 3600) * 10**6) / 10**6 / 60)

def update_elves():
	#print len(elf_list)
	for elf in elf_list:

		#current_hour_of_the_day = int(convert_minutes_to_date(current_time).split()[3])
		current_hour_of_the_day = int(the_date_time_list[3]) 

		if elf.work_time_remaining > 0:
			elf.work_time_remaining -= 1
			#if he just finished
			#print 'work remaining' + ' ' + str(elf.work_time_remaining)
			if elf.work_time_remaining <= 0:
				global numb_of_toys_completed
				numb_of_toys_completed += 1

				if printing == 0:
					if numb_of_toys_completed % 50000 == 0:
						progress_file = 'progress26/' + str(numb_of_toys_completed) + 'x.txt'
						a = [len(x) for x in groups]
						f3 = open(progress_file, 'w')
						for index in range(len(group_ranges)):

							f3.write(str(group_ranges[index]) + ': ')
							f3.write(str(a[index]) + '\n')

						f3.write('productivity:' + str(elf.productivity) + '\n')
						f3.write('current line offset: ' + str(current_line_offset) + '\n')
						f3.write('numb_of_toys_completed: ' + str(numb_of_toys_completed) + '\n')
						f3.write('To do: ' + str(current_line_offset - numb_of_toys_completed) + '\n')						
						f3.write('numb of available elves: ' + str(len(list_of_available_elves)) + '\n')
						f3.write(the_date_time)
						f3.close()

				#print 'Completed:' + str(numb_of_toys_completed)
				work_time_remaining = 0
				#elf.working = 0
				elf.productivity = elf.productivity * (1.02)**elf.sanctioned * (0.9)**elf.unsanctioned
				if elf.productivity > 4.0:
					elf.productivity = 4.0
				if elf.productivity < 0.25:
					elf.productivity = 0.25
				if elf.resting_time_remaining <= 0:
					elf.resting_time_remaining = 0
					list_of_available_elves.append(elf)
					#print 'adding ' + str(elf.elf_id) + ' to list'
					
		elif elf.work_time_remaining == 0 and elf.resting_time_remaining > 0 and not (current_hour_of_the_day >= 19 or current_hour_of_the_day < 9):
			elf.resting_time_remaining -= 1
			#if just finished resting
			if elf.resting_time_remaining <= 0:
				elf.resting_time_remaining = 0
				#print 'adding ' + str(elf.elf_id) + ' to list2'
				list_of_available_elves.append(elf)

def assign_elf(elf, toy):

	time_to_complete_toy = int(math.ceil(float(toy.duration) / elf.productivity))

	if printing == 0:
		ouput_statement = str(toy.toy_id) + ',' \
			+ str(elf.elf_id) + ',' \
			+ the_date_time + ',' \
			+ str(time_to_complete_toy)  
		f2.write('\n' + ouput_statement)


	elf.work_time_remaining = time_to_complete_toy

	if time_to_complete_toy <= time_left_in_day:
		sanctioned = time_to_complete_toy
		unsanctioned = 0
	else:
		numb_of_days_to_complete = int(time_to_complete_toy / 1440)
		minutes_left = time_to_complete_toy - (numb_of_days_to_complete * 1440)
		if minutes_left <= (time_left_in_day + 840):
			sanctioned = time_left_in_day + 600*numb_of_days_to_complete
			unsanctioned = (minutes_left - time_left_in_day) + 840*numb_of_days_to_complete
		elif minutes_left <= (time_left_in_day + 840 + 600):
			sanctioned = (minutes_left - 840) + 600*numb_of_days_to_complete
			unsanctioned = 840 + 840*numb_of_days_to_complete
		else:
			print 'SOMETHING IS WRONG'
			print the_date_time
			print time_to_complete_toy
			print time_left_in_day
			print numb_of_days_to_complete
			print toy.duration
			print elf.productivity

	elf.sanctioned = sanctioned / 60.0
	elf.unsanctioned = unsanctioned / 60.0
	elf.resting_time_remaining = unsanctioned

	list_of_available_elves.remove(elf)



print 'Making groups...'
group_ranges = range(2, 200, 1)
group_ranges1 = range(201,300, 5)
group_ranges2 = range(301, 2400, 50)
group_ranges.extend(group_ranges1)
group_ranges.extend(group_ranges2)
group_ranges.append(5760)
group_ranges.append(99999999)
print group_ranges
groups = []
for i in range(len(group_ranges)):
	groups.append([])
groups_sorted = [0]*(len(groups))
print 'Done. Now elves.'


print 'Creating elves...'
#create elves
elf_list = []
for i in range(1, numb_of_elves+1):
	elf_list.append(elf(i))
print 'Done. Now offsets.'


print 'Figuring out offsets...'
# Read in the file once and build a list of line offsets
line_offset = []
offset = 0
f = open(data_file, 'r')
#reader = csv.reader(f)
row_numb = 0
for row in f:
	#print row_numb
	line_offset.append(offset)
	offset += len(row)
	#print row
	row_numb += 1
	if row_numb > numb_of_toys:
		break
f.seek(0)
len_line_offset = len(line_offset)
print 'Done. Lets start.'




ref_time = datetime.datetime(2014, 1, 1, 0, 0)

#minutes since start
current_time = 0
current_line_offset = 1
list_of_available_elves = []
#all elves start available
for elf in elf_list:
	list_of_available_elves.append(elf)

while numb_of_toys_completed < numb_of_toys:

	current_time += 1

	#if current_time < 495120:
	#	continue

	the_date_time = convert_minutes_to_date(current_time)
	the_date_time_list = the_date_time.split()
	current_hour_of_the_day = int(the_date_time_list[3])
	current_minute_of_the_day = int(the_date_time_list[4])

	update_elves()

	#if no elves available, skip this minute
	if len(list_of_available_elves) == 0:
		if printing == 1:
			a = [len(x) for x in groups]
			print convert_minutes_to_date(current_time) + '    ' + str(current_line_offset) + ' ' + str(numb_of_toys_completed) + '    ' + str(len(list_of_available_elves)) + '   ' + str(a)
			print
		continue

	#if it is no longer working hours, I dont want to assign anyone
	if current_hour_of_the_day >= 19 or current_hour_of_the_day < 9 or (current_hour_of_the_day == 18 and current_minute_of_the_day > 57):
		if printing == 1:
			a = [len(x) for x in groups]
			print convert_minutes_to_date(current_time) + '    ' + str(current_line_offset) + ' ' + str(numb_of_toys_completed) + '    ' + str(len(list_of_available_elves)) + '   ' + str(a)
		continue

	#read in all the toys that are available
	if len_line_offset > current_line_offset:
		f.seek(line_offset[current_line_offset])

		while convert_date_to_minutes(f.readline().split(',')[1]) <= current_time:
			
			f.seek(line_offset[current_line_offset])
			a = f.readline().split(',')
			toy_id = int(a[0])
			duration = int(a[2].split('\r')[0])

			#put in proper group
			for group1 in range(len(group_ranges)):
				if duration <= group_ranges[group1]:
					groups[group1].append(toy(toy_id, duration))
					groups_sorted[group1] = 0
					break

			current_line_offset += 1
			if len_line_offset <= current_line_offset:
				break
		
	#sort elves so that the slowest is first
	list_of_available_elves.sort(key= lambda x: x.productivity)
	

	if printing == 1:
		if len(list_of_available_elves) > 4:
			a = [len(x) for x in groups]
			print convert_minutes_to_date(current_time) + '    ' + str(current_line_offset) + ' '+ str(numb_of_toys_completed) + '    ' + str(len(list_of_available_elves)) + '   ' + str(a) + '    top ' + str(list_of_available_elves[-1].productivity) + "  bottom " + str(list_of_available_elves[0].productivity)

	time_left_in_day = convert_date_to_minutes(the_date_time_list[0] + ' ' + the_date_time_list[1] + ' ' + the_date_time_list[2] + ' 19 0') - current_time




	n = len(list_of_available_elves)

	#see if I can skip some time
	if int(the_date_time_list[0]) > 2017 and n == 0:

		next_available = 99999999
		for elf2 in elf_list:
			if elf2.work_time_remaining > 0:
				if elf2.work_time_remaining < next_available:
					next_available = elf2.work_time_remaining
			elif elf2.resting_time_remaining > 0:
				if elf2.resting_time_remaining < next_available:
					next_available = elf2.resting_time_remaining

		if next_available > 3:
		#so that I dont get anyone to 0 without being in update_elf function
			next_available = next_available -1
			for elf2 in elf_list:
				if elf2.work_time_remaining > 0:
					elf2.work_time_remaining -= next_available
				elif elf2.resting_time_remaining > 0:
					elf2.resting_time_remaining -= next_available
			current_time += next_available
			continue

	#go through available elves. 
	i2 = 0
	while i2 < n:

		this_elf = list_of_available_elves[i2]

		#max duration that can be done before end of day given this elf's productivity
		#max toy duration = time left in day * productivity
		max_duration = time_left_in_day * this_elf.productivity

		#print 'Elf ' + str(this_elf.elf_id) + ' Prod ' + str(this_elf.productivity) + ' Max ' + str(max_duration) + ' i= ' + str(i2) + ' n= ' + str(n)


		assigned = 0
		if this_elf.productivity > 3.99:

			if not groups_sorted[-1]:
				groups[-1].sort(key=lambda x: x.duration, reverse=True)
				groups_sorted[-1] = 1

			if len(groups[-1]) > 0:
				assign_elf(this_elf, groups[-1][0])
				n -= 1
				assigned = 1
				groups[-1].pop(0)
				break

			if assigned == 0:
				if int(the_date_time_list[3]) <= 10:

					if not groups_sorted[-2]:
						groups[-2].sort(key=lambda x: x.duration, reverse=True)
						groups_sorted[-2] = 1

					if len(groups[-2]) > 0:
						assign_elf(this_elf, groups[-2][0])
						n -= 1
						assigned = 1
						groups[-2].pop(0)
						break


			if assigned == 0:
				for i in range(len(groups)-3, -1, -1):
					if i-1 >= 0:
						if group_ranges[i-1] < max_duration and len(groups[i]) > 0:

							if not groups_sorted[i]:
								groups[i].sort(key=lambda x: x.duration, reverse=True)
								groups_sorted[i] = 1

							for toy1 in groups[i]:
								if toy1.duration <= max_duration:
									assign_elf(this_elf, toy1)
									n -= 1
									assigned = 1
									groups[i].remove(toy1)
									break
					else:
						if len(groups[i]) > 0:
							if not groups_sorted[i]:
								groups[i].sort(key=lambda x: x.duration, reverse=True)
								groups_sorted[i] = 1

							for toy1 in groups[i]:
								if toy1.duration <= max_duration:
									assign_elf(this_elf, toy1)
									n -= 1
									assigned = 1
									groups[i].remove(toy1)
									break
					if assigned == 1:
						break
				if assigned == 0:
					i2 += 1				

		#less than 3.99
		else:
			for i in range(len(groups)-3, -1, -1):
				if i-1 >= 0:
					if group_ranges[i-1] < max_duration and len(groups[i]) > 0:

						#print 'In group <' + str(group_ranges[i]) + ' len= ' + str(len(groups[i])) 
						if not groups_sorted[i]:
							groups[i].sort(key=lambda x: x.duration, reverse=True)
							groups_sorted[i] = 1

						for toy1 in groups[i]:
							#print 'toy duration ' + str(toy1.duration)
							if toy1.duration <= max_duration:
								#print 'assigned'
								assign_elf(this_elf, toy1)
								n -= 1
								assigned = 1
								groups[i].remove(toy1)
								break
				else:
					if len(groups[i]) > 0:

						#print 'In group <' + str(group_ranges[i]) + ' len= ' + str(len(groups[i])) 
						if not groups_sorted[i]:
							groups[i].sort(key=lambda x: x.duration, reverse=True)
							groups_sorted[i] = 1

						for toy1 in groups[i]:
							#print 'toy duration ' + str(toy1.duration)
							if toy1.duration <= max_duration:
								#print 'assigned'
								assign_elf(this_elf, toy1)
								n -= 1
								assigned = 1
								groups[i].remove(toy1)
								break

				if assigned == 1:
					break

			if assigned == 0:
				if int(the_date_time_list[0]) > 2014:
					no_easy_ones_left = 0
					for group_i in groups[:140]:
						if len(group_i) > 0:
							no_easy_ones_left = 1
							break
					if no_easy_ones_left == 0:

						if not groups_sorted[-1]:
							groups[-1].sort(key=lambda x: x.duration, reverse=True)
							groups_sorted[-1] = 1

						for toy1 in groups[-1]:
							assign_elf(this_elf, toy1)
							n -= 1
							assigned = 1
							groups[-1].remove(toy1)
							break
						if assigned == 0:

							if not groups_sorted[-2]:
								groups[-2].sort(key=lambda x: x.duration, reverse=True)
								groups_sorted[-2] = 1

							for toy1 in groups[-2]:
								assign_elf(this_elf, toy1)
								n -= 1
								assigned = 1
								groups[-2].remove(toy1)
								break

			if assigned == 0:
				i2 += 1

	

	#this is at the end of assignments
	a = [len(x) for x in groups]
	if printing == 1:
		if len(list_of_available_elves) > 4:
			print convert_minutes_to_date(current_time) + '    ' + str(current_line_offset) + ' '+ str(numb_of_toys_completed) + '    ' + str(len(list_of_available_elves)) + '   ' + str(a) + '    top ' + str(list_of_available_elves[-1].productivity)  + "  bottom " + str(list_of_available_elves[0].productivity)
			print
		else:
			print convert_minutes_to_date(current_time) + '    ' + str(current_line_offset) + ' '+ str(numb_of_toys_completed) + '    ' + str(len(list_of_available_elves)) + '    ' + str(a)
			print


f.close()


print "FINISH TIME: " + str(current_time)










