import urllib
import sys
import time

def get_next_target(input,target):
	start_link = input.find(target)
	if start_link == -1:
		return None,0
	else:
		start_quote = input.find(':',start_link)

		end_quote = input.find(',',start_quote + 1)

		output = input[start_quote + 1 : end_quote]
		return output, end_quote

def get_all_id(input,targetId):
	all_id = []
	while True:
		output, end_quote = get_next_target(input,targetId)
		if output:
			all_id.append(output)
			input = input[end_quote:]
		else:
			break
	return all_id
	
def get_all_title(input,targetTitle):
	all_titles = []
	while True:
		output, end_quote = get_next_target(input,targetTitle)
		if output:
			all_titles.append(output[1:-1])
			input = input[end_quote:]
		else:
			break
	return all_titles

def get_similar_id(input, targetId):
	all_id = []
	count = 1
	while count <= 5:
		output, end_quote = get_next_target(input,targetId)
		if output:
			all_id.append(output)
			input = input[end_quote:]
		else:
			break
		count = count + 1 
	# if not all_id:
		# return "List is empty"
	# else:
	return all_id
	
# 300 science fiction movies
movie_ID_name = open(r'movie_ID_name.txt','w')
sys.stdout = movie_ID_name
result = []
movie_300_id = []
for page in range(1,16):
	strPage = str(page)
	response = urllib.urlopen('http://api.themoviedb.org/3/discover/movie?page=' + strPage + '&release_date.gte=2000-01-01&include_all_movies=true&with_genres=878&api_key=2ae2b1f3901bd779959555d0214e999d')
	pageRead = response.read()

	targetId = '"id"'	
	targetTitle = '"title"'

	movie_id = get_all_id(pageRead,targetId)
	movie_name = get_all_title(pageRead,targetTitle)


	for i in range(0,20):
		result.append(movie_id[i] + ',' + movie_name[i])
		movie_300_id.append(movie_id[i])
	
for i in result:
	print i
sys.stdout=sys.__stdout__
movie_ID_name.close()

# ###########################################################
# Find similar movies
movie_ID_sim_movie_ID = open(r'movie_ID_sim_movie_ID.txt','w')
sys.stdout = movie_ID_sim_movie_ID
result_sim = []
result1 = []
result2 = []
for id in movie_300_id:
	strID = str(id)
	response = urllib.urlopen('http://api.themoviedb.org/3/movie/' + strID + '/similar?api_key=2ae2b1f3901bd779959555d0214e999d')
	pageRead = response.read()

	targetId = '"id"'	
	movie_id_sim = get_similar_id(pageRead,targetId)

	for sim_id in movie_id_sim:
		result1.append(strID + ',' + sim_id)
		result2.append(sim_id + ',' + strID)
	
	time.sleep(0.3)
	
result_sim = result1
for i in result1:
	for j in result2:
		if i == j:
			result_sim.remove(j)
			
for i in result_sim:
	print i

sys.stdout=sys.__stdout__
movie_ID_sim_movie_ID.close()




















