(1)Platform: Windows 10
(2)Instruction:
   Run the script.py code by command promt and it will automatically generate two target files in the same folder which are movie_ID_name.txt and movie_ID_sim_movie_ID.txt.
(3)API endpoint and query parameter:
   In step b: http://api.themoviedb.org/3/discover/movie?release_date.gte=2000-01-01&include_all_movies=true&page=1&with_genres=878&api_key=2ae2b1f3901bd779959555d0214e999d
   And I can request a different page by assign a different value to the "page=" in the link above.(For now, it is showing the page = 1.)   
   In step c: http://api.themoviedb.org/3/movie/5/similar?api_key=2ae2b1f3901bd779959555d0214e999d
   And I can request a different movie by assign a different id value to the link above.(For now, it is showing the movie id = 5.)