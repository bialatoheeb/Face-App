
# #  Image and Container build on my machine 

# docker build -t d-face-pd:latest .
# docker run -p 8501:8501 face-pd:latest


# Deployment on Heroku 
heroku login
heroku container:login
heroku create face-pd
heroku container:push web --app face-pd
heroku container:release web --app face-pd
heroku open --app face-pd



# # Run this to remove all images, containers or volumes curremtly present or avialble
# docker system prune -a
