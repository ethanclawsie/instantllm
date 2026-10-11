"""Hand-written command-generation requests, grouped by area (one request per line).

None of these are the bench prompts (bench/prompts.txt); gen_prompts.py also filters
near-duplicates of the bench prompts automatically.
"""

HAND = {}

HAND["files"] = """
show hidden files in the current directory
list files sorted by modification time, newest first
list only the directories in the current folder
print the absolute path of notes.txt
create the nested directories src/components/ui in one go
copy the photos folder to /Volumes/Backup
move every pdf in Downloads into Documents/pdfs
delete all empty directories below the current folder
change the owner of /var/www to www-data recursively
create a symbolic link named python that points to /usr/local/bin/python3
how big is my Downloads folder
how many files are in this directory
count all files recursively in ~/Pictures
find every file called config.json under my home folder
find empty files in the current directory
find broken symlinks in /usr/local/bin
compare the contents of two directories dir1 and dir2
mirror ~/Music to /Volumes/Backup/Music with rsync
show the permissions of /etc/hosts
print the current working directory
create an empty file called notes.md
update the modification time of index.html without changing its contents
display data.csv with line numbers
view README.md one page at a time
concatenate part1.txt and part2.txt into combined.txt
split bigfile.bin into 100MB pieces
join the pieces bigfile.bin.aa bigfile.bin.ab back into one file
securely erase secret.txt
show the inode number of report.docx
list files with their sizes in kilobytes
show the directory tree two levels deep
find files owned by root under /usr/local
find files with 777 permissions in /var/www
delete all .pyc files and __pycache__ folders
remove every node_modules folder under ~/code
show the most recently modified file in Downloads
open the current folder in Finder
open report.pdf in Preview
reveal resume.pdf in Finder
copy the contents of id_rsa.pub to the clipboard
save the clipboard contents to clip.txt
calculate the md5 checksum of ubuntu.iso
verify the sha256 checksum of installer.dmg
what kind of file is unknown.bin
remove the quarantine attribute from /Applications/Tool.app
show the extended attributes of download.zip
find duplicate files in ~/Pictures by checksum
zip the Documents folder but leave out the .git directories
extract archive.tar.bz2
list what's inside backup.tar.gz without extracting it
extract only config.yml from settings.zip
make a password protected zip of the secrets folder
unzip photos.zip into ~/Pictures/trip
gzip every log file in /var/log/myapp
decompress data.json.gz but keep the compressed file
count the words in essay.txt
make a backup copy of .zshrc with today's date in the file name
rename all files in this folder to lowercase
add the prefix 2024_ to every file in this folder
replace spaces with underscores in all filenames here
move files older than 90 days from Downloads to ~/Archive
which folder under /var uses the most space
show free space on all mounted volumes
show what is taking up space in ~/Library/Caches
create a 1GB file filled with zeros for testing
show the number of files in each subdirectory
copy a directory and preserve permissions and timestamps
delete everything in /tmp older than 7 days
list all symlinks in the current directory
show the real path a symlink points to
find all executables in ~/bin
find files whose name contains "invoice" ignoring case
list files modified between January 1 and January 31
make all shell scripts in scripts/ executable
remove write permission for group and others on secrets.env
set permissions 644 on all files and 755 on all directories under public_html
show how much space each user's home directory uses
locate the file named hosts.allow
move up two directories
go back to the previous directory
create a temporary directory and print its path
list the 5 oldest files in this folder
find files that were accessed in the last day
delete all files in Downloads except pdfs
copy only new or changed files from src to dest
show a recursive listing of /etc/nginx
find all git repositories under ~/code
list files by extension
show the last modified date of package.json
archive and compress the www folder with xz
create a tar archive of src but exclude node_modules
unpack the tar.xz archive in Downloads
check whether two files are identical
find all files that contain a space in their name
convert all filenames to use dashes instead of spaces
how do i see file sizes in mb
"""

HAND["text"] = """
print the second column of data.csv
sort names.txt alphabetically and remove duplicates
count the unique lines in access.log
show the 10 most frequent IP addresses in access.log
remove blank lines from notes.txt
convert windows line endings to unix in script.sh
convert names.txt to lowercase
print lines 10 to 20 of server.log
delete the first line of export.csv
insert a header line at the top of data.csv
append node_modules to .gitignore
show lines that are in list1.txt but not in list2.txt
show the differences between old.conf and new.conf side by side
pretty print response.json
get the value of the name field from package.json with jq
count the number of entries in a json array in users.json
convert users.csv to json
case-insensitive grep for deprecated across the markdown docs
show lines matching ERROR in app.log with 3 lines of context around them
how many times does "timeout" appear in server.log
search for "password" recursively but only list the file names
print only the lines that start with a digit
extract all email addresses from contacts.txt
extract every url from index.html
convert tabs to 4 spaces in main.go
strip trailing whitespace from every python file in src
print a file in reverse line order
shuffle the lines of words.txt
print one random line from quotes.txt
number every line of poem.txt
join every pair of lines in pairs.txt
wrap the text in long.txt at 80 columns
show the longest line in input.txt
sum the third column of sales.tsv
compute the average of the numbers in numbers.txt
print the last field of every line in paths.txt
remove duplicate lines from history.txt while keeping the original order
show only the lines that occur more than once in ids.txt
split big.csv into files of 1000 lines each
merge two sorted files a.txt and b.txt into one sorted file
convert README.md to html with pandoc
base64 encode logo.png
decode the base64 string aGVsbG8gd29ybGQ=
url encode the string "hello world & more"
convert a json array of objects in data.json to csv with jq
grep for either foo or bar in app.log
show lines in debug.log that do not contain DEBUG
find files that contain both "import" and "async"
print the length of the string "supercalifragilistic"
convert "hello world" to uppercase
print the numbers from 1 to 100
print the 5th line of config.ini
remove comments and blank lines from nginx.conf
show non-ascii characters in a text file
count characters, words and lines in chapter1.txt
sort scores.csv numerically by the second column
sort a list of ip addresses in ips.txt numerically
align the columns of a whitespace separated table in table.txt
print the text between BEGIN and END markers in file.txt
delete lines containing "temp" from list.txt in place
show the word frequency of a text file
extract the 3rd to 5th characters of every line
print only unique words from words.txt
capitalize the first letter of every line
replace multiple spaces with a single space in text.txt
show the most common words in book.txt
show lines longer than 120 characters in main.py
count the number of commas in each line of data.csv
convert a tab separated file to comma separated
print every other line of a file
print the first and last line of a file
check if a file ends with a newline
remove the last character from every line
convert json to yaml with yq
get all keys from a json file with jq
filter a json array for objects where age is greater than 30
print the header row of a csv file
show the number of columns in a csv file
remove the BOM from a utf-8 file
convert a file from latin1 to utf-8
hexdump the first 64 bytes of a file
print the 10 longest words in a dictionary file
"""

HAND["processes"] = """
show all running processes
get the pid of Safari
show the top 5 processes by memory usage
which processes are using the most cpu right now
terminate the process with pid 4321
force quit an unresponsive app called Preview
restart the Dock
relaunch Finder
run long_task.sh in the background and keep it running after I log out
list my background jobs
bring the most recent background job to the foreground
show the process tree
show how long process 812 has been running
lower the priority of process 2201
start backup.sh with low cpu priority
watch processes live sorted by memory
list the files opened by process 991
show which process uses the most memory
wait until process 777 exits and then print done
show the threads of pid 1500
show the parent of process 345
run cleanup.sh every 5 minutes with cron
list my cron jobs
open my crontab for editing
remove all my cron jobs
shut the computer down in 30 minutes
cancel a scheduled shutdown
list all launch agents in my user folder
unload the launch agent com.example.agent
keep the mac awake for 2 hours
keep the mac awake while make runs
measure how long npm run build takes
run df -h every 2 seconds
kill whatever is using port 3000
pause process 4242
resume a stopped process with pid 4242
send SIGHUP to nginx to reload its config
show network connections opened by Chrome
count the number of running processes
list zombie processes
stop a command if it runs longer than 10 seconds
run two commands in parallel and wait for both
show the environment variables of a running process
run a command at 3pm today
list processes owned by the user www-data
show the full command line of process 2020
find which app is preventing sleep
show cpu usage per core
monitor disk activity in real time
show memory usage summary
how much ram is free
"""

HAND["network"] = """
ping google.com 5 times
trace the route to github.com
look up the DNS records for example.com
show the MX records for gmail.com
flush the dns cache on macos
list all network interfaces
show the default gateway
what wifi network am I connected to
turn wifi off
turn wifi back on
scan for nearby wifi networks
show the saved password for the wifi network HomeNet
test whether port 443 is open on example.com
open a raw tcp connection to localhost on port 6379
listen on port 9000 with netcat
download a file with curl following redirects
download a file with curl and save it under a different name
resume a partially downloaded file with curl
send a POST request with json data to http://localhost:3000/api
send a GET request with a bearer token header
print only the http status code for https://example.org
show the response headers of https://github.com
download all images linked from a web page with wget
mirror a website for offline browsing
check when the ssl certificate of example.com expires
ssh into server.example.com as admin
ssh into a server using the key ~/.ssh/work_ed25519
copy backup.tar.gz to a remote server with scp
copy the remote folder /var/log/app to my machine
forward local port 5432 to port 5432 on a remote db server over ssh
generate a new ed25519 ssh key
copy my ssh public key to user@203.0.113.10
add my ssh key to the agent
show all listening tcp ports
show established network connections
show network bandwidth used by each process
test my internet download speed
get the ip address of github.com
reverse dns lookup for 8.8.8.8
show the routing table
renew the dhcp lease on en0
start a simple web server in the current directory
check if https://example.com is up
whois lookup for example.net
show the contents of the hosts file
point myapp.local to 127.0.0.1 in /etc/hosts
scan the open ports on 192.168.1.10 with nmap
show the arp table
show my wifi signal strength
which dns servers am I using
set the dns servers to 1.1.1.1 and 8.8.8.8 on wi-fi
make an http request and time each phase
download a list of urls from urls.txt
upload a file with curl using multipart form data
show the ipv6 address of en0
measure latency to 1.1.1.1
check which ports are open on localhost
connect to an ftp server
show how many connections are in TIME_WAIT
query a specific dns server for a domain
"""

HAND["macos"] = """
show the macOS version
show a hardware overview of this mac
print the serial number of this mac
show the battery percentage
show the battery cycle count
empty the trash from the terminal
show hidden files in Finder
hide all desktop icons
save screenshots to ~/Screenshots instead of the desktop
change the screenshot format to jpg
take a screenshot of the whole screen and save it to shot.png
take a screenshot after a 5 second delay
set the computer name to dev-mac
check for macos software updates
install all available software updates
how long has this mac been running
lock the screen
put the display to sleep
restart the mac
shut down immediately
make the mac say "build finished"
play alert.wav from the terminal
set the output volume to 50 percent
mute the sound
open apple.com in Safari
open diagram.png with Preview
open this folder in VS Code
quit Spotify
list all installed applications
list all disks
eject the drive called USB
format the usb disk disk4 as exfat with the name DATA
create a dmg image from the folder MyApp
mount installer.dmg
convert photo.png to jpg with sips
resize photo.jpg to 800 pixels wide with sips
get the width and height of image.png
search spotlight for files named invoice
rebuild the spotlight index
turn off spotlight indexing for /Volumes/External
show time machine status
start a time machine backup now
list time machine backups
switch to dark mode
show a notification that says "done"
print what's on the clipboard
show the cpu model name
show how many cpu cores this mac has
show the total amount of ram
check memory pressure
free up inactive memory
check if gatekeeper is enabled
show the code signature of /Applications/Safari.app
list loaded kernel extensions
show error messages from the system log in the last hour
change my default shell to zsh
reload my zshrc
print my PATH
add ~/bin to my PATH permanently
create an alias ll for ls -la
search my command history for docker
clear the terminal screen
show the current time in UTC
show this month's calendar
convert the unix timestamp 1700000000 to a readable date
print the current unix timestamp
show the date 30 days from today
set a 10 minute timer that shows a notification
generate a random 20 character password
generate a uuid
what is 2 to the power of 32
list all environment variables
run a command with an environment variable set only for that command
remove the environment variable DEBUG
find where the git executable lives
check if ls is an alias or a binary
open the manual for rsync
show disk usage of the main disk
show info about the startup disk
verify the startup disk
show the login items
show cpu temperature
show the firmware version
prevent the mac from sleeping when the lid is closed on power
show wake reasons from the power log
list all fonts installed
show the current keyboard input source
show bluetooth devices
disable natural scrolling
set the dock to auto hide
reset the launchpad layout
show which apps have full disk access
show the current user's uid and groups
show who is logged in
show the hostname
show the system's time zone
set the time zone to America/New_York
sync the clock with an ntp server
show the display resolution
show the amount of free disk space in gigabytes
"""

HAND["git"] = """
show the status of this repo
stage all changes
commit with the message "fix typo"
change the message of the last commit
push the current branch to origin
push a new local branch and set its upstream
pull with rebase
list all branches including remote ones
delete the local branch old-feature
delete the remote branch old-feature
rename the current branch to main
switch to the develop branch
merge develop into main
rebase my branch onto main
abort the rebase in progress
continue the rebase after fixing conflicts
squash the last 3 commits into one
cherry-pick commit a1b2c3d
revert commit 9f8e7d6
show the staged diff
show the changes made to app.js
show who last changed each line of server.py
show the commit history of config.yaml
show commits authored by alice
show commits from the last week
how many commits are on this branch
show the most recent commit
list the files changed in the last commit
stash my changes including untracked files
list all stashes
apply the most recent stash
delete all stashes
discard the changes to index.html
throw away all local changes
remove untracked files and directories
unstage main.go
reset my branch to match origin/main exactly
fetch from all remotes and prune deleted branches
add a remote named upstream
show the remote urls
change the url of origin to git@github.com:me/repo.git
shallow clone https://github.com/torvalds/linux
clone only the dev branch of a repository
create the tag v1.0.0
push all tags to origin
delete the tag v0.9
list all tags
print the hash of the current commit
show the short hash of HEAD
start a git bisect to find a bad commit
list all files tracked by git
stop tracking changes to config.local.json
remove secrets.env from git but keep it on disk
set my git user name and email globally
show my git configuration
create a git alias st for status
initialize a new git repository
show how big this repository is
run garbage collection on the repo
update all git submodules
add a submodule for https://github.com/user/lib
show the reflog
restore a branch I deleted by accident
list branches already merged into main
show the commits that are in feature but not in main
find which branches contain commit abc123
create a patch file from the last commit
apply the patch fix.patch
make an empty commit
change the author of the last commit
list files with merge conflicts
resolve a conflict by taking their version of package-lock.json
show the log with dates and authors
search commit messages for "hotfix"
search the whole history for the string API_KEY
show what changed between v1.0 and v2.0
show a summary of changes per file since yesterday
undo git add for all files
push and force overwrite the remote branch safely
show the default branch of origin
list remote branches only
check out a file from another branch
show the diff between two commits
show changes ignoring whitespace
show contributors ranked by number of commits
create a new orphan branch called gh-pages
"""

HAND["docker"] = """
list running docker containers
list all containers including stopped ones
list docker images
remove all stopped containers
remove dangling docker images
clean up all unused docker data
run nginx detached and map port 80
start an interactive ubuntu container
open a bash shell in the running container web
follow the logs of the api container
stop every running container
restart the db container
build a docker image tagged myapp:latest
build the image without using the cache
push myapp:1.2 to docker hub
pull postgres version 15
show live resource usage of containers
get the ip address of a container
copy /app/log.txt from container web to the host
run a container with the current directory mounted at /src
pass environment variables to docker run
start docker compose services in the background
stop compose services and remove volumes
rebuild and restart docker compose services
show compose logs for the web service
list docker volumes
remove the volume pgdata
list docker networks
create a docker network called backend
tag image myapp:latest as registry.example.com/myapp:1.0
save the image myapp to a tar file
load an image from myapp.tar
show the layers of an image
show docker disk usage
log in to docker hub
run a container that deletes itself on exit
run a container limited to 512MB of memory
show the port mappings of the container web
run a command as root inside a container
show the environment variables of a container
inspect the health status of a container
stop all containers and remove them
remove all images
show the docker version
list running compose services
scale the worker service to 3 instances with compose
"""

HAND["brew"] = """
install wget with homebrew
update homebrew and upgrade everything
list installed homebrew packages
uninstall the htop formula
search homebrew for a package named ripgrep
show info about the node formula
list outdated brew packages
remove old versions of brew packages
install firefox with brew cask
list installed casks
start the postgresql service with brew
list brew services
pin the python formula so it doesn't upgrade
show the dependencies of ffmpeg
show which installed formulae depend on openssl
run brew doctor
reinstall git with brew
install node 18 with brew
where did brew install python
add the homebrew tap hashicorp/tap
dump all installed brew packages to a Brewfile
install everything listed in a Brewfile
upgrade only the curl package
show the homebrew version
show the brew prefix
uninstall a cask app called slack
check what would be removed by brew cleanup
list formulae installed on request
"""

HAND["media"] = """
extract the audio from video.mp4 as mp3
convert song.wav to mp3
cut a clip from 00:01:00 to 00:02:30 out of movie.mp4
scale a video down to 720p
turn clip.mp4 into an animated gif
compress a large video to make the file smaller
save one frame per second of a video as jpg images
make a video from frame_0001.png, frame_0002.png and so on
join part1.mp4 and part2.mp4 into one video
remove the audio from a video
rotate a video 90 degrees clockwise
change a video's frame rate to 30 fps
burn subtitles.srt into a video
show the duration and codecs of a video file
remux movie.mkv to mp4 without re-encoding
make a video play twice as fast
grab the first frame of a video as a thumbnail
record the screen to a video file with ffmpeg
convert album.flac to aac
normalize the loudness of podcast.mp3
combine audio.m4a and video.mp4 into one file
crop a video to a 1080x1080 square
overlay logo.png in the corner of a video
convert all heic photos in this folder to jpg
shrink every jpg in this folder to half size
remove exif metadata from photos
download a youtube video with yt-dlp
download only the audio of a youtube video as mp3
convert slides.pdf into png images
merge several pdf files into one
reduce the size of a pdf
count the pages in a pdf
convert banner.png to webp
make a 200px thumbnail of picture.jpg
show the exif data of IMG_2001.jpg
convert an mp4 to webm
extract the subtitles from an mkv file
create an audio spectrogram image from sound.wav
convert a video to black and white
loop a short video 5 times
increase the volume of an audio file by 10dB
convert stereo audio to mono
split an audio file into 10 minute chunks
generate a 1 second silent wav file
get the bitrate of a video
convert all wav files in the folder to flac
add a fade in to the start of a video
reverse a video
create a slideshow video from images with 3 seconds each
"""

HAND["python"] = """
create a python virtual environment called venv
activate the virtual environment in venv
leave the virtual environment
install requests with pip
install the packages listed in requirements.txt
write the installed packages to requirements.txt
upgrade pip
list outdated pip packages
uninstall numpy with pip
run server.py in the background and log output to server.log
start a python web server on port 8000
pretty print a json file with python
run pytest verbosely
run only the test test_login in tests/test_auth.py
format the project with black
lint the code with flake8
show the path of the python interpreter
list python versions installed with pyenv
set the local python version to 3.11 with pyenv
print today's date with a python one-liner
start an interactive python shell
profile script.py with cProfile
compile all python files in the project
install a pip package for my user only
show which version of pandas is installed
create a conda environment named ml with python 3.10
start a jupyter notebook server
convert analysis.ipynb to a python script
run a python module as a script
install a package in editable mode
check python code with mypy
generate a requirements file with pip-tools
run python with unbuffered output
time a small python snippet with timeit
show the site-packages directory
list the packages installed in the current venv
create a venv with uv
run tests and stop at the first failure
measure test coverage with pytest
"""

HAND["node"] = """
initialize a new node project with default settings
install express
install typescript as a dev dependency
install all dependencies from package.json
run the start script
run the test script with npm
list globally installed npm packages
update all npm packages
check for outdated npm packages
remove the lodash package
clear the npm cache
run create-react-app with npx
scaffold a react app with vite
use node 18 with nvm
install the latest lts version of node with nvm
print the node version
run index.js with node
install yarn globally
add axios with yarn
install dependencies with pnpm
audit dependencies and fix vulnerabilities
publish this package to npm
link a local package for development
show the npm dependency tree
explain why a package is installed
run eslint on the src folder
format the whole project with prettier
compile the typescript project
create a tsconfig.json
run a node script with 4GB of memory
list the scripts defined in package.json
bump the patch version of the package
install a specific version of react
run a script in every workspace
start nodemon on app.js
show the npm registry being used
set the npm registry to a private url
reinstall node_modules from scratch
"""

HAND["db_cloud"] = """
connect to the postgres database shop
list all postgres databases
dump the postgres database shop to shop.sql
restore shop.sql into a postgres database
open data.db with sqlite and list the tables
export the users table from app.db to csv
run a query against a sqlite database from the shell
connect to mysql as root
back up a mysql database to a file
check if redis is running
delete all keys in redis
get the value of the key session:42 in redis
start mongodb with brew services
open the mongo shell
list pods in all namespaces
describe the pod web-7d4b9
show the logs of a kubernetes pod
open a shell in a kubernetes pod
apply deployment.yaml
delete the deployment api
scale the deployment web to 3 replicas
forward local port 8080 to a kubernetes service
show the current kubectl context
switch the kubectl context to staging
list my s3 buckets
copy report.csv to an s3 bucket
sync a local folder to s3
show my aws caller identity
initialize and apply terraform
list gcloud projects
show kubernetes nodes and their status
get all services in the default namespace
restart a kubernetes deployment
show events in a namespace sorted by time
decode a kubernetes secret value
list ec2 instances with the aws cli
download a file from s3
show terraform plan without applying
"""

HAND["security"] = """
generate a self-signed tls certificate valid for a year
encrypt secrets.txt with a password using openssl
decrypt secrets.txt.enc with openssl
generate a random 32 byte hex string
show the fingerprint of my ssh public key
verify a file's gpg signature
generate a new gpg key
compute the sha256 hash of a string
create a random 16 character base64 string
check the permissions on my ssh directory
show the certificate details of a pem file
convert a pem certificate to der
list the keys in my gpg keyring
encrypt a file for a recipient with gpg
show failed login attempts
show the last logins on this machine
create an htpasswd entry for user admin
check which users have sudo rights
show the sha1 of every file in a folder
"""

HAND["misc"] = """
show the current date and time
print the weather for my city in the terminal
start a tmux session named work
attach to the tmux session work
list tmux sessions
split the tmux window vertically
start a screen session
detach from screen
show the system load average
show the number of logged in users
display a big clock in the terminal
calculate 15% of 240 on the command line
convert 100 fahrenheit to celsius in the shell
show a progress bar while copying a large file
repeat the last command with sudo
run the previous command again
create a file with the content hello world
write the output of ls to files.txt
append the date to log.txt
redirect errors of a command to errors.log
discard all output of a command
run a command only if the previous one succeeded
run a command and show output while saving to a file
count to 10 in a loop
loop over all txt files and print their names
read a file line by line in zsh
check if a file exists in a shell script
check if a directory is empty
get the extension of a filename in zsh
get the filename without the path
print the exit status of the last command
make a script that takes arguments
show the size of a variable's value in bytes
sleep for 5 seconds
beep in the terminal
print a horizontal line across the terminal width
show the width of the terminal
open a new terminal tab
change the terminal title
show which shell I'm using
show the zsh version
list all aliases
list all shell functions
measure the time of a shell loop
"""

HAND["long"] = """
I have a folder full of camera raw files with the .CR2 extension in ~/Pictures/raw and I want to convert all of them to jpg
my disk is almost full, how can I see which folders in my home directory are the biggest
I accidentally committed a large binary file to git, how do I remove it from the last commit
I need to find all the lines in my nginx access log that returned a 500 error and count them
there is a process hogging my cpu and I want to see what it is and kill it
how do I find out which application is using my microphone right now
my python script prints a lot, how do I save its output and errors to separate files
set up a cron job that backs up ~/Documents to an external drive every night at 2am
I want to rename all photos in this folder to include the date they were taken
how can I see every command I ran yesterday
I want to copy my whole home directory to an external disk but skip the Library folder
show me every file in the repo that has been changed but not committed yet
download all the pdf files linked on https://example.com/papers
I want a list of all the npm packages my project depends on directly
convert every markdown file in docs to pdf using pandoc
I need to find the IP address of my raspberry pi on the local network
check how much battery health my macbook has left
find every TODO comment in the python files of this project and show the file and line
my wifi keeps dropping, show me the wifi logs
create a new user account named guest on this mac from the command line
how can I tell which version of macOS and which build I am running
I want to download a whole youtube playlist as mp3 files
generate an ssh key and add it to my github account using the gh cli
show the size of every docker image sorted from largest to smallest
compress all the png files in assets without losing quality
list all the open ports on my machine and the programs using them
find files in Downloads that I haven't opened in over a year and move them to the trash
watch the src directory and rerun the tests whenever a file changes
make a zip of the project folder that excludes node_modules and .git
there are too many files in /tmp, delete the ones older than a week
I want to find which commit broke the build between v1.2 and v1.3
show all the git branches sorted by the date of their last commit
I need to see the http response time of my api endpoint at http://localhost:8080/health
how do I make a bootable usb installer for macOS
tell me the total size of all mp4 files in my Movies folder
turn off the startup sound and also disable automatic updates
show the top 20 largest files in my entire home directory including hidden ones
list all running services on my mac that start at login
convert a folder of wav recordings into a single mp3
check whether my machine can reach the internet and how fast
"""

HAND["terse"] = """
disk space
free memory
open ports
cpu usage
current branch
python version
node version
list usb devices
battery
uptime
whoami
date
reboot
kill port 3000
flush dns
show path
wifi password
docker cleanup
brew update
pip list
npm outdated
tar extract
unzip all
ssh keygen
weather
calendar
hostname
mac address
"""
