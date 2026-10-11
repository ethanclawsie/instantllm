"""Second batch of hand-written requests (training data expansion). One request per line."""

HAND2 = {}

HAND2["macos2"] = """
make the dock show only running apps
remove the delay when the dock auto hides
show the full path in the Finder title bar
show all file extensions in Finder
make key repeat faster
disable the press and hold accent popup
stop Photos from opening when I plug in my phone
show the ~/Library folder in Finder
set the screenshot filename prefix to shot
disable the shadow on window screenshots
restart the menu bar
restart coreaudio when sound stops working
show which apps are using the most energy
show the sleep settings
set the display to sleep after 10 minutes
disable sleep entirely while on ac power
schedule the mac to wake up at 7am every weekday
show the battery health condition
show the power adapter wattage
list usb devices connected to this mac
show thunderbolt devices
show the gpu model
show the model identifier of this mac
show the boot volume uuid
check if system integrity protection is enabled
install the xcode command line tools
show the path of the active developer directory
list installed package receipts
forget a package receipt for com.example.pkg
show the metadata of a file with mdls
find all pdf files spotlight knows about
find files spotlight tagged as Red
convert a binary plist to xml
read the value of a defaults key for the dock
delete a defaults key
show all defaults domains
open the current folder in Terminal
open a file in TextEdit
open a url in the default browser
open System Settings to the network pane
show the frontmost application with osascript
display a dialog box from the terminal
get the current volume level with osascript
empty the trash securely
show the size of the trash
eject all external disks
list apfs volumes
show the free space in the apfs container
repair a disk with diskutil
unmount a volume by name
create a 500MB ram disk
convert a dmg to an iso
create an encrypted disk image
list keychains
find a saved password in the keychain for github.com
add a generic password to the keychain
show certificates in the system keychain
show all users on this mac
add a user to the admin group
show the groups a user belongs to
show my user id
run a shortcut named Morning from the terminal
list all shortcuts
show the name of the current wifi network
turn bluetooth off
show the current proxy settings
set a web proxy on wi-fi
turn off the web proxy
show the search domains for wi-fi
list all network services
show the hardware ports
show the computer's hostname, local hostname and computer name
show the current dns configuration
show the system log live
show logs from a specific process in the last 10 minutes
show crash reports
convert a text file to rtf with textutil
convert a word document to plain text
show audio file information with afinfo
convert an aiff file to m4a with afconvert
list the voices available to say
read a text file aloud
save spoken text to an audio file
show the number of cpu threads with sysctl
show the cpu brand string
show the page size and memory statistics
show disk io statistics
show the kernel version
show the boot time of the mac
"""

HAND2["dev"] = """
build a swift package in release mode
run swift tests
create a new swift executable package
format swift files with swift-format
list available ios simulators
boot the iPhone 15 simulator
build an xcode project from the command line
clean derived data
create a new rust project called hello
build a rust project in release mode
run cargo tests
run clippy on a rust project
format rust code
update rust with rustup
initialize a go module named example.com/app
build a go program for linux amd64
run go tests with coverage
download go module dependencies
tidy go modules
compile hello.c with clang
compile a c++ file with c++17
build a cmake project in a build directory
run make with 8 parallel jobs
compile java files and run the main class
run a jar file
build a maven project skipping tests
run a gradle build
install a ruby gem
install gems from a Gemfile
create a new rails app
run rake tasks list
install composer dependencies
start the php built-in server on port 8080
run a deno script with network access
install dependencies with bun
run a bun script
create a pull request with the gh cli
list open pull requests with gh
clone a repo with gh
view issues assigned to me with gh
check out a pull request locally with gh
create a new github repo from the current folder with gh
show the status of github actions runs
run shellcheck on a script
run a bash script in debug mode
check the syntax of a bash script without running it
generate ctags for a project
count lines of code per language with cloc
search code fast with ripgrep
find files by name with fd
fuzzy find a file with fzf
open a file in vim at line 42
open two files side by side in vim
edit a file with nano
replace text in a file with perl
start a local postgres with docker
run a redis server in docker
start a jupyter lab
run a flask app in debug mode
run a django development server
apply django migrations
create a django superuser
run uvicorn with reload
run a fastapi app on port 8000
start a next.js dev server
build a next.js app for production
run storybook
lint css with stylelint
minify a javascript file with terser
bundle with esbuild
serve a static folder with npx serve
generate an openapi client
run terraform fmt
validate terraform files
destroy terraform infrastructure
list helm releases
install a helm chart
upgrade a helm release
start minikube
create a kind cluster
run an ansible playbook
ping all hosts with ansible
"""

HAND2["text2"] = """
print the lines that are common to two sorted files
join two csv files on the first column
show columns 2 and 5 of a tab separated file
fold long lines at 72 characters
reformat paragraphs in a text file
number only the non-empty lines
reverse each line of a file character by character
print a file backwards
print 5 random lines from a file
split a file at every line matching a pattern
convert a file from utf-16 to utf-8
convert dos line endings with dos2unix
convert spaces to tabs
show printable strings in a binary file
dump a file in octal
show a hex dump of a file with xxd
convert hex back to binary with xxd
generate the numbers 1 to 10 separated by commas
format a number with thousands separators
print today's date as YYYY-MM-DD
print the date 7 days ago
print the day of the week for a date
convert a date to a unix timestamp
add two numbers in the shell
do floating point math with bc
compute the square root of 2 with bc
convert decimal 255 to hex
convert hex ff to decimal
convert a number to binary
print a string 10 times
pad a number with leading zeros
get the length of a string variable
split a string on commas in zsh
trim whitespace from a string
check if a string contains a substring
generate a random number between 1 and 100
lowercase a variable in zsh
read a password without echo
prompt the user for input in a script
loop over the lines of a file
loop over numbers from 1 to 5
iterate over files with spaces in their names
find and replace text in many files using perl
count the frequency of each character in a file
print the second to last field of each line
print lines where the third column is greater than 100
print the sum of a column grouped by the first column
remove lines longer than 80 characters
delete trailing blank lines at the end of a file
insert a line after a matching line with sed
print everything after the first colon on each line
extract the domain from a list of urls
extract quoted strings from a file
remove html tags from a file
decode html entities
extract the title from an html page
pretty print xml
query xml with xmllint
convert yaml to json
validate a json file
minify a json file with jq
sort the keys of a json file
merge two json files with jq
count the lines in a gzip file without extracting
grep inside gzip compressed logs
search a pdf for a word
"""

HAND2["files2"] = """
find files modified after a reference file
find files newer than 2 days but older than 5 days
find files with the setuid bit
find directories named build and delete them
find files and exclude the .git directory
find files by size range between 10MB and 100MB
find hard links to a file
find files with no owner
list the largest subdirectories two levels deep
show the total size of all files matching *.log
copy files listed in a text file to a folder
move files in parallel with xargs
run a command on every file found by find
rename files using a counter
rename file extensions from .txt to .md
change the case of extensions to lowercase
flatten a directory tree into one folder
create a directory for each line in a file
create dated folders for every day of this month
copy a file to multiple directories at once
duplicate a directory structure without the files
compare two folders and show only the differences
sync two folders both ways
back up a file before editing it
restore a file from a time machine backup
lock a file so it can't be changed
make a file immutable
hide a file in Finder
show the creation date of a file
change a file's modification date to a specific date
set the same timestamps on all files in a folder
show how many inodes are used
show the block size of the filesystem
zip each subfolder into its own archive
create a split zip archive of 100MB parts
test the integrity of a zip file
extract a tar archive to a different directory
add a file to an existing zip
remove a file from a zip archive
list the files in a 7z archive
extract a rar file
compress a file with zstd
decompress a .zst file
compress a file with brotli
create an archive with maximum compression
show compression ratio of a gzip file
"""

HAND2["net2"] = """
show the route packets take with mtr
measure bandwidth to another machine with iperf3
capture packets on port 443 with tcpdump
capture http traffic to a file
show the open sockets with lsof
show my router's ip
get the ip address assigned by dhcp
show the subnet mask of en0
show the dns server used by the system
query txt records with dig short output
query a domain using cloudflare dns
find the authoritative name servers for a domain
check if a port is reachable with curl
show the time a website takes to respond
download a file with wget in the background
download files listed in a text file with wget
limit download speed with curl
send form data with curl
upload a file with curl PUT
follow a stream of server sent events with curl
use a proxy with curl
ignore ssl errors with curl
save cookies with curl
send a request with a custom user agent
test an http/2 connection
make a request with httpie
show my public ip using a dns query
check the open ports of a remote host with nc
transfer a directory over ssh with tar
run a remote command over ssh
keep an ssh connection alive
remove a host from known_hosts
show the ssh config for a host
create an ssh config entry for a server
mount a remote folder with sshfs
start a socks proxy over ssh
reverse ssh tunnel to expose a local port
show the certificate chain of a website
check if a website supports tls 1.3
reload nginx
test the nginx configuration
show the status of apache
"""

HAND2["linux"] = """
update the package lists on ubuntu
install nginx on ubuntu
remove a package and its config files on debian
list installed packages on ubuntu
search for a package with apt
install a package on fedora
check the status of the nginx service
restart the ssh service
enable a service to start at boot
show the logs of a systemd service
follow journal logs
show the kernel messages
show free memory in human readable form
list block devices
mount a usb drive on linux
show the linux distribution version
add a user on linux
add a user to the sudo group
change a user's password
allow port 22 through ufw
show iptables rules
show listening ports with ss
show ip addresses with the ip command
show the cpu info on linux
show the uptime and load on linux
set the hostname on linux
create a swap file
check disk health with smartctl
find which package owns a file on debian
schedule a job with at
"""

HAND2["cloud2"] = """
list lambda functions with the aws cli
invoke a lambda function
tail cloudwatch logs for a log group
list ec2 instances that are running
stop an ec2 instance
create an s3 bucket
make an s3 object public
generate a presigned url for an s3 object
delete an s3 bucket and everything in it
show the size of an s3 bucket
set the default aws region
switch aws profiles
log in to ecr with docker
list gcloud compute instances
ssh into a gcloud instance
set the gcloud project
deploy to cloud run
list azure vms
log in to azure cli
get kubernetes pods sorted by restart count
show resource usage of pods
show the yaml of a running deployment
roll back a kubernetes deployment
create a configmap from a file
create a secret from literal values
copy a file into a kubernetes pod
watch pods in real time
delete all evicted pods
get the external ip of a service
drain a kubernetes node
"""

HAND2["git2"] = """
create a worktree for a branch
list worktrees
set up sparse checkout for one folder
track large files with git lfs
show blame for lines 10 to 20 of a file
find commits that added or removed a string
show a short log grouped by author
describe the current commit with the nearest tag
export the repo as a zip archive
show the stashed changes as a diff
push with force-with-lease
fetch all tags from origin
list the remote refs
show tracking branches with ahead and behind counts
merge a branch without fast forward
revert a merge commit
interactively rebase the last 5 commits
mark a file as assume unchanged
move a file in git
show the files changed between two branches
show the number of lines changed per file in the last commit
show commits that touched a directory
show the first commit of the repo
count the total number of commits
show the date of the last commit
list authors of the repository
show unpushed commits
show commits not yet merged into main
delete all local branches that are merged
delete local branches whose remote is gone
update the current branch from main with rebase
pull all submodules recursively
clone with submodules
show the url of the origin remote
rename a remote
add all files except one
commit only part of a file
show the diff of the last commit
show a file as it was 3 commits ago
restore a deleted file from git history
"""

HAND2["media2"] = """
resize an image to exactly 1024x768 with imagemagick
crop the center 500x500 of an image with imagemagick
add a text caption to an image
create a contact sheet from images
optimize png files with pngquant
losslessly optimize jpgs with jpegoptim
optimize a gif with gifsicle
convert an svg to png
convert a pdf page to a high resolution png
extract text from a pdf
merge pdfs with qpdf
remove the password from a pdf
rotate all pages of a pdf
extract pages 3 to 5 from a pdf
compress a pdf with ghostscript
convert an audio file to 16khz mono wav
trim silence from the start of an audio file with sox
concatenate audio files with sox
change the speed of audio without changing pitch
extract subtitles from a video to srt
hardcode subtitles into a video
change a video container from mov to mkv
add padding to make a video 16:9
fade out the last 3 seconds of audio in a video
create a timelapse from a video
stabilize a shaky video
deinterlace a video
split a video into hls segments
stream a video file to an rtmp server
capture the webcam to a file with ffmpeg
list avfoundation devices with ffmpeg
extract keyframes from a video
make a video from a single image and an audio file
create a boomerang effect video
take a screenshot of a webpage with chrome headless
rename photos by their exif date with exiftool
set the gps location in a photo's metadata
remove all metadata from a video
show the frame count of a video
show the audio sample rate of a file
"""

HAND2["casual"] = """
whats eating my disk space
my mac is slow what's using cpu
how do i see whats on port 80
clear my dns
nuke node_modules
undo my last push
oops i committed to main instead of a branch
make this script runnable
see hidden stuff in this folder
how big is this folder
show me everything in this folder including hidden
whats my mac's name
go home
open finder here
copy the path of this folder
what time is it in tokyo
how long is this video
how many pages does this pdf have
is google up
whats my local ip
list wifi networks
forget a wifi network
turn off my screen
keep my laptop awake
stop the music
make a new folder called stuff
delete the folder stuff and everything in it
copy everything from a to b
backup my home folder
what shell am i using
reload my shell config
show me my aliases
where is python
which java am i using
start a server here
open localhost 3000 in the browser
kill whatever is on 8000
restart docker
what containers are running
clean up docker images
update everything with brew
what did i just install with brew
show the last command i ran
find my ssh key
print my public key
"""
